"""
utils/graph_obs_utils.py

Utilities for converting CascadingMitigationEnv observations into tensors for
Graph-DQN.

This file is designed for the first Graph-DQN version, where we do NOT use
PyTorch Geometric. Instead, we use dense adjacency matrices and implement a
simple GCN-style encoder with matrix multiplication.

Expected observation format from CascadingMitigationEnv:
    obs = {
        "node_features": np.ndarray,      # [N, F_node]
        "adj_matrix": np.ndarray,         # [N, N]
        "edge_index": np.ndarray,         # [2, E]
        "edge_features": np.ndarray,      # [E, F_edge]
        "active_edge_mask": np.ndarray,   # [E]
        "action_mask": np.ndarray,        # [A], 1 = valid, 0 = invalid
        "global_features": np.ndarray,    # [F_global]
        ...
    }

Main functions:
    - obs_to_graph_tensor(obs, device)
    - batch_graph_obs(obs_list, device)
    - normalize_adj(adj)
    - move_batch_to_device(batch, device)
    - copy_obs_to_numpy(obs)
    - infer_graph_obs_dims(obs)

Notes:
    1. For Graph-DQN V1, use --no_disconnect during training.
       The environment action space may still contain edge action slots, but
       edge-disconnection actions are invalid and action_mask will mask them.

    2. For batching, all observations in a batch must have the same shapes:
       same N, same action_dim, and same number of edges E. This is true for
       BA/WS graphs with fixed N and parameters. ER graphs may have different E
       across seeds, so avoid ER for the first Graph-DQN version unless you add
       padding.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch


ArrayLike = np.ndarray | torch.Tensor


# -----------------------------------------------------------------------------
# Basic conversion helpers
# -----------------------------------------------------------------------------
def to_numpy(x: Any, dtype: np.dtype | type | None = None) -> np.ndarray:
    """
    Convert an object to a numpy array.

    Parameters
    ----------
    x:
        Input object. Can be numpy array, torch tensor, list, etc.
    dtype:
        Optional numpy dtype.

    Returns
    -------
    np.ndarray
        Converted numpy array.
    """
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    if dtype is not None:
        arr = arr.astype(dtype)

    return arr


def to_tensor(
    x: Any,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Convert an object to a torch tensor.
    """
    if isinstance(x, torch.Tensor):
        tensor = x.to(dtype=dtype)
    else:
        tensor = torch.tensor(x, dtype=dtype)

    if device is not None:
        tensor = tensor.to(device)

    return tensor


def copy_obs_to_numpy(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Deep-copy an environment observation into numpy arrays.

    This is useful for replay buffers. Without copying, later environment steps
    may mutate arrays referenced by old transitions.

    Parameters
    ----------
    obs:
        Observation dictionary from the environment.

    Returns
    -------
    dict[str, np.ndarray]
        Copied observation dictionary.
    """
    copied = {}
    for key, value in obs.items():
        copied[key] = np.array(to_numpy(value), copy=True)
    return copied


# -----------------------------------------------------------------------------
# Shape checking
# -----------------------------------------------------------------------------
def _require_key(obs: dict[str, Any], key: str) -> None:
    if key not in obs:
        raise KeyError(
            f"Observation is missing key '{key}'. Available keys: {list(obs.keys())}"
        )


def validate_graph_obs(obs: dict[str, Any]) -> None:
    """
    Validate that an observation has the fields needed by Graph-DQN.

    Raises
    ------
    KeyError or ValueError
        If required keys are missing or shapes are inconsistent.
    """
    required_keys = [
        "node_features",
        "adj_matrix",
        "action_mask",
        "global_features",
    ]

    for key in required_keys:
        _require_key(obs, key)

    x = to_numpy(obs["node_features"])
    adj = to_numpy(obs["adj_matrix"])
    action_mask = to_numpy(obs["action_mask"])
    global_features = to_numpy(obs["global_features"])

    if x.ndim != 2:
        raise ValueError(f"node_features must have shape [N, F], got {x.shape}.")

    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adj_matrix must have shape [N, N], got {adj.shape}.")

    if adj.shape[0] != x.shape[0]:
        raise ValueError(
            f"node_features and adj_matrix disagree on N: "
            f"node_features={x.shape}, adj_matrix={adj.shape}."
        )

    if action_mask.ndim != 1:
        raise ValueError(f"action_mask must have shape [A], got {action_mask.shape}.")

    if global_features.ndim != 1:
        raise ValueError(
            f"global_features must have shape [F_global], got {global_features.shape}."
        )

    if "edge_index" in obs:
        edge_index = to_numpy(obs["edge_index"])
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must have shape [2, E], got {edge_index.shape}."
            )

    if "edge_features" in obs:
        edge_features = to_numpy(obs["edge_features"])
        if edge_features.ndim != 2:
            raise ValueError(
                f"edge_features must have shape [E, F_edge], got {edge_features.shape}."
            )


# -----------------------------------------------------------------------------
# Adjacency normalization
# -----------------------------------------------------------------------------
def normalize_adj(
    adj: torch.Tensor,
    add_self_loops: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Symmetrically normalize adjacency matrix for GCN.

    Formula:
        A_hat = A + I
        A_norm = D^{-1/2} A_hat D^{-1/2}

    Supports:
        adj: [N, N]
        adj: [B, N, N]

    Parameters
    ----------
    adj:
        Dense adjacency tensor.
    add_self_loops:
        Whether to add identity matrix.
    eps:
        Small value for numerical stability.

    Returns
    -------
    torch.Tensor
        Normalized adjacency tensor with the same shape as input.
    """
    if adj.ndim not in (2, 3):
        raise ValueError(f"adj must have shape [N,N] or [B,N,N], got {adj.shape}.")

    original_ndim = adj.ndim

    if original_ndim == 2:
        adj_work = adj.unsqueeze(0)
    else:
        adj_work = adj

    adj_work = adj_work.float()
    batch_size, num_nodes, _ = adj_work.shape

    if add_self_loops:
        eye = torch.eye(num_nodes, dtype=adj_work.dtype, device=adj_work.device)
        eye = eye.unsqueeze(0).expand(batch_size, num_nodes, num_nodes)
        adj_work = adj_work + eye

    degree = adj_work.sum(dim=-1)  # [B, N]
    degree_inv_sqrt = torch.pow(degree + eps, -0.5)

    adj_norm = (
        degree_inv_sqrt.unsqueeze(-1)
        * adj_work
        * degree_inv_sqrt.unsqueeze(-2)
    )

    if original_ndim == 2:
        return adj_norm.squeeze(0)

    return adj_norm


# -----------------------------------------------------------------------------
# Single-observation conversion
# -----------------------------------------------------------------------------
def obs_to_graph_tensor(
    obs: dict[str, Any],
    device: str | torch.device | None = None,
    add_batch_dim: bool = False,
    normalize_adjacency: bool = True,
    add_self_loops: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Convert one environment observation into tensors for Graph-DQN.

    Parameters
    ----------
    obs:
        Observation dictionary from CascadingMitigationEnv.
    device:
        Torch device.
    add_batch_dim:
        If True, return tensors with batch dimension:
            x: [1, N, F]
            adj: [1, N, N]
            action_mask: [1, A]
        If False, return:
            x: [N, F]
            adj: [N, N]
            action_mask: [A]
    normalize_adjacency:
        Whether to include normalized adjacency as 'adj_norm'.
    add_self_loops:
        Whether normalize_adj adds self-loops.

    Returns
    -------
    dict[str, torch.Tensor]
        Tensor dictionary.
    """
    validate_graph_obs(obs)

    x = to_tensor(obs["node_features"], dtype=torch.float32, device=device)
    adj = to_tensor(obs["adj_matrix"], dtype=torch.float32, device=device)
    global_features = to_tensor(
        obs["global_features"], dtype=torch.float32, device=device
    )
    action_mask = to_tensor(obs["action_mask"], dtype=torch.bool, device=device)

    graph_tensor: dict[str, torch.Tensor] = {
        "x": x,
        "adj": adj,
        "global_features": global_features,
        "action_mask": action_mask,
    }

    if normalize_adjacency:
        graph_tensor["adj_norm"] = normalize_adj(
            adj,
            add_self_loops=add_self_loops,
        )

    if "edge_index" in obs:
        graph_tensor["edge_index"] = to_tensor(
            obs["edge_index"], dtype=torch.long, device=device
        )

    if "edge_features" in obs:
        graph_tensor["edge_features"] = to_tensor(
            obs["edge_features"], dtype=torch.float32, device=device
        )

    if "active_edge_mask" in obs:
        graph_tensor["active_edge_mask"] = to_tensor(
            obs["active_edge_mask"], dtype=torch.bool, device=device
        )

    if add_batch_dim:
        for key in list(graph_tensor.keys()):
            value = graph_tensor[key]

            # edge_index is [2, E]; for dense-adj Graph-DQN V1 we usually do not
            # need it. If batched, keep it as [1, 2, E] for possible edge heads.
            graph_tensor[key] = value.unsqueeze(0)

    return graph_tensor


# -----------------------------------------------------------------------------
# Batch conversion
# -----------------------------------------------------------------------------
def _check_batch_shapes(obs_list: list[dict[str, Any]]) -> None:
    if len(obs_list) == 0:
        raise ValueError("obs_list is empty.")

    first = obs_list[0]
    validate_graph_obs(first)

    first_shapes = {
        "node_features": to_numpy(first["node_features"]).shape,
        "adj_matrix": to_numpy(first["adj_matrix"]).shape,
        "global_features": to_numpy(first["global_features"]).shape,
        "action_mask": to_numpy(first["action_mask"]).shape,
    }

    optional_keys = ["edge_index", "edge_features", "active_edge_mask"]
    for key in optional_keys:
        if key in first:
            first_shapes[key] = to_numpy(first[key]).shape

    for idx, obs in enumerate(obs_list[1:], start=1):
        validate_graph_obs(obs)

        for key, shape in first_shapes.items():
            if key not in obs:
                raise KeyError(f"obs_list[{idx}] is missing key '{key}'.")

            current_shape = to_numpy(obs[key]).shape
            if current_shape != shape:
                raise ValueError(
                    "All observations in a graph batch must have the same shapes. "
                    f"Mismatch at obs_list[{idx}] key='{key}': "
                    f"expected {shape}, got {current_shape}.\n"
                    "This often happens with ER graphs because the number of edges "
                    "can vary across seeds. For Graph-DQN V1, prefer BA or WS graphs "
                    "with fixed N and fixed parameters, or implement padding."
                )


def batch_graph_obs(
    obs_list: Iterable[dict[str, Any]],
    device: str | torch.device | None = None,
    normalize_adjacency: bool = True,
    add_self_loops: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Convert a list of observations into a batched graph tensor dictionary.

    Parameters
    ----------
    obs_list:
        List of observations.
    device:
        Torch device.
    normalize_adjacency:
        Whether to compute batched normalized adjacency.
    add_self_loops:
        Whether normalize_adj adds self-loops.

    Returns
    -------
    dict[str, torch.Tensor]
        Batched tensors:
            x: [B, N, F_node]
            adj: [B, N, N]
            adj_norm: [B, N, N]
            global_features: [B, F_global]
            action_mask: [B, A]
            edge_index: [B, 2, E] if available
            edge_features: [B, E, F_edge] if available
            active_edge_mask: [B, E] if available
    """
    obs_list = list(obs_list)
    _check_batch_shapes(obs_list)

    x = np.stack([to_numpy(obs["node_features"], np.float32) for obs in obs_list])
    adj = np.stack([to_numpy(obs["adj_matrix"], np.float32) for obs in obs_list])
    global_features = np.stack(
        [to_numpy(obs["global_features"], np.float32) for obs in obs_list]
    )
    action_mask = np.stack(
        [to_numpy(obs["action_mask"], np.bool_) for obs in obs_list]
    )

    batch: dict[str, torch.Tensor] = {
        "x": to_tensor(x, dtype=torch.float32, device=device),
        "adj": to_tensor(adj, dtype=torch.float32, device=device),
        "global_features": to_tensor(
            global_features, dtype=torch.float32, device=device
        ),
        "action_mask": to_tensor(action_mask, dtype=torch.bool, device=device),
    }

    if normalize_adjacency:
        batch["adj_norm"] = normalize_adj(
            batch["adj"],
            add_self_loops=add_self_loops,
        )

    if "edge_index" in obs_list[0]:
        edge_index = np.stack(
            [to_numpy(obs["edge_index"], np.int64) for obs in obs_list]
        )
        batch["edge_index"] = to_tensor(edge_index, dtype=torch.long, device=device)

    if "edge_features" in obs_list[0]:
        edge_features = np.stack(
            [to_numpy(obs["edge_features"], np.float32) for obs in obs_list]
        )
        batch["edge_features"] = to_tensor(
            edge_features, dtype=torch.float32, device=device
        )

    if "active_edge_mask" in obs_list[0]:
        active_edge_mask = np.stack(
            [to_numpy(obs["active_edge_mask"], np.bool_) for obs in obs_list]
        )
        batch["active_edge_mask"] = to_tensor(
            active_edge_mask, dtype=torch.bool, device=device
        )

    return batch


# -----------------------------------------------------------------------------
# Device movement
# -----------------------------------------------------------------------------
def move_batch_to_device(
    batch: dict[str, Any],
    device: str | torch.device,
) -> dict[str, Any]:
    """
    Move all tensors in a batch dictionary to a device.

    Non-tensor values are kept unchanged.
    """
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


# -----------------------------------------------------------------------------
# Dimension inference and action helpers
# -----------------------------------------------------------------------------
def infer_graph_obs_dims(obs: dict[str, Any]) -> dict[str, int]:
    """
    Infer dimensions from a single observation.

    Returns
    -------
    dict[str, int]
        {
            "num_nodes": N,
            "node_feature_dim": F_node,
            "global_feature_dim": F_global,
            "action_dim": A,
            "num_edges": E,
            "edge_feature_dim": F_edge,
            "do_nothing_action_id": A - 1
        }

    Note
    ----
    In the current environment, do-nothing action id is the last action:
        do_nothing_action_id = N + E = A - 1
    """
    validate_graph_obs(obs)

    x = to_numpy(obs["node_features"])
    global_features = to_numpy(obs["global_features"])
    action_mask = to_numpy(obs["action_mask"])

    num_nodes = int(x.shape[0])
    node_feature_dim = int(x.shape[1])
    global_feature_dim = int(global_features.shape[0])
    action_dim = int(action_mask.shape[0])

    if "edge_features" in obs:
        edge_features = to_numpy(obs["edge_features"])
        num_edges = int(edge_features.shape[0])
        edge_feature_dim = int(edge_features.shape[1])
    elif "edge_index" in obs:
        edge_index = to_numpy(obs["edge_index"])
        num_edges = int(edge_index.shape[1])
        edge_feature_dim = 0
    else:
        num_edges = max(action_dim - num_nodes - 1, 0)
        edge_feature_dim = 0

    return {
        "num_nodes": num_nodes,
        "node_feature_dim": node_feature_dim,
        "global_feature_dim": global_feature_dim,
        "action_dim": action_dim,
        "num_edges": num_edges,
        "edge_feature_dim": edge_feature_dim,
        "do_nothing_action_id": action_dim - 1,
    }


def get_valid_action_indices(obs: dict[str, Any]) -> np.ndarray:
    """
    Return valid action indices from an observation.
    """
    _require_key(obs, "action_mask")
    action_mask = to_numpy(obs["action_mask"], np.int32)
    return np.where(action_mask == 1)[0]


def get_valid_node_action_indices(obs: dict[str, Any]) -> np.ndarray:
    """
    Return valid protect-node action indices.

    In the current action encoding:
        0 ... N-1 = protect node i
    """
    dims = infer_graph_obs_dims(obs)
    valid_actions = get_valid_action_indices(obs)
    return valid_actions[valid_actions < dims["num_nodes"]]


def get_do_nothing_action_id_from_obs(obs: dict[str, Any]) -> int:
    """
    Return do-nothing action id from observation shape.
    """
    return infer_graph_obs_dims(obs)["do_nothing_action_id"]


# -----------------------------------------------------------------------------
# Debug helper
# -----------------------------------------------------------------------------
def summarize_graph_obs(obs: dict[str, Any]) -> str:
    """
    Return a concise human-readable summary of an observation.
    """
    dims = infer_graph_obs_dims(obs)
    valid_actions = get_valid_action_indices(obs)
    valid_nodes = get_valid_node_action_indices(obs)

    failed_count = None
    if "failed_mask" in obs:
        failed_count = int(np.sum(to_numpy(obs["failed_mask"], np.int32)))

    protected_count = None
    if "protected_mask" in obs:
        protected_count = int(np.sum(to_numpy(obs["protected_mask"], np.int32)))

    parts = [
        f"N={dims['num_nodes']}",
        f"E={dims['num_edges']}",
        f"node_feature_dim={dims['node_feature_dim']}",
        f"global_feature_dim={dims['global_feature_dim']}",
        f"action_dim={dims['action_dim']}",
        f"valid_actions={len(valid_actions)}",
        f"valid_node_actions={len(valid_nodes)}",
        f"do_nothing_id={dims['do_nothing_action_id']}",
    ]

    if failed_count is not None:
        parts.append(f"failed_count={failed_count}")

    if protected_count is not None:
        parts.append(f"protected_count={protected_count}")

    return "GraphObs(" + ", ".join(parts) + ")"
