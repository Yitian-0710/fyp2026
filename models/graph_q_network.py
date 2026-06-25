"""
models/graph_q_network.py

Graph Q-network for Graph-DQN V1.

Graph-DQN V1 focuses on:
    - protect node i actions
    - do-nothing action

It intentionally does NOT learn meaningful Q-values for edge-disconnection
actions yet. If the environment was created with allow_disconnect=False
(--no_disconnect), edge-disconnection actions are invalid and will be masked by
agent.update() and agent.take_action(). This lets us keep the environment's full
action dimension [N + E + 1] while training a node-action Graph-DQN.

Expected action encoding from CascadingMitigationEnv:
    0 ... N-1      : protect node i
    N ... N+E-1    : disconnect edge e  [placeholder in V1]
    N+E            : do nothing

Main output:
    q_all: [B, action_dim]

Also returns optional details:
    node_q:        [B, N]
    edge_q:        [B, E] placeholder zeros in V1
    do_nothing_q:  [B, 1]
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .graph_encoder import DenseGraphEncoder


class MLPHead(nn.Module):
    """
    Small MLP head used for node-action and graph-action Q-values.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphQNetwork(nn.Module):
    """
    Graph-DQN Q-network with node-action and do-nothing heads.

    Parameters
    ----------
    node_feature_dim:
        Dimension of node features.
    global_feature_dim:
        Dimension of global graph features.
    hidden_dim:
        Hidden dimension for GNN and Q heads.
    action_dim:
        Full environment action dimension. In current environment this is N+E+1.
    num_nodes:
        Number of nodes N. If None, it is inferred from the input batch at
        runtime. Passing it is recommended for safety.
    num_gnn_layers:
        Number of dense GCN layers.
    dropout:
        Dropout probability.
    pooling:
        Graph pooling type: "mean", "sum", "max", or "mean_max".
    use_global_features:
        Whether to fuse global features into graph embedding.
    edge_placeholder_value:
        Q-value used for edge action slots in V1. It does not matter when edge
        actions are invalid and masked. Use 0.0 by default.
    """

    def __init__(
        self,
        node_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int = 128,
        action_dim: int | None = None,
        num_nodes: int | None = None,
        num_gnn_layers: int = 2,
        dropout: float = 0.0,
        pooling: str = "mean",
        use_global_features: bool = True,
        edge_placeholder_value: float = 0.0,
    ):
        super().__init__()

        self.node_feature_dim = int(node_feature_dim)
        self.global_feature_dim = int(global_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = None if action_dim is None else int(action_dim)
        self.num_nodes = None if num_nodes is None else int(num_nodes)
        self.edge_placeholder_value = float(edge_placeholder_value)

        self.encoder = DenseGraphEncoder(
            node_feature_dim=self.node_feature_dim,
            hidden_dim=self.hidden_dim,
            num_layers=num_gnn_layers,
            global_feature_dim=self.global_feature_dim,
            pooling=pooling,
            dropout=dropout,
            activation="relu",
            layer_norm=True,
            residual=True,
            use_global_features=use_global_features,
        )

        # Q(protect node i) uses both local node embedding and global graph state.
        self.node_q_head = MLPHead(
            input_dim=2 * self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=1,
            dropout=dropout,
        )

        # Q(do nothing) uses graph-level embedding only.
        self.do_nothing_head = MLPHead(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=1,
            dropout=dropout,
        )

    def _infer_dims_from_batch(self, batch: dict[str, torch.Tensor]) -> tuple[int, int, int]:
        """
        Infer batch_size, num_nodes, and action_dim from input batch.
        """
        x = batch["x"]
        if x.ndim == 2:
            batch_size = 1
            num_nodes = int(x.shape[0])
        elif x.ndim == 3:
            batch_size = int(x.shape[0])
            num_nodes = int(x.shape[1])
        else:
            raise ValueError(f"batch['x'] must be [N,F] or [B,N,F], got {tuple(x.shape)}")

        if "action_mask" in batch:
            action_mask = batch["action_mask"]
            if action_mask.ndim == 1:
                action_dim = int(action_mask.shape[0])
            elif action_mask.ndim == 2:
                action_dim = int(action_mask.shape[1])
            else:
                raise ValueError(
                    f"batch['action_mask'] must be [A] or [B,A], got {tuple(action_mask.shape)}"
                )
        elif self.action_dim is not None:
            action_dim = self.action_dim
        else:
            # V1 fallback: no edge actions, only N node actions + do-nothing.
            action_dim = num_nodes + 1

        if self.num_nodes is not None and num_nodes != self.num_nodes:
            raise ValueError(
                f"Input num_nodes={num_nodes} differs from model num_nodes={self.num_nodes}."
            )

        if self.action_dim is not None and action_dim != self.action_dim:
            raise ValueError(
                f"Input action_dim={action_dim} differs from model action_dim={self.action_dim}."
            )

        return batch_size, num_nodes, action_dim

    @staticmethod
    def _build_graph_node_context(
        node_emb: torch.Tensor,
        graph_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatenate each node embedding with the graph embedding.

        Parameters
        ----------
        node_emb:
            [B, N, H]
        graph_emb:
            [B, H]

        Returns
        -------
        torch.Tensor
            [B, N, 2H]
        """
        batch_size, num_nodes, _ = node_emb.shape
        graph_context = graph_emb.unsqueeze(1).expand(batch_size, num_nodes, -1)
        return torch.cat([node_emb, graph_context], dim=-1)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        return_details: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Compute Q-values for all environment actions.

        Parameters
        ----------
        batch:
            Tensor batch from graph_obs_utils.batch_graph_obs.
        return_details:
            If False, return q_all directly.
            If True, return a dictionary with q_all, node_q, edge_q,
            do_nothing_q, node_emb, and graph_emb.

        Returns
        -------
        torch.Tensor or dict[str, torch.Tensor]
            q_all shape [B, action_dim], or details dictionary.
        """
        batch_size, num_nodes, action_dim = self._infer_dims_from_batch(batch)

        node_emb, graph_emb = self.encoder(batch)  # [B,N,H], [B,H]

        node_context = self._build_graph_node_context(node_emb, graph_emb)
        node_q = self.node_q_head(node_context).squeeze(-1)  # [B, N]

        do_nothing_q = self.do_nothing_head(graph_emb)  # [B, 1]

        edge_slot_dim = action_dim - num_nodes - 1
        if edge_slot_dim < 0:
            raise ValueError(
                f"action_dim={action_dim} is smaller than num_nodes+1={num_nodes + 1}."
            )

        if edge_slot_dim > 0:
            edge_q = torch.full(
                (batch_size, edge_slot_dim),
                fill_value=self.edge_placeholder_value,
                dtype=node_q.dtype,
                device=node_q.device,
            )
            q_all = torch.cat([node_q, edge_q, do_nothing_q], dim=-1)
        else:
            edge_q = torch.empty(
                (batch_size, 0),
                dtype=node_q.dtype,
                device=node_q.device,
            )
            q_all = torch.cat([node_q, do_nothing_q], dim=-1)

        if q_all.shape != (batch_size, action_dim):
            raise RuntimeError(
                f"q_all shape mismatch: expected {(batch_size, action_dim)}, "
                f"got {tuple(q_all.shape)}."
            )

        if not return_details:
            return q_all

        return {
            "q_all": q_all,
            "node_q": node_q,
            "edge_q": edge_q,
            "do_nothing_q": do_nothing_q,
            "node_emb": node_emb,
            "graph_emb": graph_emb,
        }

    def masked_q_values(
        self,
        batch: dict[str, torch.Tensor],
        invalid_value: float = -1e9,
    ) -> torch.Tensor:
        """
        Compute Q-values with invalid actions masked.

        This is mainly a convenience function for action selection.
        During DQN target computation, the agent can call forward() and apply
        masking itself.
        """
        q_all = self.forward(batch, return_details=False)

        if "action_mask" not in batch:
            return q_all

        action_mask = batch["action_mask"].bool()
        if action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)

        return q_all.masked_fill(~action_mask, invalid_value)


# Backward-compatible alias.
GraphDQNQNetwork = GraphQNetwork
