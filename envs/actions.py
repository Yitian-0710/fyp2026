"""
actions.py

Action definition and action masking for cascading failure mitigation.

Action encoding:
    0 ... N-1          : protect node i
    N ... N+E-1        : disconnect edge e
    N+E                : do nothing
"""

from __future__ import annotations

from enum import Enum
from typing import List, Tuple

import numpy as np


class ActionType(Enum):
    PROTECT_NODE = 0
    DISCONNECT_EDGE = 1
    DO_NOTHING = 2


def get_do_nothing_action_id(num_nodes: int, num_edges: int) -> int:
    """
    Return the action id of do-nothing action.
    """
    return int(num_nodes + num_edges)


def decode_action(
    action_id: int,
    num_nodes: int,
    edge_list: List[Tuple[int, int]],
) -> tuple[ActionType, int | None]:
    """
    Decode integer action id into action type and target index.

    For PROTECT_NODE, target is node id.
    For DISCONNECT_EDGE, target is edge id.
    For DO_NOTHING, target is None.
    """
    action_id = int(action_id)
    num_edges = len(edge_list)
    do_nothing_id = get_do_nothing_action_id(num_nodes, num_edges)

    if 0 <= action_id < num_nodes:
        return ActionType.PROTECT_NODE, action_id

    if num_nodes <= action_id < num_nodes + num_edges:
        return ActionType.DISCONNECT_EDGE, action_id - num_nodes

    if action_id == do_nothing_id:
        return ActionType.DO_NOTHING, None

    raise ValueError(
        f"Invalid action_id={action_id}. "
        f"Valid range: 0 to {do_nothing_id}."
    )


def build_action_mask(
    num_nodes: int,
    edge_list: List[Tuple[int, int]],
    failed_mask: np.ndarray,
    protected_mask: np.ndarray,
    active_edge_mask: np.ndarray,
    budget_left: float,
    protect_cost: float,
    disconnect_cost: float,
    allow_protect: bool = True,
    allow_disconnect: bool = True,
) -> np.ndarray:
    """
    Build action mask.

    Convention:
        1 = valid action
        0 = invalid action
    """
    failed_mask = np.asarray(failed_mask, dtype=bool)
    protected_mask = np.asarray(protected_mask, dtype=bool)
    active_edge_mask = np.asarray(active_edge_mask, dtype=bool)

    num_edges = len(edge_list)
    action_dim = num_nodes + num_edges + 1
    action_mask = np.zeros(action_dim, dtype=np.int32)

    # Protect node actions
    if allow_protect and budget_left >= protect_cost:
        for node in range(num_nodes):
            if (not failed_mask[node]) and (not protected_mask[node]):
                action_mask[node] = 1

    # Disconnect edge actions
    if allow_disconnect and budget_left >= disconnect_cost:
        for edge_id, (u, v) in enumerate(edge_list):
            action_id = num_nodes + edge_id

            if edge_id >= len(active_edge_mask):
                continue
            if not active_edge_mask[edge_id]:
                continue
            if failed_mask[u] or failed_mask[v]:
                continue

            action_mask[action_id] = 1

    # Do nothing is always valid.
    action_mask[get_do_nothing_action_id(num_nodes, num_edges)] = 1
    return action_mask


def action_to_string(
    action_id: int,
    num_nodes: int,
    edge_list: List[Tuple[int, int]],
) -> str:
    """
    Convert action id to a readable string.
    """
    action_type, target = decode_action(action_id, num_nodes, edge_list)

    if action_type == ActionType.PROTECT_NODE:
        return f"protect_node({target})"

    if action_type == ActionType.DISCONNECT_EDGE:
        u, v = edge_list[int(target)]
        return f"disconnect_edge({target}: {u}-{v})"

    return "do_nothing"
