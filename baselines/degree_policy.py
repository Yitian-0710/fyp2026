"""
baselines/degree_policy.py

Degree-based protection baseline.

This policy protects the active node with the highest current degree.
If no node-protection action is valid, it selects do-nothing.
"""

from __future__ import annotations

import numpy as np


def degree_policy(obs: dict, env) -> int:
    """
    Protect the valid node with the highest current degree.

    Node-protection actions are encoded as:
        0 ... N-1 : protect node i
    """
    action_mask = np.asarray(obs["action_mask"], dtype=np.int32)
    adj_matrix = np.asarray(obs["adj_matrix"], dtype=np.float32)

    valid_nodes = [
        node
        for node in range(env.N)
        if action_mask[node] == 1
    ]

    if len(valid_nodes) == 0:
        return int(env.do_nothing_action_id)

    degrees = adj_matrix.sum(axis=1)
    best_node = max(valid_nodes, key=lambda node: degrees[node])

    return int(best_node)
