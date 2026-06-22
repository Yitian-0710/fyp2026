"""
baselines/load_policy.py

Load-ratio-based protection baseline.

This policy protects the valid node with the highest load/capacity ratio.
If no node-protection action is valid, it selects do-nothing.
"""

from __future__ import annotations

import numpy as np


def load_policy(obs: dict, env) -> int:
    """
    Protect the valid node with the highest load ratio.

    load_ratio = load / capacity
    """
    action_mask = np.asarray(obs["action_mask"], dtype=np.int32)
    load_ratio = np.asarray(obs["load_ratio"], dtype=np.float32)

    valid_nodes = [
        node
        for node in range(env.N)
        if action_mask[node] == 1
    ]

    if len(valid_nodes) == 0:
        return int(env.do_nothing_action_id)

    best_node = max(valid_nodes, key=lambda node: load_ratio[node])

    return int(best_node)
