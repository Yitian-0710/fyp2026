"""
baselines/greedy_policy.py

One-step greedy baseline.

For each valid action, this policy simulates one step and chooses the action
with the highest immediate reward.

This baseline is stronger than random/degree/load baselines, but it is slower.
"""

from __future__ import annotations

import copy

import numpy as np


def greedy_policy(obs: dict, env, max_candidates: int | None = None) -> int:
    """
    One-step greedy policy.

    Parameters
    ----------
    obs:
        Current observation.
    env:
        Current environment.
    max_candidates:
        Optional maximum number of actions to evaluate.
        If None, evaluate all valid actions.

    Returns
    -------
    int
        Best action id.
    """
    action_mask = np.asarray(obs["action_mask"], dtype=np.int32)
    valid_actions = np.where(action_mask == 1)[0]

    if len(valid_actions) == 0:
        return int(env.do_nothing_action_id)

    if max_candidates is not None and len(valid_actions) > max_candidates:
        valid_actions = np.random.choice(
            valid_actions,
            size=max_candidates,
            replace=False,
        )

    best_action = int(env.do_nothing_action_id)
    best_reward = -float("inf")

    for action in valid_actions:
        env_copy = copy.deepcopy(env)

        try:
            _, reward, _, _, _ = env_copy.step(int(action))
        except Exception:
            continue

        if reward > best_reward:
            best_reward = reward
            best_action = int(action)

    return best_action
