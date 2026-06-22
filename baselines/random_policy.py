"""
baselines/random_policy.py

Random valid-action baseline.
"""

from __future__ import annotations

import numpy as np


def random_policy(obs: dict, env) -> int:
    """
    Randomly select one valid action.

    action_mask convention:
        1 = valid action
        0 = invalid action
    """
    action_mask = np.asarray(obs["action_mask"], dtype=np.int32)
    valid_actions = np.where(action_mask == 1)[0]

    if len(valid_actions) == 0:
        return int(env.do_nothing_action_id)

    return int(np.random.choice(valid_actions))
