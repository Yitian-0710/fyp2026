"""
baselines/no_action.py

No-action baseline.

The policy always selects the do-nothing action.
"""

from __future__ import annotations


def no_action_policy(obs: dict, env) -> int:
    """
    Always choose do-nothing.

    Parameters
    ----------
    obs:
        Current observation.
    env:
        CascadingMitigationEnv instance.

    Returns
    -------
    int
        Do-nothing action id.
    """
    return int(env.do_nothing_action_id)