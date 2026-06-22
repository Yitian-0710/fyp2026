"""
test_env_basic.py

Minimal test script for CascadingMitigationEnv.

Run from the project root:

    python experiments/test_env_basic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running from experiments/ or project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.cascading_mitigation_env import CascadingMitigationEnv


def main() -> None:
    env = CascadingMitigationEnv(
        N=20,
        network_type="BA",
        max_steps=5,
        budget=3.0,
        alpha=0.2,
        failure_model="deterministic",
        redistribution_mode="uniform",
        initial_failures=1,
        initial_failure_strategy="random",
        allow_disconnect=True,
        seed=42,
    )

    obs, info = env.reset()

    print("Initial info:")
    print(info)
    print("Action dim:", env.action_dim)
    print("Do-nothing action id:", env.do_nothing_action_id)
    print("Valid actions:", np.where(obs["action_mask"] == 1)[0])

    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        valid_actions = np.where(obs["action_mask"] == 1)[0]
        action = int(np.random.choice(valid_actions))

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print("\nAction:", action)
        print("Reward:", reward)
        print("Info:", info)

    print("\nEpisode finished.")
    print("Total reward:", total_reward)
    env.render()


if __name__ == "__main__":
    main()
