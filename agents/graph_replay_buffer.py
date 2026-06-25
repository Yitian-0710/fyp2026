"""
agents/graph_replay_buffer.py

Replay buffer for Graph-DQN.

Different from Vector-DQN, this buffer stores the original graph observation
instead of a flattened vector. Each observation is copied into numpy arrays when
it is inserted, so later environment steps will not mutate old transitions.

Each transition contains:
    obs
    action
    reward
    next_obs
    done

The action masks are already inside obs["action_mask"] and
next_obs["action_mask"]. GraphDQNAgent.update() will use them when computing
current Q and target Q values.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import numpy as np

from utils.graph_obs_utils import copy_obs_to_numpy


class GraphReplayBuffer:
    """
    Replay buffer for graph observations.

    Parameters
    ----------
    capacity:
        Maximum number of transitions stored in the buffer.
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

    def add(
        self,
        obs: dict[str, Any],
        action: int,
        reward: float,
        next_obs: dict[str, Any],
        done: bool,
    ) -> None:
        """
        Add one transition.

        Parameters
        ----------
        obs:
            Current graph observation.
        action:
            Selected action id.
        reward:
            Reward after taking the action.
        next_obs:
            Next graph observation.
        done:
            Whether the episode is finished.
        """
        transition = {
            "obs": copy_obs_to_numpy(obs),
            "action": int(action),
            "reward": float(reward),
            "next_obs": copy_obs_to_numpy(next_obs),
            "done": bool(done),
        }
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> dict[str, Any]:
        """
        Sample a mini-batch.

        Returns
        -------
        dict
            {
                "obs_list": list[dict],
                "actions": np.ndarray[int64],
                "rewards": np.ndarray[float32],
                "next_obs_list": list[dict],
                "dones": np.ndarray[float32]
            }
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample batch_size={batch_size} from buffer size={len(self.buffer)}."
            )

        transitions = random.sample(self.buffer, batch_size)

        obs_list = [item["obs"] for item in transitions]
        actions = np.asarray([item["action"] for item in transitions], dtype=np.int64)
        rewards = np.asarray([item["reward"] for item in transitions], dtype=np.float32)
        next_obs_list = [item["next_obs"] for item in transitions]
        dones = np.asarray([item["done"] for item in transitions], dtype=np.float32)

        return {
            "obs_list": obs_list,
            "actions": actions,
            "rewards": rewards,
            "next_obs_list": next_obs_list,
            "dones": dones,
        }

    def size(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)
