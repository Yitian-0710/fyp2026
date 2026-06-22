"""
agents/replay_buffer.py

Replay buffer for Vector-DQN.

Each transition contains:
    state
    action
    reward
    next_state
    done
    action_mask
    next_action_mask

The action masks are important because the DQN must not select invalid actions.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
    ) -> None:
        transition = (
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
            np.asarray(action_mask, dtype=np.int32),
            np.asarray(next_action_mask, dtype=np.int32),
        )
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> dict:
        transitions = random.sample(self.buffer, batch_size)

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            action_masks,
            next_action_masks,
        ) = zip(*transitions)

        return {
            "states": np.stack(states).astype(np.float32),
            "actions": np.asarray(actions, dtype=np.int64),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "next_states": np.stack(next_states).astype(np.float32),
            "dones": np.asarray(dones, dtype=np.float32),
            "action_masks": np.stack(action_masks).astype(np.int32),
            "next_action_masks": np.stack(next_action_masks).astype(np.int32),
        }

    def size(self) -> int:
        return len(self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)