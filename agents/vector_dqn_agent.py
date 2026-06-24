"""
agents/vector_dqn_agent.py

Vector-based DQN baseline for cascading failure mitigation.

This is not Graph-DQN yet.
It flattens the graph observation into a vector:

    node_features
    adjacency matrix
    global_features

Then it uses an MLP Q-network to output Q-values for all actions.

Important:
    action_mask == 1 means valid action
    action_mask == 0 means invalid action
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def flatten_obs(obs: dict[str, Any]) -> np.ndarray:
    """
    Flatten environment observation into a 1D vector.

    The current vector-DQN baseline uses:
        1. node_features: [N, F_node]
        2. adj_matrix: [N, N]
        3. global_features: [F_global]

    Parameters
    ----------
    obs:
        Observation returned by CascadingMitigationEnv.

    Returns
    -------
    np.ndarray
        Flattened state vector.
    """
    node_features = np.asarray(obs["node_features"], dtype=np.float32).reshape(-1)
    adj_matrix = np.asarray(obs["adj_matrix"], dtype=np.float32).reshape(-1)
    global_features = np.asarray(obs["global_features"], dtype=np.float32).reshape(-1)
    action_mask = np.asarray(obs["action_mask"], dtype=np.float32).reshape(-1)

    state = np.concatenate(
        [
            node_features,
            adj_matrix,
            global_features,
            action_mask,
        ],
        axis=0,
    ).astype(np.float32)

    return state


def infer_state_dim(obs: dict[str, Any]) -> int:
    """
    Infer flattened state dimension from one observation.
    """
    return int(flatten_obs(obs).shape[0])


class MLPQNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VectorDQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update: int = 100,
        device: str | torch.device = "cpu",
        double_dqn: bool = True,
        grad_clip_norm: float = 1.0,
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)

        self.target_update = int(target_update)
        self.double_dqn = bool(double_dqn)
        self.grad_clip_norm = float(grad_clip_norm)

        self.device = torch.device(device)

        self.q_net = MLPQNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.target_q_net = MLPQNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.update_count = 0

    def take_action(
        self,
        obs: dict[str, Any],
        evaluate: bool = False,
    ) -> int:
        """
        Select an action using epsilon-greedy strategy.

        If evaluate=True, epsilon is ignored and the greedy action is selected.
        """
        action_mask = np.asarray(obs["action_mask"], dtype=np.int32)
        valid_actions = np.where(action_mask == 1)[0]

        if len(valid_actions) == 0:
            raise RuntimeError("No valid actions available. Check action_mask logic.")

        if (not evaluate) and (np.random.random() < self.epsilon):
            return int(np.random.choice(valid_actions))

        state = flatten_obs(obs)
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_net(state_tensor).squeeze(0)

            mask_tensor = torch.tensor(
                action_mask,
                dtype=torch.bool,
                device=self.device,
            )

            # action_mask == 0 means invalid, so set those Q-values to -inf
            q_values = q_values.masked_fill(~mask_tensor, -1e9)

            action = int(torch.argmax(q_values).item())

        return action

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """
        One DQN update step.

        This implementation masks invalid next actions when computing target Q.
        """
        states = torch.tensor(
            batch["states"],
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.tensor(
            batch["actions"],
            dtype=torch.int64,
            device=self.device,
        ).view(-1, 1)

        rewards = torch.tensor(
            batch["rewards"],
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)

        next_states = torch.tensor(
            batch["next_states"],
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.tensor(
            batch["dones"],
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)

        next_action_masks = torch.tensor(
            batch["next_action_masks"],
            dtype=torch.bool,
            device=self.device,
        )

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            if self.double_dqn:
                # Online network selects the best valid action.
                next_q_online = self.q_net(next_states)
                next_q_online = next_q_online.masked_fill(~next_action_masks, -1e9)
                best_next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)

                # Target network evaluates that action.
                next_q_target = self.target_q_net(next_states)
                next_q_target = next_q_target.masked_fill(~next_action_masks, -1e9)
                max_next_q = next_q_target.gather(1, best_next_actions)

            else:
                next_q_target = self.target_q_net(next_states)
                next_q_target = next_q_target.masked_fill(~next_action_masks, -1e9)
                max_next_q = torch.max(next_q_target, dim=1, keepdim=True)[0]

            q_targets = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_values, q_targets)

        self.optimizer.zero_grad()
        loss.backward()

        if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.q_net.parameters(),
                self.grad_clip_norm,
            )

        self.optimizer.step()

        self.update_count += 1

        if self.update_count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return {
            "loss": float(loss.item()),
            "mean_q": float(q_values.mean().item()),
            "mean_target_q": float(q_targets.mean().item()),
        }

    def decay_epsilon(self) -> None:
        """
        Decay epsilon once after each episode.
        """
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "q_net": self.q_net.state_dict(),
            "target_q_net": self.target_q_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "update_count": self.update_count,
        }

        torch.save(checkpoint, path)

    def load(self, path: str, map_location: str | torch.device | None = None) -> None:
        checkpoint = torch.load(
            path,
            map_location=map_location or self.device,
        )

        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_q_net.load_state_dict(checkpoint["target_q_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
        self.update_count = int(checkpoint.get("update_count", 0))
