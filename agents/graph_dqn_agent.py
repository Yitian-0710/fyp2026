"""
agents/graph_dqn_agent.py

Graph-DQN agent for cascading failure mitigation.

Graph-DQN replaces the Vector-DQN pipeline:
    obs -> flatten -> MLP -> Q-values

with:
    obs -> graph tensors -> GNN encoder -> node/global Q heads -> Q-values

Graph-DQN V1 focuses on:
    - protect node i actions
    - do-nothing action

Edge-disconnection actions can remain in the environment's action dimension, but
for Graph-DQN V1 you should train with:
    --no_disconnect

This means edge actions are invalid and will be masked by obs["action_mask"].
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from models.graph_q_network import GraphQNetwork
from utils.graph_obs_utils import (
    batch_graph_obs,
    infer_graph_obs_dims,
    obs_to_graph_tensor,
)


class GraphDQNAgent:
    """
    Graph-DQN agent.

    Parameters
    ----------
    node_feature_dim:
        Dimension of obs["node_features"].
    global_feature_dim:
        Dimension of obs["global_features"].
    action_dim:
        Full action dimension of the environment.
    num_nodes:
        Number of nodes N.
    hidden_dim:
        Hidden dimension for GNN and Q heads.
    lr:
        Learning rate.
    gamma:
        Discount factor.
    epsilon_start:
        Initial epsilon for epsilon-greedy exploration.
    epsilon_end:
        Minimum epsilon.
    epsilon_decay:
        Multiplicative epsilon decay after each episode.
    target_update:
        Number of gradient updates between target network syncs.
    device:
        torch device.
    double_dqn:
        Whether to use Double DQN target calculation.
    grad_clip_norm:
        Gradient clipping norm. Set <= 0 or None to disable.
    """

    def __init__(
        self,
        node_feature_dim: int,
        global_feature_dim: int,
        action_dim: int,
        num_nodes: int,
        hidden_dim: int = 128,
        num_gnn_layers: int = 2,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update: int = 100,
        device: str | torch.device = "cpu",
        double_dqn: bool = True,
        grad_clip_norm: float | None = 1.0,
        dropout: float = 0.0,
        pooling: str = "mean",
        use_global_features: bool = True,
    ):
        self.node_feature_dim = int(node_feature_dim)
        self.global_feature_dim = int(global_feature_dim)
        self.action_dim = int(action_dim)
        self.num_nodes = int(num_nodes)
        self.hidden_dim = int(hidden_dim)
        self.num_gnn_layers = int(num_gnn_layers)

        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)

        self.target_update = int(target_update)
        self.double_dqn = bool(double_dqn)
        self.grad_clip_norm = grad_clip_norm

        self.dropout = float(dropout)
        self.pooling = pooling
        self.use_global_features = bool(use_global_features)

        self.device = torch.device(device)

        self.q_net = GraphQNetwork(
            node_feature_dim=self.node_feature_dim,
            global_feature_dim=self.global_feature_dim,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            num_nodes=self.num_nodes,
            num_gnn_layers=self.num_gnn_layers,
            dropout=self.dropout,
            pooling=self.pooling,
            use_global_features=self.use_global_features,
        ).to(self.device)

        self.target_q_net = GraphQNetwork(
            node_feature_dim=self.node_feature_dim,
            global_feature_dim=self.global_feature_dim,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            num_nodes=self.num_nodes,
            num_gnn_layers=self.num_gnn_layers,
            dropout=self.dropout,
            pooling=self.pooling,
            use_global_features=self.use_global_features,
        ).to(self.device)

        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.update_count = 0

    @classmethod
    def from_obs(
        cls,
        obs: dict[str, Any],
        hidden_dim: int = 128,
        num_gnn_layers: int = 2,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update: int = 100,
        device: str | torch.device = "cpu",
        double_dqn: bool = True,
        grad_clip_norm: float | None = 1.0,
        dropout: float = 0.0,
        pooling: str = "mean",
        use_global_features: bool = True,
    ) -> "GraphDQNAgent":
        """
        Build an agent by inferring dimensions from one environment observation.
        """
        dims = infer_graph_obs_dims(obs)
        return cls(
            node_feature_dim=dims["node_feature_dim"],
            global_feature_dim=dims["global_feature_dim"],
            action_dim=dims["action_dim"],
            num_nodes=dims["num_nodes"],
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            target_update=target_update,
            device=device,
            double_dqn=double_dqn,
            grad_clip_norm=grad_clip_norm,
            dropout=dropout,
            pooling=pooling,
            use_global_features=use_global_features,
        )

    def take_action(
        self,
        obs: dict[str, Any],
        evaluate: bool = False,
    ) -> int:
        """
        Select action using epsilon-greedy exploration.

        Invalid actions are masked with -inf before argmax.
        """
        action_mask = np.asarray(obs["action_mask"], dtype=np.int32)
        valid_actions = np.where(action_mask == 1)[0]

        if len(valid_actions) == 0:
            raise RuntimeError("No valid actions available. Check action_mask logic.")

        if (not evaluate) and (np.random.random() < self.epsilon):
            return int(np.random.choice(valid_actions))

        graph_batch = obs_to_graph_tensor(
            obs,
            device=self.device,
            add_batch_dim=True,
            normalize_adjacency=True,
        )

        with torch.no_grad():
            q_values = self.q_net(graph_batch)  # [1, A]
            mask = graph_batch["action_mask"].bool()  # [1, A]
            q_values = q_values.masked_fill(~mask, -1e9)
            action = int(torch.argmax(q_values, dim=1).item())

        return action

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """
        One Graph-DQN update step.

        Parameters
        ----------
        batch:
            A mini-batch sampled from GraphReplayBuffer.

        Returns
        -------
        dict[str, float]
            Training diagnostics: loss, mean_q, mean_target_q.
        """
        obs_batch = batch_graph_obs(
            batch["obs_list"],
            device=self.device,
            normalize_adjacency=True,
        )
        next_obs_batch = batch_graph_obs(
            batch["next_obs_list"],
            device=self.device,
            normalize_adjacency=True,
        )

        actions = torch.tensor(
            batch["actions"],
            dtype=torch.long,
            device=self.device,
        ).view(-1, 1)

        rewards = torch.tensor(
            batch["rewards"],
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)

        dones = torch.tensor(
            batch["dones"],
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)

        q_all = self.q_net(obs_batch)  # [B, A]
        q_values = q_all.gather(1, actions)

        with torch.no_grad():
            next_action_mask = next_obs_batch["action_mask"].bool()

            if self.double_dqn:
                # Online network selects best valid next action.
                next_q_online = self.q_net(next_obs_batch)
                next_q_online = next_q_online.masked_fill(~next_action_mask, -1e9)
                best_next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)

                # Target network evaluates selected action.
                next_q_target = self.target_q_net(next_obs_batch)
                next_q_target = next_q_target.masked_fill(~next_action_mask, -1e9)
                max_next_q = next_q_target.gather(1, best_next_actions)
            else:
                next_q_target = self.target_q_net(next_obs_batch)
                next_q_target = next_q_target.masked_fill(~next_action_mask, -1e9)
                max_next_q = torch.max(next_q_target, dim=1, keepdim=True)[0]

            q_targets = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_values, q_targets)

        self.optimizer.zero_grad()
        loss.backward()

        if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.q_net.parameters(),
                float(self.grad_clip_norm),
            )

        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return {
            "loss": float(loss.item()),
            "mean_q": float(q_values.mean().item()),
            "mean_target_q": float(q_targets.mean().item()),
            "epsilon": float(self.epsilon),
        }

    def decay_epsilon(self) -> None:
        """
        Decay epsilon once after each episode.
        """
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )

    def get_q_details(self, obs: dict[str, Any]) -> dict[str, Any]:
        """
        Return Q-value details for debugging.

        This is useful for understanding whether Graph-DQN prefers protecting
        nodes or taking do-nothing.
        """
        graph_batch = obs_to_graph_tensor(
            obs,
            device=self.device,
            add_batch_dim=True,
            normalize_adjacency=True,
        )

        with torch.no_grad():
            details = self.q_net(graph_batch, return_details=True)
            q_all = details["q_all"].squeeze(0).detach().cpu().numpy()
            node_q = details["node_q"].squeeze(0).detach().cpu().numpy()
            do_nothing_q = float(details["do_nothing_q"].squeeze().item())

        action_mask = np.asarray(obs["action_mask"], dtype=np.int32)
        q_masked = np.where(action_mask == 1, q_all, -np.inf)

        return {
            "q_all": q_all,
            "q_masked": q_masked,
            "node_q": node_q,
            "do_nothing_q": do_nothing_q,
            "best_action": int(np.argmax(q_masked)),
            "valid_actions": np.where(action_mask == 1)[0],
        }

    def save(self, path: str) -> None:
        """
        Save checkpoint.
        """
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        checkpoint = {
            "node_feature_dim": self.node_feature_dim,
            "global_feature_dim": self.global_feature_dim,
            "action_dim": self.action_dim,
            "num_nodes": self.num_nodes,
            "hidden_dim": self.hidden_dim,
            "num_gnn_layers": self.num_gnn_layers,
            "dropout": self.dropout,
            "pooling": self.pooling,
            "use_global_features": self.use_global_features,
            "q_net": self.q_net.state_dict(),
            "target_q_net": self.target_q_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "update_count": self.update_count,
        }

        torch.save(checkpoint, path)

    def load(
        self,
        path: str,
        map_location: str | torch.device | None = None,
        load_optimizer: bool = True,
    ) -> None:
        """
        Load checkpoint into an existing agent.
        """
        checkpoint = torch.load(path, map_location=map_location or self.device)

        expected = {
            "node_feature_dim": self.node_feature_dim,
            "global_feature_dim": self.global_feature_dim,
            "action_dim": self.action_dim,
            "num_nodes": self.num_nodes,
            "hidden_dim": self.hidden_dim,
            "num_gnn_layers": self.num_gnn_layers,
        }

        for key, value in expected.items():
            if key in checkpoint and int(checkpoint[key]) != int(value):
                raise ValueError(
                    f"Checkpoint {key}={checkpoint[key]} does not match current {key}={value}. "
                    "Use the same environment/model parameters or retrain."
                )

        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_q_net.load_state_dict(checkpoint["target_q_net"])

        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
        self.update_count = int(checkpoint.get("update_count", 0))

    @staticmethod
    def load_from_checkpoint(
        path: str,
        device: str | torch.device = "cpu",
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 0.0,
        epsilon_end: float = 0.0,
        epsilon_decay: float = 1.0,
        target_update: int = 100,
        double_dqn: bool = True,
        grad_clip_norm: float | None = 1.0,
        load_optimizer: bool = False,
    ) -> "GraphDQNAgent":
        """
        Build and load an agent directly from a checkpoint.

        This is useful for evaluation scripts.
        """
        device = torch.device(device)
        checkpoint = torch.load(path, map_location=device)

        agent = GraphDQNAgent(
            node_feature_dim=int(checkpoint["node_feature_dim"]),
            global_feature_dim=int(checkpoint["global_feature_dim"]),
            action_dim=int(checkpoint["action_dim"]),
            num_nodes=int(checkpoint["num_nodes"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            num_gnn_layers=int(checkpoint.get("num_gnn_layers", 2)),
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            target_update=target_update,
            device=device,
            double_dqn=double_dqn,
            grad_clip_norm=grad_clip_norm,
            dropout=float(checkpoint.get("dropout", 0.0)),
            pooling=str(checkpoint.get("pooling", "mean")),
            use_global_features=bool(checkpoint.get("use_global_features", True)),
        )
        agent.load(path, map_location=device, load_optimizer=load_optimizer)
        return agent
