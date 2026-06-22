"""
cascading_mitigation_env.py

Gymnasium-style environment for intervention-aware cascading failure mitigation.

Old attack setting:
    action = directly remove a node

New mitigation setting:
    initial failure happens first,
    then the agent chooses:
        1. protect a node
        2. disconnect an edge
        3. do nothing

Goal:
    reduce final cascade damage under intervention budget.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np

from .actions import (
    ActionType,
    action_to_string,
    build_action_mask,
    decode_action,
    get_do_nothing_action_id,
)
from .cascade_engine import (
    fail_nodes_and_redistribute,
    has_overload,
    propagate_one_generation,
)


class CascadingMitigationEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        N: int = 30,
        network_type: str = "BA",
        m: int = 2,
        p: float = 0.1,
        k: int = 4,
        rewiring_p: float = 0.1,
        alpha: float = 0.2,
        max_steps: int = 10,
        budget: float = 5.0,
        protect_cost: float = 1.0,
        disconnect_cost: float = 1.0,
        do_nothing_cost: float = 0.0,
        protect_strength: float = 0.5,
        protect_duration: int = 2,
        failure_model: str = "deterministic",
        redistribution_mode: str = "uniform",
        failure_gamma: float = 1.5,
        failure_sharpness: float = 10.0,
        load_type: str = "degree",
        initial_failures: int = 1,
        initial_failure_strategy: str = "random",
        allow_disconnect: bool = True,
        seed: int | None = 0,
        reward_weights: dict[str, float] | None = None,
        invalid_action_penalty: float = 5.0,
    ):
        super().__init__()

        self.N = int(N)
        self.network_type = network_type
        self.m = int(m)
        self.p = float(p)
        self.k = int(k)
        self.rewiring_p = float(rewiring_p)

        self.alpha = float(alpha)
        self.max_steps = int(max_steps)

        self.initial_budget = float(budget)
        self.budget_left = float(budget)

        self.protect_cost = float(protect_cost)
        self.disconnect_cost = float(disconnect_cost)
        self.do_nothing_cost = float(do_nothing_cost)
        self.protect_strength = float(protect_strength)
        self.protect_duration = int(protect_duration)

        self.failure_model = failure_model
        self.redistribution_mode = redistribution_mode
        self.failure_gamma = float(failure_gamma)
        self.failure_sharpness = float(failure_sharpness)

        self.load_type = load_type
        self.initial_failures = int(initial_failures)
        self.initial_failure_strategy = initial_failure_strategy

        self.allow_disconnect = bool(allow_disconnect)
        self.invalid_action_penalty = float(invalid_action_penalty)

        self.seed_value = seed
        self.rng = np.random.default_rng(seed)

        self.reward_weights = reward_weights or {
            "failed": 1.0,
            "lcc": 2.0,
            "lost_load": 1.0,
            "action_cost": 0.1,
        }

        self.base_graph = self._build_base_graph()
        self.base_adj = nx.to_numpy_array(self.base_graph, dtype=float)
        self.edge_list = list(self.base_graph.edges())
        self.num_edges = len(self.edge_list)

        self.do_nothing_action_id = get_do_nothing_action_id(self.N, self.num_edges)
        self.action_dim = self.N + self.num_edges + 1
        self.action_space = spaces.Discrete(self.action_dim)

        edge_count_for_space = max(self.num_edges, 1)
        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.N, 6),
                    dtype=np.float32,
                ),
                "node_load": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(self.N,),
                    dtype=np.float32,
                ),
                "node_capacity": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(self.N,),
                    dtype=np.float32,
                ),
                "load_ratio": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(self.N,),
                    dtype=np.float32,
                ),
                "failed_mask": spaces.MultiBinary(self.N),
                "protected_mask": spaces.MultiBinary(self.N),
                "adj_matrix": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.N, self.N),
                    dtype=np.float32,
                ),
                "edge_index": spaces.Box(
                    low=0,
                    high=max(self.N - 1, 0),
                    shape=(2, edge_count_for_space),
                    dtype=np.int64,
                ),
                "edge_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(edge_count_for_space, 2),
                    dtype=np.float32,
                ),
                "active_edge_mask": spaces.MultiBinary(edge_count_for_space),
                "action_mask": spaces.MultiBinary(self.action_dim),
                "global_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(5,),
                    dtype=np.float32,
                ),
            }
        )

        self.current_step = 0
        self.current_adj: np.ndarray | None = None
        self.initial_load: np.ndarray | None = None
        self.load: np.ndarray | None = None
        self.capacity: np.ndarray | None = None
        self.failed_mask: np.ndarray | None = None
        self.protected_timer: np.ndarray | None = None
        self.initial_failed_nodes: list[int] = []
        self.prev_damage = 0.0

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def _build_base_graph(self) -> nx.Graph:
        network_type = self.network_type.lower()

        if network_type in {"ba", "barabasi", "barabasi_albert"}:
            m = max(1, min(self.m, self.N - 1))
            graph = nx.barabasi_albert_graph(self.N, m, seed=self.seed_value)

        elif network_type in {"er", "erdos", "erdos_renyi"}:
            graph = nx.erdos_renyi_graph(self.N, self.p, seed=self.seed_value)
            retry = 0
            while not nx.is_connected(graph) and retry < 20:
                graph = nx.erdos_renyi_graph(
                    self.N,
                    self.p,
                    seed=None if self.seed_value is None else self.seed_value + retry + 1,
                )
                retry += 1

        elif network_type in {"ws", "watts", "watts_strogatz"}:
            k = min(self.k, self.N - 1)
            if k % 2 == 1:
                k += 1
            graph = nx.watts_strogatz_graph(
                self.N,
                k,
                self.rewiring_p,
                seed=self.seed_value,
            )

        else:
            raise ValueError(
                f"Unknown network_type={self.network_type}. "
                "Expected 'BA', 'ER', or 'WS'."
            )

        return nx.convert_node_labels_to_integers(graph)

    def _compute_initial_load(self, adj_matrix: np.ndarray) -> np.ndarray:
        graph = nx.from_numpy_array(adj_matrix)

        if self.load_type == "degree":
            load = adj_matrix.sum(axis=1).astype(float)

        elif self.load_type == "betweenness":
            load_dict = nx.betweenness_centrality(graph, normalized=False)
            load = np.array([load_dict[i] for i in range(self.N)], dtype=float)

        else:
            raise ValueError(
                f"Unknown load_type={self.load_type}. "
                "Expected 'degree' or 'betweenness'."
            )

        if load.max() <= 1e-12:
            load = np.ones(self.N, dtype=float)

        return load

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        if seed is not None:
            self.seed_value = seed
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.current_adj = self.base_adj.copy()

        self.initial_load = self._compute_initial_load(self.current_adj)
        self.load = self.initial_load.copy()
        self.capacity = np.maximum((1.0 + self.alpha) * self.initial_load, 1e-6)

        self.failed_mask = np.zeros(self.N, dtype=bool)
        self.protected_timer = np.zeros(self.N, dtype=int)
        self.budget_left = self.initial_budget

        self.initial_failed_nodes = self._sample_initial_failures()

        degrees = self.current_adj.sum(axis=1)
        self.current_adj, self.load, self.failed_mask, actually_failed = (
            fail_nodes_and_redistribute(
                adj_matrix=self.current_adj,
                load=self.load,
                failed_mask=self.failed_mask,
                nodes_to_fail=self.initial_failed_nodes,
                redistribution_mode=self.redistribution_mode,
                rng=self.rng,
                degrees=degrees,
            )
        )

        self.initial_failed_nodes = actually_failed
        self.prev_damage = self._compute_damage()

        return self._get_obs(), self._get_info()

    def _sample_initial_failures(self) -> list[int]:
        if self.initial_failures <= 0:
            return []

        candidate_nodes = np.arange(self.N)

        if self.initial_failure_strategy == "random":
            chosen = self.rng.choice(
                candidate_nodes,
                size=min(self.initial_failures, self.N),
                replace=False,
            )
            return [int(x) for x in chosen]

        if self.initial_failure_strategy == "highest_load":
            order = np.argsort(-self.initial_load)
            return [int(x) for x in order[: self.initial_failures]]

        raise ValueError(
            f"Unknown initial_failure_strategy={self.initial_failure_strategy}. "
            "Expected 'random' or 'highest_load'."
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: int):
        action = int(action)
        previous_damage = self._compute_damage()

        action_info = self._apply_action(action)

        if action_info["invalid"]:
            reward = -self.invalid_action_penalty
            return (
                self._get_obs(),
                float(reward),
                False,
                False,
                self._get_info(extra_info=action_info),
            )

        degrees = self.current_adj.sum(axis=1)

        propagation_result = propagate_one_generation(
            adj_matrix=self.current_adj,
            load=self.load,
            capacity=self.capacity,
            failed_mask=self.failed_mask,
            protected_timer=self.protected_timer,
            protect_strength=self.protect_strength,
            failure_model=self.failure_model,
            redistribution_mode=self.redistribution_mode,
            rng=self.rng,
            failure_gamma=self.failure_gamma,
            failure_sharpness=self.failure_sharpness,
            degrees=degrees,
        )

        self.current_adj = propagation_result["adj_matrix"]
        self.load = propagation_result["load"]
        self.failed_mask = propagation_result["failed_mask"]
        new_failed_nodes = propagation_result["new_failed_nodes"]

        self.protected_timer = np.maximum(self.protected_timer - 1, 0)
        self.current_step += 1

        current_damage = self._compute_damage()
        action_cost = float(action_info["cost"])

        reward = (
            previous_damage
            - current_damage
            - self.reward_weights["action_cost"] * action_cost
        )

        terminated = self._check_terminated()
        truncated = False

        extra_info = {
            **action_info,
            "new_failed_nodes": new_failed_nodes,
            "previous_damage": previous_damage,
            "current_damage": current_damage,
        }

        self.prev_damage = current_damage

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            self._get_info(extra_info=extra_info),
        )

    def _apply_action(self, action: int) -> dict[str, Any]:
        action_mask = self._build_action_mask()

        if action < 0 or action >= len(action_mask) or action_mask[action] == 0:
            return {
                "action_id": action,
                "action_name": "invalid_action",
                "action_type": "invalid",
                "target": None,
                "cost": 0.0,
                "invalid": True,
            }

        action_type, target = decode_action(action, self.N, self.edge_list)

        if action_type == ActionType.PROTECT_NODE:
            node = int(target)
            self.protected_timer[node] = self.protect_duration
            self.budget_left -= self.protect_cost

            return {
                "action_id": action,
                "action_name": action_to_string(action, self.N, self.edge_list),
                "action_type": "protect_node",
                "target": node,
                "cost": self.protect_cost,
                "invalid": False,
            }

        if action_type == ActionType.DISCONNECT_EDGE:
            edge_id = int(target)
            u, v = self.edge_list[edge_id]
            self.current_adj[u, v] = 0.0
            self.current_adj[v, u] = 0.0
            self.budget_left -= self.disconnect_cost

            return {
                "action_id": action,
                "action_name": action_to_string(action, self.N, self.edge_list),
                "action_type": "disconnect_edge",
                "target": edge_id,
                "edge": (u, v),
                "cost": self.disconnect_cost,
                "invalid": False,
            }

        if action_type == ActionType.DO_NOTHING:
            self.budget_left -= self.do_nothing_cost

            return {
                "action_id": action,
                "action_name": "do_nothing",
                "action_type": "do_nothing",
                "target": None,
                "cost": self.do_nothing_cost,
                "invalid": False,
            }

        raise RuntimeError(f"Unhandled action type: {action_type}")

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(self) -> dict[str, np.ndarray]:
        active_edge_mask = self._active_edge_mask()
        protected_mask = self.protected_timer > 0
        action_mask = self._build_action_mask()

        node_load = self.load.astype(np.float32)
        node_capacity = self.capacity.astype(np.float32)

        load_ratio = node_load / np.maximum(node_capacity, 1e-6)
        load_ratio = np.nan_to_num(load_ratio, nan=0.0, posinf=0.0, neginf=0.0)

        degrees = self.current_adj.sum(axis=1).astype(np.float32)
        degree_norm = degrees / max(float(degrees.max()), 1.0)

        load_norm = node_load / max(float(node_load.max()), 1.0)
        cap_norm = node_capacity / max(float(node_capacity.max()), 1.0)

        node_features = np.stack(
            [
                load_norm,
                cap_norm,
                load_ratio.astype(np.float32),
                self.failed_mask.astype(np.float32),
                protected_mask.astype(np.float32),
                degree_norm,
            ],
            axis=1,
        ).astype(np.float32)

        if self.num_edges > 0:
            edge_index = np.array(self.edge_list, dtype=np.int64).T
            edge_features = np.zeros((self.num_edges, 2), dtype=np.float32)

            for edge_id, _ in enumerate(self.edge_list):
                is_active = float(active_edge_mask[edge_id])
                is_disconnected = 1.0 - is_active
                edge_features[edge_id] = np.array([is_active, is_disconnected])
        else:
            edge_index = np.zeros((2, 1), dtype=np.int64)
            edge_features = np.zeros((1, 2), dtype=np.float32)
            active_edge_mask = np.zeros(1, dtype=np.int32)

        metrics = self._compute_metrics()

        global_features = np.array(
            [
                self.current_step / max(self.max_steps, 1),
                self.budget_left / max(self.initial_budget, 1e-6),
                metrics["failed_fraction"],
                metrics["lcc_ratio"],
                metrics["lost_load_ratio"],
            ],
            dtype=np.float32,
        )

        return {
            "node_features": node_features,
            "node_load": node_load,
            "node_capacity": node_capacity,
            "load_ratio": load_ratio.astype(np.float32),
            "failed_mask": self.failed_mask.astype(np.int32),
            "protected_mask": protected_mask.astype(np.int32),
            "adj_matrix": self.current_adj.astype(np.float32),
            "edge_index": edge_index,
            "edge_features": edge_features,
            "active_edge_mask": active_edge_mask.astype(np.int32),
            "action_mask": action_mask.astype(np.int32),
            "global_features": global_features,
        }

    def _active_edge_mask(self) -> np.ndarray:
        mask = np.zeros(self.num_edges, dtype=bool)

        for edge_id, (u, v) in enumerate(self.edge_list):
            mask[edge_id] = (
                self.current_adj[u, v] > 0
                and not self.failed_mask[u]
                and not self.failed_mask[v]
            )

        return mask

    def _build_action_mask(self) -> np.ndarray:
        protected_mask = self.protected_timer > 0
        active_edge_mask = self._active_edge_mask()

        return build_action_mask(
            num_nodes=self.N,
            edge_list=self.edge_list,
            failed_mask=self.failed_mask,
            protected_mask=protected_mask,
            active_edge_mask=active_edge_mask,
            budget_left=self.budget_left,
            protect_cost=self.protect_cost,
            disconnect_cost=self.disconnect_cost,
            allow_protect=True,
            allow_disconnect=self.allow_disconnect,
        )

    # ------------------------------------------------------------------
    # Metrics and reward
    # ------------------------------------------------------------------
    def _compute_metrics(self) -> dict[str, float]:
        failed_fraction = float(np.mean(self.failed_mask))

        active_nodes = np.where(~self.failed_mask)[0]

        if len(active_nodes) == 0:
            lcc_ratio = 0.0
        else:
            sub_adj = self.current_adj[np.ix_(active_nodes, active_nodes)]
            graph = nx.from_numpy_array(sub_adj)
            components = list(nx.connected_components(graph))
            largest_component = max((len(c) for c in components), default=0)
            lcc_ratio = float(largest_component / self.N)

        total_initial_load = max(float(np.sum(self.initial_load)), 1e-6)
        failed_load = float(np.sum(self.initial_load[self.failed_mask]))
        lost_load_ratio = failed_load / total_initial_load
        served_load_ratio = 1.0 - lost_load_ratio
        budget_used = self.initial_budget - self.budget_left

        return {
            "failed_fraction": failed_fraction,
            "lcc_ratio": lcc_ratio,
            "lost_load_ratio": lost_load_ratio,
            "served_load_ratio": served_load_ratio,
            "budget_left": float(self.budget_left),
            "budget_used": float(budget_used),
        }

    def _compute_damage(self) -> float:
        metrics = self._compute_metrics()

        damage = (
            self.reward_weights["failed"] * metrics["failed_fraction"]
            + self.reward_weights["lcc"] * (1.0 - metrics["lcc_ratio"])
            + self.reward_weights["lost_load"] * metrics["lost_load_ratio"]
        )

        return float(damage)

    def _check_terminated(self) -> bool:
        if self.current_step >= self.max_steps:
            return True

        if np.all(self.failed_mask):
            return True

        still_overloaded = has_overload(
            load=self.load,
            capacity=self.capacity,
            failed_mask=self.failed_mask,
            protected_timer=self.protected_timer,
            protect_strength=self.protect_strength,
        )

        if not still_overloaded:
            return True

        return False

    # ------------------------------------------------------------------
    # Info and render
    # ------------------------------------------------------------------
    def _get_info(self, extra_info: dict[str, Any] | None = None) -> dict[str, Any]:
        metrics = self._compute_metrics()

        info = {
            "step": self.current_step,
            "initial_failed_nodes": self.initial_failed_nodes,
            "budget_left": float(self.budget_left),
            "damage": self._compute_damage(),
            **metrics,
        }

        if extra_info:
            info.update(extra_info)

        return info

    def render(self):
        metrics = self._compute_metrics()

        print("=" * 50)
        print(f"Step: {self.current_step}")
        print(f"Budget left: {self.budget_left:.3f}")
        print(f"Failed fraction: {metrics['failed_fraction']:.3f}")
        print(f"LCC ratio: {metrics['lcc_ratio']:.3f}")
        print(f"Lost load ratio: {metrics['lost_load_ratio']:.3f}")
        print(f"Damage: {self._compute_damage():.3f}")
        print("=" * 50)
