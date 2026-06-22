"""
cascade_engine.py

Core cascading failure propagation engine.

This module updates:
- adjacency matrix
- load
- failed mask

It does not define the RL environment.
"""

from __future__ import annotations

import numpy as np

from .failure_models import sample_failed_nodes
from .redistribution import get_redistribution_weights


def compute_effective_capacity(
    capacity: np.ndarray,
    protected_timer: np.ndarray,
    protect_strength: float = 0.5,
) -> np.ndarray:
    """
    Compute effective capacity under temporary protection.
    """
    capacity = np.asarray(capacity, dtype=float)
    protected_timer = np.asarray(protected_timer, dtype=int)

    effective_capacity = capacity.copy()
    protected_nodes = protected_timer > 0
    effective_capacity[protected_nodes] *= 1.0 + protect_strength
    return effective_capacity


def get_active_neighbors(
    adj_matrix: np.ndarray,
    node: int,
    failed_mask: np.ndarray,
    exclude_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Get active neighbors of a node.
    """
    adj_matrix = np.asarray(adj_matrix)
    failed_mask = np.asarray(failed_mask, dtype=bool)

    neighbor_mask = adj_matrix[int(node)] > 0
    valid_mask = neighbor_mask & (~failed_mask)

    if exclude_mask is not None:
        exclude_mask = np.asarray(exclude_mask, dtype=bool)
        valid_mask = valid_mask & (~exclude_mask)

    return np.where(valid_mask)[0]


def fail_nodes_and_redistribute(
    adj_matrix: np.ndarray,
    load: np.ndarray,
    failed_mask: np.ndarray,
    nodes_to_fail: np.ndarray | list[int],
    redistribution_mode: str = "uniform",
    rng: np.random.Generator | None = None,
    degrees: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    Mark nodes as failed and redistribute their load to active neighbors.

    Nodes failing in the same generation do not receive redistributed load.
    """
    if rng is None:
        rng = np.random.default_rng()

    adj_matrix = np.asarray(adj_matrix, dtype=float).copy()
    load = np.asarray(load, dtype=float).copy()
    failed_mask = np.asarray(failed_mask, dtype=bool).copy()

    nodes_to_fail = np.asarray(nodes_to_fail, dtype=int)
    nodes_to_fail = np.unique(nodes_to_fail)

    valid_nodes = [
        int(node)
        for node in nodes_to_fail
        if 0 <= int(node) < len(load) and not failed_mask[int(node)]
    ]

    if not valid_nodes:
        return adj_matrix, load, failed_mask, []

    scheduled_fail_mask = failed_mask.copy()
    scheduled_fail_mask[valid_nodes] = True

    for node in valid_nodes:
        neighbors = get_active_neighbors(
            adj_matrix=adj_matrix,
            node=node,
            failed_mask=failed_mask,
            exclude_mask=scheduled_fail_mask,
        )

        failed_load = float(load[node])

        if len(neighbors) > 0 and failed_load > 0:
            weights = get_redistribution_weights(
                neighbors=neighbors,
                mode=redistribution_mode,
                rng=rng,
                degrees=degrees,
                loads=load,
            )

            for weight, neighbor in zip(weights, neighbors):
                load[neighbor] += failed_load * float(weight)

        load[node] = 0.0

    for node in valid_nodes:
        failed_mask[node] = True
        adj_matrix[node, :] = 0.0
        adj_matrix[:, node] = 0.0

    return adj_matrix, load, failed_mask, valid_nodes


def propagate_one_generation(
    adj_matrix: np.ndarray,
    load: np.ndarray,
    capacity: np.ndarray,
    failed_mask: np.ndarray,
    protected_timer: np.ndarray,
    protect_strength: float = 0.5,
    failure_model: str = "deterministic",
    redistribution_mode: str = "uniform",
    rng: np.random.Generator | None = None,
    failure_gamma: float = 1.5,
    failure_sharpness: float = 10.0,
    degrees: np.ndarray | None = None,
) -> dict:
    """
    Propagate one generation of cascading failures.
    """
    if rng is None:
        rng = np.random.default_rng()

    effective_capacity = compute_effective_capacity(
        capacity=capacity,
        protected_timer=protected_timer,
        protect_strength=protect_strength,
    )

    new_failed_nodes = sample_failed_nodes(
        load=load,
        capacity=effective_capacity,
        failed_mask=failed_mask,
        model=failure_model,
        rng=rng,
        gamma=failure_gamma,
        sharpness=failure_sharpness,
    )

    adj_matrix, load, failed_mask, actually_failed = fail_nodes_and_redistribute(
        adj_matrix=adj_matrix,
        load=load,
        failed_mask=failed_mask,
        nodes_to_fail=new_failed_nodes,
        redistribution_mode=redistribution_mode,
        rng=rng,
        degrees=degrees,
    )

    return {
        "adj_matrix": adj_matrix,
        "load": load,
        "failed_mask": failed_mask,
        "new_failed_nodes": actually_failed,
    }


def propagate_until_stable(
    adj_matrix: np.ndarray,
    load: np.ndarray,
    capacity: np.ndarray,
    failed_mask: np.ndarray,
    protected_timer: np.ndarray,
    protect_strength: float = 0.5,
    failure_model: str = "deterministic",
    redistribution_mode: str = "uniform",
    rng: np.random.Generator | None = None,
    failure_gamma: float = 1.5,
    failure_sharpness: float = 10.0,
    degrees: np.ndarray | None = None,
    max_generations: int = 100,
) -> dict:
    """
    Propagate until no new failures occur or max_generations is reached.
    """
    all_new_failed: list[int] = []

    for _ in range(max_generations):
        result = propagate_one_generation(
            adj_matrix=adj_matrix,
            load=load,
            capacity=capacity,
            failed_mask=failed_mask,
            protected_timer=protected_timer,
            protect_strength=protect_strength,
            failure_model=failure_model,
            redistribution_mode=redistribution_mode,
            rng=rng,
            failure_gamma=failure_gamma,
            failure_sharpness=failure_sharpness,
            degrees=degrees,
        )

        adj_matrix = result["adj_matrix"]
        load = result["load"]
        failed_mask = result["failed_mask"]
        new_failed_nodes = result["new_failed_nodes"]

        if len(new_failed_nodes) == 0:
            break

        all_new_failed.extend(new_failed_nodes)

    return {
        "adj_matrix": adj_matrix,
        "load": load,
        "failed_mask": failed_mask,
        "new_failed_nodes": all_new_failed,
    }


def has_overload(
    load: np.ndarray,
    capacity: np.ndarray,
    failed_mask: np.ndarray,
    protected_timer: np.ndarray,
    protect_strength: float = 0.5,
) -> bool:
    """
    Check whether any active node is still overloaded.
    """
    effective_capacity = compute_effective_capacity(
        capacity=capacity,
        protected_timer=protected_timer,
        protect_strength=protect_strength,
    )

    active_mask = ~np.asarray(failed_mask, dtype=bool)
    overloaded = (np.asarray(load, dtype=float) > effective_capacity) & active_mask
    return bool(np.any(overloaded))
