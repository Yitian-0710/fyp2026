"""
failure_models.py

Failure rules for intervention-aware cascading failure mitigation.

Supported models:
1. deterministic:
   A node fails when load > effective capacity.

2. logistic:
   Inspired by logistic failure probability models. Overloaded nodes do not
   always fail immediately. Their failure probability increases nonlinearly
   as load exceeds capacity.
"""

from __future__ import annotations

import numpy as np


def deterministic_failure(
    load: np.ndarray,
    capacity: np.ndarray,
    failed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Deterministic overload failure.

    Parameters
    ----------
    load:
        Current node load, shape [N].
    capacity:
        Effective node capacity, shape [N].
    failed_mask:
        Boolean mask. True means the node has already failed.

    Returns
    -------
    np.ndarray
        Boolean mask. True means newly failed.
    """
    load = np.asarray(load, dtype=float)
    capacity = np.asarray(capacity, dtype=float)

    new_failed = load > capacity

    if failed_mask is not None:
        failed_mask = np.asarray(failed_mask, dtype=bool)
        new_failed = new_failed & (~failed_mask)

    return new_failed


def logistic_failure_probability(
    load: np.ndarray | float,
    capacity: np.ndarray | float,
    gamma: float = 1.5,
    sharpness: float = 10.0,
) -> np.ndarray | float:
    """
    Logistic failure probability.

    Rule:
    - load <= capacity: probability = 0
    - load >= gamma * capacity: probability = 1
    - otherwise: probability follows a logistic curve

    Parameters
    ----------
    load:
        Node load.
    capacity:
        Effective node capacity.
    gamma:
        Removal threshold multiplier.
    sharpness:
        Controls the steepness of the logistic curve.

    Returns
    -------
    np.ndarray or float
        Failure probability.
    """
    scalar_input = np.isscalar(load) and np.isscalar(capacity)

    load_arr = np.asarray(load, dtype=float)
    cap_arr = np.asarray(capacity, dtype=float)
    cap_arr = np.maximum(cap_arr, 1e-8)

    prob = np.zeros_like(load_arr, dtype=float)

    below_capacity = load_arr <= cap_arr
    above_threshold = load_arr >= gamma * cap_arr
    middle = (~below_capacity) & (~above_threshold)

    prob[below_capacity] = 0.0
    prob[above_threshold] = 1.0

    midpoint = 0.5 * (cap_arr + gamma * cap_arr)
    scale = np.maximum(gamma * cap_arr - cap_arr, 1e-8)

    z = sharpness * (load_arr - midpoint) / scale
    prob[middle] = 1.0 / (1.0 + np.exp(-z[middle]))

    prob = np.clip(prob, 0.0, 1.0)

    if scalar_input:
        return float(prob)

    return prob


def sample_failed_nodes(
    load: np.ndarray,
    capacity: np.ndarray,
    failed_mask: np.ndarray,
    model: str = "deterministic",
    rng: np.random.Generator | None = None,
    gamma: float = 1.5,
    sharpness: float = 10.0,
) -> np.ndarray:
    """
    Sample newly failed nodes according to the selected failure model.

    Parameters
    ----------
    load:
        Current node load, shape [N].
    capacity:
        Effective node capacity, shape [N].
    failed_mask:
        Boolean mask. True means the node has already failed.
    model:
        "deterministic" or "logistic".
    rng:
        NumPy random generator.
    gamma:
        Removal threshold multiplier for logistic model.
    sharpness:
        Logistic curve sharpness.

    Returns
    -------
    np.ndarray
        Indices of newly failed nodes.
    """
    if rng is None:
        rng = np.random.default_rng()

    load = np.asarray(load, dtype=float)
    capacity = np.asarray(capacity, dtype=float)
    failed_mask = np.asarray(failed_mask, dtype=bool)

    if model == "deterministic":
        new_failed_mask = deterministic_failure(load, capacity, failed_mask)

    elif model == "logistic":
        prob = logistic_failure_probability(
            load=load,
            capacity=capacity,
            gamma=gamma,
            sharpness=sharpness,
        )
        prob = np.asarray(prob, dtype=float)
        prob[failed_mask] = 0.0
        new_failed_mask = rng.random(len(load)) < prob

    else:
        raise ValueError(
            f"Unknown failure model: {model}. "
            "Expected 'deterministic' or 'logistic'."
        )

    return np.where(new_failed_mask)[0]
