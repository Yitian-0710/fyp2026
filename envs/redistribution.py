"""
redistribution.py

Load redistribution rules for cascading failure simulation.

Supported modes:
1. uniform
2. stochastic
3. degree_weighted
4. load_weighted
"""

from __future__ import annotations

import numpy as np


def get_redistribution_weights(
    neighbors: np.ndarray,
    mode: str = "uniform",
    rng: np.random.Generator | None = None,
    degrees: np.ndarray | None = None,
    loads: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute redistribution weights over active neighbors.

    Parameters
    ----------
    neighbors:
        Neighbor node indices.
    mode:
        Redistribution mode.
    rng:
        NumPy random generator.
    degrees:
        Degree array, required by degree_weighted mode.
    loads:
        Load array, required by load_weighted mode.

    Returns
    -------
    np.ndarray
        Weights that sum to one.
    """
    neighbors = np.asarray(neighbors, dtype=int)
    n = len(neighbors)

    if n == 0:
        return np.array([], dtype=float)

    if rng is None:
        rng = np.random.default_rng()

    if mode == "uniform":
        return np.ones(n, dtype=float) / n

    if mode == "stochastic":
        return rng.dirichlet(np.ones(n, dtype=float))

    if mode == "degree_weighted":
        if degrees is None:
            raise ValueError("degrees must be provided for degree_weighted redistribution.")
        raw = np.asarray(degrees, dtype=float)[neighbors]
        if raw.sum() <= 1e-12:
            return np.ones(n, dtype=float) / n
        return raw / raw.sum()

    if mode == "load_weighted":
        if loads is None:
            raise ValueError("loads must be provided for load_weighted redistribution.")
        raw = np.asarray(loads, dtype=float)[neighbors]
        if raw.sum() <= 1e-12:
            return np.ones(n, dtype=float) / n
        return raw / raw.sum()

    raise ValueError(
        f"Unknown redistribution mode: {mode}. "
        "Expected 'uniform', 'stochastic', 'degree_weighted', or 'load_weighted'."
    )


def redistribute_failed_load(
    load: np.ndarray,
    failed_node: int,
    neighbors: np.ndarray,
    mode: str = "uniform",
    rng: np.random.Generator | None = None,
    degrees: np.ndarray | None = None,
) -> np.ndarray:
    """
    Redistribute the load of one failed node to active neighbors.

    This function returns a new load array and does not mutate the input.

    Parameters
    ----------
    load:
        Current node load, shape [N].
    failed_node:
        Failed node id.
    neighbors:
        Active neighbor ids that can receive redistributed load.
    mode:
        Redistribution mode.
    rng:
        NumPy random generator.
    degrees:
        Degree array for degree_weighted mode.

    Returns
    -------
    np.ndarray
        Updated load array.
    """
    new_load = np.asarray(load, dtype=float).copy()
    neighbors = np.asarray(neighbors, dtype=int)

    failed_load = float(new_load[failed_node])

    if len(neighbors) == 0 or failed_load <= 0:
        new_load[failed_node] = 0.0
        return new_load

    weights = get_redistribution_weights(
        neighbors=neighbors,
        mode=mode,
        rng=rng,
        degrees=degrees,
        loads=new_load,
    )

    for weight, node in zip(weights, neighbors):
        new_load[node] += failed_load * float(weight)

    new_load[failed_node] = 0.0
    return new_load
