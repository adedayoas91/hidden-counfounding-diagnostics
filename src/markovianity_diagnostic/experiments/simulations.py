"""Synthetic scenarios for Markovianity-diagnostic experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ScenarioResult:
    """Container for synthetic data, compact ground truth, and metadata."""

    X: np.ndarray
    ground_truth_compact: np.ndarray | None
    metadata: dict[str, Any]


def spectral_radius(matrix: np.ndarray) -> float:
    """Return the spectral radius of a square matrix."""

    values = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(values)))


def scale_to_stability(matrix: np.ndarray, target_radius: float = 0.85) -> np.ndarray:
    """Scale a coefficient matrix to a target spectral radius."""

    radius = spectral_radius(matrix)
    if radius <= 1e-12:
        return matrix.copy()
    return matrix * (target_radius / radius)


def make_sparse_matrix(
    d: int,
    edge_prob: float,
    *,
    weight_low: float = 0.15,
    weight_high: float = 0.35,
    allow_self: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a random sparse signed weight matrix."""

    rng = np.random.default_rng(seed)
    matrix = np.zeros((d, d), dtype=float)
    mask = rng.uniform(size=(d, d)) < edge_prob
    if not allow_self:
        np.fill_diagonal(mask, 0)
    weights = rng.uniform(weight_low, weight_high, size=(d, d))
    signs = rng.choice([-1.0, 1.0], size=(d, d))
    matrix[mask] = weights[mask] * signs[mask]
    return matrix


def compact_ground_truth(lag_matrices: list[np.ndarray]) -> np.ndarray:
    """Collapse multi-lag ground truth into a binary adjacency matrix."""

    truth = np.zeros_like(lag_matrices[0], dtype=int)
    for lag_matrix in lag_matrices:
        truth = np.logical_or(truth, np.abs(lag_matrix) > 1e-12)
    truth = truth.astype(int)
    np.fill_diagonal(truth, 0)
    return truth


def simulate_var(
    T: int,
    lag_matrices: list[np.ndarray],
    *,
    burn_in: int = 300,
    noise_scale: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate a stable VAR process with additive Gaussian noise."""

    rng = np.random.default_rng(seed)
    order = len(lag_matrices)
    d = lag_matrices[0].shape[0]
    total = T + burn_in

    X = np.zeros((total + order, d), dtype=float)
    noise = rng.normal(scale=noise_scale, size=(total + order, d))

    for t in range(order, total + order):
        current = np.zeros(d, dtype=float)
        for lag_index, lag_matrix in enumerate(lag_matrices, start=1):
            current += lag_matrix @ X[t - lag_index]
        X[t] = current + noise[t]

    return X[order + burn_in : order + burn_in + T]


def scenario_order1_unconfounded(
    *,
    T: int = 2000,
    d: int = 10,
    edge_prob: float = 0.12,
    noise_scale: float = 1.0,
    seed: int = 0,
) -> ScenarioResult:
    """Clean order-1 Markov null."""

    a1 = scale_to_stability(make_sparse_matrix(d, edge_prob=edge_prob, seed=seed))
    X = simulate_var(T=T, lag_matrices=[a1], noise_scale=noise_scale, seed=seed + 1)
    return ScenarioResult(
        X=X,
        ground_truth_compact=compact_ground_truth([a1]),
        metadata={
            "scenario": "order1_unconfounded",
            "latent": False,
            "true_order": 1,
            "edge_prob": edge_prob,
            "noise_scale": noise_scale,
        },
    )


def scenario_order3_unconfounded(
    *,
    T: int = 2000,
    d: int = 10,
    noise_scale: float = 1.0,
    seed: int = 0,
) -> ScenarioResult:
    """Higher-order but unconfounded control."""

    a1 = make_sparse_matrix(d, edge_prob=0.10, seed=seed + 1) * 0.60
    a2 = make_sparse_matrix(d, edge_prob=0.08, seed=seed + 2) * 0.35
    a3 = make_sparse_matrix(d, edge_prob=0.06, seed=seed + 3) * 0.20
    total_radius = spectral_radius(a1) + spectral_radius(a2) + spectral_radius(a3)
    if total_radius > 1e-12:
        scale = 0.80 / total_radius
        a1, a2, a3 = scale * a1, scale * a2, scale * a3

    X = simulate_var(
        T=T,
        lag_matrices=[a1, a2, a3],
        noise_scale=noise_scale,
        seed=seed + 10,
    )
    return ScenarioResult(
        X=X,
        ground_truth_compact=compact_ground_truth([a1, a2, a3]),
        metadata={
            "scenario": "order3_unconfounded",
            "latent": False,
            "true_order": 3,
            "noise_scale": noise_scale,
        },
    )


def scenario_latent_common_driver(
    *,
    T: int = 2000,
    d: int = 10,
    conf_strength: float = 0.35,
    latent_ar: float = 0.80,
    noise_scale: float = 1.0,
    seed: int = 0,
) -> ScenarioResult:
    """Observed order-1 system perturbed by a latent AR(1) common driver."""

    rng = np.random.default_rng(seed)
    a1 = scale_to_stability(make_sparse_matrix(d, edge_prob=0.10, seed=seed + 1) * 0.50)

    total = T + 400
    latent = np.zeros(total, dtype=float)
    latent_noise = rng.normal(scale=1.0, size=total)
    for t in range(1, total):
        latent[t] = latent_ar * latent[t - 1] + latent_noise[t]

    conf_weights = rng.normal(scale=conf_strength, size=d)
    X = np.zeros((total, d), dtype=float)
    noise = rng.normal(scale=noise_scale, size=(total, d))
    for t in range(1, total):
        X[t] = a1 @ X[t - 1] + conf_weights * latent[t - 1] + noise[t]

    return ScenarioResult(
        X=X[400:],
        ground_truth_compact=compact_ground_truth([a1]),
        metadata={
            "scenario": "latent_common_driver",
            "latent": True,
            "true_order": 1,
            "conf_strength": conf_strength,
            "latent_ar": latent_ar,
            "noise_scale": noise_scale,
        },
    )


def scenario_variable_lag_unconfounded(
    *,
    T: int = 2500,
    d: int = 10,
    noise_scale: float = 1.0,
    seed: int = 0,
) -> ScenarioResult:
    """Lag-heterogeneous but unconfounded control."""

    a1 = make_sparse_matrix(d, edge_prob=0.08, seed=seed + 1) * 0.60
    a2 = make_sparse_matrix(d, edge_prob=0.08, seed=seed + 2) * 0.35
    a3 = make_sparse_matrix(d, edge_prob=0.05, seed=seed + 3) * 0.20
    total_radius = spectral_radius(a1) + spectral_radius(a2) + spectral_radius(a3)
    if total_radius > 1e-12:
        scale = 0.80 / total_radius
        a1, a2, a3 = scale * a1, scale * a2, scale * a3

    X = simulate_var(
        T=T,
        lag_matrices=[a1, a2, a3],
        noise_scale=noise_scale,
        seed=seed + 10,
    )
    return ScenarioResult(
        X=X,
        ground_truth_compact=compact_ground_truth([a1, a2, a3]),
        metadata={
            "scenario": "variable_lag_unconfounded",
            "latent": False,
            "true_order": 3,
            "noise_scale": noise_scale,
        },
    )


def scenario_hidden_nodes(
    *,
    T: int = 2000,
    d_total: int = 16,
    d_observed: int = 10,
    noise_scale: float = 1.0,
    seed: int = 0,
) -> ScenarioResult:
    """Partial-observation latent-confounding scenario."""

    a1 = scale_to_stability(make_sparse_matrix(d_total, edge_prob=0.10, seed=seed))
    X_full = simulate_var(T=T, lag_matrices=[a1], noise_scale=noise_scale, seed=seed + 1)
    observed_indices = np.arange(d_observed)
    hidden_indices = np.arange(d_observed, d_total)
    observed_truth = (np.abs(a1[np.ix_(observed_indices, observed_indices)]) > 1e-12).astype(int)
    np.fill_diagonal(observed_truth, 0)
    return ScenarioResult(
        X=X_full[:, observed_indices],
        ground_truth_compact=observed_truth,
        metadata={
            "scenario": "hidden_nodes",
            "latent": True,
            "true_order": 1,
            "d_total": d_total,
            "d_observed": d_observed,
            "hidden_indices": hidden_indices.tolist(),
            "noise_scale": noise_scale,
        },
    )


SCENARIOS = {
    "order1_unconfounded": scenario_order1_unconfounded,
    "order3_unconfounded": scenario_order3_unconfounded,
    "latent_common_driver": scenario_latent_common_driver,
    "variable_lag_unconfounded": scenario_variable_lag_unconfounded,
    "hidden_nodes": scenario_hidden_nodes,
}
