"""Bootstrap calibration for graph-instability diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .graph_metrics import compute_graph_instability


def fit_null_var(X: np.ndarray, p0: int) -> dict[str, Any]:
    """Fit a simple order-``p0`` linear autoregressive null model."""

    T, d = X.shape
    if p0 < 1:
        raise ValueError("p0 must be >= 1.")
    if T <= p0 + 5:
        raise ValueError("Time series is too short for the requested p0.")

    Y = X[p0:]
    rows = []
    for t in range(p0, T):
        rows.append(np.concatenate([X[t - lag] for lag in range(1, p0 + 1)]))
    Z = np.asarray(rows)
    coef, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    residuals = Y - Z @ coef
    init = X[:p0].copy()
    return {"coef": coef, "residuals": residuals, "p0": p0, "d": d, "init": init}


def sample_residuals(
    residuals: np.ndarray,
    n: int,
    *,
    rng: np.random.Generator,
    block_length: int = 1,
) -> np.ndarray:
    """Sample residuals either iid or in moving blocks."""

    if block_length <= 1:
        indices = rng.integers(low=0, high=len(residuals), size=n)
        return residuals[indices]

    chunks = []
    while sum(len(chunk) for chunk in chunks) < n:
        start = int(rng.integers(low=0, high=max(1, len(residuals) - block_length + 1)))
        chunks.append(residuals[start : start + block_length])
    return np.vstack(chunks)[:n]


def simulate_from_null(
    fit: dict[str, Any],
    T: int,
    *,
    seed: int,
    burn_in: int = 200,
    block_length: int = 1,
) -> np.ndarray:
    """Simulate a synthetic time series from a fitted null model."""

    rng = np.random.default_rng(seed)
    coef = fit["coef"]
    residuals = fit["residuals"]
    p0 = int(fit["p0"])
    d = int(fit["d"])

    total = T + burn_in
    X = np.zeros((total + p0, d), dtype=float)
    X[:p0] = fit["init"]
    sampled_residuals = sample_residuals(
        residuals,
        total,
        rng=rng,
        block_length=block_length,
    )

    for t in range(p0, total + p0):
        z_t = np.concatenate([X[t - lag] for lag in range(1, p0 + 1)])
        X[t] = z_t @ coef + sampled_residuals[t - p0]

    return X[p0 + burn_in : p0 + burn_in + T]


def max_instability_after_p0(
    adjacencies: dict[int, np.ndarray],
    *,
    p0: int,
) -> tuple[float, dict[int, float]]:
    """Extract ``T_obs`` and the subset of ``D_p`` values for ``p > p0``."""

    d_values = compute_graph_instability(adjacencies)
    selected = {p_value: value for p_value, value in d_values.items() if p_value > p0}
    return (max(selected.values()) if selected else 0.0), selected


def bootstrap_global_test(
    X: np.ndarray,
    *,
    analyze_fn: Callable[[np.ndarray, list[int]], dict[int, np.ndarray]],
    p_values: list[int],
    p0: int,
    B: int = 200,
    block_length: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    """Calibrate ``T_obs = max_{p > p0} D_p`` under an order-``p0`` null."""

    observed_adjacencies = analyze_fn(X, p_values)
    T_obs, D_obs = max_instability_after_p0(observed_adjacencies, p0=p0)

    fit = fit_null_var(X, p0=p0)
    T_boot = []
    D_boot: list[dict[int, float]] = []
    for index in range(B):
        X_star = simulate_from_null(
            fit,
            T=len(X),
            seed=seed + index,
            block_length=block_length,
        )
        adjacencies_star = analyze_fn(X_star, p_values)
        T_star, D_star = max_instability_after_p0(adjacencies_star, p0=p0)
        T_boot.append(float(T_star))
        D_boot.append(D_star)

    T_boot_array = np.asarray(T_boot)
    p_value = float((1 + np.sum(T_boot_array >= T_obs)) / (B + 1))
    critical_95 = float(np.quantile(T_boot_array, 0.95)) if len(T_boot_array) else 0.0

    pointwise_95: dict[int, float] = {}
    for p_value_key in sorted(D_obs):
        values = [d_values.get(p_value_key, 0.0) for d_values in D_boot]
        pointwise_95[p_value_key] = float(np.quantile(values, 0.95)) if values else 0.0

    return {
        "T_obs": float(T_obs),
        "D_obs": {int(key): float(value) for key, value in D_obs.items()},
        "T_boot": [float(value) for value in T_boot],
        "p_value": p_value,
        "critical_95": critical_95,
        "pointwise_95": pointwise_95,
        "p0": int(p0),
        "B": int(B),
        "block_length": int(block_length),
    }
