"""Deterministic data generation for cross-method benchmark notebooks.

All active method notebooks (c-GC, c-GC*, and PCMCI+) call
:func:`generate_trial_data` so that, for a given ``(trial_id, seed)``, every
method observes the *exact same* simulated data. This enables apples-to-apples
comparison across methods.

Determinism is guaranteed by reseeding the global NumPy RNG with
``seed + trial_id`` at the start of each trial, immediately before any random
draw. Because the reseed happens per trial, the generated data is independent
of whatever the inference algorithms consume from the global RNG between
trials.

The number of variables and the time-series length are always the first two
draws in every scenario, so they are identical across *all* notebooks for a
given ``trial_id`` (only the realised series ``X`` differs by scenario, as
dictated by each data-generating process).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from markovianity_diagnostic.core.utils import adj_mtx, continuous_noise_fun

DEFAULT_SEED: int = 42
DEFAULT_N_VARS_RANGE: tuple[int, int] = (18, 25)  # inclusive of both bounds
DEFAULT_LENGTH_RANGE: tuple[int, int] = (3000, 6000)  # inclusive of both bounds

VALID_SCENARIOS: tuple[str, ...] = (
    "singleLag-Markovian",
    "singleLag-NonMarkovian",
    "varLags-Markovian",
    "varLags-NonMarkovian",
)


@dataclass(frozen=True)
class TrialData:
    """Container for one trial's simulated data and ground truth.

    Attributes:
        X: Observed series with shape ``(n_vars, length)``.
        A: Ground-truth adjacency used for scoring (weighted for single-lag,
            boolean for variable-lag scenarios).
        n_vars: Number of variables drawn for this trial.
        length: Number of timepoints drawn for this trial.
        scenario: Scenario name (see :data:`VALID_SCENARIOS`).
        trial_id: Trial index.
        seed: Master seed; the effective per-trial seed is ``seed + trial_id``.
    """

    X: np.ndarray
    A: np.ndarray
    n_vars: int
    length: int
    scenario: str
    trial_id: int
    seed: int


def _draw_inclusive(low: int, high: int) -> int:
    """Draw an integer uniformly from the inclusive range ``[low, high]``."""

    return int(np.random.randint(low, high + 1))


def generate_trial_data(
    scenario: str,
    trial_id: int,
    *,
    seed: int = DEFAULT_SEED,
    n_vars_range: tuple[int, int] = DEFAULT_N_VARS_RANGE,
    length_range: tuple[int, int] = DEFAULT_LENGTH_RANGE,
) -> TrialData:
    """Generate deterministic benchmark data for a single trial.

    Args:
        scenario: One of :data:`VALID_SCENARIOS`. Selects the data-generating
            process: ``Markovian`` uses white process noise only, while
            ``NonMarkovian`` adds a smooth (colored) confounding signal;
            ``singleLag`` uses a first-order process, while ``varLags`` splits
            the coupling into lag-1 and lag-2 components.
        trial_id: Trial index; combined with ``seed`` to seed the RNG.
        seed: Master seed. The effective per-trial seed is ``seed + trial_id``.
        n_vars_range: Inclusive ``(low, high)`` range for the variable count.
        length_range: Inclusive ``(low, high)`` range for the series length.

    Returns:
        A :class:`TrialData` instance.

    Raises:
        ValueError: If ``scenario`` is not a recognised scenario name.
    """

    if scenario not in VALID_SCENARIOS:
        raise ValueError(
            f"Unknown scenario {scenario!r}; expected one of {VALID_SCENARIOS}."
        )

    # Reseed per trial so the data is identical across methods regardless of
    # how each inference algorithm uses the global RNG between trials.
    np.random.seed(seed + trial_id)

    # n_vars and length are always the first two draws -> identical across all
    # scenarios/notebooks for a given (seed, trial_id).
    n_vars = _draw_inclusive(*n_vars_range)
    length = _draw_inclusive(*length_range)

    non_markovian = "NonMarkovian" in scenario
    var_lags = scenario.startswith("varLags")

    noise = continuous_noise_fun(num=n_vars, length=length) if non_markovian else None

    A = adj_mtx(n_vars)

    A_lag1 = A_lag2 = None
    if var_lags:
        mask = np.random.random((A.shape[0], A.shape[1])) > 0.55
        np.fill_diagonal(mask, 1)
        A_lag2 = np.multiply(A, 1 - mask)
        A_lag1 = np.multiply(A, mask)
        A = np.logical_or(A_lag1, A_lag2)

    X = np.zeros((A.shape[0], length)).T
    X[0] = np.random.randn(A.shape[0])
    for i in range(length - 1):
        if var_lags:
            step = A_lag1 @ X[i] + A_lag2 @ X[i - 1]
        else:
            step = A @ X[i]
        step = step + np.random.normal(0, 0.25, A.shape[0])
        if non_markovian:
            step = step + noise[:, i]
        X[i + 1] = step
    X = X.T

    return TrialData(
        X=X,
        A=A,
        n_vars=n_vars,
        length=length,
        scenario=scenario,
        trial_id=trial_id,
        seed=seed,
    )


__all__ = [
    "TrialData",
    "generate_trial_data",
    "DEFAULT_SEED",
    "DEFAULT_N_VARS_RANGE",
    "DEFAULT_LENGTH_RANGE",
    "VALID_SCENARIOS",
]
