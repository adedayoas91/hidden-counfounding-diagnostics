"""Positive and negative controls for Markovianity diagnostic validation.

This module implements synthetic and real-data controls that serve as
validation baselines for the Markovianity diagnostic:

- Synthetic Markovian nulls: Order-p VAR processes without confounding
- Synthetic latent positive: Systems with known latent confounders
- Real-data controls: Shuffled/randomized versions of empirical data

Reference:
    Suggestion Implementation Plan, Phase 5: Positive and Negative Controls
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from markovianity_diagnostic.experiments.simulations import (
    ScenarioResult,
    make_sparse_matrix,
    scale_to_stability,
    simulate_var,
    compact_ground_truth,
    spectral_radius,
)


class SyntheticMarkovianNull:
    """Order-p Markov null control without latent confounding.

    Generates a stable VAR(p) process with known Markovian structure.
    Used to validate that the diagnostic shows low instability after the true order.

    Parameters
    ----------
    order : int, default=1
        Markov order p for the VAR(p) process.
    edge_prob : float, default=0.12
        Probability of edge in the coefficient matrix.
    """

    def __init__(self, order: int = 1, edge_prob: float = 0.12):
        """Initialize Markovian null control."""
        self.order = order
        self.edge_prob = edge_prob

    def generate(
        self,
        T: int = 2000,
        d: int = 10,
        noise_scale: float = 1.0,
        seed: int = 0,
    ) -> ScenarioResult:
        """Generate order-p Markovian null data.

        Parameters
        ----------
        T : int, default=2000
            Time series length.
        d : int, default=10
            Number of variables.
        noise_scale : float, default=1.0
            Noise standard deviation.
        seed : int, default=0
            Random seed for reproducibility.

        Returns
        -------
        ScenarioResult
            Generated data with ground truth adjacency and metadata.
        """
        rng = np.random.default_rng(seed)

        # Generate lag matrices with decreasing magnitude
        lag_matrices = []
        for lag_idx in range(1, self.order + 1):
            edge_prob = max(0.04, self.edge_prob - 0.02 * lag_idx)
            a = make_sparse_matrix(
                d,
                edge_prob=edge_prob,
                weight_low=0.10,
                weight_high=0.30,
                seed=seed + lag_idx,
            )
            magnitude = 0.60 - 0.15 * lag_idx
            lag_matrices.append(a * magnitude)

        # Scale to stability
        total_radius = sum(spectral_radius(a) for a in lag_matrices)
        if total_radius > 1e-12:
            scale = 0.80 / total_radius
            lag_matrices = [scale * a for a in lag_matrices]

        # Generate VAR process
        X = simulate_var(
            T=T,
            lag_matrices=lag_matrices,
            noise_scale=noise_scale,
            seed=seed + 100,
        )

        return ScenarioResult(
            X=X,
            ground_truth_compact=compact_ground_truth(lag_matrices),
            metadata={
                "scenario": "markovian_null",
                "latent": False,
                "true_order": self.order,
                "edge_prob": self.edge_prob,
                "noise_scale": noise_scale,
            },
        )


class SyntheticLatentControlPositive:
    """Latent common driver positive control.

    Generates order-1 observed system perturbed by a latent AR(1) common driver.
    Used to validate that the diagnostic detects hidden confounding.

    Parameters
    ----------
    conf_strength : float, default=0.35
        Strength of latent confounder coupling to observed variables.
    latent_ar : float, default=0.80
        AR coefficient of latent confounder process.
    """

    def __init__(self, conf_strength: float = 0.35, latent_ar: float = 0.80):
        """Initialize latent common driver control."""
        self.conf_strength = conf_strength
        self.latent_ar = latent_ar

    def generate(
        self,
        T: int = 2000,
        d: int = 10,
        noise_scale: float = 1.0,
        seed: int = 0,
    ) -> ScenarioResult:
        """Generate latent common driver data.

        Parameters
        ----------
        T : int, default=2000
            Time series length (not including burn-in).
        d : int, default=10
            Number of observed variables.
        noise_scale : float, default=1.0
            Observed noise standard deviation.
        seed : int, default=0
            Random seed for reproducibility.

        Returns
        -------
        ScenarioResult
            Generated data with ground truth adjacency and metadata.
        """
        rng = np.random.default_rng(seed)

        # Base order-1 system
        a1 = scale_to_stability(
            make_sparse_matrix(d, edge_prob=0.10, seed=seed + 1) * 0.50
        )

        # Latent confounder: AR(1) process
        total_len = T + 400
        latent = np.zeros(total_len)
        latent_noise = rng.normal(scale=1.0, size=total_len)
        for t in range(1, total_len):
            latent[t] = self.latent_ar * latent[t - 1] + latent_noise[t]

        # Confounding coupling weights
        conf_weights = rng.normal(scale=self.conf_strength, size=d)

        # Generate observed data with confounding
        X = np.zeros((total_len, d))
        obs_noise = rng.normal(scale=noise_scale, size=(total_len, d))
        for t in range(1, total_len):
            X[t] = a1 @ X[t - 1] + conf_weights * latent[t - 1] + obs_noise[t]

        return ScenarioResult(
            X=X[400:],
            ground_truth_compact=compact_ground_truth([a1]),
            metadata={
                "scenario": "latent_common_driver",
                "latent": True,
                "true_order": 1,
                "conf_strength": self.conf_strength,
                "latent_ar": self.latent_ar,
                "noise_scale": noise_scale,
            },
        )


class SyntheticHiddenNodesPositive:
    """Hidden nodes positive control.

    Generates a larger system with some nodes unobserved.
    Used to validate that the diagnostic detects confounding from hidden variables.

    Parameters
    ----------
    d_total : int, default=16
        Total number of nodes (observed + hidden).
    d_observed : int, default=10
        Number of observed nodes.
    """

    def __init__(self, d_total: int = 16, d_observed: int = 10):
        """Initialize hidden nodes control."""
        self.d_total = d_total
        self.d_observed = d_observed

    def generate(
        self,
        T: int = 2000,
        d: int | None = None,
        noise_scale: float = 1.0,
        seed: int = 0,
    ) -> ScenarioResult:
        """Generate hidden nodes data.

        Parameters
        ----------
        T : int, default=2000
            Time series length.
        d : int, optional
            Ignored; output dimension is self.d_observed.
        noise_scale : float, default=1.0
            Noise standard deviation.
        seed : int, default=0
            Random seed for reproducibility.

        Returns
        -------
        ScenarioResult
            Generated observed data with ground truth adjacency over observed nodes.
        """
        # Generate full system
        a1 = scale_to_stability(
            make_sparse_matrix(self.d_total, edge_prob=0.10, seed=seed)
        )
        X_full = simulate_var(
            T=T, lag_matrices=[a1], noise_scale=noise_scale, seed=seed + 1
        )

        # Extract observed nodes
        observed_indices = np.arange(self.d_observed)
        hidden_indices = np.arange(self.d_observed, self.d_total)

        # Ground truth for observed system
        observed_truth = (
            np.abs(a1[np.ix_(observed_indices, observed_indices)]) > 1e-12
        ).astype(int)
        np.fill_diagonal(observed_truth, 0)

        return ScenarioResult(
            X=X_full[:, observed_indices],
            ground_truth_compact=observed_truth,
            metadata={
                "scenario": "hidden_nodes",
                "latent": True,
                "true_order": 1,
                "d_total": self.d_total,
                "d_observed": self.d_observed,
                "hidden_indices": hidden_indices.tolist(),
                "noise_scale": noise_scale,
            },
        )


class TimeShuffledControl:
    """Time-shuffled control for real data.

    Shuffles time indices to break temporal structure while preserving
    marginal distributions. Used as a negative control on real data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (T, d).
    """

    def __init__(self, X: np.ndarray):
        """Initialize time-shuffled control."""
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        self.X = X

    def generate(self, seed: int = 0) -> np.ndarray:
        """Generate time-shuffled version.

        Parameters
        ----------
        seed : int, default=0
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Shuffled data with same shape as input.
        """
        rng = np.random.default_rng(seed)
        T = self.X.shape[0]
        shuffled_indices = rng.permutation(T)
        return self.X[shuffled_indices]


class BlockShuffledControl:
    """Block-shuffled control for real data.

    Shuffles contiguous blocks of data to partially preserve short-range
    temporal structure while breaking long-range dependencies.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (T, d).
    block_size : int, default=50
        Size of blocks to shuffle.
    """

    def __init__(self, X: np.ndarray, block_size: int = 50):
        """Initialize block-shuffled control."""
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        self.X = X
        self.block_size = block_size

    def generate(self, seed: int = 0) -> np.ndarray:
        """Generate block-shuffled version.

        Parameters
        ----------
        seed : int, default=0
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Block-shuffled data with same shape as input.
        """
        rng = np.random.default_rng(seed)
        T = self.X.shape[0]

        # Determine block positions
        n_blocks = (T + self.block_size - 1) // self.block_size
        block_indices = list(range(n_blocks))
        rng.shuffle(block_indices)

        # Assemble shuffled data
        X_shuffled = np.zeros_like(self.X)
        idx = 0
        for block_idx in block_indices:
            start = block_idx * self.block_size
            end = min(start + self.block_size, T)
            block_len = end - start
            X_shuffled[idx : idx + block_len] = self.X[start:end]
            idx += block_len

        return X_shuffled


class PhaseRandomizedControl:
    """Phase-randomized control for real data.

    Randomizes the phase of the Fourier transform while preserving
    power spectrum, to break autocorrelations while maintaining spectral properties.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (T, d).
    """

    def __init__(self, X: np.ndarray):
        """Initialize phase-randomized control."""
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        self.X = X

    def generate(self, seed: int = 0) -> np.ndarray:
        """Generate phase-randomized version.

        Parameters
        ----------
        seed : int, default=0
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Phase-randomized data with same shape as input.
        """
        rng = np.random.default_rng(seed)
        T, d = self.X.shape

        X_randomized = np.zeros_like(self.X)

        for col_idx in range(d):
            # Compute FFT
            fft_vals = np.fft.fft(self.X[:, col_idx])

            # Extract magnitude and phase
            magnitude = np.abs(fft_vals)
            phase = np.angle(fft_vals)

            # Randomize phase
            phase_random = rng.uniform(0, 2 * np.pi, size=T)

            # Reconstruct with randomized phase
            fft_randomized = magnitude * np.exp(1j * phase_random)
            X_randomized[:, col_idx] = np.real(np.fft.ifft(fft_randomized))

        return X_randomized


class CircularlyShiftedControl:
    """Circularly shifted control for real data.

    Shifts the time series by a fixed lag with wraparound, breaking
    temporal dependencies while preserving values.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (T, d).
    lag : int, default=1
        Number of time steps to shift.
    """

    def __init__(self, X: np.ndarray, lag: int = 1):
        """Initialize circularly shifted control."""
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        self.X = X
        self.lag = lag

    def generate(self, seed: int = 0) -> np.ndarray:
        """Generate circularly shifted version.

        Parameters
        ----------
        seed : int, default=0
            Unused (shift is deterministic).

        Returns
        -------
        np.ndarray
            Circularly shifted data with same shape as input.
        """
        return np.roll(self.X, self.lag, axis=0)
