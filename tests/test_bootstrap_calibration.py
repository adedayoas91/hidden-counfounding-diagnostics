"""Tests for bootstrap calibration module and NullModel base class.

TDD Approach: Tests written FIRST, implementation follows.
"""

import numpy as np
import pytest
from pathlib import Path
from typing import Optional

from markovianity_diagnostic.experiments.calibration import NullModel


class SimpleTestNull(NullModel):
    """Simple concrete implementation of NullModel for testing."""

    def __init__(self):
        self._fitted = False
        self._X = None
        self._p0 = None

    def fit(self, X: np.ndarray, p0: int) -> "SimpleTestNull":
        """Fit model to data."""
        self._fitted = True
        self._X = X.copy()
        self._p0 = p0
        return self

    def sample(self, T: int, seed: int) -> np.ndarray:
        """Sample from fitted model."""
        if not self._fitted:
            raise ValueError("Model must be fitted before sampling")

        np.random.seed(seed)
        d = self._X.shape[1]
        return np.random.randn(T, d)

    @property
    def metadata(self) -> dict:
        """Return model metadata."""
        return {
            "fitted": self._fitted,
            "n_samples": self._X.shape[0] if self._fitted else None,
            "n_features": self._X.shape[1] if self._fitted else None,
        }


class TestNullModelFitReturnsSelf:
    """Test that NullModel.fit() returns self."""

    def test_fit_returns_self(self, synthetic_markov_data):
        """fit() should return the model instance for chaining."""
        model = SimpleTestNull()
        result = model.fit(synthetic_markov_data, p0=1)

        assert result is model, "fit() must return self"


class TestNullModelSampleShape:
    """Test that NullModel.sample() returns correct shape."""

    def test_sample_shape(self, synthetic_markov_data):
        """sample() should return (T, d) shaped array."""
        d = synthetic_markov_data.shape[1]
        model = SimpleTestNull().fit(synthetic_markov_data, p0=1)

        T = 100
        samples = model.sample(T=T, seed=42)

        assert samples.shape == (T, d), f"Expected shape {(T, d)}, got {samples.shape}"


class TestNullModelReproducibility:
    """Test that NullModel sampling is reproducible with fixed seed."""

    def test_reproducibility_with_seed(self, synthetic_markov_data):
        """sample() with same seed should produce identical arrays."""
        model = SimpleTestNull().fit(synthetic_markov_data, p0=1)

        # Sample twice with same seed
        sample1 = model.sample(T=100, seed=42)
        sample2 = model.sample(T=100, seed=42)

        np.testing.assert_array_equal(
            sample1, sample2,
            err_msg="Sampling with same seed must be reproducible"
        )


class TestNullModelMetadata:
    """Test that NullModel.metadata property exists and contains expected keys."""

    def test_metadata_exists(self, synthetic_markov_data):
        """metadata should return dict with required keys."""
        model = SimpleTestNull().fit(synthetic_markov_data, p0=1)

        metadata = model.metadata

        assert isinstance(metadata, dict), "metadata must be a dict"
        assert "fitted" in metadata, "metadata must have 'fitted' key"
        assert "n_samples" in metadata, "metadata must have 'n_samples' key"
        assert "n_features" in metadata, "metadata must have 'n_features' key"

    def test_metadata_values(self, synthetic_markov_data):
        """metadata values should be correct after fit."""
        T, d = synthetic_markov_data.shape
        model = SimpleTestNull().fit(synthetic_markov_data, p0=1)

        metadata = model.metadata

        assert metadata["fitted"] is True
        assert metadata["n_samples"] == T
        assert metadata["n_features"] == d
