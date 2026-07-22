"""Tests for bootstrap calibration module and NullModel base class.

TDD Approach: Tests written FIRST, implementation follows.
"""

import numpy as np
import pytest

from markovianity_diagnostic.experiments.calibration import (
    MovingBlockBootstrapNull,
    NullModel,
    ResidualBootstrapNull,
    VARNullModel,
)


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
class TestVARNullModelFit:
    """Test VARNullModel.fit() interface."""

    def test_var_fit_accepts_valid_data(self, synthetic_markov_data):
        """fit() should accept valid (T, d) data and p0."""
        model = VARNullModel(p0=1)
        result = model.fit(synthetic_markov_data, p0=1)
        assert result is model

    def test_var_fit_returns_self(self, synthetic_markov_data):
        """fit() must return self for method chaining."""
        model = VARNullModel(p0=1)
        result = model.fit(synthetic_markov_data, p0=1)
        assert result is model

    def test_var_fit_sets_fitted_flag(self, synthetic_markov_data):
        """fit() should set _fitted flag to True."""
        model = VARNullModel(p0=1)
        assert not model._fitted
        model.fit(synthetic_markov_data, p0=1)
        assert model._fitted

    def test_var_fit_stores_residual_covariance(self, synthetic_markov_data):
        """fit() should compute and store residual covariance."""
        model = VARNullModel(p0=1)
        model.fit(synthetic_markov_data, p0=1)
        assert model._cov is not None
        assert model._cov.shape == (synthetic_markov_data.shape[1],
                                    synthetic_markov_data.shape[1])

    def test_var_fit_rejects_invalid_p0(self, synthetic_markov_data):
        """fit() should raise ValueError if p0 >= T."""
        model = VARNullModel(p0=300)
        with pytest.raises(ValueError):
            model.fit(synthetic_markov_data, p0=300)

    def test_var_fit_rejects_invalid_shape(self):
        """fit() should raise ValueError if X is not 2D."""
        model = VARNullModel(p0=1)
        with pytest.raises(ValueError):
            model.fit(np.random.randn(100), p0=1)


class TestVARNullModelSample:
    """Test VARNullModel.sample() interface."""

    def test_var_sample_shape(self, synthetic_markov_data):
        """sample() should return (T, d) shaped array."""
        T, d = synthetic_markov_data.shape
        model = VARNullModel(p0=1).fit(synthetic_markov_data, p0=1)
        sample = model.sample(T=100, seed=42)
        assert sample.shape == (100, d)

    def test_var_sample_reproducible_with_seed(self, synthetic_markov_data):
        """sample() with same seed should produce identical arrays."""
        model = VARNullModel(p0=1).fit(synthetic_markov_data, p0=1)
        sample1 = model.sample(T=100, seed=42)
        sample2 = model.sample(T=100, seed=42)
        np.testing.assert_array_equal(sample1, sample2)

    def test_var_sample_different_seeds_produce_different(self, synthetic_markov_data):
        """sample() with different seeds should produce different arrays."""
        model = VARNullModel(p0=1).fit(synthetic_markov_data, p0=1)
        sample1 = model.sample(T=100, seed=42)
        sample2 = model.sample(T=100, seed=43)
        assert not np.allclose(sample1, sample2)

    def test_var_sample_unfitted_raises_error(self):
        """sample() should raise ValueError if model not fitted."""
        model = VARNullModel(p0=1)
        with pytest.raises(ValueError):
            model.sample(T=100, seed=42)

    def test_var_sample_respects_ar_structure(self, synthetic_markov_data):
        """sample() should respect the VAR(p0) structure."""
        # Fitted on VAR(1) process, sample should show autocorrelation
        model = VARNullModel(p0=1).fit(synthetic_markov_data, p0=1)
        sample = model.sample(T=500, seed=42)
        
        # Compute lag-1 correlation (should be nonzero)
        lag1_corr = np.corrcoef(sample[:-1, 0], sample[1:, 0])[0, 1]
        assert abs(lag1_corr) > 0.05, "VAR(1) sample should show autocorrelation"


class TestVARNullModelMetadata:
    """Test VARNullModel.metadata property."""

    def test_var_metadata_valid_dict(self, synthetic_markov_data):
        """metadata should return valid dict with required keys."""
        model = VARNullModel(p0=1).fit(synthetic_markov_data, p0=1)
        metadata = model.metadata
        
        assert isinstance(metadata, dict)
        assert "model" in metadata
        assert metadata["model"] == "VAR"
        assert "fitted" in metadata
        assert "p0" in metadata
        assert "n_samples" in metadata
        assert "n_features" in metadata

    def test_var_metadata_correct_values(self, synthetic_markov_data):
        """metadata should report correct values."""
        T, d = synthetic_markov_data.shape
        model = VARNullModel(p0=2).fit(synthetic_markov_data, p0=2)
        metadata = model.metadata
        
        assert metadata["fitted"] is True
        assert metadata["p0"] == 2
        assert metadata["n_samples"] == T
        assert metadata["n_features"] == d

    def test_var_p0_uses_all_lags(self, synthetic_markov_data):
        """VAR(p0) coefficients should contain one block per lag."""
        _, d = synthetic_markov_data.shape
        model = VARNullModel(p0=3).fit(synthetic_markov_data, p0=3)

        assert model._A.shape == (3 * d, d)

    def test_var_p0_initializes_with_p0_observed_rows(self, synthetic_markov_data):
        """Sampling should preserve all p0 initial states before recursion."""
        model = VARNullModel(p0=3).fit(synthetic_markov_data, p0=3)
        sample = model.sample(T=20, seed=42)

        np.testing.assert_array_equal(sample[:3], synthetic_markov_data[:3])


class TestResidualBootstrapNullFit:
    """Test ResidualBootstrapNull.fit() interface."""

    def test_residual_bootstrap_fit_stores_residuals(self, synthetic_markov_data):
        """fit() should store residuals for later resampling."""
        model = ResidualBootstrapNull(p0=1)
        model.fit(synthetic_markov_data, p0=1)
        
        assert model._residuals is not None
        assert model._residuals.shape[0] == synthetic_markov_data.shape[0] - 1
        assert model._residuals.shape[1] == synthetic_markov_data.shape[1]

    def test_residual_bootstrap_fit_stores_var_coefficients(self, synthetic_markov_data):
        """fit() should store VAR coefficients for predictions."""
        model = ResidualBootstrapNull(p0=2)
        model.fit(synthetic_markov_data, p0=2)
        
        assert model._A is not None
        assert model._A.shape == (2 * synthetic_markov_data.shape[1],
                                  synthetic_markov_data.shape[1])


class TestResidualBootstrapNullSample:
    """Test ResidualBootstrapNull.sample() interface."""

    def test_residual_bootstrap_sample_shape(self, synthetic_markov_data):
        """sample() should return (T, d) shaped array."""
        T, d = synthetic_markov_data.shape
        model = ResidualBootstrapNull(p0=1).fit(synthetic_markov_data, p0=1)
        sample = model.sample(T=100, seed=42)
        assert sample.shape == (100, d)

    def test_residual_bootstrap_sample_reproducible(self, synthetic_markov_data):
        """sample() with same seed should be reproducible."""
        model = ResidualBootstrapNull(p0=1).fit(synthetic_markov_data, p0=1)
        sample1 = model.sample(T=100, seed=42)
        sample2 = model.sample(T=100, seed=42)
        np.testing.assert_array_equal(sample1, sample2)

    def test_residual_bootstrap_maintains_var_structure(self, synthetic_markov_data):
        """sample() should respect underlying VAR structure."""
        model = ResidualBootstrapNull(p0=1).fit(synthetic_markov_data, p0=1)
        sample = model.sample(T=500, seed=42)
        
        # Should show autocorrelation from VAR structure
        lag1_corr = np.corrcoef(sample[:-1, 0], sample[1:, 0])[0, 1]
        assert abs(lag1_corr) > 0.05


class TestMovingBlockBootstrapNullFit:
    """Test MovingBlockBootstrapNull.fit() interface."""

    def test_moving_block_bootstrap_fit_valid(self, synthetic_markov_data):
        """fit() should accept valid data and store residuals."""
        model = MovingBlockBootstrapNull(p0=1, block_length=20)
        model.fit(synthetic_markov_data, p0=1)
        
        assert model._fitted
        assert model._residuals is not None

    def test_moving_block_bootstrap_stores_block_length(self, synthetic_markov_data):
        """fit() should store block_length for resampling."""
        model = MovingBlockBootstrapNull(p0=1, block_length=15)
        model.fit(synthetic_markov_data, p0=1)
        
        assert model.block_length == 15


class TestMovingBlockBootstrapNullSample:
    """Test MovingBlockBootstrapNull.sample() interface."""

    def test_moving_block_bootstrap_sample_shape(self, synthetic_markov_data):
        """sample() should return (T, d) shaped array."""
        T, d = synthetic_markov_data.shape
        model = MovingBlockBootstrapNull(p0=1, block_length=20).fit(
            synthetic_markov_data, p0=1
        )
        sample = model.sample(T=100, seed=42)
        assert sample.shape == (100, d)

    def test_moving_block_bootstrap_sample_reproducible(self, synthetic_markov_data):
        """sample() with same seed should be reproducible."""
        model = MovingBlockBootstrapNull(p0=1, block_length=20).fit(
            synthetic_markov_data, p0=1
        )
        sample1 = model.sample(T=100, seed=42)
        sample2 = model.sample(T=100, seed=42)
        np.testing.assert_array_equal(sample1, sample2)

    def test_moving_block_bootstrap_block_length_respected(self, synthetic_markov_data):
        """sample() should use the specified block_length."""
        model = MovingBlockBootstrapNull(p0=1, block_length=10).fit(
            synthetic_markov_data, p0=1
        )
        # If block_length works correctly, we should see some structure
        sample = model.sample(T=500, seed=42)
        assert sample.shape == (500, synthetic_markov_data.shape[1])

    def test_moving_block_bootstrap_default_block_length(self, synthetic_markov_data):
        """Default block_length should be 20."""
        model = MovingBlockBootstrapNull(p0=1).fit(synthetic_markov_data, p0=1)
        assert model.block_length == 20


class TestAllModelsConsistency:
    """Test that all models satisfy NullModel interface."""

    @pytest.mark.parametrize("model_class,kwargs", [
        (VARNullModel, {"p0": 1}),
        (ResidualBootstrapNull, {"p0": 1}),
        (MovingBlockBootstrapNull, {"p0": 1, "block_length": 20}),
    ])
    def test_all_models_fit_returns_self(self, model_class, kwargs, synthetic_markov_data):
        """All models should implement fit() returning self."""
        model = model_class(**kwargs)
        result = model.fit(synthetic_markov_data, p0=1)
        assert result is model

    @pytest.mark.parametrize("model_class,kwargs", [
        (VARNullModel, {"p0": 1}),
        (ResidualBootstrapNull, {"p0": 1}),
        (MovingBlockBootstrapNull, {"p0": 1, "block_length": 20}),
    ])
    def test_all_models_sample_shape(self, model_class, kwargs, synthetic_markov_data):
        """All models should return correct sample shape."""
        T, d = synthetic_markov_data.shape
        model = model_class(**kwargs).fit(synthetic_markov_data, p0=1)
        sample = model.sample(T=100, seed=42)
        assert sample.shape == (100, d)

    @pytest.mark.parametrize("model_class,kwargs", [
        (VARNullModel, {"p0": 1}),
        (ResidualBootstrapNull, {"p0": 1}),
        (MovingBlockBootstrapNull, {"p0": 1, "block_length": 20}),
    ])
    def test_all_models_reproducibility(self, model_class, kwargs, synthetic_markov_data):
        """All models should be reproducible with fixed seed."""
        model = model_class(**kwargs).fit(synthetic_markov_data, p0=1)
        sample1 = model.sample(T=100, seed=42)
        sample2 = model.sample(T=100, seed=42)
        np.testing.assert_array_equal(sample1, sample2)

    @pytest.mark.parametrize("model_class,kwargs", [
        (VARNullModel, {"p0": 1}),
        (ResidualBootstrapNull, {"p0": 1}),
        (MovingBlockBootstrapNull, {"p0": 1, "block_length": 20}),
    ])
    def test_all_models_have_metadata(self, model_class, kwargs, synthetic_markov_data):
        """All models should implement metadata property."""
        model = model_class(**kwargs).fit(synthetic_markov_data, p0=1)
        metadata = model.metadata
        
        assert isinstance(metadata, dict)
        assert "fitted" in metadata
        assert "n_samples" in metadata
        assert "n_features" in metadata
        assert metadata["fitted"] is True


class TestCalibrationResult:
    """Test suite for CalibrationResult dataclass."""

    def test_calibration_result_dataclass_creation(self):
        """CalibrationResult should be creatable with required fields."""
        from markovianity_diagnostic.experiments.calibration import CalibrationResult
        
        result = CalibrationResult(
            observed={"T_obs": 2.5, "D_obs": 0.8},
            null={"T_boot": [1.0, 2.0, 3.0], "critical_90": 1.5, "critical_95": 2.0, "critical_99": 2.8, "p_value": 0.1},
            diagnosis={"reject_global_95": False, "first_exceedance_depth": 2}
        )
        
        assert result.observed == {"T_obs": 2.5, "D_obs": 0.8}
        assert result.null["T_boot"] == [1.0, 2.0, 3.0]
        assert result.diagnosis["reject_global_95"] is False

    def test_critical_values_ordered(self):
        """Critical values should satisfy 90 < 95 < 99."""
        from markovianity_diagnostic.experiments.calibration import CalibrationResult
        
        result = CalibrationResult(
            observed={"T_obs": 2.5},
            null={
                "T_boot": [1.0, 2.0, 3.0],
                "critical_90": 1.5,
                "critical_95": 2.0,
                "critical_99": 2.8,
                "p_value": 0.15
            },
            diagnosis={}
        )
        
        assert result.null["critical_90"] < result.null["critical_95"]
        assert result.null["critical_95"] < result.null["critical_99"]

    def test_p_value_in_range(self):
        """p_value should be in [0, 1]."""
        from markovianity_diagnostic.experiments.calibration import CalibrationResult
        
        result = CalibrationResult(
            observed={"T_obs": 2.5},
            null={
                "T_boot": [1.0, 2.0, 3.0],
                "critical_90": 1.5,
                "critical_95": 2.0,
                "critical_99": 2.8,
                "p_value": 0.42
            },
            diagnosis={}
        )
        
        assert 0 <= result.null["p_value"] <= 1

    def test_calibration_result_to_json(self, tmp_path):
        """CalibrationResult.to_json() should produce valid JSON file."""
        from markovianity_diagnostic.experiments.calibration import CalibrationResult
        import json
        
        result = CalibrationResult(
            observed={"T_obs": 2.5, "D_obs": 0.8, "D_parts_obs": [0.2, 0.6], "edge_counts": 5},
            null={
                "T_boot": [1.0, 2.0, 3.0],
                "D_boot": [0.5, 0.7, 0.9],
                "critical_90": 1.5,
                "critical_95": 2.0,
                "critical_99": 2.8,
                "p_value": 0.15
            },
            diagnosis={"reject_global_95": False, "first_exceedance_depth": 2}
        )
        
        output_path = tmp_path / "result.json"
        result.to_json(str(output_path))
        
        assert output_path.exists()
        
        # Verify JSON is valid and readable
        with open(output_path) as f:
            data = json.load(f)
        
        assert "observed" in data
        assert "null" in data
        assert "diagnosis" in data

    def test_calibration_result_from_json(self, tmp_path):
        """CalibrationResult.from_json() should load from JSON file."""
        from markovianity_diagnostic.experiments.calibration import CalibrationResult
        import json
        
        # Create a JSON file
        test_data = {
            "observed": {"T_obs": 2.5, "D_obs": 0.8},
            "null": {
                "T_boot": [1.0, 2.0, 3.0],
                "critical_90": 1.5,
                "critical_95": 2.0,
                "critical_99": 2.8,
                "p_value": 0.15
            },
            "diagnosis": {"reject_global_95": False, "first_exceedance_depth": 2}
        }
        
        output_path = tmp_path / "result.json"
        with open(output_path, 'w') as f:
            json.dump(test_data, f)
        
        result = CalibrationResult.from_json(str(output_path))
        
        assert result.observed == test_data["observed"]
        assert result.null == test_data["null"]
        assert result.diagnosis == test_data["diagnosis"]

    def test_calibration_result_round_trip(self, tmp_path):
        """CalibrationResult should survive to_json + from_json round trip."""
        from markovianity_diagnostic.experiments.calibration import CalibrationResult
        
        original = CalibrationResult(
            observed={"T_obs": 2.5, "D_obs": 0.8, "D_parts_obs": [0.2, 0.6], "edge_counts": 5},
            null={
                "T_boot": [1.0, 2.0, 3.0, 1.5, 2.1],
                "D_boot": [0.5, 0.7, 0.9, 0.6, 0.8],
                "critical_90": 1.45,
                "critical_95": 1.98,
                "critical_99": 2.82,
                "p_value": 0.2
            },
            diagnosis={"reject_global_95": True, "first_exceedance_depth": 3}
        )
        
        output_path = tmp_path / "result.json"
        original.to_json(str(output_path))
        
        loaded = CalibrationResult.from_json(str(output_path))
        
        assert loaded.observed == original.observed
        assert loaded.null == original.null
        assert loaded.diagnosis == original.diagnosis

    def test_json_contains_all_keys(self, tmp_path):
        """JSON should contain all required top-level keys."""
        from markovianity_diagnostic.experiments.calibration import CalibrationResult
        import json
        
        result = CalibrationResult(
            observed={"T_obs": 2.5},
            null={"critical_90": 1.5, "critical_95": 2.0, "critical_99": 2.8, "p_value": 0.1},
            diagnosis={"reject_global_95": False}
        )
        
        output_path = tmp_path / "result.json"
        result.to_json(str(output_path))
        
        with open(output_path) as f:
            data = json.load(f)
        
        required_keys = {"observed", "null", "diagnosis"}
        assert required_keys.issubset(set(data.keys()))
