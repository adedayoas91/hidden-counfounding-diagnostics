"""Test suite for Phase 3.1 mechanism ablation scenarios."""

from __future__ import annotations

import numpy as np
import pytest

from markovianity_diagnostic.experiments.simulations import (
    ScenarioResult,
    scenario_measurement_noise,
    scenario_nonlinear_hidden_driver,
    scenario_nonlinear_markov,
    scenario_observation_filtering,
    scenario_omitted_lag_order,
    scenario_partial_observation_hidden_nodes_strong,
    scenario_regime_shift,
    scenario_time_varying_coefficients,
    scenario_undersampled_markov,
)


class TestScenarioInterface:
    """All scenarios must return ScenarioResult with valid structure."""

    SCENARIOS = [
        ("omitted_lag_order", scenario_omitted_lag_order),
        ("undersampled_markov", scenario_undersampled_markov),
        ("measurement_noise", scenario_measurement_noise),
        ("time_varying_coefficients", scenario_time_varying_coefficients),
        ("regime_shift", scenario_regime_shift),
        ("nonlinear_markov", scenario_nonlinear_markov),
        ("nonlinear_hidden_driver", scenario_nonlinear_hidden_driver),
        ("observation_filtering", scenario_observation_filtering),
        ("partial_observation_hidden_nodes_strong", scenario_partial_observation_hidden_nodes_strong),
    ]

    @pytest.mark.parametrize("scenario_name,scenario_func", SCENARIOS)
    def test_returns_scenario_result(self, scenario_name: str, scenario_func) -> None:
        """Each scenario returns a valid ScenarioResult."""
        result = scenario_func(T=500, seed=42)
        assert isinstance(result, ScenarioResult)

    @pytest.mark.parametrize("scenario_name,scenario_func", SCENARIOS)
    def test_output_shape(self, scenario_name: str, scenario_func) -> None:
        """X shape is (T, d) and metadata is a dict."""
        T = 500
        if scenario_name == "partial_observation_hidden_nodes_strong":
            result = scenario_func(T=T, d_observed=10, seed=42)
        elif scenario_name == "nonlinear_markov":
            result = scenario_func(T=T, d=6, seed=42)
        else:
            result = scenario_func(T=T, d=10, seed=42)
        assert result.X.shape[0] == T, f"{scenario_name}: Expected T={T}, got {result.X.shape[0]}"
        assert len(result.X.shape) == 2, f"{scenario_name}: Expected 2D array"
        assert isinstance(result.metadata, dict)

    @pytest.mark.parametrize("scenario_name,scenario_func", SCENARIOS)
    def test_no_nans_in_X(self, scenario_name: str, scenario_func) -> None:
        """X contains no NaN values."""
        result = scenario_func(T=1000, seed=42)
        assert not np.isnan(result.X).any(), f"{scenario_name}: Found NaN values in X"

    @pytest.mark.parametrize("scenario_name,scenario_func", SCENARIOS)
    def test_no_nans_in_ground_truth(self, scenario_name: str, scenario_func) -> None:
        """ground_truth_compact contains no NaN values (if provided)."""
        result = scenario_func(T=500, seed=42)
        if result.ground_truth_compact is not None:
            assert not np.isnan(result.ground_truth_compact).any(), f"{scenario_name}: Found NaN in ground_truth"

    @pytest.mark.parametrize("scenario_name,scenario_func", SCENARIOS)
    def test_metadata_has_required_keys(self, scenario_name: str, scenario_func) -> None:
        """Metadata contains required keys: scenario, latent, true_order, observed_order, mechanism, expected_signature."""
        result = scenario_func(T=500, seed=42)
        metadata = result.metadata
        required_keys = {"scenario", "latent", "true_order", "observed_order", "mechanism", "expected_signature"}
        missing = required_keys - set(metadata.keys())
        assert not missing, f"{scenario_name}: Missing metadata keys: {missing}"

    @pytest.mark.parametrize("scenario_name,scenario_func", SCENARIOS)
    def test_metadata_scenario_matches_function(self, scenario_name: str, scenario_func) -> None:
        """metadata['scenario'] matches the scenario function name."""
        result = scenario_func(T=500, seed=42)
        assert result.metadata["scenario"] == scenario_name

    @pytest.mark.parametrize("scenario_name,scenario_func", SCENARIOS)
    def test_metadata_latent_is_bool(self, scenario_name: str, scenario_func) -> None:
        """metadata['latent'] is a boolean."""
        result = scenario_func(T=500, seed=42)
        assert isinstance(result.metadata["latent"], bool)

    @pytest.mark.parametrize("scenario_name,scenario_func", SCENARIOS)
    def test_metadata_orders_are_positive_int(self, scenario_name: str, scenario_func) -> None:
        """true_order and observed_order are positive integers."""
        result = scenario_func(T=500, seed=42)
        assert isinstance(result.metadata["true_order"], int) and result.metadata["true_order"] > 0
        assert isinstance(result.metadata["observed_order"], int) and result.metadata["observed_order"] > 0

    @pytest.mark.parametrize("scenario_name,scenario_func", SCENARIOS)
    def test_metadata_mechanism_is_str(self, scenario_name: str, scenario_func) -> None:
        """metadata['mechanism'] is a non-empty string."""
        result = scenario_func(T=500, seed=42)
        assert isinstance(result.metadata["mechanism"], str) and len(result.metadata["mechanism"]) > 0

    @pytest.mark.parametrize("scenario_name,scenario_func", SCENARIOS)
    def test_metadata_expected_signature_is_dict(self, scenario_name: str, scenario_func) -> None:
        """metadata['expected_signature'] is a dict describing expected diagnostic signatures."""
        result = scenario_func(T=500, seed=42)
        assert isinstance(result.metadata["expected_signature"], dict)

    @pytest.mark.parametrize("scenario_name,scenario_func", SCENARIOS)
    def test_reproducibility_with_seed(self, scenario_name: str, scenario_func) -> None:
        """Same seed produces identical output."""
        result1 = scenario_func(T=500, seed=42)
        result2 = scenario_func(T=500, seed=42)
        np.testing.assert_array_equal(result1.X, result2.X, err_msg=f"{scenario_name}: Not reproducible")

    @pytest.mark.parametrize("scenario_name,scenario_func", SCENARIOS)
    def test_different_seed_produces_different_data(self, scenario_name: str, scenario_func) -> None:
        """Different seed produces different (but valid) output."""
        result1 = scenario_func(T=1000, seed=42)
        result2 = scenario_func(T=1000, seed=43)
        assert not np.allclose(result1.X, result2.X, atol=1e-10)

    @pytest.mark.parametrize("scenario_name,scenario_func", SCENARIOS)
    def test_ground_truth_diagonal_zero(self, scenario_name: str, scenario_func) -> None:
        """ground_truth_compact has zero diagonal (no self-loops)."""
        result = scenario_func(T=500, seed=42)
        if result.ground_truth_compact is not None:
            diag = np.diag(result.ground_truth_compact)
            assert np.all(diag == 0), f"{scenario_name}: Diagonal is not zero: {diag}"

    @pytest.mark.parametrize("scenario_name,scenario_func", SCENARIOS)
    def test_ground_truth_is_binary(self, scenario_name: str, scenario_func) -> None:
        """ground_truth_compact is binary (0 or 1)."""
        result = scenario_func(T=500, seed=42)
        if result.ground_truth_compact is not None:
            unique_vals = np.unique(result.ground_truth_compact)
            assert np.all(np.isin(unique_vals, [0, 1])), f"{scenario_name}: Non-binary values: {unique_vals}"


class TestScenariosIntegration:
    """Integration tests across multiple scenarios."""

    def test_all_scenarios_have_finite_variance(self) -> None:
        """All scenarios produce finite variance."""
        scenarios = [
            scenario_omitted_lag_order,
            scenario_undersampled_markov,
            scenario_measurement_noise,
            scenario_time_varying_coefficients,
            scenario_regime_shift,
            scenario_nonlinear_markov,
            scenario_nonlinear_hidden_driver,
            scenario_observation_filtering,
            scenario_partial_observation_hidden_nodes_strong,
        ]
        for scenario_func in scenarios:
            result = scenario_func(T=500, seed=42)
            var = np.var(result.X)
            assert np.isfinite(var) and var > 0, f"{scenario_func.__name__}: Variance not finite or zero"

    def test_latent_confounder_scenarios_marked_true(self) -> None:
        """Scenarios with hidden confounders are marked as latent=True."""
        latent_scenarios = [
            scenario_nonlinear_hidden_driver,
            scenario_partial_observation_hidden_nodes_strong,
        ]
        for scenario_func in latent_scenarios:
            result = scenario_func(T=500, seed=42)
            assert result.metadata["latent"] is True, f"{scenario_func.__name__}: Should have latent=True"

    def test_clean_mechanisms_marked_latent_false(self) -> None:
        """Pure mechanism scenarios without latent confounders are marked latent=False."""
        clean_scenarios = [
            scenario_omitted_lag_order,
            scenario_undersampled_markov,
            scenario_measurement_noise,
            scenario_time_varying_coefficients,
            scenario_regime_shift,
            scenario_nonlinear_markov,
            scenario_observation_filtering,
        ]
        for scenario_func in clean_scenarios:
            result = scenario_func(T=500, seed=42)
            assert result.metadata["latent"] is False, f"{scenario_func.__name__}: Should have latent=False"
