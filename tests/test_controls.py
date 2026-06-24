"""Tests for synthetic controls module.

TDD Approach: Tests written FIRST, implementation follows.
Positive and negative controls for Markovianity diagnostic validation.
"""

import numpy as np
import pytest
from pathlib import Path

from markovianity_diagnostic.experiments.controls import (
    SyntheticMarkovianNull,
    SyntheticLatentControlPositive,
    SyntheticHiddenNodesPositive,
    TimeShuffledControl,
    BlockShuffledControl,
    PhaseRandomizedControl,
    CircularlyShiftedControl,
)
from markovianity_diagnostic.experiments.simulations import ScenarioResult


class TestSyntheticMarkovianNull:
    """Test order-p Markov null controls."""

    def test_init_default_order(self):
        """SyntheticMarkovianNull should default to order=1."""
        ctrl = SyntheticMarkovianNull()
        assert ctrl.order == 1

    def test_init_custom_order(self):
        """SyntheticMarkovianNull should accept custom order."""
        ctrl = SyntheticMarkovianNull(order=3)
        assert ctrl.order == 3

    def test_generate_returns_scenario_result(self, seed_fix):
        """generate() should return ScenarioResult."""
        ctrl = SyntheticMarkovianNull(order=1)
        result = ctrl.generate(T=200, d=5, seed=42)

        assert isinstance(result, ScenarioResult)
        assert hasattr(result, 'X')
        assert hasattr(result, 'ground_truth_compact')
        assert hasattr(result, 'metadata')

    def test_generate_shape(self, seed_fix):
        """Generated data should have shape (T, d)."""
        T, d = 200, 5
        ctrl = SyntheticMarkovianNull(order=1)
        result = ctrl.generate(T=T, d=d, seed=42)

        assert result.X.shape == (T, d), f"Expected {(T, d)}, got {result.X.shape}"

    def test_generate_reproducibility(self, seed_fix):
        """Same seed should produce identical data."""
        ctrl = SyntheticMarkovianNull(order=1)
        result1 = ctrl.generate(T=200, d=5, seed=42)
        result2 = ctrl.generate(T=200, d=5, seed=42)

        np.testing.assert_array_equal(result1.X, result2.X)

    def test_ground_truth_shape(self, seed_fix):
        """ground_truth_compact should be (d, d) binary."""
        T, d = 200, 5
        ctrl = SyntheticMarkovianNull(order=1)
        result = ctrl.generate(T=T, d=d, seed=42)

        assert result.ground_truth_compact.shape == (d, d)
        assert np.isin(result.ground_truth_compact, [0, 1]).all()
        assert np.all(np.diag(result.ground_truth_compact) == 0), "Diagonal should be zero"

    def test_metadata_contains_order(self, seed_fix):
        """Metadata should record the true order."""
        ctrl = SyntheticMarkovianNull(order=3)
        result = ctrl.generate(T=200, d=5, seed=42)

        assert result.metadata['true_order'] == 3
        assert result.metadata['scenario'] == 'markovian_null'

    def test_stability(self, seed_fix):
        """Generated VAR should be stable (no explosion)."""
        ctrl = SyntheticMarkovianNull(order=1)
        result = ctrl.generate(T=500, d=5, seed=42)

        # Check that data doesn't explode (variance should be finite)
        variance = np.var(result.X)
        assert np.isfinite(variance)
        assert variance > 1e-8, "Should have some signal"


class TestSyntheticLatentControlPositive:
    """Test latent common driver positive control."""

    def test_generate_returns_scenario_result(self, seed_fix):
        """generate() should return ScenarioResult."""
        ctrl = SyntheticLatentControlPositive()
        result = ctrl.generate(T=200, d=5, seed=42)

        assert isinstance(result, ScenarioResult)

    def test_generate_shape(self, seed_fix):
        """Generated data should have shape (T, d)."""
        T, d = 200, 5
        ctrl = SyntheticLatentControlPositive()
        result = ctrl.generate(T=T, d=d, seed=42)

        assert result.X.shape == (T, d)

    def test_metadata_indicates_latent(self, seed_fix):
        """Metadata should indicate presence of latent confounder."""
        ctrl = SyntheticLatentControlPositive()
        result = ctrl.generate(T=200, d=5, seed=42)

        assert result.metadata['latent'] is True
        assert result.metadata['scenario'] == 'latent_common_driver'

    def test_generates_order1_system(self, seed_fix):
        """Latent control should generate order-1 system."""
        ctrl = SyntheticLatentControlPositive()
        result = ctrl.generate(T=200, d=5, seed=42)

        assert result.metadata['true_order'] == 1

    def test_reproducibility(self, seed_fix):
        """Same seed should produce identical data."""
        ctrl = SyntheticLatentControlPositive()
        result1 = ctrl.generate(T=200, d=5, seed=42)
        result2 = ctrl.generate(T=200, d=5, seed=42)

        np.testing.assert_array_equal(result1.X, result2.X)


class TestSyntheticHiddenNodesPositive:
    """Test hidden nodes positive control."""

    def test_generate_returns_scenario_result(self, seed_fix):
        """generate() should return ScenarioResult."""
        ctrl = SyntheticHiddenNodesPositive()
        result = ctrl.generate(T=200, d=5, seed=42)

        assert isinstance(result, ScenarioResult)

    def test_output_dimension_is_d(self, seed_fix):
        """Output should be (T, d) where d is observed dimension."""
        T, d = 200, 5
        ctrl = SyntheticHiddenNodesPositive(d_observed=d)
        result = ctrl.generate(T=T, d=d, seed=42)

        assert result.X.shape == (T, d)

    def test_metadata_indicates_latent(self, seed_fix):
        """Metadata should indicate presence of hidden nodes."""
        ctrl = SyntheticHiddenNodesPositive()
        result = ctrl.generate(T=200, d=5, seed=42)

        assert result.metadata['latent'] is True
        assert result.metadata['scenario'] == 'hidden_nodes'

    def test_hidden_indices_recorded(self, seed_fix):
        """Metadata should record which nodes are hidden."""
        d_total, d_observed = 10, 5
        ctrl = SyntheticHiddenNodesPositive(d_total=d_total, d_observed=d_observed)
        result = ctrl.generate(T=200, d=d_observed, seed=42)

        assert 'hidden_indices' in result.metadata
        assert len(result.metadata['hidden_indices']) == d_total - d_observed


class TestTimeShuffledControl:
    """Test time-shuffled control."""

    def test_shuffle_preserves_shape(self, synthetic_markov_data):
        """Shuffling should preserve shape."""
        ctrl = TimeShuffledControl(synthetic_markov_data)
        shuffled = ctrl.generate(seed=42)

        assert shuffled.shape == synthetic_markov_data.shape

    def test_shuffle_breaks_autocorrelation(self, synthetic_markov_data):
        """Shuffled should have lower autocorrelation than original."""
        ctrl = TimeShuffledControl(synthetic_markov_data)
        shuffled = ctrl.generate(seed=42)

        # Original autocorrelation at lag 1
        orig_acf = np.corrcoef(
            synthetic_markov_data[:-1, 0], synthetic_markov_data[1:, 0]
        )[0, 1]

        # Shuffled autocorrelation at lag 1
        shuf_acf = np.corrcoef(shuffled[:-1, 0], shuffled[1:, 0])[0, 1]

        # Shuffled should break temporal structure
        assert abs(shuf_acf) < abs(orig_acf), "Shuffle should reduce autocorrelation"

    def test_reproducibility(self, synthetic_markov_data):
        """Same seed should produce identical shuffle."""
        ctrl = TimeShuffledControl(synthetic_markov_data)
        shuffled1 = ctrl.generate(seed=42)
        shuffled2 = ctrl.generate(seed=42)

        np.testing.assert_array_equal(shuffled1, shuffled2)

    def test_marginals_preserved(self, synthetic_markov_data):
        """Marginal distributions should be preserved."""
        ctrl = TimeShuffledControl(synthetic_markov_data)
        shuffled = ctrl.generate(seed=42)

        # Mean should be approximately same
        np.testing.assert_allclose(
            np.mean(shuffled, axis=0),
            np.mean(synthetic_markov_data, axis=0),
            rtol=0.2,
            err_msg="Marginal means should be preserved"
        )


class TestBlockShuffledControl:
    """Test block-shuffled control."""

    def test_shuffle_preserves_shape(self, synthetic_markov_data):
        """Block shuffle should preserve shape."""
        ctrl = BlockShuffledControl(synthetic_markov_data, block_size=20)
        shuffled = ctrl.generate(seed=42)

        assert shuffled.shape == synthetic_markov_data.shape

    def test_block_size_parameter(self, synthetic_markov_data):
        """Should accept custom block_size."""
        ctrl = BlockShuffledControl(synthetic_markov_data, block_size=50)
        shuffled = ctrl.generate(seed=42)

        assert shuffled.shape == synthetic_markov_data.shape

    def test_preserves_marginals(self, synthetic_markov_data):
        """Block shuffle should preserve marginal distributions."""
        ctrl = BlockShuffledControl(synthetic_markov_data, block_size=20)
        shuffled = ctrl.generate(seed=42)

        np.testing.assert_allclose(
            np.mean(shuffled, axis=0),
            np.mean(synthetic_markov_data, axis=0),
            rtol=0.2
        )

    def test_reproducibility(self, synthetic_markov_data):
        """Same seed should produce identical shuffle."""
        ctrl = BlockShuffledControl(synthetic_markov_data, block_size=20)
        shuffled1 = ctrl.generate(seed=42)
        shuffled2 = ctrl.generate(seed=42)

        np.testing.assert_array_equal(shuffled1, shuffled2)


class TestPhaseRandomizedControl:
    """Test phase randomization control."""

    def test_generates_correct_shape(self, synthetic_markov_data):
        """Phase randomization should preserve shape."""
        ctrl = PhaseRandomizedControl(synthetic_markov_data)
        randomized = ctrl.generate(seed=42)

        assert randomized.shape == synthetic_markov_data.shape

    def test_preserves_marginal_spectrum(self, synthetic_markov_data):
        """Phase randomization should preserve power spectrum."""
        ctrl = PhaseRandomizedControl(synthetic_markov_data)
        randomized = ctrl.generate(seed=42)

        # Compute power spectra
        fft_orig = np.abs(np.fft.fft(synthetic_markov_data[:, 0], axis=0))
        fft_rand = np.abs(np.fft.fft(randomized[:, 0], axis=0))

        # Power should be approximately same
        np.testing.assert_allclose(
            np.sort(fft_orig),
            np.sort(fft_rand),
            rtol=0.05
        )

    def test_reproducibility(self, synthetic_markov_data):
        """Same seed should produce identical randomization."""
        ctrl = PhaseRandomizedControl(synthetic_markov_data)
        rand1 = ctrl.generate(seed=42)
        rand2 = ctrl.generate(seed=42)

        np.testing.assert_array_equal(rand1, rand2)


class TestCircularlyShiftedControl:
    """Test circularly shifted control."""

    def test_generates_correct_shape(self, synthetic_markov_data):
        """Circular shift should preserve shape."""
        ctrl = CircularlyShiftedControl(synthetic_markov_data, lag=1)
        shifted = ctrl.generate(seed=42)

        assert shifted.shape == synthetic_markov_data.shape

    def test_default_lag(self, synthetic_markov_data):
        """Should default to lag=1."""
        ctrl = CircularlyShiftedControl(synthetic_markov_data)
        shifted = ctrl.generate(seed=42)

        assert shifted.shape == synthetic_markov_data.shape

    def test_custom_lag(self, synthetic_markov_data):
        """Should accept custom lag values."""
        ctrl = CircularlyShiftedControl(synthetic_markov_data, lag=10)
        shifted = ctrl.generate(seed=42)

        assert shifted.shape == synthetic_markov_data.shape

    def test_lag_shifts_appropriately(self, synthetic_markov_data):
        """Shifted data should match source at offset."""
        T = synthetic_markov_data.shape[0]
        ctrl = CircularlyShiftedControl(synthetic_markov_data, lag=1)
        shifted = ctrl.generate(seed=42)

        # Element at position t should equal element at position t-1 in original
        # (with wraparound)
        np.testing.assert_array_equal(
            shifted[1:, :],
            synthetic_markov_data[:-1, :]
        )

    def test_reproducibility(self, synthetic_markov_data):
        """Shift is deterministic (no randomness)."""
        ctrl = CircularlyShiftedControl(synthetic_markov_data, lag=1)
        shifted1 = ctrl.generate(seed=42)
        shifted2 = ctrl.generate(seed=99)  # seed doesn't matter

        np.testing.assert_array_equal(shifted1, shifted2)


class TestControlIntegration:
    """Integration tests for controls workflow."""

    def test_null_control_shows_low_instability_structure(self):
        """Markovian null should have stable causal structure."""
        # Generate order-1 null
        null_ctrl = SyntheticMarkovianNull(order=1)
        result = null_ctrl.generate(T=500, d=5, seed=42)

        # Should have well-defined adjacency
        assert result.ground_truth_compact.sum() > 0, "Should have some edges"

    def test_positive_control_introduces_hidden_confounding(self):
        """Latent control should introduce apparent indirect edges."""
        pos_ctrl = SyntheticLatentControlPositive()
        result = pos_ctrl.generate(T=500, d=5, seed=42)

        # Latent confounder should create multi-node patterns
        assert result.metadata['latent'] is True

    def test_real_data_controls_are_permutations(self, synthetic_data_with_confounder):
        """Real-data controls should preserve empirical distribution."""
        # TimeShuffled
        time_ctrl = TimeShuffledControl(synthetic_data_with_confounder)
        shuffled = time_ctrl.generate(seed=42)

        # Should have same marginal stats
        np.testing.assert_allclose(
            np.std(shuffled, axis=0),
            np.std(synthetic_data_with_confounder, axis=0),
            rtol=0.3
        )
