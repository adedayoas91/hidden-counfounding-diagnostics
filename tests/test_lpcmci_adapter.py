"""Tests for LPCMCI adapter that normalizes output to standard schema."""

import numpy as np
import pytest

from markovianity_diagnostic.methods.lpcmci_adapter import LPCMCIAdapter


class TestLPCMCIAdapterInit:
    """Test LPCMCIAdapter initialization."""

    def test_init_with_default_params(self):
        """LPCMCIAdapter initializes with defaults."""
        adapter = LPCMCIAdapter()
        assert adapter.tau_min == 1
        assert adapter.tau_max == 5
        assert adapter.pc_alpha == 0.05

    def test_init_with_custom_params(self):
        """LPCMCIAdapter initializes with custom parameters."""
        adapter = LPCMCIAdapter(tau_min=2, tau_max=10, pc_alpha=0.01)
        assert adapter.tau_min == 2
        assert adapter.tau_max == 10
        assert adapter.pc_alpha == 0.01


class TestLPCMCIAdapterFit:
    """Test LPCMCIAdapter.fit() method."""

    def test_fit_accepts_valid_data(self, synthetic_markov_data):
        """fit() accepts valid X and p_values."""
        X = synthetic_markov_data
        adapter = LPCMCIAdapter()
        p_values = [1, 2, 3]

        # This should not raise an error
        result = adapter.fit(X, p_values)
        assert result is not None

    def test_fit_returns_dict(self, synthetic_markov_data):
        """fit() returns a dict mapping p to output schema."""
        X = synthetic_markov_data
        adapter = LPCMCIAdapter()
        p_values = [1, 2]

        result = adapter.fit(X, p_values)
        assert isinstance(result, dict)
        assert set(result.keys()) == {1, 2}

    def test_fit_output_has_required_schema(self, synthetic_markov_data):
        """fit() output has adjacency, edge_marks, raw_pag, metadata for each p."""
        X = synthetic_markov_data
        adapter = LPCMCIAdapter()
        p_values = [1, 2]

        result = adapter.fit(X, p_values)

        for p, output in result.items():
            assert isinstance(output, dict)
            assert "adjacency" in output
            assert "edge_marks" in output
            assert "raw_pag" in output
            assert "metadata" in output


class TestLPCMCIAdapterSchema:
    """Test output schema correctness."""

    def test_adjacency_is_binary_matrix(self, synthetic_markov_data):
        """adjacency is a binary (0/1) matrix."""
        X = synthetic_markov_data
        adapter = LPCMCIAdapter()
        p_values = [1]

        result = adapter.fit(X, p_values)
        adjacency = result[1]["adjacency"]

        assert isinstance(adjacency, np.ndarray)
        assert adjacency.dtype in [np.int32, np.int64, int]
        assert np.all((adjacency == 0) | (adjacency == 1))

    def test_adjacency_is_square(self, synthetic_markov_data):
        """adjacency is a square matrix."""
        X = synthetic_markov_data
        d = X.shape[1]
        adapter = LPCMCIAdapter()
        p_values = [1]

        result = adapter.fit(X, p_values)
        adjacency = result[1]["adjacency"]

        assert adjacency.shape == (d, d)

    def test_adjacency_diagonal_is_zero(self, synthetic_markov_data):
        """adjacency matrix diagonal is all zeros."""
        X = synthetic_markov_data
        adapter = LPCMCIAdapter()
        p_values = [1, 2]

        result = adapter.fit(X, p_values)

        for p, output in result.items():
            adjacency = output["adjacency"]
            np.testing.assert_array_equal(np.diag(adjacency), 0)

    def test_edge_marks_is_dict(self, synthetic_markov_data):
        """edge_marks is a dict (preserves bidirected, uncertain marks)."""
        X = synthetic_markov_data
        adapter = LPCMCIAdapter()
        p_values = [1]

        result = adapter.fit(X, p_values)
        edge_marks = result[1]["edge_marks"]

        assert isinstance(edge_marks, dict)

    def test_raw_pag_exists(self, synthetic_markov_data):
        """raw_pag key exists in output (may be None if no relationships)."""
        X = synthetic_markov_data
        adapter = LPCMCIAdapter()
        p_values = [1]

        result = adapter.fit(X, p_values)

        assert "raw_pag" in result[1]
        # raw_pag can be None if LPCMCI finds no relationships

    def test_metadata_contains_algorithm_and_params(self, synthetic_markov_data):
        """metadata contains algorithm, params, p_value, runtime."""
        X = synthetic_markov_data
        adapter = LPCMCIAdapter()
        p_values = [1]

        result = adapter.fit(X, p_values)
        metadata = result[1]["metadata"]

        assert isinstance(metadata, dict)
        assert "algorithm" in metadata
        assert "params" in metadata
        assert "p_value" in metadata
        assert metadata["algorithm"] == "LPCMCI"
        assert metadata["p_value"] == 1

    def test_metadata_params_include_tau_pc_alpha(self, synthetic_markov_data):
        """metadata params include tau_min, tau_max, pc_alpha."""
        X = synthetic_markov_data
        adapter = LPCMCIAdapter(tau_min=1, tau_max=5, pc_alpha=0.05)
        p_values = [1]

        result = adapter.fit(X, p_values)
        params = result[1]["metadata"]["params"]

        assert "tau_min" in params
        assert "tau_max" in params
        assert "pc_alpha" in params
        assert params["tau_min"] == 1
        assert params["tau_max"] == 5
        assert params["pc_alpha"] == 0.05


class TestLPCMCIAdapterEdgeCases:
    """Test edge cases and error handling."""

    def test_fit_with_empty_p_values(self, synthetic_markov_data):
        """fit() handles empty p_values list gracefully."""
        X = synthetic_markov_data
        adapter = LPCMCIAdapter()

        result = adapter.fit(X, [])
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_fit_with_single_p_value(self, synthetic_markov_data):
        """fit() works with single p value."""
        X = synthetic_markov_data
        adapter = LPCMCIAdapter()
        p_values = [1]

        result = adapter.fit(X, p_values)
        assert len(result) == 1
        assert 1 in result

    def test_fit_with_multiple_p_values(self, synthetic_markov_data):
        """fit() works with multiple p values."""
        X = synthetic_markov_data
        adapter = LPCMCIAdapter()
        p_values = [1, 2, 3, 4, 5]

        result = adapter.fit(X, p_values)
        assert len(result) == 5
        assert set(result.keys()) == {1, 2, 3, 4, 5}

    def test_fit_output_keys_match_input_p_values(self, synthetic_markov_data):
        """fit() output keys exactly match input p_values."""
        X = synthetic_markov_data
        adapter = LPCMCIAdapter()
        p_values = [1, 3, 5]

        result = adapter.fit(X, p_values)
        assert set(result.keys()) == set(p_values)
