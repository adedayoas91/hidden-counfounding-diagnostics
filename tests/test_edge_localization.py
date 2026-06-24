"""Tests for edge localization module.

TDD Approach: Tests written FIRST, implementation follows.
Tests verify that compute_edge_instability_metrics produces correct edge-level metrics.
"""

import numpy as np
import pandas as pd
import pytest
from typing import Optional
from pathlib import Path


class TestComputeEdgeInstabilityMetrics:
    """Test compute_edge_instability_metrics function."""

    def test_basic_output_shape(self, simple_adjacency_dict):
        """Function should return DataFrame with one row per unique edge."""
        from markovianity_diagnostic.experiments.edge_localization import (
            compute_edge_instability_metrics
        )

        df = compute_edge_instability_metrics(
            adj_dict=simple_adjacency_dict,
            recording="test_recording",
            method="test_method"
        )

        assert isinstance(df, pd.DataFrame), "Output must be a DataFrame"
        assert len(df) > 0, "Output must have at least one row"

    def test_output_columns(self, simple_adjacency_dict):
        """DataFrame must contain all required columns."""
        from markovianity_diagnostic.experiments.edge_localization import (
            compute_edge_instability_metrics
        )

        df = compute_edge_instability_metrics(
            adj_dict=simple_adjacency_dict,
            recording="test_recording",
            method="test_method"
        )

        required_columns = [
            "recording", "method", "source", "target",
            "n_depths_present", "n_deletions", "n_additions",
            "first_appearance_p", "first_deletion_p",
            "instability_frequency", "null_frequency",
            "p_value", "q_value", "status"
        ]

        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"

    def test_edge_appearance_tracking(self, simple_adjacency_dict):
        """Should correctly track which edges appear at which depths."""
        from markovianity_diagnostic.experiments.edge_localization import (
            compute_edge_instability_metrics
        )

        df = compute_edge_instability_metrics(
            adj_dict=simple_adjacency_dict,
            recording="test_recording",
            method="test_method"
        )

        # Edge (0, 1) appears at depths 0, 1 (fixture has it disappear at depth 2)
        edge_01 = df[(df["source"] == 0) & (df["target"] == 1)]
        assert len(edge_01) == 1, "Should have one row for edge (0, 1)"
        assert edge_01.iloc[0]["n_depths_present"] == 2, "Edge (0, 1) should appear at 2 depths"

    def test_edge_deletion_counting(self, simple_adjacency_dict):
        """Should correctly count edge deletions between depths."""
        from markovianity_diagnostic.experiments.edge_localization import (
            compute_edge_instability_metrics
        )

        df = compute_edge_instability_metrics(
            adj_dict=simple_adjacency_dict,
            recording="test_recording",
            method="test_method"
        )

        # Edge (0, 1) exists at depths 0 and 1, but disappears at depth 2 -> 1 deletion
        edge_01 = df[(df["source"] == 0) & (df["target"] == 1)]
        assert edge_01.iloc[0]["n_deletions"] == 1, "Edge (0, 1) should have 1 deletion"

    def test_edge_addition_counting(self, simple_adjacency_dict):
        """Should correctly count edge additions between depths."""
        from markovianity_diagnostic.experiments.edge_localization import (
            compute_edge_instability_metrics
        )

        df = compute_edge_instability_metrics(
            adj_dict=simple_adjacency_dict,
            recording="test_recording",
            method="test_method"
        )

        # Edge (1, 0) appears at depths 1, 2 -> 1 addition
        edge_10 = df[(df["source"] == 1) & (df["target"] == 0)]
        if len(edge_10) > 0:
            assert edge_10.iloc[0]["n_additions"] >= 1, "Edge (1, 0) should have at least 1 addition"

    def test_first_appearance_tracking(self, simple_adjacency_dict):
        """Should correctly identify first depth where edge appears."""
        from markovianity_diagnostic.experiments.edge_localization import (
            compute_edge_instability_metrics
        )

        df = compute_edge_instability_metrics(
            adj_dict=simple_adjacency_dict,
            recording="test_recording",
            method="test_method"
        )

        # Edge (0, 1) appears first at depth 0
        edge_01 = df[(df["source"] == 0) & (df["target"] == 1)]
        assert edge_01.iloc[0]["first_appearance_p"] == 0, "Edge (0, 1) should first appear at depth 0"

    def test_first_deletion_tracking(self, simple_adjacency_dict):
        """Should correctly identify first depth where edge disappears."""
        from markovianity_diagnostic.experiments.edge_localization import (
            compute_edge_instability_metrics
        )

        df = compute_edge_instability_metrics(
            adj_dict=simple_adjacency_dict,
            recording="test_recording",
            method="test_method"
        )

        # Edge (0, 1) exists at 0, 1 then disappears at 2 -> first_deletion_p = 2
        edge_01 = df[(df["source"] == 0) & (df["target"] == 1)]
        assert edge_01.iloc[0]["first_deletion_p"] == 2, "Edge (0, 1) should first disappear at depth 2"

    def test_instability_frequency(self, simple_adjacency_dict):
        """instability_frequency should be (n_deletions + n_additions) / (n_depths - 1)."""
        from markovianity_diagnostic.experiments.edge_localization import (
            compute_edge_instability_metrics
        )

        df = compute_edge_instability_metrics(
            adj_dict=simple_adjacency_dict,
            recording="test_recording",
            method="test_method"
        )

        # For edge (0, 1): n_deletions = 1, n_additions = 0, n_depths = 3
        # instability_frequency = (1 + 0) / 2 = 0.5
        edge_01 = df[(df["source"] == 0) & (df["target"] == 1)]
        expected_freq = (edge_01.iloc[0]["n_deletions"] + edge_01.iloc[0]["n_additions"]) / 2
        assert np.isclose(
            edge_01.iloc[0]["instability_frequency"],
            expected_freq
        ), "instability_frequency calculation is incorrect"

    def test_metadata_fields_populated(self, simple_adjacency_dict):
        """recording and method fields should be populated."""
        from markovianity_diagnostic.experiments.edge_localization import (
            compute_edge_instability_metrics
        )

        recording = "fish_001"
        method = "c-GC"

        df = compute_edge_instability_metrics(
            adj_dict=simple_adjacency_dict,
            recording=recording,
            method=method
        )

        assert all(df["recording"] == recording), "All rows should have same recording"
        assert all(df["method"] == method), "All rows should have same method"

    def test_no_self_loops_in_edges(self, simple_adjacency_dict):
        """Should not include self-loops (source == target) in output."""
        from markovianity_diagnostic.experiments.edge_localization import (
            compute_edge_instability_metrics
        )

        df = compute_edge_instability_metrics(
            adj_dict=simple_adjacency_dict,
            recording="test_recording",
            method="test_method"
        )

        assert not (df["source"] == df["target"]).any(), "Output should not contain self-loops"

    def test_null_frequency_defaults_to_nan(self, simple_adjacency_dict):
        """null_frequency should default to NaN when bootstrap_adj_dict is not provided."""
        from markovianity_diagnostic.experiments.edge_localization import (
            compute_edge_instability_metrics
        )

        df = compute_edge_instability_metrics(
            adj_dict=simple_adjacency_dict,
            recording="test_recording",
            method="test_method",
            bootstrap_adj_dict=None
        )

        assert df["null_frequency"].isna().all(), "null_frequency should be NaN when no bootstrap data"

    def test_p_value_defaults_to_nan(self, simple_adjacency_dict):
        """p_value should default to NaN when bootstrap_adj_dict is not provided."""
        from markovianity_diagnostic.experiments.edge_localization import (
            compute_edge_instability_metrics
        )

        df = compute_edge_instability_metrics(
            adj_dict=simple_adjacency_dict,
            recording="test_recording",
            method="test_method",
            bootstrap_adj_dict=None
        )

        assert df["p_value"].isna().all(), "p_value should be NaN when no bootstrap data"

    def test_q_value_defaults_to_nan(self, simple_adjacency_dict):
        """q_value should default to NaN when bootstrap_adj_dict is not provided."""
        from markovianity_diagnostic.experiments.edge_localization import (
            compute_edge_instability_metrics
        )

        df = compute_edge_instability_metrics(
            adj_dict=simple_adjacency_dict,
            recording="test_recording",
            method="test_method",
            bootstrap_adj_dict=None
        )

        assert df["q_value"].isna().all(), "q_value should be NaN when no bootstrap data"

    def test_status_field_populated(self, simple_adjacency_dict):
        """status field should be populated with edge status (stable/unstable)."""
        from markovianity_diagnostic.experiments.edge_localization import (
            compute_edge_instability_metrics
        )

        df = compute_edge_instability_metrics(
            adj_dict=simple_adjacency_dict,
            recording="test_recording",
            method="test_method"
        )

        assert "status" in df.columns
        assert not df["status"].isna().all(), "status column should not be all NaN"


class TestComputeEdgeInstabilityMetricsWithBootstrap:
    """Test compute_edge_instability_metrics with bootstrap null data."""

    def test_null_frequency_computed(self, simple_adjacency_dict):
        """null_frequency should be computed when bootstrap_adj_dict is provided."""
        from markovianity_diagnostic.experiments.edge_localization import (
            compute_edge_instability_metrics
        )

        # Create bootstrap null adjacencies
        bootstrap_adj_dict = {
            0: np.array([[0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]]),
            1: np.array([[0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]]),
            2: np.array([[0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]]),
        }

        df = compute_edge_instability_metrics(
            adj_dict=simple_adjacency_dict,
            recording="test_recording",
            method="test_method",
            bootstrap_adj_dict=bootstrap_adj_dict
        )

        # null_frequency should not be NaN
        assert not df["null_frequency"].isna().all(), "null_frequency should be computed"

    def test_p_value_computed(self, simple_adjacency_dict):
        """p_value should be computed when bootstrap_adj_dict is provided."""
        from markovianity_diagnostic.experiments.edge_localization import (
            compute_edge_instability_metrics
        )

        # Create bootstrap null adjacencies with some edges
        bootstrap_adj_dict = {
            0: np.array([[0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]]),
            1: np.array([[0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]]),
            2: np.array([[0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]]),
        }

        df = compute_edge_instability_metrics(
            adj_dict=simple_adjacency_dict,
            recording="test_recording",
            method="test_method",
            bootstrap_adj_dict=bootstrap_adj_dict
        )

        # p_value should be computed
        assert not df["p_value"].isna().all(), "p_value should be computed"
        assert (df["p_value"] >= 0.0).all() and (df["p_value"] <= 1.0).all(), "p_value should be in [0, 1]"

    def test_q_value_is_fdr_corrected(self, simple_adjacency_dict):
        """q_value should be FDR-corrected p_value."""
        from markovianity_diagnostic.experiments.edge_localization import (
            compute_edge_instability_metrics
        )

        # Create bootstrap null adjacencies
        bootstrap_adj_dict = {
            0: simple_adjacency_dict[0].copy(),
            1: simple_adjacency_dict[1].copy(),
            2: simple_adjacency_dict[2].copy(),
        }

        df = compute_edge_instability_metrics(
            adj_dict=simple_adjacency_dict,
            recording="test_recording",
            method="test_method",
            bootstrap_adj_dict=bootstrap_adj_dict
        )

        # q_value should be computed
        assert not df["q_value"].isna().all(), "q_value should be computed"
        assert (df["q_value"] >= 0.0).all() and (df["q_value"] <= 1.0).all(), "q_value should be in [0, 1]"

        # q_value should be >= p_value (conservative correction)
        assert (df["q_value"] >= df["p_value"]).all(), "q_value should be >= p_value"

    def test_bootstrap_replicates_are_used_per_edge(self):
        """P-values should compare each edge against bootstrap replicate frequencies."""
        from markovianity_diagnostic.experiments.edge_localization import (
            compute_edge_instability_metrics
        )

        obs = {
            0: np.array([[0, 1], [0, 0]]),
            1: np.array([[0, 1], [0, 0]]),
            2: np.array([[0, 0], [0, 0]]),
        }
        stable_null = {
            0: np.array([[0, 1], [0, 0]]),
            1: np.array([[0, 1], [0, 0]]),
            2: np.array([[0, 1], [0, 0]]),
        }
        unstable_null = {
            0: np.array([[0, 1], [0, 0]]),
            1: np.array([[0, 1], [0, 0]]),
            2: np.array([[0, 0], [0, 0]]),
        }

        df = compute_edge_instability_metrics(
            adj_dict=obs,
            recording="test_recording",
            method="test_method",
            bootstrap_adj_dict=[stable_null, unstable_null],
        )
        edge = df[(df["source"] == 0) & (df["target"] == 1)].iloc[0]

        assert edge["null_frequency"] == 0.25
        assert edge["p_value"] == pytest.approx(2 / 3)


class TestEdgeLocalizationEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_adjacency_dict(self):
        """Should handle empty adjacency dictionary gracefully."""
        from markovianity_diagnostic.experiments.edge_localization import (
            compute_edge_instability_metrics
        )

        df = compute_edge_instability_metrics(
            adj_dict={},
            recording="test_recording",
            method="test_method"
        )

        assert isinstance(df, pd.DataFrame), "Output must be a DataFrame even for empty input"
        # Should either be empty or have default columns
        assert len(df) == 0 or all(col in df.columns for col in ["source", "target"])

    def test_single_depth(self):
        """Should handle single depth adjacency dict."""
        from markovianity_diagnostic.experiments.edge_localization import (
            compute_edge_instability_metrics
        )

        single_depth_dict = {
            0: np.array([[0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0]])
        }

        df = compute_edge_instability_metrics(
            adj_dict=single_depth_dict,
            recording="test_recording",
            method="test_method"
        )

        assert isinstance(df, pd.DataFrame), "Should handle single depth"
        # With single depth, n_deletions and n_additions should be 0
        if len(df) > 0:
            assert (df["n_deletions"] == 0).all(), "Single depth: no deletions"
            assert (df["n_additions"] == 0).all(), "Single depth: no additions"

    def test_all_zeros_adjacency(self):
        """Should handle all-zero adjacency matrices."""
        from markovianity_diagnostic.experiments.edge_localization import (
            compute_edge_instability_metrics
        )

        zeros_dict = {
            0: np.zeros((5, 5)),
            1: np.zeros((5, 5)),
            2: np.zeros((5, 5)),
        }

        df = compute_edge_instability_metrics(
            adj_dict=zeros_dict,
            recording="test_recording",
            method="test_method"
        )

        assert isinstance(df, pd.DataFrame), "Should handle all-zero matrices"
        assert len(df) == 0, "All-zero matrices should produce no edges"
