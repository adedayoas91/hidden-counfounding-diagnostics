"""Tests for v2a-RSN notebook persistence utilities."""

import json

import numpy as np
import pandas as pd

from markovianity_diagnostic.experiments.v2a_rsn_utils import (
    completed_p_values,
    get_v2a_selected_cell_indices,
    load_adjacency_checkpoint,
    save_adjacency_checkpoint,
    subset_v2a_cells,
    upsert_method_outputs,
    upsert_summary_json,
)


def test_adjacency_checkpoint_round_trips_integer_p_values(tmp_path):
    """Adjacency checkpoints should load as dict[int, ndarray]."""
    checkpoint = tmp_path / "recording.pkl"
    adjacencies = {
        2: np.array([[0, 1], [0, 0]]),
        1: np.array([[0, 0], [1, 0]]),
    }

    save_adjacency_checkpoint(checkpoint, adjacencies)
    loaded = load_adjacency_checkpoint(checkpoint)

    assert sorted(loaded) == [1, 2]
    assert completed_p_values(loaded, [1, 2, 3]) == [1, 2]
    np.testing.assert_array_equal(loaded[1], adjacencies[1])
    np.testing.assert_array_equal(loaded[2], adjacencies[2])


def test_load_adjacency_checkpoint_returns_empty_for_missing_file(tmp_path):
    """Missing checkpoints should mean no completed p-values."""
    assert load_adjacency_checkpoint(tmp_path / "missing.pkl") == {}


def test_upsert_method_outputs_replaces_existing_recording_rows(tmp_path):
    """Rerunning one notebook should not duplicate aggregate method rows."""
    method_output_dir = tmp_path / "c-GC"

    first_summary = pd.DataFrame(
        [{"dataset": "fish_a", "T_obs": 1.0}, {"dataset": "fish_b", "T_obs": 2.0}]
    )
    first_transition = pd.DataFrame(
        [
            {"dataset": "fish_a", "P": 1, "edge_count": 10},
            {"dataset": "fish_b", "P": 1, "edge_count": 20},
        ]
    )
    upsert_method_outputs(first_summary, first_transition, method_output_dir)

    updated_summary = pd.DataFrame([{"dataset": "fish_a", "T_obs": 3.0}])
    updated_transition = pd.DataFrame([{"dataset": "fish_a", "P": 1, "edge_count": 30}])
    upsert_method_outputs(updated_summary, updated_transition, method_output_dir)

    summary = pd.read_csv(method_output_dir / "summary.csv")
    transitions = pd.read_csv(method_output_dir / "transitions.csv")

    assert summary.to_dict("records") == [
        {"dataset": "fish_a", "T_obs": 3.0},
        {"dataset": "fish_b", "T_obs": 2.0},
    ]
    assert transitions.to_dict("records") == [
        {"dataset": "fish_a", "P": 1, "edge_count": 30},
        {"dataset": "fish_b", "P": 1, "edge_count": 20},
    ]


def test_upsert_summary_json_replaces_recording_entry(tmp_path):
    """Method summary JSON should stay one record per dataset."""
    path = tmp_path / "summary.json"
    upsert_summary_json({"dataset": "fish_a", "T_obs": 1.0}, path)
    upsert_summary_json({"dataset": "fish_b", "T_obs": 2.0}, path)
    upsert_summary_json({"dataset": "fish_a", "T_obs": 3.0}, path)

    payload = json.loads(path.read_text())

    assert payload == [
        {"dataset": "fish_a", "T_obs": 3.0},
        {"dataset": "fish_b", "T_obs": 2.0},
    ]


def test_get_v2a_selected_cell_indices_uses_ordered_role_union(tmp_path):
    """Standard recordings should keep emitters first and append new receivers."""
    recording_id = "fish_a"
    np.save(tmp_path / "fish_a_emitter_cells.npy", np.array([5, 1, 3]))
    np.save(tmp_path / "fish_a_receiver_cells.npy", np.array([3, 8, 2]))

    selected = get_v2a_selected_cell_indices(tmp_path, recording_id)

    np.testing.assert_array_equal(selected, np.array([5, 1, 3, 8, 2]))


def test_220210_f1_run6_indices_are_reduced_to_35_emitters_65_receivers(tmp_path):
    """The high-dimensional v2a recording should use the configured 100-cell subset."""
    recording_id = "220210_F1_run6"
    emitters = np.arange(0, 201)
    receivers = np.arange(300, 511)
    np.save(tmp_path / "220210_F1_F1_run6_emitter_cells.npy", emitters)
    np.save(tmp_path / "220210_F1_F1_run6_receiver_cells.npy", receivers)

    selected = get_v2a_selected_cell_indices(tmp_path, recording_id)

    expected = np.concatenate([emitters[:35], receivers[:65]])
    assert selected.shape == (100,)
    np.testing.assert_array_equal(selected, expected)


def test_220210_f1_run6_trace_subset_has_100_rows(tmp_path):
    """Trace subsetting should apply the same deterministic high-dimensional subset."""
    recording_id = "220210_F1_run6"
    emitters = np.arange(0, 201)
    receivers = np.arange(300, 511)
    traces = np.arange(600 * 4).reshape(600, 4)
    np.save(tmp_path / "220210_F1_F1_run6_emitter_cells.npy", emitters)
    np.save(tmp_path / "220210_F1_F1_run6_receiver_cells.npy", receivers)

    subset = subset_v2a_cells(traces, tmp_path, recording_id)

    expected_indices = np.concatenate([emitters[:35], receivers[:65]])
    assert subset.shape == (100, 4)
    np.testing.assert_array_equal(subset, traces[expected_indices, :])
