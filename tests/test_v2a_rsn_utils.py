"""Tests for v2a-RSN notebook persistence utilities."""

import json

import numpy as np
import pandas as pd

from markovianity_diagnostic.experiments.v2a_rsn_utils import (
    completed_p_values,
    load_adjacency_checkpoint,
    save_adjacency_checkpoint,
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
