"""Tests for v2a-RSN notebook persistence utilities."""

import json

import numpy as np
import pandas as pd
import pytest

from markovianity_diagnostic.experiments.v2a_rsn_utils import (
    V2A_ANALYSIS_PROFILE,
    V2A_PROFILE_RECORDINGS,
    completed_p_values,
    get_v2a_selection_metadata,
    get_v2a_selected_cell_indices,
    load_adjacency_checkpoint,
    save_adjacency_checkpoint,
    setup_recording_paths,
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


@pytest.mark.parametrize("recording_id", sorted(V2A_PROFILE_RECORDINGS))
def test_profile_recordings_use_reproducible_random_18_32_role_sample(
    tmp_path,
    recording_id,
):
    """Every biological recording should use the same seeded 50-cell profile."""
    emitters = np.arange(0, 80)
    receivers = np.arange(100, 180)
    np.save(tmp_path / f"{recording_id}_emitter_cells.npy", emitters)
    np.save(tmp_path / f"{recording_id}_receiver_cells.npy", receivers)

    first = get_v2a_selected_cell_indices(tmp_path, recording_id)
    second = get_v2a_selected_cell_indices(tmp_path, recording_id)
    metadata = get_v2a_selection_metadata(tmp_path, recording_id)

    assert first.shape == (50,)
    np.testing.assert_array_equal(first, second)
    assert np.isin(first[:18], emitters).all()
    assert np.isin(first[18:], receivers).all()
    assert not np.array_equal(first[:18], emitters[:18])
    assert not np.array_equal(first[18:], receivers[:32])
    assert metadata["analysis_profile"] == V2A_ANALYSIS_PROFILE
    assert metadata["selection_strategy"] == "seeded_stratified_without_replacement"
    assert metadata["n_emitters"] == 18
    assert metadata["n_receivers"] == 32
    assert metadata["expected_total"] == 50
    assert isinstance(metadata["selection_seed"], int)
    assert metadata["selected_emitter_indices"] == first[:18].tolist()
    assert metadata["selected_receiver_indices"] == first[18:].tolist()
    assert metadata["selected_cell_indices"] == first.tolist()


def test_profile_trace_subset_has_50_rows(tmp_path):
    """Trace subsetting should apply the seeded 18-emitter/32-receiver sample."""
    recording_id = "220210_F1_run6"
    emitters = np.arange(0, 80)
    receivers = np.arange(100, 180)
    traces = np.arange(200 * 4).reshape(200, 4)
    np.save(tmp_path / "220210_F1_F1_run6_emitter_cells.npy", emitters)
    np.save(tmp_path / "220210_F1_F1_run6_receiver_cells.npy", receivers)

    subset = subset_v2a_cells(traces, tmp_path, recording_id)
    selected = get_v2a_selected_cell_indices(tmp_path, recording_id)

    assert subset.shape == (50, 4)
    np.testing.assert_array_equal(subset, traces[selected, :])


def test_profile_selection_rejects_insufficient_role_counts(tmp_path):
    """The fixed profile must fail instead of silently changing its role ratio."""
    recording_id = "220119_F2_run11"
    np.save(tmp_path / f"{recording_id}_emitter_cells.npy", np.arange(17))
    np.save(tmp_path / f"{recording_id}_receiver_cells.npy", np.arange(100, 140))

    with pytest.raises(ValueError, match="at least 18 emitters and 32 receivers"):
        get_v2a_selected_cell_indices(tmp_path, recording_id)


def test_setup_recording_paths_isolates_profile_outputs(tmp_path):
    paths = setup_recording_paths(
        tmp_path,
        "220119_F2_run11",
        "c-GC",
        analysis_profile=V2A_ANALYSIS_PROFILE,
    )

    expected_method_dir = (
        tmp_path / "outputs" / "v2a-RSNs" / V2A_ANALYSIS_PROFILE / "c-GC"
    )
    assert paths["method_output_dir"] == expected_method_dir
    assert paths["output_dir"] == expected_method_dir / "220119_F2_run11"
