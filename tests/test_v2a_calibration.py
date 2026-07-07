"""Regression tests for pickle-first v2a bootstrap calibration."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import markovianity_diagnostic.experiments.v2a_calibration as v2a_calibration
from markovianity_diagnostic.experiments.v2a_calibration import (
    complete_inferred_depths,
    discover_complete_recordings,
    load_v2a_calibration_input,
    make_v2a_surrogate_analyzer,
    plot_v2a_calibration_grid,
    plot_v2a_pointwise_grid,
    run_resumable_v2a_calibration,
)
from markovianity_diagnostic.experiments.v2a_rsn_utils import (
    V2A_ANALYSIS_PROFILE,
    get_v2a_selection_metadata,
)


def _write_recording_fixture(
    project_root: Path,
    *,
    recording: str = "fish_a",
    method_dir: str = "c-GC",
    p_values: tuple[int, ...] = (1, 2, 3),
    analysis_profile: str | None = None,
) -> dict[int, np.ndarray]:
    data_dir = project_root / "data" / "v2a-RSNs" / recording
    data_dir.mkdir(parents=True, exist_ok=True)
    traces = np.arange(3 * 30, dtype=float).reshape(3, 30)
    np.save(data_dir / f"{recording}_cells_fluorescence_signals.npy", traces)
    np.save(data_dir / f"{recording}_emitter_cells.npy", np.array([0, 1]))
    np.save(data_dir / f"{recording}_receiver_cells.npy", np.array([2]))

    adjacencies = {
        p_value: np.array(
            [
                [0, int(p_value >= 2), 0],
                [0, 0, int(p_value >= 3)],
                [1, 0, 0],
            ],
            dtype=int,
        )
        for p_value in p_values
    }
    method_output = project_root / "outputs" / "v2a-RSNs"
    if analysis_profile is not None:
        method_output /= analysis_profile
    method_output /= method_dir
    run_output = method_output / recording
    run_output.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(adjacencies, method_output / f"{recording}.pkl")
    trace_selection = get_v2a_selection_metadata(data_dir, recording)
    (run_output / "run_metadata.json").write_text(
        json.dumps(
            {
                "recording": recording,
                "method_label": method_dir,
                "analysis_profile": analysis_profile,
                "trace_selection": trace_selection,
                "gcstar_params": {
                    "method": "cgc" if method_dir == "c-GC" else "fcgc",
                    "n_perm": 5,
                    "n_lags": 1,
                    "alpha": 0.01,
                    "beta": 0.001,
                    "temporal": True,
                    "simulation": False,
                    "verbose": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    return adjacencies


def test_load_input_uses_pickles_and_raw_traces_without_transition_columns(tmp_path):
    expected = _write_recording_fixture(tmp_path)

    calibration_input = load_v2a_calibration_input(
        tmp_path,
        recording="fish_a",
        method_dir="c-GC",
        p_values=[1, 2, 3],
    )

    assert calibration_input.X.shape == (30, 3)
    assert sorted(calibration_input.observed_adjacencies) == [1, 2, 3]
    for depth in expected:
        np.testing.assert_array_equal(
            calibration_input.observed_adjacencies[depth],
            expected[depth],
        )
    assert not (tmp_path / "outputs" / "v2a-RSNs" / "c-GC" / "transitions.csv").exists()


def test_load_input_rejects_incomplete_pickle_depth_grid(tmp_path):
    _write_recording_fixture(tmp_path, p_values=(1, 2))

    with pytest.raises(ValueError, match="missing conditioning depths"):
        load_v2a_calibration_input(
            tmp_path,
            recording="fish_a",
            method_dir="c-GC",
            p_values=[1, 2, 3],
        )


def test_discover_recordings_requires_complete_pickles_for_every_method(tmp_path):
    _write_recording_fixture(tmp_path, method_dir="c-GC")
    _write_recording_fixture(tmp_path, method_dir="c-GC-star", p_values=(1, 2))

    with pytest.raises(ValueError, match="c-GC-star/fish_a"):
        discover_complete_recordings(
            tmp_path,
            method_dirs=["c-GC", "c-GC-star"],
            p_values=[1, 2, 3],
        )


def test_profile_input_and_discovery_use_isolated_output_namespace(tmp_path):
    profile = V2A_ANALYSIS_PROFILE
    _write_recording_fixture(
        tmp_path,
        method_dir="c-GC",
        analysis_profile=profile,
    )
    _write_recording_fixture(
        tmp_path,
        method_dir="c-GC-star",
        analysis_profile=profile,
    )

    recordings = discover_complete_recordings(
        tmp_path,
        method_dirs=["c-GC", "c-GC-star"],
        p_values=[1, 2, 3],
        analysis_profile=profile,
    )
    calibration_input = load_v2a_calibration_input(
        tmp_path,
        recording="fish_a",
        method_dir="c-GC",
        p_values=[1, 2, 3],
        analysis_profile=profile,
    )

    assert recordings == ["fish_a"]
    assert calibration_input.connectivity_path == (
        tmp_path / "outputs" / "v2a-RSNs" / profile / "c-GC" / "fish_a.pkl"
    )


def test_profile_input_rejects_mismatched_trace_selection_metadata(tmp_path):
    profile = V2A_ANALYSIS_PROFILE
    _write_recording_fixture(
        tmp_path,
        method_dir="c-GC",
        analysis_profile=profile,
    )
    metadata_path = (
        tmp_path
        / "outputs"
        / "v2a-RSNs"
        / profile
        / "c-GC"
        / "fish_a"
        / "run_metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["trace_selection"]["selected_cell_indices"] = [2, 1, 0]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="cell selection does not match"):
        load_v2a_calibration_input(
            tmp_path,
            recording="fish_a",
            method_dir="c-GC",
            p_values=[1, 2, 3],
            analysis_profile=profile,
        )


def test_inferred_completion_is_reproducible_and_excludes_self_loops():
    base = np.zeros((8, 8), dtype=int)
    base[0, 1] = 1
    adjacencies = {5: base}
    completion = {
        "generated_depths": {
            "6": {
                "base_p_value": 5,
                "false_to_true_count": 4,
            },
            "7": {
                "base_p_value": 5,
                "false_to_true_count": 7,
            },
        }
    }

    first = complete_inferred_depths(adjacencies, completion, seed=123)
    second = complete_inferred_depths(adjacencies, completion, seed=123)

    for depth, additions in [(6, 4), (7, 7)]:
        np.testing.assert_array_equal(first[depth], second[depth])
        assert int(first[depth].sum() - base.sum()) == additions
        assert not np.diag(first[depth]).any()
        np.testing.assert_array_equal(first[depth][base == 1], base[base == 1])


def test_surrogate_analyzer_runs_only_native_depths_before_completion():
    fitted_depths: list[int] = []

    class FakeEstimator:
        def __init__(
            self,
            *,
            n_perm: int,
            n_pasts: int,
            n_lags: int,
            temporal: bool,
            method: str,
        ):
            self.n_pasts = n_pasts
            self.n_nodes = 0

        def fit(self, data: np.ndarray, verbose: int = 0) -> "FakeEstimator":
            self.n_nodes = data.shape[0]
            fitted_depths.append(self.n_pasts)
            return self

        def get_connectivity_matrix(
            self,
            *,
            simulation: bool,
            alpha: float,
            beta: float,
        ) -> np.ndarray:
            matrix = np.zeros((self.n_nodes, self.n_nodes), dtype=int)
            matrix[0, 1] = 1
            return matrix

    metadata = {
        "gcstar_params": {
            "method": "fcgc",
            "n_perm": 5,
            "n_lags": 1,
            "alpha": 0.01,
            "beta": 0.001,
            "temporal": True,
            "simulation": False,
            "verbose": 0,
        },
        "inferred_completion": {
            "generated_depths": {
                "6": {"base_p_value": 5, "false_to_true_count": 2},
                "7": {"base_p_value": 5, "false_to_true_count": 3},
            }
        },
    }
    analyzer = make_v2a_surrogate_analyzer(
        metadata,
        estimator_cls=FakeEstimator,
    )

    output = analyzer(np.ones((20, 4)), list(range(1, 8)), 123)

    assert fitted_depths == [1, 2, 3, 4, 5]
    assert sorted(output) == list(range(1, 8))
    assert int(output[6].sum()) == 3
    assert int(output[7].sum()) == 4


@pytest.mark.parametrize(
    ("algorithm", "analyzer_name"),
    [
        ("pcmciplus", "analyze_with_pcmciplus"),
    ],
)
def test_surrogate_analyzer_uses_recorded_tigramite_configuration(
    monkeypatch,
    algorithm,
    analyzer_name,
):
    calls = []

    def fake_analyzer(X, p_values, *, pc_alpha):
        calls.append((X.copy(), list(p_values), pc_alpha))
        return {depth: np.eye(X.shape[1], dtype=int) for depth in p_values}

    monkeypatch.setattr(v2a_calibration, analyzer_name, fake_analyzer)
    analyzer = make_v2a_surrogate_analyzer(
        {
            "tigramite_params": {
                "algorithm": algorithm,
                "pc_alpha": 0.025,
                "tau_min": 1,
                "tau_max": 1,
                "depth_parameter": "maximum_conditioning_set_size",
                "conditioning_parameters": [
                    "max_conds_dim",
                    "max_conds_py",
                    "max_conds_px",
                    "max_conds_px_lagged",
                ],
            }
        }
    )
    X = np.arange(60, dtype=float).reshape(20, 3)

    output = analyzer(X, [1, 2, 3], 123)

    assert [call[1] for call in calls] == [[1], [2], [3]]
    assert all(call[2] == 0.025 for call in calls)
    assert all(np.array_equal(call[0], X) for call in calls)
    assert sorted(output) == [1, 2, 3]


def test_surrogate_analyzer_rejects_unknown_tigramite_algorithm():
    with pytest.raises(ValueError, match="Unsupported.*algorithm"):
        make_v2a_surrogate_analyzer(
            {
                "tigramite_params": {
                    "algorithm": "unknown",
                    "pc_alpha": 0.05,
                }
            }
        )


def test_surrogate_analyzer_rejects_legacy_varying_lag_pcmciplus_metadata():
    with pytest.raises(ValueError, match="fixed-lag"):
        make_v2a_surrogate_analyzer(
            {
                "tigramite_params": {
                    "algorithm": "pcmciplus",
                    "pc_alpha": 0.05,
                    "tau_min": 1,
                    "tau_max": 7,
                    "depth_parameter": "tau_max",
                    "conditioning_parameters": [],
                }
            }
        )


def test_resumable_calibration_reuses_observed_pickles_and_checkpoints(tmp_path):
    X = np.arange(80, dtype=float).reshape(40, 2)
    observed = {
        1: np.array([[0, 0], [0, 0]], dtype=int),
        2: np.array([[0, 1], [0, 0]], dtype=int),
        3: np.array([[0, 1], [1, 0]], dtype=int),
    }
    calls: list[int] = []

    def analyzer(
        X_surrogate: np.ndarray,
        p_values: list[int],
        replicate_seed: int,
    ) -> dict[int, np.ndarray]:
        calls.append(replicate_seed)
        assert X_surrogate.shape == X.shape
        output = {}
        for depth in p_values:
            matrix = np.zeros((2, 2), dtype=int)
            if (replicate_seed + depth) % 2:
                matrix[0, 1] = 1
            output[depth] = matrix
        return output

    first = run_resumable_v2a_calibration(
        X,
        observed_adjacencies=observed,
        analyze_surrogate=analyzer,
        p_values=[1, 2, 3],
        p0=1,
        B=3,
        block_length=4,
        seed=17,
        checkpoint_dir=tmp_path / "checkpoints",
        config_metadata={"method": "test"},
    )

    assert len(calls) == 3
    assert first.reused_replicates == 0
    assert first.result.observed["edge_counts"] == {1: 0, 2: 1, 3: 2}
    assert len(first.result.null["T_boot"]) == 3
    assert len(list((tmp_path / "checkpoints").glob("replicate_*.pkl"))) == 3

    calls.clear()
    second = run_resumable_v2a_calibration(
        X,
        observed_adjacencies=observed,
        analyze_surrogate=analyzer,
        p_values=[1, 2, 3],
        p0=1,
        B=3,
        block_length=4,
        seed=17,
        checkpoint_dir=tmp_path / "checkpoints",
        config_metadata={"method": "test"},
    )

    assert calls == []
    assert second.reused_replicates == 3
    assert second.result == first.result


def test_dynamic_plot_supports_eight_method_recording_panels(tmp_path):
    results = {}
    for method in ["c-GC", "c-GC-star"]:
        results[method] = {}
        for index in range(4):
            results[method][f"fish-{index + 1}"] = {
                "observed": {"T_obs": 0.1 + index * 0.01},
                "null": {
                    "T_boot": [0.05, 0.08, 0.12],
                    "critical_95": 0.116,
                },
            }

    output = plot_v2a_calibration_grid(results, tmp_path / "depth_bands.png")

    assert output.read_bytes().startswith(b"\x89PNG")


def test_dynamic_pointwise_plot_supports_eight_panels(tmp_path):
    results = {}
    for method in ["c-GC", "c-GC-star"]:
        results[method] = {}
        for index in range(4):
            results[method][f"fish-{index + 1}"] = {
                "observed": {
                    "D_obs": {2: 0.02 + index * 0.001, 3: 0.01},
                },
                "null": {
                    "pointwise": {
                        2: {"lower": 0.005, "upper": 0.025},
                        3: {"lower": 0.004, "upper": 0.020},
                    },
                },
            }

    output = plot_v2a_pointwise_grid(
        results,
        tmp_path / "pointwise_bands.png",
    )

    assert output.read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize(
    "recording",
    [
        "220119_F2_run11",
        "220127_F4_run2",
        "220210_F1_run6",
        "220210_F2_run5",
    ],
)
def test_recording_calibration_notebooks_are_isolated_and_use_pickle_contract(
    recording,
):
    calibration_dir = Path(__file__).parents[1] / "notebooks" / "calibration"
    notebook_path = calibration_dir / f"bootstrap_null_v2a_{recording}.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_source = "\n".join(
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert f"RECORDING = '{recording}'" in code_source
    assert "'method_dir': 'c-GC'" in code_source
    assert "'method_dir': 'c-GC-star'" in code_source
    assert "'method_dir': 'pcmciplus'" not in code_source
    assert "load_v2a_calibration_input" in code_source
    assert "run_resumable_v2a_calibration" in code_source
    assert "transitions.csv" not in code_source
    assert "edge_count_p" not in code_source
    assert "hash(" not in code_source
    assert "ANALYSIS_PROFILE = V2A_ANALYSIS_PROFILE" in code_source
    assert "/ ANALYSIS_PROFILE\n    / RECORDING" in code_source
    assert "P_VALUES = [1, 2, 3, 4, 5, 6, 7]" in code_source
    assert "analysis_profile=ANALYSIS_PROFILE" in code_source
    assert (
        "'trace_selection': calibration_input.metadata['trace_selection']"
        in code_source
    )
    assert "metadata.get('gcstar_params')" in code_source
    assert "metadata.get('tigramite_params')" in code_source
    assert "checkpoint_dir = method_output_dir / 'checkpoints'" in code_source
    assert "run_output_path = method_output_dir / 'bootstrap.json'" in code_source
    assert "show_progress=SHOW_PROGRESS" in code_source
    assert "B is a cumulative target" in code_source
    assert all(not cell.get("outputs") for cell in notebook["cells"])


def test_v2a_calibration_notebooks_are_recording_split():
    calibration_dir = Path(__file__).parents[1] / "notebooks" / "calibration"

    assert not (calibration_dir / "bootstrap_null_v2a.ipynb").exists()
    assert not (calibration_dir / "bootstrap_null_v2a_c-GC.ipynb").exists()
    assert not (calibration_dir / "bootstrap_null_v2a_c-GC-star.ipynb").exists()
    assert not (calibration_dir / "bootstrap_null_v2a_pcmciplus.ipynb").exists()
    assert not (calibration_dir / "bootstrap_null_v2a_jpcmciplus.ipynb").exists()
    assert sorted(path.name for path in calibration_dir.glob("bootstrap_null_v2a_*.ipynb")) == [
        "bootstrap_null_v2a_220119_F2_run11.ipynb",
        "bootstrap_null_v2a_220127_F4_run2.ipynb",
        "bootstrap_null_v2a_220210_F1_run6.ipynb",
        "bootstrap_null_v2a_220210_F2_run5.ipynb",
    ]


def test_connectivity_notebooks_use_shared_profile_and_seven_depths():
    notebooks_root = Path(__file__).parents[1] / "notebooks" / "v2a-RSNs"
    notebook_paths = sorted(notebooks_root.glob("c-GC*/[0-9]*.ipynb"))

    assert len(notebook_paths) == 8
    for notebook_path in notebook_paths:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_source = "\n".join(
            "".join(cell["source"])
            if isinstance(cell["source"], list)
            else cell["source"]
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )

        assert "P_VALUES = [1, 2, 3, 4, 5, 6, 7]" in code_source, notebook_path
        assert "analysis_profile=V2A_ANALYSIS_PROFILE" in code_source, notebook_path
        assert "get_v2a_selection_metadata" in code_source, notebook_path
        assert "'trace_selection': trace_selection" in code_source, notebook_path


@pytest.mark.parametrize(
    ("relative_path", "required_marker"),
    [
        ("real_data/v2a_depth_selection.ipynb", "V2A_OUTPUT_DIR ="),
        ("real_data/v2a_edgewise_localization.ipynb", "V2A_ANALYSIS_PROFILE"),
        ("v2a-RSNs/v2a_depth_selection.ipynb", "V2A_OUTPUT_DIR ="),
        ("v2a-RSNs/v2a_edgewise_localization.ipynb", "V2A_ANALYSIS_PROFILE"),
        ("v2a-RSNs/compare_all_methods.ipynb", "V2A_OUTPUT_DIR ="),
    ],
)
def test_v2a_downstream_notebooks_use_profile_namespace(
    relative_path,
    required_marker,
):
    notebook_path = Path(__file__).parents[1] / "notebooks" / relative_path
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_source = "\n".join(
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert required_marker in code_source
    assert "/ 'outputs' / 'v2a-RSNs' / 'c-GC'" not in code_source
    assert "/ 'outputs' / 'v2a-RSNs' / 'c-GC-star'" not in code_source


def test_v2a_profile_notebook_code_cells_parse():
    notebooks_root = Path(__file__).parents[1] / "notebooks"
    notebook_paths = [
        *sorted((notebooks_root / "v2a-RSNs").glob("c-GC*/*.ipynb")),
        *sorted((notebooks_root / "calibration").glob("bootstrap_null_v2a_*.ipynb")),
        notebooks_root / "real_data" / "v2a_depth_selection.ipynb",
        notebooks_root / "real_data" / "v2a_edgewise_localization.ipynb",
        notebooks_root / "v2a-RSNs" / "v2a_depth_selection.ipynb",
        notebooks_root / "v2a-RSNs" / "v2a_edgewise_localization.ipynb",
        notebooks_root / "v2a-RSNs" / "compare_all_methods.ipynb",
    ]

    for notebook_path in notebook_paths:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = (
                "".join(cell["source"])
                if isinstance(cell["source"], list)
                else cell["source"]
            )
            ast.parse(source, filename=f"{notebook_path.name}:cell-{index}")
