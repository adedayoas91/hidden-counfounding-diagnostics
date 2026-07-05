from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from markovianity_diagnostic.experiments import simulation_extensions
from markovianity_diagnostic.experiments import adapters
from markovianity_diagnostic.experiments.adapters import (
    FastGcStar,
    METHODS,
    _collapse_tigramite_graph,
)
from markovianity_diagnostic.experiments.simulation_extensions import (
    load_trial_adjacencies,
    run_extension_simulations,
)
from markovianity_diagnostic.experiments.graph_metrics import recovery_metrics


def test_gcstar_registry_uses_fast_implementation():
    assert METHODS["gcstar_cgc"].__name__ == "analyze_with_fast_gcstar_cgc"
    assert METHODS["gcstar_fcgc"].__name__ == "analyze_with_fast_gcstar_fcgc"
    assert FastGcStar.__name__ == "FastGcStar"


def test_tigramite_graph_collapses_to_target_by_source():
    graph = np.full((2, 2, 2), "", dtype=object)
    graph[0, 1, 1] = "-->"
    adjacency = _collapse_tigramite_graph(graph, directed_only=True)
    assert adjacency[1, 0] == 1
    assert adjacency[0, 1] == 0


def test_pcmciplus_adapter_forwards_pc_alpha(monkeypatch):
    calls = []

    def fake_analyzer(X, p_values, *, algorithm, pc_alpha):
        calls.append((algorithm, pc_alpha))
        return {p_values[0]: np.zeros((X.shape[1], X.shape[1]), dtype=int)}

    monkeypatch.setattr(adapters, "_analyze_with_tigramite", fake_analyzer)
    adapters.analyze_with_pcmciplus(
        np.zeros((20, 3)),
        [2],
        pc_alpha=0.025,
    )

    assert calls == [("pcmciplus", 0.025)]


def test_pcmciplus_depth_controls_conditioning_at_fixed_lag():
    assert adapters._pcmciplus_run_kwargs(4, pc_alpha=0.025) == {
        "tau_min": 1,
        "tau_max": 1,
        "pc_alpha": 0.025,
        "max_conds_dim": 4,
        "max_conds_py": 4,
        "max_conds_px": 4,
        "max_conds_px_lagged": 4,
    }


def test_jpcmciplus_is_not_an_active_simulation_method():
    assert "jpcmciplus" not in METHODS
    assert not hasattr(adapters, "analyze_with_jpcmciplus")


def test_recovery_metrics_exclude_diagonal_true_negatives():
    truth = np.array([[0, 1], [0, 0]])
    predicted = np.zeros((2, 2), dtype=int)
    metrics = recovery_metrics(predicted, truth)
    assert metrics["tn"] == 1
    assert metrics["fn"] == 1
    assert metrics["accuracy"] == 0.5


def test_recovery_metrics_binarize_weighted_truth_and_include_derived_scores():
    truth = np.array(
        [
            [0.0, 0.5, 0.0],
            [-0.4, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    predicted = np.array(
        [
            [0, 1, 1],
            [1, 0, 0],
            [0, 0, 0],
        ]
    )

    metrics = recovery_metrics(predicted, truth)

    assert metrics["tp"] == 2
    assert metrics["fp"] == 1
    assert metrics["tn"] == 3
    assert metrics["fn"] == 0
    assert metrics["balanced_accuracy"] == 0.875
    assert metrics["f1"] == 0.8


def test_extension_runner_preserves_all_downstream_artifacts(
    monkeypatch,
    tmp_path,
    capsys,
):
    analyzed_depths = []

    def analyzer(X: np.ndarray, p_values: list[int]) -> dict[int, np.ndarray]:
        d = X.shape[1]
        output = {}
        for depth in p_values:
            analyzed_depths.append(depth)
            graph = np.zeros((d, d), dtype=int)
            if depth % 2:
                graph[1, 0] = 1
            output[depth] = graph
        return output

    monkeypatch.setitem(simulation_extensions.METHODS, "test_method", analyzer)
    result = run_extension_simulations(
        method="test_method",
        scenario_names=["order1_unconfounded"],
        p_values=[1, 2, 3, 4],
        n_repeats=1,
        T=100,
        d=4,
        seed=42,
        output_dir=tmp_path,
        show_progress=True,
    )

    metrics = pd.read_csv(result["metrics_path"])
    required = {
        "edge_count",
        "D_p",
        "D_minus",
        "D_plus",
        "cumulative_instability",
        "T_obs",
        "selected_absolute",
        "selected_relative",
        "selected_bootstrap_band",
        "accuracy",
        "precision",
        "recall",
        "fpr",
    }
    assert required.issubset(metrics.columns)

    payload = json.loads((tmp_path / "extension_results.json").read_text())
    run = payload["runs"][0]
    assert run["calibration_status"] == "ready_for_surrogate_rerun"
    assert np.load(tmp_path / run["data_path"], allow_pickle=False).shape == (
        100,
        4,
    )
    adjacency = load_trial_adjacencies(tmp_path / run["adjacency_path"])
    assert sorted(adjacency) == [1, 2, 3, 4]
    assert (tmp_path / "manifest.json").exists()
    assert result["reused_trials"] == 0
    assert result["computed_trials"] == 1
    assert analyzed_depths == [1, 2, 3, 4]

    resumed = run_extension_simulations(
        method="test_method",
        scenario_names=["order1_unconfounded"],
        p_values=[1, 2, 3, 4],
        n_repeats=1,
        T=100,
        d=4,
        seed=42,
        output_dir=tmp_path,
        show_progress=False,
    )
    assert resumed["reused_trials"] == 1
    assert resumed["computed_trials"] == 0
    assert analyzed_depths == [1, 2, 3, 4]

    resumed_payload = json.loads(
        (tmp_path / "extension_results.json").read_text()
    )
    assert resumed_payload["runs"][0]["reused_trial_artifacts"] is True

    captured = capsys.readouterr()
    assert "Extension-compatible simulation metrics" in captured.out
    assert "Total trials: 1" in captured.out
    assert "Extension metrics complete" in captured.out
    assert "Completed trials: 1" in captured.out


def test_reusable_trial_accepts_matching_nan_positions(tmp_path):
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    data = np.array([[0.0, np.nan], [1.0, 2.0]])
    truth = np.array([[0, 1], [0, 0]])
    np.save(trial_dir / "data.npy", data, allow_pickle=False)
    np.save(trial_dir / "ground_truth.npy", truth, allow_pickle=False)
    np.savez_compressed(
        trial_dir / "adjacencies.npz",
        p_1=np.zeros((2, 2), dtype=int),
        p_2=np.zeros((2, 2), dtype=int),
    )

    reused = simulation_extensions._load_reusable_trial(
        trial_dir,
        expected_data=data.copy(),
        expected_truth=truth.copy(),
        p_values=[1, 2],
    )

    assert reused is not None
    assert sorted(reused) == [1, 2]


def test_extension_metric_notebooks_expose_progress_and_workload_configuration():
    project_root = Path(__file__).parents[1]
    notebooks = {
        "c-GC/extension_metrics.ipynb": "fast_gcstar_cgc",
        "c-GC-star/extension_metrics.ipynb": "fast_gcstar_fcgc",
    }

    for relative_path, method in notebooks.items():
        notebook_path = (
            project_root / "notebooks" / "simulations" / relative_path
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"])
            if isinstance(cell["source"], list)
            else cell["source"]
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        markdown = "\n".join(
            "".join(cell["source"])
            if isinstance(cell["source"], list)
            else cell["source"]
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )

        assert f"METHOD = '{method}'" in code
        assert "N_REPEATS = 10" in code
        assert "P_VALUES = [1, 2, 3, 4, 5, 6, 7]" in code
        assert "Total single-depth fits" in code
        assert "'-u'" in code
        assert "progress" in markdown.lower()

        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = (
                "".join(cell["source"])
                if isinstance(cell["source"], list)
                else cell["source"]
            )
            ast.parse(source, filename=f"{notebook_path.name}:cell-{index}")
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
