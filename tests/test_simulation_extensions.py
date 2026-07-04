from __future__ import annotations

import json

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


def test_jpcmciplus_adapter_forwards_pc_alpha(monkeypatch):
    calls = []

    def fake_analyzer(X, p_values, *, algorithm, pc_alpha):
        calls.append((algorithm, pc_alpha))
        return {p_values[0]: np.zeros((X.shape[1], X.shape[1]), dtype=int)}

    monkeypatch.setattr(adapters, "_analyze_with_tigramite", fake_analyzer)
    adapters.analyze_with_jpcmciplus(
        np.zeros((20, 3)),
        [2],
        pc_alpha=0.025,
    )

    assert calls == [("jpcmciplus", 0.025)]


def test_recovery_metrics_exclude_diagonal_true_negatives():
    truth = np.array([[0, 1], [0, 0]])
    predicted = np.zeros((2, 2), dtype=int)
    metrics = recovery_metrics(predicted, truth)
    assert metrics["tn"] == 1
    assert metrics["fn"] == 1
    assert metrics["accuracy"] == 0.5


def test_extension_runner_preserves_all_downstream_artifacts(
    monkeypatch, tmp_path
):
    def analyzer(X: np.ndarray, p_values: list[int]) -> dict[int, np.ndarray]:
        d = X.shape[1]
        output = {}
        for depth in p_values:
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
