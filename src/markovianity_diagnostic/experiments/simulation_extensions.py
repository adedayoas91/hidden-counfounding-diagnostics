"""Extension-compatible simulation sweeps and artifacts.

This module is the package-first replacement for notebook-local result
aggregation. It preserves each trial's input, ground truth, and inferred graph
at every conditioning depth so calibration, localization, depth selection, and
reporting can consume the same frozen run.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from markovianity_diagnostic.reporting.manifest import Manifest

from .adapters import METHODS, METHOD_METADATA
from .depth_selection import DepthSelector
from .graph_metrics import summarize_run
from .simulations import SCENARIOS


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def load_trial_adjacencies(path: str | Path) -> dict[int, np.ndarray]:
    """Load a trial adjacency artifact written by this module."""
    with np.load(path, allow_pickle=False) as payload:
        return {
            int(key.removeprefix("p_")): np.asarray(value, dtype=int)
            for key, value in payload.items()
        }


def _scenario_arguments(
    scenario_name: str,
    *,
    T: int,
    d: int,
    seed: int,
) -> dict[str, int]:
    signature = inspect.signature(SCENARIOS[scenario_name])
    available = {"T": T, "d": d, "seed": seed}
    return {
        name: value
        for name, value in available.items()
        if name in signature.parameters
    }


def _depth_rows(
    *,
    scenario: str,
    method: str,
    repeat: int,
    summary: dict[str, Any],
    selection: dict[str, int | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cumulative = 0.0
    for depth in sorted(summary["edge_counts"]):
        d_value = summary["D_p"].get(depth)
        if d_value is not None:
            cumulative += float(d_value)
        decomposition = summary["D_parts"].get(depth, {})
        recovery = summary.get("metrics_by_p", {}).get(depth, {})
        rows.append(
            {
                "scenario": scenario,
                "method": method,
                "repeat": repeat,
                "depth": depth,
                "edge_count": summary["edge_counts"][depth],
                "D_p": d_value,
                "D_minus": decomposition.get("D_minus"),
                "D_plus": decomposition.get("D_plus"),
                "cumulative_instability": cumulative,
                "T_obs": summary["T_obs"],
                "selected_absolute": selection.get("absolute"),
                "selected_relative": selection.get("relative"),
                "selected_bootstrap_band": selection.get("bootstrap_band"),
                **recovery,
            }
        )
    return rows


def run_extension_simulations(
    *,
    method: str,
    scenario_names: list[str],
    p_values: list[int],
    n_repeats: int,
    T: int,
    d: int,
    seed: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run simulations and write all artifacts required by the extension plan."""
    if method not in METHODS or method in {"user_method"}:
        raise ValueError(f"Unknown runnable method {method!r}")
    if not scenario_names:
        raise ValueError("scenario_names cannot be empty")
    missing_scenarios = sorted(set(scenario_names).difference(SCENARIOS))
    if missing_scenarios:
        raise ValueError(f"Unknown scenarios: {missing_scenarios}")
    if sorted(set(p_values)) != p_values or len(p_values) < 2:
        raise ValueError("p_values must contain at least two unique sorted depths")
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")

    output_dir = Path(output_dir)
    trial_root = output_dir / "trials"
    trial_root.mkdir(parents=True, exist_ok=True)
    analyzer = METHODS[method]
    selector = DepthSelector()
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    input_paths: list[str] = []

    for scenario_name in scenario_names:
        scenario_fn = SCENARIOS[scenario_name]
        for repeat in range(n_repeats):
            local_seed = seed + repeat
            sample = scenario_fn(
                **_scenario_arguments(
                    scenario_name,
                    T=T,
                    d=d,
                    seed=local_seed,
                )
            )
            adjacencies = analyzer(sample.X, p_values)
            summary = summarize_run(
                adjacencies,
                sample.ground_truth_compact,
            )
            summary["cumulative_instability"] = float(
                sum(summary["D_p"].values())
            )
            selection_result = selector.apply_all_rules(
                summary["D_p"],
                D_boot_pointwise={},
            )

            trial_dir = trial_root / scenario_name / f"repeat_{repeat:03d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            data_path = trial_dir / "data.npy"
            truth_path = trial_dir / "ground_truth.npy"
            adjacency_path = trial_dir / "adjacencies.npz"
            np.save(data_path, sample.X, allow_pickle=False)
            np.save(
                truth_path,
                sample.ground_truth_compact,
                allow_pickle=False,
            )
            np.savez_compressed(
                adjacency_path,
                **{f"p_{depth}": graph for depth, graph in adjacencies.items()},
            )
            input_paths.append(str(data_path))
            relative_data_path = data_path.relative_to(output_dir)
            relative_truth_path = truth_path.relative_to(output_dir)
            relative_adjacency_path = adjacency_path.relative_to(output_dir)

            records.append(
                {
                    "scenario": scenario_name,
                    "method": method,
                    "repeat": repeat,
                    "seed": local_seed,
                    "metadata": sample.metadata,
                    "data_path": str(relative_data_path),
                    "ground_truth_path": str(relative_truth_path),
                    "adjacency_path": str(relative_adjacency_path),
                    "summary": summary,
                    "depth_selection": {
                        "selected": selection_result.selected,
                        "warnings": selection_result.warnings,
                    },
                    "calibration_status": "ready_for_surrogate_rerun",
                }
            )
            rows.extend(
                _depth_rows(
                    scenario=scenario_name,
                    method=method,
                    repeat=repeat,
                    summary=summary,
                    selection=selection_result.selected,
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "extension_results.json"
    metrics_path = output_dir / "extension_metrics.csv"
    summary_path = output_dir / "extension_summary.csv"
    results_path.write_text(
        json.dumps({"runs": records}, indent=2, default=_json_default)
    )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(metrics_path, index=False)
    numeric_columns = [
        column
        for column in (
            "edge_count",
            "D_p",
            "D_minus",
            "D_plus",
            "cumulative_instability",
            "T_obs",
            "accuracy",
            "precision",
            "recall",
            "fpr",
            "tp",
            "fp",
            "tn",
            "fn",
        )
        if column in metrics
    ]
    aggregate = (
        metrics.groupby(["scenario", "method", "depth"], dropna=False)[
            numeric_columns
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in aggregate.columns
    ]
    aggregate.to_csv(summary_path, index=False)

    manifest_path = output_dir / "manifest.json"
    Manifest(
        analysis="extension-compatible simulation metrics",
        input_paths=[
            str(Path(path).relative_to(output_dir)) for path in input_paths
        ],
        output_paths=[
            results_path.name,
            metrics_path.name,
            summary_path.name,
            trial_root.name,
        ],
        method=method,
        method_params={
            "analyzer_registry_key": method,
            **METHOD_METADATA.get(method, {}),
            "scenarios": scenario_names,
            "n_repeats": n_repeats,
            "T": T,
            "d": d,
        },
        p_values=p_values,
        random_seed=seed,
    ).to_json(manifest_path)

    return {
        "results_path": str(results_path),
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
        "n_runs": len(records),
    }


__all__ = [
    "load_trial_adjacencies",
    "run_extension_simulations",
]
