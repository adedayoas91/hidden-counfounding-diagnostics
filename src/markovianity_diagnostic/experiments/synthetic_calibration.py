"""Resumable bootstrap calibration for the matched synthetic depth sweeps."""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .adapters import METHODS, METHOD_METADATA
from .simulations import (
    scenario_latent_common_driver,
    scenario_order1_unconfounded,
    scenario_order3_unconfounded,
)
from .v2a_calibration import (
    calibration_payload,
    plot_v2a_calibration_grid,
    plot_v2a_pointwise_grid,
    run_resumable_v2a_calibration,
    stable_seed,
    write_calibration_payload,
)


ELIGIBLE_METHODS = {
    "c-GC": "fast_gcstar_cgc",
    "c-GC-star": "fast_gcstar_fcgc",
}

SYNTHETIC_SCENARIOS = {
    "order1_unconfounded": scenario_order1_unconfounded,
    "order1_with_latent_confounder": scenario_latent_common_driver,
    "order3_unconfounded": scenario_order3_unconfounded,
}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _manifest_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _validate_requested_names(
    requested: Sequence[str],
    available: dict[str, Any],
    *,
    label: str,
) -> list[str]:
    normalized = [str(name) for name in requested]
    if not normalized:
        raise ValueError(f"{label} cannot be empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must be unique")
    unknown = sorted(set(normalized).difference(available))
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}")
    return normalized


def _analyze_depth_grid(
    analyzer: Callable[[np.ndarray, list[int]], dict[int, np.ndarray]],
    X: np.ndarray,
    depths: list[int],
    *,
    label: str,
    show_progress: bool,
) -> dict[int, np.ndarray]:
    iterator: Any = depths
    if show_progress:
        iterator = tqdm(
            depths,
            desc=label,
            unit="depth",
            dynamic_ncols=True,
            leave=False,
            position=2,
        )

    output: dict[int, np.ndarray] = {}
    for depth in iterator:
        output.update(analyzer(X, [int(depth)]))
    return output


def run_synthetic_calibration(
    project_root: str | Path,
    *,
    output_dir: str | Path,
    method_names: Sequence[str] = ("c-GC", "c-GC-star"),
    scenario_names: Sequence[str] = tuple(SYNTHETIC_SCENARIOS),
    p_values: Sequence[int] = tuple(range(1, 8)),
    p0: int = 1,
    B: int = 5,
    block_length: int = 1,
    seed: int = 42,
    T: int = 2000,
    d: int = 10,
    n_jobs: int = 1,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run or resume synthetic calibration for c-GC and c-GC*.

    ``B`` is the target cumulative replicate count per method and scenario.
    Existing compatible replicate checkpoints are reused, so increasing
    ``B`` from 5 to 10 computes only replicates 6--10.
    """

    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    methods = _validate_requested_names(
        method_names,
        ELIGIBLE_METHODS,
        label="method_names",
    )
    scenarios = _validate_requested_names(
        scenario_names,
        SYNTHETIC_SCENARIOS,
        label="scenario_names",
    )
    depths = [int(depth) for depth in p_values]
    if depths != sorted(set(depths)) or len(depths) < 2:
        raise ValueError("p_values must contain at least two sorted unique depths")
    if p0 not in depths:
        raise ValueError("p0 must be included in p_values")
    if B < 1:
        raise ValueError("B must be at least 1")
    if T <= p0 + 5:
        raise ValueError("T is too short for the requested p0")
    if d < 2:
        raise ValueError("d must be at least 2")

    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, dict[str, Any]]] = {}
    summary_rows: list[dict[str, Any]] = []
    run_output_paths: list[Path] = []
    suite_start = time.perf_counter()
    jobs = [
        (method_name, scenario_name)
        for method_name in methods
        for scenario_name in scenarios
    ]
    maximum_depth_fits = len(jobs) * (B + 1) * len(depths)

    print("=" * 72)
    print("Synthetic bootstrap-null calibration")
    print(f"Methods: {', '.join(methods)}")
    print(f"Scenarios: {', '.join(scenarios)}")
    print(f"Depth grid: {depths}; p0={p0}")
    print(
        f"Cumulative target: B={B} per method/scenario; "
        f"block_length={block_length}; n_jobs={n_jobs}"
    )
    print(f"Data shape per scenario: T={T}, d={d}")
    print(f"Output root: {output_dir}")
    print(
        "Maximum learner workload if no checkpoints exist: "
        f"{maximum_depth_fits} single-depth fits"
    )
    print("Compatible replicate checkpoints will be reused automatically.")
    print("=" * 72)

    results = {method_name: {} for method_name in methods}
    job_iterator: Any = enumerate(jobs, start=1)
    overall_progress = None
    if show_progress:
        overall_progress = tqdm(
            job_iterator,
            total=len(jobs),
            desc="Overall method/scenario progress",
            unit="job",
            dynamic_ncols=True,
            leave=True,
            position=0,
        )
        job_iterator = overall_progress

    for job_index, (method_name, scenario_name) in job_iterator:
        registry_key = ELIGIBLE_METHODS[method_name]
        analyzer = METHODS[registry_key]

        status = tqdm.write if show_progress else print
        status(
            f"\n[{job_index}/{len(jobs)}] {method_name} | {scenario_name}"
        )
        status(
            f"Generating controlled data (T={T}, d={d}, seed={seed})..."
        )
        scenario_fn = SYNTHETIC_SCENARIOS[scenario_name]
        sample = scenario_fn(T=T, d=d, seed=seed)
        status(f"Observed data ready: shape={sample.X.shape}")
        status(f"Inferring observed graph grid at {len(depths)} depths...")
        observed_start = time.perf_counter()
        observed_adjacencies = _analyze_depth_grid(
            analyzer,
            sample.X,
            depths,
            label=f"{method_name} observed depths",
            show_progress=show_progress and n_jobs == 1,
        )
        status(
            "Observed graph grid complete "
            f"({time.perf_counter() - observed_start:.1f}s)."
        )

        run_seed = stable_seed(seed, method_name, scenario_name)
        checkpoint_dir = output_dir / "checkpoints" / method_name / scenario_name
        status(f"Checkpoint directory: {checkpoint_dir}")
        config_metadata = {
            "method": method_name,
            "method_registry_key": registry_key,
            "method_metadata": METHOD_METADATA.get(registry_key, {}),
            "scenario": scenario_name,
            "scenario_metadata": sample.metadata,
            "T": int(T),
            "d": int(d),
        }

        def analyze_surrogate(
            X_surrogate: np.ndarray,
            requested_depths: list[int],
            replicate_seed: int,
            *,
            _analyzer=analyzer,
        ) -> dict[int, np.ndarray]:
            del replicate_seed
            return _analyze_depth_grid(
                _analyzer,
                X_surrogate,
                requested_depths,
                label=f"{method_name} surrogate depths",
                show_progress=show_progress and n_jobs == 1,
            )

        status(
            f"Running or reusing {B} bootstrap replicates "
            f"across depths {depths}..."
        )
        run_start = time.perf_counter()
        outcome = run_resumable_v2a_calibration(
            sample.X,
            observed_adjacencies=observed_adjacencies,
            analyze_surrogate=analyze_surrogate,
            p_values=depths,
            p0=p0,
            B=B,
            block_length=block_length,
            seed=run_seed,
            checkpoint_dir=checkpoint_dir,
            n_jobs=n_jobs,
            config_metadata=config_metadata,
            show_progress=show_progress,
            progress_label=f"{method_name} | {scenario_name}",
            progress_position=1,
        )
        elapsed = time.perf_counter() - run_start
        run_metadata = {
            **config_metadata,
            "p_values": depths,
            "p0": int(p0),
            "B": int(B),
            "block_length": int(block_length),
            "seed": int(run_seed),
            "elapsed_seconds": float(elapsed),
        }
        run_output_path = output_dir / method_name / scenario_name / "bootstrap.json"
        write_calibration_payload(
            run_output_path,
            outcome,
            metadata=run_metadata,
        )
        payload = calibration_payload(outcome, metadata=run_metadata)
        results[method_name][scenario_name] = payload
        run_output_paths.append(run_output_path)

        summary_rows.append(
            {
                "method": method_name,
                "scenario": scenario_name,
                "T_obs": outcome.result.observed["T_obs"],
                "critical_90": outcome.result.null["critical_90"],
                "critical_95": outcome.result.null["critical_95"],
                "critical_99": outcome.result.null["critical_99"],
                "p_value": outcome.result.null["p_value"],
                "reject_global_95": outcome.result.diagnosis[
                    "reject_global_95"
                ],
                "first_exceedance_depth": outcome.result.diagnosis[
                    "first_exceedance_depth"
                ],
                "B": int(B),
                "reused_replicates": outcome.reused_replicates,
                "elapsed_seconds": float(elapsed),
            }
        )
        status(
            f"Completed {method_name} | {scenario_name}: "
            f"T_obs={outcome.result.observed['T_obs']:.6f}; "
            f"critical_95={outcome.result.null['critical_95']:.6f}; "
            f"p={outcome.result.null['p_value']:.4f}; "
            f"reused={outcome.reused_replicates}/{B}; "
            f"computed={B - outcome.reused_replicates}; "
            f"elapsed={elapsed:.1f}s"
        )
        status(f"Saved scenario result: {run_output_path}")
        if overall_progress is not None:
            overall_progress.set_postfix(
                method=method_name,
                scenario=scenario_name,
                refresh=True,
            )

    report_dir = (
        output_dir / methods[0]
        if len(methods) == 1
        else output_dir / "combined"
    )
    results_path = report_dir / "bootstrap_results.json"
    summary_path = report_dir / "bootstrap_summary.csv"
    null_plot_path = report_dir / "bootstrap_T_obs.png"
    pointwise_plot_path = report_dir / "bootstrap_pointwise.png"
    manifest_path = report_dir / "manifest.json"

    _atomic_write_json(results_path, results)
    _atomic_write_csv(summary_path, pd.DataFrame(summary_rows))
    plot_v2a_calibration_grid(results, null_plot_path)
    plot_v2a_pointwise_grid(results, pointwise_plot_path)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis": "bootstrap_null_synthetic_matched_cgc",
        "input_paths": [],
        "output_paths": [
            _manifest_path(path, project_root)
            for path in (
                *run_output_paths,
                results_path,
                summary_path,
                null_plot_path,
                pointwise_plot_path,
            )
        ],
        "methods": methods,
        "method_registry_keys": [ELIGIBLE_METHODS[name] for name in methods],
        "scenarios": scenarios,
        "p_values": depths,
        "p0": int(p0),
        "B": int(B),
        "block_length": int(block_length),
        "random_seed": int(seed),
        "T": int(T),
        "d": int(d),
        "n_jobs": int(n_jobs),
        "elapsed_seconds": float(time.perf_counter() - suite_start),
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    _atomic_write_json(manifest_path, manifest)

    print("=" * 72)
    print("Synthetic calibration complete")
    print(f"Elapsed: {time.perf_counter() - suite_start:.1f}s")
    print(f"Summary: {summary_path}")
    print(f"Null-distribution plot: {null_plot_path}")
    print(f"Pointwise plot: {pointwise_plot_path}")
    print(f"Manifest: {manifest_path}")
    print("=" * 72)

    return {
        "results": results,
        "summary": pd.DataFrame(summary_rows),
        "results_path": results_path,
        "summary_path": summary_path,
        "null_plot_path": null_plot_path,
        "pointwise_plot_path": pointwise_plot_path,
        "manifest_path": manifest_path,
    }


__all__ = [
    "ELIGIBLE_METHODS",
    "SYNTHETIC_SCENARIOS",
    "run_synthetic_calibration",
]
