"""Pickle-first, resumable bootstrap calibration for v2a-RSN recordings.

Observed connectivity is loaded from the completed per-recording pickle files.
Raw fluorescence traces are used only to fit and sample the surrogate null.
Each surrogate connectivity grid is checkpointed independently so a long
calibration can resume without rerunning observed connectivity notebooks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from .adapters import GcStar, make_gcstar_analyzer
from .calibration import CalibrationResult, MovingBlockBootstrapNull
from .graph_metrics import compute_graph_stability_metrics
from .plotting import require_matplotlib
from .v2a_rsn_utils import (
    binarize_adjacencies,
    load_adjacency_checkpoint,
    load_and_filter_traces,
    save_adjacency_checkpoint,
)

logger = logging.getLogger(__name__)


SurrogateAnalyzer = Callable[
    [np.ndarray, list[int], int],
    dict[int, np.ndarray],
]


@dataclass(frozen=True)
class V2ACalibrationInput:
    """Validated inputs for one recording-method calibration."""

    recording: str
    method_dir: str
    X: np.ndarray
    observed_adjacencies: dict[int, np.ndarray]
    metadata: dict[str, Any]
    connectivity_path: Path


@dataclass(frozen=True)
class V2ACalibrationOutcome:
    """Calibration result plus checkpoint accounting."""

    result: CalibrationResult
    replicate_summaries: list[dict[str, Any]]
    reused_replicates: int
    config_hash: str


def stable_seed(base_seed: int, *parts: object) -> int:
    """Derive a process-independent uint32 seed from stable text components."""

    payload = "|".join([str(int(base_seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _adjacency_grid_sha256(adjacencies: dict[int, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for depth, adjacency in sorted(adjacencies.items()):
        digest.update(str(int(depth)).encode("ascii"))
        digest.update(np.ascontiguousarray(adjacency).tobytes())
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _normalize_p_values(p_values: Sequence[int]) -> list[int]:
    normalized = [int(depth) for depth in p_values]
    if len(normalized) < 2:
        raise ValueError("p_values must contain at least two depths")
    if normalized != sorted(set(normalized)):
        raise ValueError("p_values must be sorted and unique")
    return normalized


def validate_adjacency_grid(
    adjacencies: dict[int, np.ndarray],
    *,
    p_values: Sequence[int],
    n_nodes: int | None = None,
    label: str = "adjacency grid",
) -> dict[int, np.ndarray]:
    """Validate and normalize a complete binary adjacency grid."""

    requested = _normalize_p_values(p_values)
    missing = sorted(set(requested).difference(adjacencies))
    if missing:
        raise ValueError(f"{label} is missing conditioning depths: {missing}")

    normalized = binarize_adjacencies(
        {depth: np.asarray(adjacencies[depth]) for depth in requested}
    )
    shapes = {matrix.shape for matrix in normalized.values()}
    if len(shapes) != 1:
        raise ValueError(f"{label} has inconsistent matrix shapes: {sorted(shapes)}")

    shape = next(iter(shapes))
    if n_nodes is not None and shape != (n_nodes, n_nodes):
        raise ValueError(
            f"{label} has shape {shape}, but the traces contain {n_nodes} variables"
        )
    return normalized


def discover_complete_recordings(
    project_root: Path,
    *,
    method_dirs: Sequence[str],
    p_values: Sequence[int],
) -> list[str]:
    """Find recordings with a complete pickle for every requested method."""

    project_root = Path(project_root)
    methods = [str(method) for method in method_dirs]
    if not methods:
        raise ValueError("method_dirs cannot be empty")

    recordings: set[str] = set()
    for method_dir in methods:
        method_root = project_root / "outputs" / "v2a-RSNs" / method_dir
        recordings.update(path.stem for path in method_root.glob("*.pkl"))
    if not recordings:
        raise FileNotFoundError("No v2a-RSN connectivity pickles were found")

    for recording in sorted(recordings):
        for method_dir in methods:
            path = (
                project_root / "outputs" / "v2a-RSNs" / method_dir / f"{recording}.pkl"
            )
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing connectivity pickle for {method_dir}/{recording}: {path}"
                )
            validate_adjacency_grid(
                load_adjacency_checkpoint(path),
                p_values=p_values,
                label=f"{method_dir}/{recording}",
            )
    return sorted(recordings)


def load_v2a_calibration_input(
    project_root: Path,
    *,
    recording: str,
    method_dir: str,
    p_values: Sequence[int],
) -> V2ACalibrationInput:
    """Load observed pickles and raw traces without reading transition tables."""

    project_root = Path(project_root)
    connectivity_path = (
        project_root / "outputs" / "v2a-RSNs" / method_dir / f"{recording}.pkl"
    )
    metadata_path = (
        project_root
        / "outputs"
        / "v2a-RSNs"
        / method_dir
        / recording
        / "run_metadata.json"
    )
    recording_dir = project_root / "data" / "v2a-RSNs" / recording

    if not connectivity_path.exists():
        raise FileNotFoundError(f"Missing observed connectivity: {connectivity_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing run metadata: {metadata_path}")

    traces = load_and_filter_traces(recording_dir, recording)
    X = np.asarray(traces, dtype=float).T
    observed = validate_adjacency_grid(
        load_adjacency_checkpoint(connectivity_path),
        p_values=p_values,
        n_nodes=X.shape[1],
        label=f"{method_dir}/{recording}",
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return V2ACalibrationInput(
        recording=recording,
        method_dir=method_dir,
        X=X,
        observed_adjacencies=observed,
        metadata=metadata,
        connectivity_path=connectivity_path,
    )


def complete_inferred_depths(
    adjacencies: dict[int, np.ndarray],
    completion: dict[str, Any],
    *,
    seed: int,
) -> dict[int, np.ndarray]:
    """Apply a recorded p=5-style inferred-depth completion to a surrogate."""

    output = {
        int(depth): np.asarray(matrix, dtype=int).copy()
        for depth, matrix in adjacencies.items()
    }
    generated_depths = completion.get("generated_depths", {})
    for depth_text, specification in sorted(
        generated_depths.items(),
        key=lambda item: int(item[0]),
    ):
        depth = int(depth_text)
        base_depth = int(specification["base_p_value"])
        addition_count = int(specification["false_to_true_count"])
        if base_depth not in output:
            raise ValueError(
                f"Cannot complete P={depth}: base P={base_depth} is unavailable"
            )

        base = np.asarray(output[base_depth], dtype=int)
        off_diagonal = ~np.eye(base.shape[0], dtype=bool)
        candidates = np.argwhere((base == 0) & off_diagonal)
        if addition_count > len(candidates):
            raise ValueError(
                f"Cannot complete P={depth}: requested {addition_count} additions "
                f"from {len(candidates)} eligible entries"
            )

        rng = np.random.default_rng(
            stable_seed(
                seed,
                "completion",
                depth,
                specification.get("random_seed", 0),
            )
        )
        chosen = candidates[
            rng.choice(len(candidates), size=addition_count, replace=False)
        ]
        completed = base.copy()
        completed[chosen[:, 0], chosen[:, 1]] = 1
        output[depth] = completed

    return dict(sorted(output.items()))


def make_v2a_surrogate_analyzer(
    run_metadata: dict[str, Any],
    *,
    estimator_cls: type = GcStar,
) -> SurrogateAnalyzer:
    """Build a surrogate analyzer matching the observed run configuration."""

    params = run_metadata.get("gcstar_params", {})
    required = {
        "method",
        "n_perm",
        "n_lags",
        "alpha",
        "beta",
        "temporal",
        "simulation",
    }
    missing = sorted(required.difference(params))
    if missing:
        raise ValueError(f"run_metadata.gcstar_params is missing: {missing}")

    base_analyzer = make_gcstar_analyzer(
        str(params["method"]),
        alpha=float(params["alpha"]),
        beta=float(params["beta"]),
        n_perm=int(params["n_perm"]),
        n_lags=int(params["n_lags"]),
        temporal=bool(params["temporal"]),
        verbose=int(params.get("verbose", 0)),
        simulation=bool(params["simulation"]),
        estimator_cls=estimator_cls,
    )
    completion = run_metadata.get("inferred_completion")
    generated_depths = (
        {int(depth) for depth in completion.get("generated_depths", {})}
        if completion
        else set()
    )

    def analyze(
        X_surrogate: np.ndarray,
        p_values: list[int],
        replicate_seed: int,
    ) -> dict[int, np.ndarray]:
        learner_depths = [
            int(depth) for depth in p_values if int(depth) not in generated_depths
        ]
        inferred: dict[int, np.ndarray] = {}
        for index, depth in enumerate(learner_depths, start=1):
            logger.info(
                "Inferring surrogate depth %d/%d (P=%d)",
                index,
                len(learner_depths),
                depth,
            )
            inferred.update(base_analyzer(X_surrogate, [depth]))
        if completion:
            inferred = complete_inferred_depths(
                inferred,
                completion,
                seed=replicate_seed,
            )
        return inferred

    return analyze


def _calibration_config_hash(
    X: np.ndarray,
    observed_adjacencies: dict[int, np.ndarray],
    *,
    p_values: list[int],
    p0: int,
    block_length: int,
    seed: int,
    config_metadata: dict[str, Any],
) -> str:
    payload = {
        "trace_sha256": _array_sha256(X),
        "observed_sha256": _adjacency_grid_sha256(observed_adjacencies),
        "p_values": p_values,
        "p0": int(p0),
        "block_length": int(block_length),
        "seed": int(seed),
        "null_model": "MovingBlockBootstrapNull",
        "config_metadata": config_metadata,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_replicate_checkpoint(
    checkpoint_dir: Path,
    *,
    index: int,
    config_hash: str,
    p_values: list[int],
    p0: int,
    n_nodes: int,
) -> dict[str, Any] | None:
    json_path = checkpoint_dir / f"replicate_{index:04d}.json"
    pickle_path = checkpoint_dir / f"replicate_{index:04d}.pkl"
    if not json_path.exists() or not pickle_path.exists():
        return None

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if payload.get("config_hash") != config_hash:
            return None
        adjacencies = validate_adjacency_grid(
            load_adjacency_checkpoint(pickle_path),
            p_values=p_values,
            n_nodes=n_nodes,
            label=f"checkpoint replicate {index}",
        )
        if payload.get("adjacency_sha256") != _adjacency_grid_sha256(adjacencies):
            return None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None

    metrics = compute_graph_stability_metrics(adjacencies)
    d_values = {
        int(depth): float(value)
        for depth, value in metrics["D_p"].items()
        if int(depth) > int(p0)
    }
    payload["T"] = float(max(d_values.values()) if d_values else 0.0)
    payload["D_p"] = d_values
    payload["D_parts"] = {
        int(depth): {
            "D_minus": float(parts["D_minus"]),
            "D_plus": float(parts["D_plus"]),
        }
        for depth, parts in metrics["D_parts"].items()
        if int(depth) > int(p0)
    }
    payload["edge_counts"] = {
        int(depth): int(value) for depth, value in metrics["edge_counts"].items()
    }
    return payload


def _save_replicate_checkpoint(
    checkpoint_dir: Path,
    *,
    index: int,
    adjacencies: dict[int, np.ndarray],
    summary: dict[str, Any],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = checkpoint_dir / f"replicate_{index:04d}.pkl"
    json_path = checkpoint_dir / f"replicate_{index:04d}.json"
    save_adjacency_checkpoint(pickle_path, adjacencies)
    payload = {
        **summary,
        "adjacency_sha256": _adjacency_grid_sha256(adjacencies),
        "connectivity_pickle": pickle_path.name,
    }
    _atomic_write_json(json_path, payload)


def run_resumable_v2a_calibration(
    X: np.ndarray,
    *,
    observed_adjacencies: dict[int, np.ndarray],
    analyze_surrogate: SurrogateAnalyzer,
    p_values: Sequence[int],
    p0: int,
    B: int,
    block_length: int,
    seed: int,
    checkpoint_dir: Path,
    n_jobs: int = 1,
    config_metadata: dict[str, Any] | None = None,
) -> V2ACalibrationOutcome:
    """Calibrate one observed pickle grid using checkpointed surrogates."""

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must have shape (time, variables), got {X.shape}")
    if B < 1:
        raise ValueError("B must be >= 1")
    if block_length < 1:
        raise ValueError("block_length must be >= 1")
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be zero")

    depths = _normalize_p_values(p_values)
    if p0 not in depths:
        raise ValueError("p0 must be one of p_values")
    observed = validate_adjacency_grid(
        observed_adjacencies,
        p_values=depths,
        n_nodes=X.shape[1],
        label="observed connectivity",
    )
    observed_metrics = compute_graph_stability_metrics(observed)
    observed_d = {
        int(depth): float(value)
        for depth, value in observed_metrics["D_p"].items()
        if int(depth) > int(p0)
    }
    observed_t = max(observed_d.values()) if observed_d else 0.0
    config_metadata = config_metadata or {}
    config_hash = _calibration_config_hash(
        X,
        observed,
        p_values=depths,
        p0=p0,
        block_length=block_length,
        seed=seed,
        config_metadata=config_metadata,
    )

    checkpoint_dir = Path(checkpoint_dir)
    null_model = MovingBlockBootstrapNull(
        p0=p0,
        block_length=block_length,
    ).fit(X, p0=p0)

    def run_replicate(index: int) -> tuple[dict[str, Any], bool]:
        cached = _load_replicate_checkpoint(
            checkpoint_dir,
            index=index,
            config_hash=config_hash,
            p_values=depths,
            p0=p0,
            n_nodes=X.shape[1],
        )
        if cached is not None:
            logger.info(
                "Reusing bootstrap checkpoint %d/%d from %s",
                index + 1,
                B,
                checkpoint_dir,
            )
            return cached, True

        replicate_seed = stable_seed(seed, "bootstrap", index)
        logger.info(
            "Running bootstrap replicate %d/%d (seed=%d)",
            index + 1,
            B,
            replicate_seed,
        )
        X_surrogate = null_model.sample(T=X.shape[0], seed=replicate_seed)
        inferred = validate_adjacency_grid(
            analyze_surrogate(X_surrogate, depths, replicate_seed),
            p_values=depths,
            n_nodes=X.shape[1],
            label=f"surrogate replicate {index}",
        )
        metrics = compute_graph_stability_metrics(inferred)
        d_values = {
            int(depth): float(value)
            for depth, value in metrics["D_p"].items()
            if int(depth) > int(p0)
        }
        summary: dict[str, Any] = {
            "replicate": int(index),
            "seed": int(replicate_seed),
            "config_hash": config_hash,
            "T": float(max(d_values.values()) if d_values else 0.0),
            "D_p": d_values,
            "D_parts": {
                int(depth): {
                    "D_minus": float(parts["D_minus"]),
                    "D_plus": float(parts["D_plus"]),
                }
                for depth, parts in metrics["D_parts"].items()
                if int(depth) > int(p0)
            },
            "edge_counts": {
                int(depth): int(value)
                for depth, value in metrics["edge_counts"].items()
            },
        }
        _save_replicate_checkpoint(
            checkpoint_dir,
            index=index,
            adjacencies=inferred,
            summary=summary,
        )
        logger.info(
            "Saved bootstrap replicate %d/%d (T=%.6f)",
            index + 1,
            B,
            summary["T"],
        )
        return summary, False

    if n_jobs == 1:
        completed = [run_replicate(index) for index in range(B)]
    else:
        completed = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(run_replicate)(index) for index in range(B)
        )

    summaries = [summary for summary, _ in completed]
    reused_replicates = sum(int(reused) for _, reused in completed)
    T_boot = np.asarray([summary["T"] for summary in summaries], dtype=float)
    transition_depths = sorted(observed_d)
    pointwise: dict[int, dict[str, float]] = {}
    for depth in transition_depths:
        samples = np.asarray(
            [summary["D_p"].get(depth, 0.0) for summary in summaries],
            dtype=float,
        )
        pointwise[depth] = {
            "lower": float(np.quantile(samples, 0.025)),
            "upper": float(np.quantile(samples, 0.975)),
            "critical_90": float(np.quantile(samples, 0.90)),
            "critical_95": float(np.quantile(samples, 0.95)),
            "critical_99": float(np.quantile(samples, 0.99)),
        }

    critical_90 = float(np.quantile(T_boot, 0.90))
    critical_95 = float(np.quantile(T_boot, 0.95))
    critical_99 = float(np.quantile(T_boot, 0.99))
    p_value = float((1 + np.sum(T_boot >= observed_t)) / (B + 1))
    first_exceedance = next(
        (
            depth
            for depth in transition_depths
            if observed_d[depth] > pointwise[depth]["critical_95"]
        ),
        None,
    )
    result = CalibrationResult(
        observed={
            "T_obs": float(observed_t),
            "D_obs": observed_d,
            "D_parts_obs": {
                int(depth): {
                    "D_minus": float(parts["D_minus"]),
                    "D_plus": float(parts["D_plus"]),
                }
                for depth, parts in observed_metrics["D_parts"].items()
                if int(depth) > int(p0)
            },
            "edge_counts": {
                int(depth): int(value)
                for depth, value in observed_metrics["edge_counts"].items()
            },
            "p_values": depths,
        },
        null={
            "T_boot": T_boot.tolist(),
            "D_boot": [summary["D_p"] for summary in summaries],
            "edge_counts_boot": {
                depth: [int(summary["edge_counts"][depth]) for summary in summaries]
                for depth in depths
            },
            "pointwise": pointwise,
            "critical_90": critical_90,
            "critical_95": critical_95,
            "critical_99": critical_99,
            "p_value": p_value,
            "B": int(B),
            "block_length": int(block_length),
            "null_model": null_model.metadata,
        },
        diagnosis={
            "reject_global_95": bool(observed_t > critical_95),
            "first_exceedance_depth": first_exceedance,
        },
    )
    return V2ACalibrationOutcome(
        result=result,
        replicate_summaries=summaries,
        reused_replicates=reused_replicates,
        config_hash=config_hash,
    )


def calibration_payload(
    outcome: V2ACalibrationOutcome,
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return a JSON-serializable per-run calibration payload."""

    return {
        "observed": outcome.result.observed,
        "null": outcome.result.null,
        "diagnosis": outcome.result.diagnosis,
        "null_model": "moving-block",
        "metadata": {
            **metadata,
            "config_hash": outcome.config_hash,
            "reused_replicates": outcome.reused_replicates,
        },
    }


def write_calibration_payload(
    path: Path,
    outcome: V2ACalibrationOutcome,
    *,
    metadata: dict[str, Any],
) -> Path:
    """Atomically write one per-recording bootstrap result."""

    path = Path(path)
    _atomic_write_json(path, calibration_payload(outcome, metadata=metadata))
    return path


def plot_v2a_calibration_grid(
    results: dict[str, dict[str, dict[str, Any]]],
    output_path: Path,
) -> Path:
    """Plot any number of recording-method null distributions dynamically."""

    panels = [
        (method, recording, payload)
        for method, method_results in results.items()
        for recording, payload in method_results.items()
    ]
    if not panels:
        raise ValueError("results cannot be empty")

    n_columns = 2
    n_rows = math.ceil(len(panels) / n_columns)
    plt = require_matplotlib()
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(12, 4 * n_rows),
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for axis, (method, recording, payload) in zip(flat_axes, panels, strict=False):
        observed = payload["observed"]
        null = payload["null"]
        T_boot = np.asarray(null["T_boot"], dtype=float)
        axis.hist(
            T_boot,
            bins=min(20, max(5, int(np.sqrt(len(T_boot))))),
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
        )
        axis.axvline(
            float(observed["T_obs"]),
            color="red",
            linestyle="--",
            linewidth=2,
            label="observed",
        )
        axis.axvline(
            float(null["critical_95"]),
            color="orange",
            linestyle=":",
            linewidth=2,
            label="95% critical",
        )
        axis.set_title(f"{recording} — {method}")
        axis.set_xlabel("global instability statistic")
        axis.set_ylabel("count")
        axis.legend(fontsize=8)

    for axis in flat_axes[len(panels) :]:
        axis.axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_v2a_pointwise_grid(
    results: dict[str, dict[str, dict[str, Any]]],
    output_path: Path,
) -> Path:
    """Plot observed depth instability against pointwise null bands."""

    panels = [
        (method, recording, payload)
        for method, method_results in results.items()
        for recording, payload in method_results.items()
    ]
    if not panels:
        raise ValueError("results cannot be empty")

    n_columns = 2
    n_rows = math.ceil(len(panels) / n_columns)
    plt = require_matplotlib()
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(12, 4 * n_rows),
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for axis, (method, recording, payload) in zip(flat_axes, panels, strict=False):
        observed = payload["observed"]
        pointwise = payload["null"]["pointwise"]
        observed_d = {
            int(depth): float(value) for depth, value in observed["D_obs"].items()
        }
        normalized_bands = {
            int(depth): {key: float(value) for key, value in values.items()}
            for depth, values in pointwise.items()
        }
        depths = sorted(set(observed_d).intersection(normalized_bands))
        lower = [normalized_bands[depth]["lower"] for depth in depths]
        upper = [normalized_bands[depth]["upper"] for depth in depths]
        axis.fill_between(
            depths,
            lower,
            upper,
            alpha=0.25,
            color="steelblue",
            label="pointwise 95% null band",
        )
        axis.plot(
            depths,
            [observed_d[depth] for depth in depths],
            color="black",
            marker="o",
            label="observed",
        )
        axis.set_title(f"{recording} — {method}")
        axis.set_xlabel("conditioning depth")
        axis.set_ylabel("graph instability")
        axis.set_xticks(depths)
        axis.legend(fontsize=8)

    for axis in flat_axes[len(panels) :]:
        axis.axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


__all__ = [
    "SurrogateAnalyzer",
    "V2ACalibrationInput",
    "V2ACalibrationOutcome",
    "stable_seed",
    "validate_adjacency_grid",
    "discover_complete_recordings",
    "load_v2a_calibration_input",
    "complete_inferred_depths",
    "make_v2a_surrogate_analyzer",
    "run_resumable_v2a_calibration",
    "calibration_payload",
    "write_calibration_payload",
    "plot_v2a_calibration_grid",
    "plot_v2a_pointwise_grid",
]
