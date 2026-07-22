"""Plotting helpers for experiment outputs."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from statistics import mean

import numpy as np


def require_matplotlib():
    """Import matplotlib with a local writable config cache."""

    cache_dir = Path(__file__).resolve().parent / ".mpl_cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "matplotlib is required for plotting. Install the 'plots' extra or inspect JSON/CSV outputs."
        ) from exc
    return plt


def read_summary(path: Path) -> list[dict[str, str]]:
    """Read a CSV summary into a list of dictionaries."""

    with path.open() as handle:
        return list(csv.DictReader(handle))


def mean_by_p(rows: list[dict[str, str]], prefix: str) -> tuple[list[int], list[float]]:
    """Aggregate fields with a shared prefix over conditioning depth."""

    fields = [name for name in rows[0] if name.startswith(prefix)]
    pairs = []
    for field in fields:
        p_value = int(field.replace(prefix, ""))
        values = [float(row[field]) for row in rows if row[field] not in ("", "None")]
        pairs.append((p_value, mean(values) if values else 0.0))
    pairs.sort()
    return [p_value for p_value, _ in pairs], [value for _, value in pairs]


def plot_summary(summary_csv: Path, output_dir: Path) -> None:
    """Plot mean instability and edge counts from a run summary."""

    plt = require_matplotlib()
    rows = read_summary(summary_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    p_d, d_values = mean_by_p(rows, "D_p")
    if p_d:
        plt.figure(figsize=(6, 4))
        plt.plot(p_d, d_values, marker="o")
        plt.xlabel("conditioning depth p")
        plt.ylabel("mean graph instability D_p")
        plt.title("Graph Instability Across Conditioning Depth")
        plt.tight_layout()
        plt.savefig(output_dir / "mean_D_p.png", dpi=200)
        plt.close()

    p_e, edge_values = mean_by_p(rows, "edge_count_p")
    if p_e:
        plt.figure(figsize=(6, 4))
        plt.plot(p_e, edge_values, marker="o")
        plt.xlabel("conditioning depth p")
        plt.ylabel("mean inferred edge count")
        plt.title("Inferred Edge Count Across Conditioning Depth")
        plt.tight_layout()
        plt.savefig(output_dir / "mean_edge_count.png", dpi=200)
        plt.close()


def plot_bootstrap(bootstrap_json: Path, output_dir: Path) -> None:
    """Plot the bootstrap null distribution for ``T_obs``."""

    plt = require_matplotlib()
    data = json.loads(bootstrap_json.read_text())
    result = data["bootstrap"]
    output_dir.mkdir(parents=True, exist_ok=True)

    T_boot = result["T_boot"]
    T_obs = result["T_obs"]
    critical = result["critical_95"]
    plt.figure(figsize=(6, 4))
    plt.hist(T_boot, bins=25, alpha=0.8)
    plt.axvline(T_obs, color="red", linestyle="--", label="T_obs")
    plt.axvline(critical, color="black", linestyle=":", label="95% critical")
    plt.xlabel("bootstrap T")
    plt.ylabel("count")
    plt.title("Bootstrap Null Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "bootstrap_T_obs.png", dpi=200)
    plt.close()


def plot_T_boot_histogram(
    T_boot: list[float] | np.ndarray,
    T_obs: float,
    critical_values: dict[float | str, float],
    output_path: str | Path,
) -> Path:
    """Write a calibrated global-statistic histogram."""
    values = np.asarray(T_boot, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("T_boot must be a non-empty finite one-dimensional array")

    plt = require_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _, axis = plt.subplots(figsize=(6, 4))
    axis.hist(values, bins=min(30, max(5, int(np.sqrt(values.size)))), alpha=0.8)
    axis.axvline(float(T_obs), color="red", linestyle="--", label="observed")
    for level, critical in sorted(critical_values.items(), key=lambda item: str(item[0])):
        axis.axvline(float(critical), linestyle=":", label=f"{level} critical")
    axis.set(xlabel="global instability statistic", ylabel="count")
    axis.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_D_p_with_bands(
    D_p: dict[int, float],
    D_boot_pointwise: dict[int, dict[str, float]],
    p_values: list[int],
    output_path: str | Path,
) -> Path:
    """Write the observed depth curve with pointwise bootstrap bands."""
    depths = [int(p) for p in p_values if p in D_p and p in D_boot_pointwise]
    if not depths:
        raise ValueError("No shared depths across D_p, bands, and p_values")

    observed = np.asarray([D_p[p] for p in depths], dtype=float)
    lower = np.asarray([D_boot_pointwise[p]["lower"] for p in depths], dtype=float)
    upper = np.asarray([D_boot_pointwise[p]["upper"] for p in depths], dtype=float)
    if np.any(lower > upper):
        raise ValueError("Pointwise lower bands cannot exceed upper bands")

    plt = require_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _, axis = plt.subplots(figsize=(6, 4))
    axis.fill_between(depths, lower, upper, alpha=0.25, label="null band")
    axis.plot(depths, observed, marker="o", label="observed")
    axis.set(xlabel="conditioning depth", ylabel="graph instability")
    axis.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_edge_count_trajectory(
    edge_counts: dict[int, float],
    edge_counts_boot: dict[int, list[float] | np.ndarray],
    output_path: str | Path,
) -> Path:
    """Write observed edge counts with bootstrap 95% intervals."""
    depths = sorted(set(edge_counts).intersection(edge_counts_boot))
    if not depths:
        raise ValueError("No shared edge-count depths")
    samples = [np.asarray(edge_counts_boot[p], dtype=float) for p in depths]
    if any(values.size == 0 for values in samples):
        raise ValueError("Bootstrap edge-count samples cannot be empty")

    observed = [edge_counts[p] for p in depths]
    lower = [float(np.quantile(values, 0.025)) for values in samples]
    upper = [float(np.quantile(values, 0.975)) for values in samples]

    plt = require_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _, axis = plt.subplots(figsize=(6, 4))
    axis.fill_between(depths, lower, upper, alpha=0.25, label="bootstrap 95%")
    axis.plot(depths, observed, marker="o", label="observed")
    axis.set(xlabel="conditioning depth", ylabel="edge count")
    axis.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path
