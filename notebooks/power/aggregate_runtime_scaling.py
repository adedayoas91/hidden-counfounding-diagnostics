"""Aggregate method-sharded runtime-scaling outputs."""

from __future__ import annotations

import json
import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = ["c-GC", "c-GC-star"]


def find_project_root() -> Path:
    current = Path.cwd().resolve()
    for path in (current, *current.parents):
        if (path / "src" / "markovianity_diagnostic").exists():
            return path
    raise FileNotFoundError("Could not find hidden-confounding-diagnostics root")


def get_git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            check=False,
            cwd=project_root,
            text=True,
            timeout=5,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def plot_runtime_grid(runtime_df: pd.DataFrame, output_path: Path) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    axis = axes[0, 0]
    for label in METHOD_LABELS:
        subset = runtime_df[runtime_df["method_label"] == label]
        grouped = subset.groupby("T")["runtime_seconds"].mean()
        axis.plot(grouped.index, grouped.values, marker="o", label=label, linewidth=2)
    axis.set_xlabel("Time Series Length (T)")
    axis.set_ylabel("Runtime (seconds)")
    axis.set_title("Runtime Scaling with Time Series Length")
    axis.legend()
    axis.grid(True, alpha=0.3)

    axis = axes[0, 1]
    for label in METHOD_LABELS:
        subset = runtime_df[runtime_df["method_label"] == label]
        grouped = subset.groupby("d")["runtime_seconds"].mean()
        axis.plot(grouped.index, grouped.values, marker="s", label=label, linewidth=2)
    axis.set_xlabel("Dimensionality (d)")
    axis.set_ylabel("Runtime (seconds)")
    axis.set_title("Runtime Scaling with Dimensionality")
    axis.legend()
    axis.grid(True, alpha=0.3)

    axis = axes[1, 0]
    for label in METHOD_LABELS:
        for n_jobs in sorted(runtime_df["n_jobs"].unique()):
            subset = runtime_df[
                (runtime_df["method_label"] == label)
                & (runtime_df["n_jobs"] == n_jobs)
            ]
            grouped = subset.groupby("T")["runtime_seconds"].mean()
            axis.plot(
                grouped.index,
                grouped.values,
                marker="o",
                linestyle="--" if int(n_jobs) != 1 else "-",
                linewidth=2,
                alpha=0.75,
                label=f"{label} (n_jobs={n_jobs})",
            )
    axis.set_xlabel("Time Series Length (T)")
    axis.set_ylabel("Runtime (seconds)")
    axis.set_title("Parallelization Effect")
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.3)

    axis = axes[1, 1]
    for label in METHOD_LABELS:
        speedups: list[float] = []
        t_values: list[int] = []
        for T in sorted(runtime_df["T"].unique()):
            subset = runtime_df[
                (runtime_df["method_label"] == label) & (runtime_df["T"] == T)
            ]
            t1 = subset[subset["n_jobs"] == 1]["runtime_seconds"].mean()
            t4 = subset[subset["n_jobs"] == 4]["runtime_seconds"].mean()
            if t1 > 0 and t4 > 0:
                speedups.append(float(t1 / t4))
                t_values.append(int(T))
        if speedups:
            axis.plot(t_values, speedups, marker="D", label=label, linewidth=2)
    axis.axhline(y=1.0, color="gray", linestyle=":", linewidth=1)
    axis.axhline(y=4.0, color="gray", linestyle=":", linewidth=1)
    axis.set_xlabel("Time Series Length (T)")
    axis.set_ylabel("Speedup")
    axis.set_title("Parallelization Speedup")
    axis.legend()
    axis.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    project_root = find_project_root()
    output_dir = project_root / "outputs" / "power" / "runtime_scaling"
    shard_root = output_dir / "by_method"

    frames: list[pd.DataFrame] = []
    shard_paths: list[str] = []
    for label in METHOD_LABELS:
        shard_path = shard_root / label / "runtime_grid.csv"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing runtime shard for {label}: {shard_path}")
        frame = pd.read_csv(shard_path)
        if frame.empty:
            raise ValueError(f"Runtime shard is empty: {shard_path}")
        frames.append(frame)
        shard_paths.append(str(shard_path))

    runtime_df = pd.concat(frames, ignore_index=True)
    runtime_df = runtime_df.sort_values(
        ["method_label", "T", "d", "p_grid_max", "n_jobs"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_path = output_dir / "runtime_grid.csv"
    plot_path = output_dir / "runtime_plot.png"
    manifest_path = output_dir / "manifest.json"
    runtime_df.to_csv(runtime_path, index=False)
    plot_runtime_grid(runtime_df, plot_path)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(project_root),
        "analysis": "runtime_scaling",
        "sharded_by": "method",
        "methods": METHOD_LABELS,
        "input_paths": shard_paths,
        "output_paths": [str(runtime_path), str(plot_path), str(manifest_path)],
        "measurement_grid": {
            "T_values": sorted(int(value) for value in runtime_df["T"].unique()),
            "d_values": sorted(int(value) for value in runtime_df["d"].unique()),
            "methods": METHOD_LABELS,
            "p_grid_max_values": sorted(
                int(value) for value in runtime_df["p_grid_max"].unique()
            ),
            "n_jobs_values": sorted(int(value) for value in runtime_df["n_jobs"].unique()),
            "successful_measurements": int(len(runtime_df)),
        },
        "scenario": "order1_unconfounded",
        "software_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
        "runtime_statistics": {
            "min_seconds": float(runtime_df["runtime_seconds"].min()),
            "max_seconds": float(runtime_df["runtime_seconds"].max()),
            "mean_seconds": float(runtime_df["runtime_seconds"].mean()),
            "median_seconds": float(runtime_df["runtime_seconds"].median()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {runtime_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {manifest_path}")
    print(runtime_df.groupby("method_label")["runtime_seconds"].describe().to_string())


if __name__ == "__main__":
    main()
