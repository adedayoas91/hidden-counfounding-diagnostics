"""Aggregate method-sharded sample-size power outputs."""

from __future__ import annotations

import json
import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def read_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def plot_power_outputs(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_paths: list[Path] = []

    figure, axes = plt.subplots(1, 3, figsize=(16, 4))
    for axis, d in zip(axes, sorted(summary["d"].unique()), strict=False):
        subset = summary[summary["d"] == d]
        for label in METHOD_LABELS:
            method_subset = subset[subset["method_label"] == label]
            axis.plot(
                method_subset["T"],
                method_subset["mean_tpr"],
                marker="o",
                linewidth=2,
                label=label,
            )
        axis.set_xlabel("Sample Size T")
        axis.set_ylabel("Mean TPR")
        axis.set_title(f"d={d}")
        axis.set_ylim([0, 1])
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    path = output_dir / "tpr_by_sample_size.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    output_paths.append(path)

    for metric, ylabel, filename, ylim in [
        ("mean_tpr", "Mean TPR", "tpr_by_dimension.png", [0, 1]),
        ("mean_fpr", "Mean FPR", "fpr_by_dimension.png", [0, 1]),
        ("mean_p_star", "Mean p_star", "pstar_by_dimension.png", None),
        ("mean_runtime", "Mean Runtime (seconds)", "runtime_scaling.png", None),
    ]:
        figure, axis = plt.subplots(figsize=(10, 6))
        for label in METHOD_LABELS:
            for T in sorted(summary["T"].unique()):
                subset = summary[
                    (summary["method_label"] == label) & (summary["T"] == T)
                ]
                axis.plot(
                    subset["d"],
                    subset[metric],
                    marker="o",
                    linewidth=2,
                    label=f"{label}, T={T}",
                )
        axis.set_xlabel("Dimension d")
        axis.set_ylabel(ylabel)
        axis.set_title(ylabel + " vs Dimension")
        if ylim is not None:
            axis.set_ylim(ylim)
        if metric == "mean_runtime":
            axis.set_yscale("log")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(path)

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    project_root = find_project_root()
    output_dir = project_root / "outputs" / "power" / "sample_size_power"
    shard_root = output_dir / "by_method"

    summaries: list[pd.DataFrame] = []
    results: list[dict[str, Any]] = []
    aggregated: list[dict[str, Any]] = []
    input_paths: list[str] = []

    for label in METHOD_LABELS:
        shard_dir = shard_root / label
        summary_path = shard_dir / "summary.csv"
        results_path = shard_dir / "results.json"
        aggregated_path = shard_dir / "aggregated.json"
        for path in (summary_path, results_path, aggregated_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing sample-size power shard file: {path}")
            input_paths.append(str(path))
        summaries.append(pd.read_csv(summary_path))
        results.extend(read_json_list(results_path))
        aggregated.extend(read_json_list(aggregated_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.concat(summaries, ignore_index=True)
    summary = summary.sort_values(["method_label", "T", "d"])

    summary_path = output_dir / "summary.csv"
    results_path = output_dir / "results.json"
    aggregated_path = output_dir / "aggregated.json"
    manifest_path = output_dir / "manifest.json"
    summary.to_csv(summary_path, index=False)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    aggregated_path.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
    plot_paths = plot_power_outputs(summary, output_dir)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(project_root),
        "analysis": "sample_size_power",
        "sharded_by": "method",
        "methods": METHOD_LABELS,
        "input_paths": input_paths,
        "output_paths": [
            str(summary_path),
            str(results_path),
            str(aggregated_path),
            *(str(path) for path in plot_paths),
            str(manifest_path),
        ],
        "grid": {
            "T_values": sorted(int(value) for value in summary["T"].unique()),
            "d_values": sorted(int(value) for value in summary["d"].unique()),
            "methods": METHOD_LABELS,
            "grid_points": int(len(summary)),
            "total_runs": int(len(results)),
        },
        "software_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {summary_path}")
    print(f"Wrote {results_path}")
    print(f"Wrote {aggregated_path}")
    for path in plot_paths:
        print(f"Wrote {path}")
    print(f"Wrote {manifest_path}")
    print(summary.groupby("method_label")[["mean_tpr", "mean_fpr", "mean_runtime"]].mean().to_string())


if __name__ == "__main__":
    main()
