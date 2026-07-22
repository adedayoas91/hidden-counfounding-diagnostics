"""Run sample-size power analysis for one method shard."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METHOD_CONFIG = {
    "c-GC": {"key": "gcstar_cgc", "plot_label": "c-GC"},
    "c-GC-star": {"key": "gcstar_fcgc", "plot_label": "c-GC-star"},
}

T_VALUES = [500, 1000, 2000]
D_VALUES = [5, 10, 20]
P_VALUES = [1, 2, 3, 4, 5, 6]
REPEATS = 10
SEED = 42


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


def write_results(
    *,
    output_dir: Path,
    method_key: str,
    method_label: str,
    results: list[Any],
    aggregated: list[Any],
) -> None:
    """Write detailed and aggregated power outputs with method columns."""

    output_dir.mkdir(parents=True, exist_ok=True)

    result_rows = [
        {
            "method": method_key,
            "method_label": method_label,
            **asdict(result),
        }
        for result in results
    ]
    (output_dir / "results.json").write_text(
        json.dumps(result_rows, indent=2),
        encoding="utf-8",
    )

    aggregate_rows = [
        {
            "method": method_key,
            "method_label": method_label,
            **asdict(result),
        }
        for result in aggregated
    ]
    (output_dir / "aggregated.json").write_text(
        json.dumps(aggregate_rows, indent=2),
        encoding="utf-8",
    )

    fieldnames = [
        "method",
        "method_label",
        "T",
        "d",
        "edge_density",
        "confounder_strength",
        "latent_autocorr",
        "noise_scale",
        "n_repeats",
        "mean_tpr",
        "std_tpr",
        "mean_fpr",
        "std_fpr",
        "mean_p_star",
        "std_p_star",
        "mean_runtime",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)


def run_method(method_label: str, *, force: bool = False) -> None:
    project_root = find_project_root()
    sys.path.insert(0, str(project_root / "src"))

    from markovianity_diagnostic.experiments.power import PowerAnalyzer, PowerGrid

    method_key = METHOD_CONFIG[method_label]["key"]
    output_dir = (
        project_root
        / "outputs"
        / "power"
        / "sample_size_power"
        / "by_method"
        / method_label
    )
    summary_path = output_dir / "summary.csv"
    results_path = output_dir / "results.json"
    aggregated_path = output_dir / "aggregated.json"
    manifest_path = output_dir / "manifest.json"
    outputs_exist = (
        summary_path.exists()
        and results_path.exists()
        and aggregated_path.exists()
        and manifest_path.exists()
    )

    if outputs_exist and not force:
        summary = pd.read_csv(summary_path)
        print(f"Loaded cached {method_label} power shard: {summary_path}")
        print(summary.to_string(index=False))
        return

    grid = PowerGrid(T_values=T_VALUES, d_values=D_VALUES)
    analyzer = PowerAnalyzer(
        grid=grid,
        method=method_key,
        repeats=REPEATS,
        p_values=P_VALUES,
        verbose=True,
        seed=SEED,
    )

    print(
        f"Running sample-size power shard for {method_label}: "
        f"{grid.total_combinations()} grid points x {REPEATS} repeats"
    )
    started = datetime.now(timezone.utc)
    results = analyzer.run()
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    aggregated = analyzer.aggregate(results)
    write_results(
        output_dir=output_dir,
        method_key=method_key,
        method_label=method_label,
        results=results,
        aggregated=aggregated,
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(project_root),
        "analysis": "sample_size_power_method_shard",
        "method": method_key,
        "method_label": method_label,
        "grid": {
            "T_values": T_VALUES,
            "d_values": D_VALUES,
            "total_combinations": grid.total_combinations(),
            "repeats": REPEATS,
            "total_runs": len(results),
            "p_values": P_VALUES,
        },
        "random_seed": SEED,
        "elapsed_seconds": float(elapsed),
        "output_paths": [
            str(summary_path),
            str(results_path),
            str(aggregated_path),
            str(manifest_path),
        ],
        "software_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {summary_path}")
    print(f"Wrote {results_path}")
    print(f"Wrote {aggregated_path}")
    print(f"Wrote {manifest_path}")
    print(f"Completed {len(results)} runs in {elapsed:.1f}s")
    print(pd.read_csv(summary_path).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHOD_CONFIG), required=True)
    parser.add_argument("--force", action="store_true", help="recompute even if shard outputs exist")
    args = parser.parse_args()
    run_method(args.method, force=args.force)


if __name__ == "__main__":
    main()
