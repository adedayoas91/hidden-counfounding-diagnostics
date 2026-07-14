"""Run runtime-scaling measurements for one method shard.

This script is called by the method-specific runtime notebooks. It writes
method-isolated artifacts so c-GC and c-GC-star can run asynchronously without
clobbering the canonical combined outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_CONFIG = {
    "c-GC": {"key": "cgc", "plot_label": "c-GC"},
    "c-GC-star": {"key": "fcgc", "plot_label": "c-GC-star"},
}

T_VALUES = [500, 1000, 2000, 5000]
D_VALUES = [5, 10, 20, 50]
P_GRID_MAX_VALUES = [5, 10]
N_JOBS_VALUES = [1, 4]
SEED = 42


def find_project_root() -> Path:
    """Return the hidden-confounding-diagnostics repository root."""

    current = Path.cwd().resolve()
    for path in (current, *current.parents):
        if (path / "src" / "markovianity_diagnostic").exists():
            return path
    raise FileNotFoundError("Could not find hidden-confounding-diagnostics root")


def stable_seed(*parts: object) -> int:
    """Derive a stable uint32 seed from configuration parts."""

    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def get_git_commit(project_root: Path) -> str:
    """Return the current short git commit when available."""

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


def measure_runtime(
    *,
    T: int,
    d: int,
    method_key: str,
    p_grid_max: int,
    n_jobs: int,
    seed: int,
) -> dict[str, Any]:
    """Measure runtime for one grid configuration."""

    from markovianity_diagnostic.experiments.adapters import (
        analyze_with_gcstar_cgc,
        analyze_with_gcstar_fcgc,
    )
    from markovianity_diagnostic.experiments.simulations import (
        scenario_order1_unconfounded,
    )

    analyzers = {
        "cgc": analyze_with_gcstar_cgc,
        "fcgc": analyze_with_gcstar_fcgc,
    }
    try:
        scenario_result = scenario_order1_unconfounded(
            T=T,
            d=d,
            seed=stable_seed(seed, T, d, method_key, p_grid_max, n_jobs),
        )
        p_values = list(range(1, min(p_grid_max + 1, T // 10)))
        if not p_values:
            p_values = [1]

        started = time.perf_counter()
        analyzers[method_key](scenario_result.X, p_values)
        elapsed = time.perf_counter() - started
        return {"runtime_seconds": float(elapsed), "error": None}
    except Exception as error:
        return {"runtime_seconds": None, "error": str(error)}


def plot_runtime_grid(runtime_df: pd.DataFrame, output_path: Path) -> Path:
    """Write a runtime diagnostic plot for one method shard."""

    method_label = str(runtime_df["method_label"].iloc[0])
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    axis = axes[0, 0]
    grouped = runtime_df.groupby("T")["runtime_seconds"].mean()
    axis.plot(grouped.index, grouped.values, marker="o", linewidth=2)
    axis.set_xlabel("Time Series Length (T)")
    axis.set_ylabel("Runtime (seconds)")
    axis.set_title(f"{method_label}: runtime vs T")
    axis.grid(True, alpha=0.3)

    axis = axes[0, 1]
    grouped = runtime_df.groupby("d")["runtime_seconds"].mean()
    axis.plot(grouped.index, grouped.values, marker="s", linewidth=2)
    axis.set_xlabel("Dimensionality (d)")
    axis.set_ylabel("Runtime (seconds)")
    axis.set_title(f"{method_label}: runtime vs d")
    axis.grid(True, alpha=0.3)

    axis = axes[1, 0]
    for n_jobs in sorted(runtime_df["n_jobs"].unique()):
        subset = runtime_df[runtime_df["n_jobs"] == n_jobs]
        grouped = subset.groupby("T")["runtime_seconds"].mean()
        axis.plot(
            grouped.index,
            grouped.values,
            marker="o",
            linestyle="--" if int(n_jobs) != 1 else "-",
            linewidth=2,
            label=f"n_jobs={n_jobs}",
        )
    axis.set_xlabel("Time Series Length (T)")
    axis.set_ylabel("Runtime (seconds)")
    axis.set_title(f"{method_label}: n_jobs effect")
    axis.legend()
    axis.grid(True, alpha=0.3)

    axis = axes[1, 1]
    speedups: list[float] = []
    t_values: list[int] = []
    for T in sorted(runtime_df["T"].unique()):
        subset = runtime_df[runtime_df["T"] == T]
        t1 = subset[subset["n_jobs"] == 1]["runtime_seconds"].mean()
        t4 = subset[subset["n_jobs"] == 4]["runtime_seconds"].mean()
        if t1 > 0 and t4 > 0:
            speedups.append(float(t1 / t4))
            t_values.append(int(T))
    if speedups:
        axis.plot(t_values, speedups, marker="D", linewidth=2)
    axis.axhline(y=1.0, color="gray", linestyle=":", linewidth=1)
    axis.axhline(y=4.0, color="gray", linestyle=":", linewidth=1)
    axis.set_xlabel("Time Series Length (T)")
    axis.set_ylabel("Speedup")
    axis.set_title(f"{method_label}: n_jobs 1 to 4")
    axis.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def run_method(method_label: str, *, force: bool = False) -> None:
    """Run or load the requested method shard."""

    project_root = find_project_root()
    sys.path.insert(0, str(project_root / "src"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    method = METHOD_CONFIG[method_label]
    method_key = method["key"]
    output_dir = project_root / "outputs" / "power" / "runtime_scaling" / "by_method" / method_label
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_path = output_dir / "runtime_grid.csv"
    plot_path = output_dir / "runtime_plot.png"
    manifest_path = output_dir / "manifest.json"
    outputs_exist = runtime_path.exists() and plot_path.exists() and manifest_path.exists()

    if outputs_exist and not force:
        runtime_df = pd.read_csv(runtime_path)
        print(f"Loaded cached {method_label} runtime shard: {runtime_path}")
    else:
        measurements: list[dict[str, Any]] = []
        total = len(T_VALUES) * len(D_VALUES) * len(P_GRID_MAX_VALUES) * len(N_JOBS_VALUES)
        completed = 0
        started = time.time()
        print(f"Running runtime shard for {method_label}: {total} grid points")

        for T in T_VALUES:
            for d in D_VALUES:
                for p_grid_max in P_GRID_MAX_VALUES:
                    for n_jobs in N_JOBS_VALUES:
                        completed += 1
                        print(
                            f"[{completed}/{total}] T={T}, d={d}, "
                            f"p_max={p_grid_max}, n_jobs={n_jobs}",
                            end=" ",
                            flush=True,
                        )
                        result = measure_runtime(
                            T=T,
                            d=d,
                            method_key=method_key,
                            p_grid_max=p_grid_max,
                            n_jobs=n_jobs,
                            seed=SEED,
                        )
                        if result["error"] is None:
                            runtime = float(result["runtime_seconds"])
                            print(f"-> {runtime:.3f}s")
                            measurements.append(
                                {
                                    "T": int(T),
                                    "d": int(d),
                                    "method": method_key,
                                    "method_label": method_label,
                                    "p_grid_max": int(p_grid_max),
                                    "n_jobs": int(n_jobs),
                                    "runtime_seconds": runtime,
                                }
                            )
                        else:
                            print(f"error: {result['error']}")
                            logger.warning("Runtime measurement failed: %s", result["error"])

        runtime_df = pd.DataFrame(measurements)
        runtime_df.to_csv(runtime_path, index=False)
        logger.info("Wrote %s", runtime_path)
        print(f"Wrote {runtime_path}")

        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit(project_root),
            "analysis": "runtime_scaling_method_shard",
            "method": method_key,
            "method_label": method_label,
            "measurement_grid": {
                "T_values": T_VALUES,
                "d_values": D_VALUES,
                "p_grid_max_values": P_GRID_MAX_VALUES,
                "n_jobs_values": N_JOBS_VALUES,
                "total_combinations": total,
                "successful_measurements": int(len(runtime_df)),
            },
            "random_seed": SEED,
            "scenario": "order1_unconfounded",
            "elapsed_seconds": float(time.time() - started),
            "output_paths": [str(runtime_path), str(plot_path)],
            "software_versions": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "matplotlib": plt.matplotlib.__version__,
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote {manifest_path}")

    if runtime_df.empty:
        raise RuntimeError(f"No runtime measurements available for {method_label}")
    plot_runtime_grid(runtime_df, plot_path)
    print(f"Wrote {plot_path}")
    print(runtime_df.describe(include="all").to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHOD_CONFIG), required=True)
    parser.add_argument("--force", action="store_true", help="recompute even if shard outputs exist")
    args = parser.parse_args()
    run_method(args.method, force=args.force)


if __name__ == "__main__":
    main()
