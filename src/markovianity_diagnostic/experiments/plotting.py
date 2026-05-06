"""Plotting helpers for experiment outputs."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from statistics import mean


def require_matplotlib():
    """Import matplotlib with a local writable config cache."""

    cache_dir = Path(__file__).resolve().parent / ".mpl_cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    try:
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
