"""Aggregate method-sharded v2a edge-localization outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return BH-adjusted q-values in the original order."""

    if len(p_values) == 0:
        return np.array([])
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    ranks = np.arange(1, len(sorted_p) + 1)
    adjusted = sorted_p * len(sorted_p) / ranks
    for index in range(len(adjusted) - 2, -1, -1):
        adjusted[index] = min(adjusted[index], adjusted[index + 1])
    adjusted = np.minimum(adjusted, 1.0)
    q_values = np.empty_like(adjusted)
    q_values[order] = adjusted
    return q_values


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    project_root = find_project_root()
    sys.path.insert(0, str(project_root / "src"))
    from markovianity_diagnostic.experiments.v2a_rsn_utils import V2A_ANALYSIS_PROFILE

    output_dir = (
        project_root
        / "outputs"
        / "v2a-RSNs"
        / V2A_ANALYSIS_PROFILE
        / "edge_localization"
    )
    shard_root = output_dir / "by_method"

    frames: list[pd.DataFrame] = []
    input_paths: list[str] = []
    shard_manifests: dict[str, dict[str, Any]] = {}
    for method_label in METHOD_LABELS:
        shard_dir = shard_root / method_label
        edge_path = shard_dir / "edge_instability.csv"
        manifest_path = shard_dir / "manifest.json"
        if not edge_path.exists():
            raise FileNotFoundError(f"Missing edge-localization shard: {edge_path}")
        frame = pd.read_csv(edge_path)
        if frame.empty:
            raise ValueError(f"Edge-localization shard is empty: {edge_path}")
        frames.append(frame)
        input_paths.append(str(edge_path.relative_to(project_root)))
        shard_manifests[method_label] = read_manifest(manifest_path)
        if manifest_path.exists():
            input_paths.append(str(manifest_path.relative_to(project_root)))

    edge_df = pd.concat(frames, ignore_index=True)
    if "q_value" in edge_df.columns and "q_value_recording_method" not in edge_df.columns:
        edge_df["q_value_recording_method"] = edge_df["q_value"]

    edge_df["q_value_method"] = np.nan
    valid = edge_df["p_value"].notna() if "p_value" in edge_df.columns else pd.Series(False, index=edge_df.index)
    for method_label, method_index in edge_df.loc[valid].groupby("method").groups.items():
        p_values = edge_df.loc[method_index, "p_value"].to_numpy(dtype=float)
        edge_df.loc[method_index, "q_value_method"] = benjamini_hochberg(p_values)
    edge_df["q_value"] = edge_df["q_value_method"]
    edge_df.loc[edge_df["p_value"].notna(), "fdr_scope"] = "method"

    output_dir.mkdir(parents=True, exist_ok=True)
    edge_path = output_dir / "edge_instability.csv"
    fdr_path = output_dir / "edge_fdr.csv"
    manifest_path = output_dir / "manifest.json"
    edge_df.to_csv(edge_path, index=False)

    fdr_columns = [
        "recording",
        "method",
        "source",
        "target",
        "instability_frequency",
        "null_frequency",
        "p_value",
        "q_value",
        "q_value_recording_method",
        "q_value_method",
        "bootstrap_replicates",
        "fdr_scope",
    ]
    available_fdr_columns = [column for column in fdr_columns if column in edge_df.columns]
    edge_df[available_fdr_columns].dropna(subset=["p_value"]).to_csv(
        fdr_path,
        index=False,
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(project_root),
        "analysis": "v2a_edgewise_localization_aggregate",
        "analysis_profile": V2A_ANALYSIS_PROFILE,
        "sharded_by": "method",
        "methods": METHOD_LABELS,
        "input_paths": input_paths,
        "output_paths": [
            str(edge_path.relative_to(project_root)),
            str(fdr_path.relative_to(project_root)),
            str(manifest_path.relative_to(project_root)),
        ],
        "fdr_scope": "method",
        "q_value_recording_method_retained": True,
        "shard_manifests": shard_manifests,
        "recordings_processed": int(edge_df["recording"].nunique()),
        "total_edges": int(len(edge_df)),
        "edge_status_counts": {
            status: int(count)
            for status, count in edge_df.get("status", pd.Series(dtype=str))
            .value_counts()
            .items()
        },
        "calibrated_rows": int(edge_df["p_value"].notna().sum())
        if "p_value" in edge_df.columns
        else 0,
        "software_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {edge_path}")
    print(f"Wrote {fdr_path}")
    print(f"Wrote {manifest_path}")
    print(edge_df.groupby(["method", "calibration_status"]).size().to_string())


if __name__ == "__main__":
    main()
