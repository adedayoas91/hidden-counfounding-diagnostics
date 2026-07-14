"""Run calibrated v2a edge localization for one method shard.

The shard reads observed adjacency pickles from Phase 0 and, when present,
bootstrap replicate adjacency pickles from Phase 1 calibration. Outputs are
method-isolated so c-GC and c-GC-star can run asynchronously.
"""

from __future__ import annotations

import argparse
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
    "c-GC": {"method_dir": "c-GC"},
    "c-GC-star": {"method_dir": "c-GC-star"},
}

FDR_COLUMNS = [
    "recording",
    "method",
    "source",
    "target",
    "instability_frequency",
    "null_frequency",
    "p_value",
    "q_value",
    "q_value_recording_method",
    "bootstrap_replicates",
    "fdr_scope",
]


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


def load_bootstrap_replicates(
    *,
    project_root: Path,
    analysis_profile: str,
    recording: str,
    method_dir: str,
    expected_depths: list[int],
) -> tuple[list[dict[int, np.ndarray]], Path, list[str]]:
    """Load compatible bootstrap adjacency replicate pickles for one run."""

    from markovianity_diagnostic.experiments.v2a_rsn_utils import (
        load_adjacency_checkpoint,
    )

    checkpoint_dir = (
        project_root
        / "outputs"
        / "calibration"
        / "v2a"
        / analysis_profile
        / recording
        / method_dir
        / "checkpoints"
    )
    if not checkpoint_dir.exists():
        return [], checkpoint_dir, []

    replicates: list[dict[int, np.ndarray]] = []
    skipped: list[str] = []
    required_depths = set(int(depth) for depth in expected_depths)
    for replicate_path in sorted(checkpoint_dir.glob("replicate_*.pkl")):
        try:
            replicate = load_adjacency_checkpoint(replicate_path)
            missing = required_depths.difference(replicate)
            if missing:
                skipped.append(f"{replicate_path.name}: missing depths {sorted(missing)}")
                continue
            replicates.append({depth: replicate[depth] for depth in expected_depths})
        except Exception as error:
            skipped.append(f"{replicate_path.name}: {error}")
    return replicates, checkpoint_dir, skipped


def create_edge_heatmap(
    adj_dict: dict[int, np.ndarray],
    *,
    recording: str,
    method_label: str,
    output_dir: Path,
) -> Path | None:
    """Write a compact additions/deletions heatmap for one recording."""

    depths = sorted(adj_dict)
    if len(depths) < 2:
        return None

    edges: set[tuple[int, int]] = set()
    for adjacency in adj_dict.values():
        rows, cols = np.where(adjacency == 1)
        for source, target in zip(rows, cols, strict=False):
            if int(source) != int(target):
                edges.add((int(source), int(target)))
    if not edges:
        return None

    edge_list = sorted(edges)
    deletion_matrix = np.zeros((len(edge_list), len(depths) - 1))
    addition_matrix = np.zeros((len(edge_list), len(depths) - 1))
    labels = [f"{source}->{target}" for source, target in edge_list]

    for edge_index, (source, target) in enumerate(edge_list):
        for depth_index, depth in enumerate(depths[:-1]):
            next_depth = depths[depth_index + 1]
            present = bool(adj_dict[depth][source, target] == 1)
            next_present = bool(adj_dict[next_depth][source, target] == 1)
            if present and not next_present:
                deletion_matrix[edge_index, depth_index] = 1
            elif not present and next_present:
                addition_matrix[edge_index, depth_index] = 1

    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(14, min(12, max(4, len(labels) * 0.2))))
    transition_labels = [
        f"{depth}->{depths[index + 1]}" for index, depth in enumerate(depths[:-1])
    ]
    yticklabels = labels if len(labels) <= 50 else []
    deletion_image = axes[0].imshow(deletion_matrix, aspect="auto", cmap="Reds")
    figure.colorbar(deletion_image, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_xticks(range(len(transition_labels)))
    axes[0].set_xticklabels(transition_labels, rotation=45, ha="right")
    axes[0].set_yticks(range(len(yticklabels)))
    axes[0].set_yticklabels(yticklabels)
    axes[0].set_title(f"Edge deletions - {recording} ({method_label})")
    axes[0].set_xlabel("Depth transition")
    axes[0].set_ylabel("Edge")
    addition_image = axes[1].imshow(addition_matrix, aspect="auto", cmap="Blues")
    figure.colorbar(addition_image, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_xticks(range(len(transition_labels)))
    axes[1].set_xticklabels(transition_labels, rotation=45, ha="right")
    axes[1].set_yticks(range(len(yticklabels)))
    axes[1].set_yticklabels(yticklabels)
    axes[1].set_title(f"Edge additions - {recording} ({method_label})")
    axes[1].set_xlabel("Depth transition")
    axes[1].set_ylabel("Edge")
    figure.tight_layout()
    path = output_dir / f"{recording}_{method_label}_edge_heatmaps.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def run_method(method_label: str, *, force: bool = False) -> None:
    project_root = find_project_root()
    sys.path.insert(0, str(project_root / "src"))

    from markovianity_diagnostic.experiments.edge_localization import (
        compute_edge_instability_metrics,
    )
    from markovianity_diagnostic.experiments.v2a_rsn_utils import (
        V2A_ANALYSIS_PROFILE,
        load_adjacency_checkpoint,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    method_dir = METHOD_CONFIG[method_label]["method_dir"]
    outputs_v2a = project_root / "outputs" / "v2a-RSNs" / V2A_ANALYSIS_PROFILE
    observed_dir = outputs_v2a / method_dir
    output_dir = outputs_v2a / "edge_localization" / "by_method" / method_label
    heatmap_dir = output_dir / "heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)

    edge_path = output_dir / "edge_instability.csv"
    fdr_path = output_dir / "edge_fdr.csv"
    manifest_path = output_dir / "manifest.json"
    outputs_exist = edge_path.exists() and manifest_path.exists()
    if outputs_exist and not force:
        frame = pd.read_csv(edge_path)
        print(f"Loaded cached {method_label} edge-localization shard: {edge_path}")
        print(f"Rows: {len(frame)}")
        return

    observed_paths = sorted(observed_dir.glob("*.pkl"))
    if not observed_paths:
        raise FileNotFoundError(f"No observed adjacency pickles found in {observed_dir}")

    started = time.time()
    rows: list[pd.DataFrame] = []
    input_paths: list[str] = []
    heatmap_paths: list[str] = []
    calibration_summary: dict[str, Any] = {}

    for recording_index, observed_path in enumerate(observed_paths, start=1):
        recording = observed_path.stem
        print(f"[{recording_index}/{len(observed_paths)}] {method_label}: {recording}")
        observed = load_adjacency_checkpoint(observed_path)
        if not observed:
            logger.warning("Skipping empty observed adjacency file: %s", observed_path)
            continue
        depths = sorted(int(depth) for depth in observed)
        replicates, checkpoint_dir, skipped = load_bootstrap_replicates(
            project_root=project_root,
            analysis_profile=V2A_ANALYSIS_PROFILE,
            recording=recording,
            method_dir=method_dir,
            expected_depths=depths,
        )
        input_paths.append(str(observed_path.relative_to(project_root)))
        calibration_summary[recording] = {
            "checkpoint_dir": str(checkpoint_dir.relative_to(project_root))
            if checkpoint_dir.exists()
            else str(checkpoint_dir),
            "bootstrap_replicates": len(replicates),
            "skipped_replicates": skipped[:20],
            "skipped_replicate_count": len(skipped),
        }

        edge_frame = compute_edge_instability_metrics(
            adj_dict=observed,
            recording=recording,
            method=method_label,
            bootstrap_adj_dict=replicates if replicates else None,
        )
        edge_frame["bootstrap_replicates"] = len(replicates)
        edge_frame["bootstrap_checkpoint_dir"] = (
            str(checkpoint_dir.relative_to(project_root))
            if checkpoint_dir.exists()
            else ""
        )
        edge_frame["calibration_status"] = (
            "bootstrap_calibrated" if replicates else "descriptive_no_bootstrap"
        )
        edge_frame["fdr_scope"] = "recording_method" if replicates else ""
        if "q_value" in edge_frame:
            edge_frame["q_value_recording_method"] = edge_frame["q_value"]
        rows.append(edge_frame)

        if recording_index <= 2:
            heatmap_path = create_edge_heatmap(
                observed,
                recording=recording,
                method_label=method_label,
                output_dir=heatmap_dir,
            )
            if heatmap_path is not None:
                heatmap_paths.append(str(heatmap_path.relative_to(project_root)))

        print(
            f"  edges={len(edge_frame)}, bootstrap_replicates={len(replicates)}, "
            f"status={edge_frame['calibration_status'].iloc[0] if len(edge_frame) else 'empty'}"
        )

    edge_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    edge_df.to_csv(edge_path, index=False)

    if not edge_df.empty and "p_value" in edge_df:
        fdr_cols = [column for column in FDR_COLUMNS if column in edge_df.columns]
        fdr_df = edge_df[fdr_cols].dropna(subset=["p_value"])
        fdr_df.to_csv(fdr_path, index=False)
    else:
        pd.DataFrame(columns=FDR_COLUMNS).to_csv(fdr_path, index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(project_root),
        "analysis": "v2a_edgewise_localization_method_shard",
        "analysis_profile": V2A_ANALYSIS_PROFILE,
        "method": method_label,
        "method_dir": method_dir,
        "input_paths": input_paths,
        "output_paths": [
            str(edge_path.relative_to(project_root)),
            str(fdr_path.relative_to(project_root)),
            str(manifest_path.relative_to(project_root)),
            *heatmap_paths,
        ],
        "calibration_summary": calibration_summary,
        "fdr_scope": "recording_method",
        "recordings_processed": int(len(calibration_summary)),
        "total_edges": int(len(edge_df)),
        "edge_status_counts": {
            status: int(count)
            for status, count in edge_df.get("status", pd.Series(dtype=str))
            .value_counts()
            .items()
        },
        "elapsed_seconds": float(time.time() - started),
        "software_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {edge_path}")
    print(f"Wrote {fdr_path}")
    print(f"Wrote {manifest_path}")
    print(edge_df.groupby("calibration_status").size().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHOD_CONFIG), required=True)
    parser.add_argument("--force", action="store_true", help="recompute even if shard outputs exist")
    args = parser.parse_args()
    run_method(args.method, force=args.force)


if __name__ == "__main__":
    main()
