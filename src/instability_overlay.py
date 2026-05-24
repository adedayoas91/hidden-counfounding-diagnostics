#!/usr/bin/env python
"""Plot observed v2a-RSN graph instability for c-GC and c-GC*."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v2a-RSNs"

METHODS = {
    "c-GC": {
        "path": OUTPUT_DIR / "c-GC" / "transition_diagnostics.csv",
        "linestyle": "-",
        "fillstyle": "full",
    },
    "c-GC*": {
        "path": OUTPUT_DIR / "c-GC-star" / "transition_diagnostics.csv",
        "linestyle": "--",
        "fillstyle": "none",
    },
}
METRICS = (
    ("D_p", r"$D_p$", "#1f77b4", "o"),
    ("D_minus", r"$D_p^-$", "#e66101", "v"),
    ("D_plus", r"$D_p^+$", "#2ca02c", "^"),
)


def load_transitions(path: Path) -> dict[str, list[dict[str, float]]]:
    """Load adjacent-depth instability rows grouped by recording."""
    grouped: dict[str, list[dict[str, float]]] = {}
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            if not row["D_p"]:
                continue
            values = {
                "P": float(row["P"]),
                "D_p": float(row["D_p"]),
                "D_minus": float(row["D_minus"]),
                "D_plus": float(row["D_plus"]),
            }
            grouped.setdefault(row["dataset"], []).append(values)

    for rows in grouped.values():
        rows.sort(key=lambda values: values["P"])
    return grouped


def recording_labels(data: dict[str, dict[str, list[dict[str, float]]]]) -> dict[str, str]:
    """Return stable anonymized recording labels shared by both methods."""
    method_recordings = [set(method_data) for method_data in data.values()]
    if not method_recordings or any(names != method_recordings[0] for names in method_recordings[1:]):
        raise ValueError("c-GC and c-GC* must contain the same recordings")
    return {
        recording: f"fish-{index}"
        for index, recording in enumerate(sorted(method_recordings[0]), start=1)
    }


def plot_method(
    method: str,
    data: dict[str, list[dict[str, float]]],
    labels: dict[str, str],
) -> Path:
    """Regenerate a single-method instability decomposition figure."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for axis, recording in zip(axes.ravel(), sorted(data)):
        rows = data[recording]
        x_values = [row["P"] for row in rows]
        for key, label, color, marker in METRICS:
            axis.plot(x_values, [row[key] for row in rows], marker=marker, color=color, label=label)
        axis.set_title(labels[recording])
        axis.set_xlabel(r"$n_{\mathrm{pasts}}$")
        axis.set_ylabel("Normalized instability")
        axis.set_xticks(range(2, 8))

    handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    method_directory = "c-GC" if method == "c-GC" else "c-GC-star"
    output_path = OUTPUT_DIR / method_directory / "instability_decomposition.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_overlay(
    data: dict[str, dict[str, list[dict[str, float]]]],
    labels: dict[str, str],
) -> Path:
    """Create the combined method overlay for each recording."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), sharex=True, sharey=True)
    recordings = sorted(labels)

    for axis, recording in zip(axes.ravel(), recordings):
        for method, method_data in data.items():
            rows = method_data[recording]
            x_values = [row["P"] for row in rows]
            style = METHODS[method]
            for key, _, color, marker in METRICS:
                axis.plot(
                    x_values,
                    [row[key] for row in rows],
                    color=color,
                    linestyle=style["linestyle"],
                    marker=marker,
                    markerfacecolor=color if style["fillstyle"] == "full" else "white",
                    linewidth=1.8,
                    markersize=5.5,
                )
        axis.set_title(labels[recording])
        axis.set_xlabel(r"$n_{\mathrm{pasts}}$")
        axis.set_ylabel("Normalized instability")
        axis.set_xticks(range(2, 8))
        axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.45)

    metric_handles = [
        Line2D([0], [0], color=color, marker=marker, linewidth=1.8, label=label)
        for _, label, color, marker in METRICS
    ]
    method_handles = [
        Line2D([0], [0], color="black", linestyle=style["linestyle"], linewidth=1.8, label=method)
        for method, style in METHODS.items()
    ]
    first_legend = fig.legend(metric_handles, [handle.get_label() for handle in metric_handles],
                              loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.add_artist(first_legend)
    fig.legend(method_handles, [handle.get_label() for handle in method_handles],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.89))

    output_path = OUTPUT_DIR / "c-GC_vs_c-GC-star_instability_overlay.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    """Generate standalone and overlay v2a-RSN instability figures."""
    data = {method: load_transitions(config["path"]) for method, config in METHODS.items()}
    labels = recording_labels(data)
    for method, method_data in data.items():
        print(f"Saved {plot_method(method, method_data, labels)}")
    print(f"Saved {plot_overlay(data, labels)}")


if __name__ == "__main__":
    main()
