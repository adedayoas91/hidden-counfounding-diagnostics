"""Metrics for graph instability and graph recovery."""

from __future__ import annotations

from typing import Any

import numpy as np


def zero_diagonal(adjacency: np.ndarray) -> np.ndarray:
    """Return a copy of ``adjacency`` with its diagonal set to zero."""

    out = np.array(adjacency, copy=True)
    np.fill_diagonal(out, 0)
    return out


def compact_graph_instability(adjacencies: dict[int, np.ndarray]) -> dict[int, float]:
    """Compute ``D_p`` between successive conditioning depths."""

    p_values = sorted(adjacencies)
    if len(p_values) < 2:
        return {}

    d = adjacencies[p_values[0]].shape[0]
    m = d * (d - 1)
    output: dict[int, float] = {}
    for index in range(1, len(p_values)):
        previous_p = p_values[index - 1]
        current_p = p_values[index]
        previous = zero_diagonal(adjacencies[previous_p])
        current = zero_diagonal(adjacencies[current_p])
        output[current_p] = float(np.abs(current - previous).sum() / m)
    return output


def instability_decomposition(
    adjacencies: dict[int, np.ndarray],
) -> dict[int, dict[str, float]]:
    """Decompose instability into edge deletions and additions."""

    p_values = sorted(adjacencies)
    if len(p_values) < 2:
        return {}

    d = adjacencies[p_values[0]].shape[0]
    m = d * (d - 1)
    output: dict[int, dict[str, float]] = {}
    for index in range(1, len(p_values)):
        previous_p = p_values[index - 1]
        current_p = p_values[index]
        previous = zero_diagonal(adjacencies[previous_p]).astype(int)
        current = zero_diagonal(adjacencies[current_p]).astype(int)
        deletions = np.logical_and(previous == 1, current == 0).sum() / m
        additions = np.logical_and(previous == 0, current == 1).sum() / m
        output[current_p] = {
            "D_minus": float(deletions),
            "D_plus": float(additions),
        }
    return output


def edge_counts(adjacencies: dict[int, np.ndarray]) -> dict[int, int]:
    """Count directed off-diagonal edges at each conditioning depth."""

    return {
        p_value: int(zero_diagonal(adjacency).sum())
        for p_value, adjacency in adjacencies.items()
    }


def recovery_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Compute accuracy, precision, recall, and FPR for binary graphs."""

    predicted = zero_diagonal(predicted).astype(int)
    truth = zero_diagonal(truth).astype(int)

    tp = int(np.logical_and(predicted == 1, truth == 1).sum())
    tn = int(np.logical_and(predicted == 0, truth == 0).sum())
    fp = int(np.logical_and(predicted == 1, truth == 0).sum())
    fn = int(np.logical_and(predicted == 0, truth == 1).sum())

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def summarize_run(
    adjacencies: dict[int, np.ndarray],
    truth: np.ndarray | None,
) -> dict[str, Any]:
    """Summarize one experiment run into manuscript-facing statistics."""

    d_stats = compact_graph_instability(adjacencies)
    d_parts = instability_decomposition(adjacencies)
    counts = edge_counts(adjacencies)
    summary: dict[str, Any] = {
        "D_p": d_stats,
        "D_parts": d_parts,
        "edge_counts": counts,
        "T_obs": max(d_stats.values()) if d_stats else 0.0,
    }
    if truth is not None:
        summary["metrics_by_p"] = {
            p_value: recovery_metrics(adjacency, truth)
            for p_value, adjacency in adjacencies.items()
        }
    return summary
