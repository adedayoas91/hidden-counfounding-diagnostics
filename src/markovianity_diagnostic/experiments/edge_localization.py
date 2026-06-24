"""Edge localization and instability metrics for causal inference.

This module computes edgewise instability metrics that identify which specific
edges drive global graph instability across conditioning depths. It supports
optional bootstrap-based calibration for FDR-corrected significance testing.

Main function:
- compute_edge_instability_metrics(): Compute per-edge metrics across depths.
"""

from __future__ import annotations

from typing import Optional, Dict, Sequence
import numpy as np
import pandas as pd


def compute_edge_instability_metrics(
    adj_dict: Dict[int, np.ndarray],
    recording: str,
    method: str,
    bootstrap_adj_dict: Optional[Dict[int, np.ndarray] | Sequence[Dict[int, np.ndarray]]] = None,
) -> pd.DataFrame:
    """Compute edgewise instability metrics across conditioning depths.

    For each edge (source, target) observed in adj_dict, compute:
    - Number of depths where edge appears
    - Number of deletions and additions between depths
    - First depth where edge appears/disappears
    - Instability frequency: (n_deletions + n_additions) / (n_depths - 1)
    - Bootstrap-based null frequency and FDR-corrected p-values (if bootstrap provided)

    Parameters
    ----------
    adj_dict : Dict[int, np.ndarray]
        Dictionary mapping conditioning depth (n_past) to adjacency matrix.
        Adjacency matrices must be (d, d) square binary arrays (values in {0, 1}).
    recording : str
        Recording identifier (e.g., 'fish_001').
    method : str
        Method name (e.g., 'c-GC').
    bootstrap_adj_dict : Optional[Dict[int, np.ndarray] | Sequence[Dict[int, np.ndarray]]]
        Optional bootstrap null adjacency output. Pass a sequence of replicate
        dictionaries, each mapping depth to adjacency matrix. A single dict is
        still accepted for backward compatibility and is treated as one replicate.

    Returns
    -------
    pd.DataFrame
        Edge-level metrics with columns:
        - recording: Recording identifier
        - method: Method name
        - source: Source node index
        - target: Target node index
        - n_depths_present: Number of depths where edge appears
        - n_deletions: Number of times edge is deleted between successive depths
        - n_additions: Number of times edge is added between successive depths
        - first_appearance_p: First depth where edge appears
        - first_deletion_p: First depth where edge is deleted (or NaN if never deleted)
        - instability_frequency: (n_deletions + n_additions) / (n_depths - 1)
        - null_frequency: Bootstrap null instability frequency (NaN if no bootstrap)
        - p_value: FDR p-value (NaN if no bootstrap)
        - q_value: FDR-corrected q-value (NaN if no bootstrap)
        - status: Edge status ('stable', 'unstable', 'novel', 'lost')

    Raises
    ------
    ValueError
        If any adjacency matrix is not square or not binary.
    """
    if not adj_dict:
        # Return empty DataFrame with proper columns
        return pd.DataFrame(columns=[
            "recording", "method", "source", "target",
            "n_depths_present", "n_deletions", "n_additions",
            "first_appearance_p", "first_deletion_p",
            "instability_frequency", "null_frequency",
            "p_value", "q_value", "status"
        ])

    # Validate adjacency matrices
    for depth, adj_matrix in adj_dict.items():
        if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError(
                f"Depth {depth}: adjacency matrix must be square, got shape {adj_matrix.shape}"
            )
        if not np.all(np.isin(adj_matrix, [0, 1])):
            raise ValueError(
                f"Depth {depth}: adjacency matrix must contain only binary values {{0, 1}}"
            )

    # Sort depths for consistent ordering
    depths = sorted(adj_dict.keys())
    n_depths = len(depths)

    # Collect all edges across all depths
    edges = set()
    for adj_matrix in adj_dict.values():
        # Extract edges (exclude self-loops)
        rows, cols = np.where(adj_matrix == 1)
        for src, tgt in zip(rows, cols):
            if src != tgt:
                edges.add((int(src), int(tgt)))

    # If no edges found, return empty DataFrame
    if not edges:
        return pd.DataFrame(columns=[
            "recording", "method", "source", "target",
            "n_depths_present", "n_deletions", "n_additions",
            "first_appearance_p", "first_deletion_p",
            "instability_frequency", "null_frequency",
            "p_value", "q_value", "status"
        ])

    # Compute metrics for each edge
    rows_data = []
    edge_instabilities = []

    for source, target in sorted(edges):
        # Track whether edge appears at each depth
        appearances = []
        for depth in depths:
            is_present = bool(adj_dict[depth][source, target] == 1)
            appearances.append(is_present)

        # Count depths where edge appears
        n_depths_present = sum(appearances)

        # Count deletions and additions
        n_deletions = 0
        n_additions = 0
        first_deletion_p = np.nan

        for i in range(1, len(appearances)):
            if appearances[i - 1] and not appearances[i]:  # Deletion
                n_deletions += 1
                if np.isnan(first_deletion_p):
                    first_deletion_p = depths[i]
            elif not appearances[i - 1] and appearances[i]:  # Addition
                n_additions += 1

        # Find first appearance
        first_appearance_p = np.nan
        for i, is_present in enumerate(appearances):
            if is_present:
                first_appearance_p = depths[i]
                break

        # Compute instability frequency
        if n_depths > 1:
            instability_frequency = (n_deletions + n_additions) / (n_depths - 1)
        else:
            instability_frequency = 0.0

        edge_instabilities.append(instability_frequency)

        # Determine status
        if n_deletions > 0 or n_additions > 0:
            status = "unstable"
            if n_depths_present == 1:
                status = "novel" if first_appearance_p == depths[-1] else "unstable"
            elif n_deletions > 0 and n_additions == 0:
                status = "lost" if not appearances[-1] else "unstable"
        else:
            status = "stable"

        # Bootstrap metrics (computed later)
        null_frequency = np.nan
        p_value = np.nan
        q_value = np.nan

        rows_data.append({
            "recording": recording,
            "method": method,
            "source": source,
            "target": target,
            "n_depths_present": n_depths_present,
            "n_deletions": n_deletions,
            "n_additions": n_additions,
            "first_appearance_p": first_appearance_p,
            "first_deletion_p": first_deletion_p,
            "instability_frequency": instability_frequency,
            "null_frequency": null_frequency,
            "p_value": p_value,
            "q_value": q_value,
            "status": status,
        })

    # Create DataFrame
    df = pd.DataFrame(rows_data)

    # Compute bootstrap metrics if provided
    if bootstrap_adj_dict is not None:
        df = _compute_bootstrap_metrics(df, adj_dict, bootstrap_adj_dict, depths)

    return df


def _compute_bootstrap_metrics(
    df: pd.DataFrame,
    adj_dict: Dict[int, np.ndarray],
    bootstrap_adj_dict: Dict[int, np.ndarray] | Sequence[Dict[int, np.ndarray]],
    depths: list,
) -> pd.DataFrame:
    """Compute bootstrap-based null frequency and FDR-corrected p-values.

    Parameters
    ----------
    df : pd.DataFrame
        Edge metrics DataFrame from compute_edge_instability_metrics.
    adj_dict : Dict[int, np.ndarray]
        Observed adjacency matrices.
    bootstrap_adj_dict : Dict[int, np.ndarray] | Sequence[Dict[int, np.ndarray]]
        Bootstrap null adjacency matrices. The preferred form is one dictionary
        per bootstrap replicate.
    depths : list
        Sorted list of conditioning depths.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with null_frequency, p_value, q_value filled in.
    """
    bootstrap_replicates = _coerce_bootstrap_replicates(bootstrap_adj_dict)

    p_values = []
    null_freqs = []

    for _, row in df.iterrows():
        source, target = int(row["source"]), int(row["target"])
        obs_instability = row["instability_frequency"]
        null_instabilities = [
            _edge_instability_for_replicate(replicate, depths, source, target)
            for replicate in bootstrap_replicates
        ]

        null_freq = float(np.mean(null_instabilities)) if null_instabilities else np.nan
        null_freqs.append(null_freq)

        if null_instabilities:
            n_null_with_higher = sum(inst >= obs_instability for inst in null_instabilities)
            p_val = (1 + n_null_with_higher) / (len(null_instabilities) + 1)
        else:
            p_val = np.nan

        p_values.append(p_val)

    df["null_frequency"] = null_freqs
    df["p_value"] = p_values

    # Compute FDR-corrected q-values using Benjamini-Hochberg
    if len(p_values) > 0 and any(not np.isnan(p) for p in p_values):
        # Handle NaN values
        valid_mask = ~pd.isna(df["p_value"])
        if valid_mask.any():
            _, q_values_valid = _benjamini_hochberg_fdr(
                df.loc[valid_mask, "p_value"].values
            )
            df["q_value"] = np.nan
            df.loc[valid_mask, "q_value"] = q_values_valid
    else:
        df["q_value"] = np.nan

    return df


def _coerce_bootstrap_replicates(
    bootstrap_adj_dict: Dict[int, np.ndarray] | Sequence[Dict[int, np.ndarray]]
) -> list[Dict[int, np.ndarray]]:
    if isinstance(bootstrap_adj_dict, dict):
        return [bootstrap_adj_dict] if bootstrap_adj_dict else []
    return [replicate for replicate in bootstrap_adj_dict if replicate]


def _edge_instability_for_replicate(
    replicate: Dict[int, np.ndarray],
    depths: list,
    source: int,
    target: int,
) -> float:
    appearances = []
    for depth in depths:
        adj = replicate.get(depth)
        if adj is None:
            appearances.append(False)
            continue
        if source >= adj.shape[0] or target >= adj.shape[1]:
            appearances.append(False)
            continue
        appearances.append(bool(adj[source, target] == 1))

    changes = sum(
        appearances[index - 1] != appearances[index]
        for index in range(1, len(appearances))
    )
    return float(changes / (len(appearances) - 1)) if len(appearances) > 1 else 0.0


def _benjamini_hochberg_fdr(p_values: np.ndarray, alpha: float = 0.05) -> tuple[float | np.floating, np.ndarray]:
    """Compute Benjamini-Hochberg FDR-corrected q-values.

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values, shape (n,).
    alpha : float
        Target FDR level (default 0.05).

    Returns
    -------
    tuple[float | np.floating, np.ndarray]
        (threshold, q_values) where:
        - threshold: FDR threshold (scalar or np.nan)
        - q_values: FDR-corrected q-values, shape (n,), sorted back to input order
    """
    n = len(p_values)
    if n == 0:
        return np.nan, np.array([])

    # Sort p-values and get ranks
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # Compute q-values: adjusted p-value = p * n / rank
    ranks = np.arange(1, n + 1)
    q_values_sorted = sorted_p * n / ranks

    # Ensure q-values are non-decreasing from right to left
    for i in range(n - 2, -1, -1):
        q_values_sorted[i] = min(q_values_sorted[i], q_values_sorted[i + 1])

    # Ensure q-values don't exceed 1.0
    q_values_sorted = np.minimum(q_values_sorted, 1.0)

    # Unsort back to original order
    q_values = np.empty_like(q_values_sorted)
    q_values[sorted_indices] = q_values_sorted

    # Find FDR threshold (largest rank where q <= alpha)
    threshold_idx = np.where(q_values_sorted <= alpha)[0]
    threshold = sorted_p[threshold_idx[-1]] if len(threshold_idx) > 0 else np.nan

    return threshold, q_values
