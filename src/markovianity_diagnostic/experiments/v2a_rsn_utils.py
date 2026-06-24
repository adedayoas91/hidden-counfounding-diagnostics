"""Utilities for v2a-RSN stability analysis notebooks.

Provides:
- Checkpoint and result logging (append-safe CSV/JSON operations)
- Metadata handling (neuron selection, bad frame filtering)
- Common functions extracted from notebook cells
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Checkpoint & Logging
# ============================================================================


def load_checkpoint(checkpoint_file: Path) -> dict[str, Any]:
    """Load checkpoint tracking all processing runs."""
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {'processing_runs': [], 'total_runs': 0}


def save_checkpoint(
    checkpoint_file: Path,
    recording_name: str,
    processing_runs: list,
    total_runs: int,
) -> None:
    """Save checkpoint of all processing runs."""
    checkpoint = {
        'recording': recording_name,
        'processing_runs': processing_runs,
        'total_runs': total_runs,
        'last_updated': datetime.now().isoformat(),
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def log_processing_run(
    result_log_file: Path,
    recording_name: str,
    summary: dict[str, Any],
    transition_df_len: int,
) -> None:
    """Log this processing run to JSONL file."""
    run_log = {
        'timestamp': datetime.now().isoformat(),
        'recording': recording_name,
        'p_values_processed': [int(p) for p in sorted(summary['D_p'].keys())],
        'num_p_values': len(summary['D_p']),
        'transition_rows': transition_df_len,
        'T_obs': summary['T_obs'],
        'edge_count_p1': summary['edge_count_p1'],
        'edge_count_pmax': summary['edge_count_pmax'],
    }
    with open(result_log_file, 'a') as f:
        f.write(json.dumps(run_log) + '\n')


# ============================================================================
# CSV Append-Safe Operations
# ============================================================================


def append_csv_safe(
    df: pd.DataFrame,
    csv_path: Path,
    index: bool = False,
) -> None:
    """Append DataFrame to CSV file (create with header if not exists)."""
    if csv_path.exists():
        df.to_csv(csv_path, mode='a', header=False, index=index)
    else:
        df.to_csv(csv_path, index=index)


def append_multiple_csvs(
    summary_df: pd.DataFrame,
    transition_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Append summary and transition dataframes to their respective CSV files."""
    append_csv_safe(summary_df, output_dir / 'summary.csv')
    append_csv_safe(transition_df, output_dir / 'transitions.csv')
    append_csv_safe(summary_df, output_dir / 'diagnostic_summary.csv')
    append_csv_safe(transition_df, output_dir / 'transition_diagnostics.csv')


def upsert_csv_by_keys(
    df: pd.DataFrame,
    csv_path: Path,
    key_columns: list[str],
    *,
    sort_columns: list[str] | None = None,
) -> None:
    """Write ``df`` into ``csv_path``, replacing matching key rows if present."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        missing = [column for column in key_columns if column not in existing.columns]
        if missing:
            logger.warning(
                "Replacing %s because it lacks key columns: %s",
                csv_path,
                missing,
            )
            combined = df.copy()
        else:
            incoming_keys = {
                tuple(row[column] for column in key_columns)
                for _, row in df.iterrows()
            }
            keep_mask = [
                tuple(row[column] for column in key_columns) not in incoming_keys
                for _, row in existing.iterrows()
            ]
            combined = pd.concat(
                [existing.loc[keep_mask], df],
                ignore_index=True,
                sort=False,
            )
    else:
        combined = df.copy()

    if sort_columns:
        available = [column for column in sort_columns if column in combined.columns]
        if available:
            combined = combined.sort_values(available).reset_index(drop=True)

    combined.to_csv(csv_path, index=False)


def upsert_method_outputs(
    summary_df: pd.DataFrame,
    transition_df: pd.DataFrame,
    method_output_dir: Path,
) -> None:
    """Update method-level v2a-RSN summary and transition outputs.

    The recording notebooks are restartable, so repeated executions should replace
    the current recording's rows rather than append duplicates.
    """
    upsert_csv_by_keys(
        summary_df,
        method_output_dir / 'summary.csv',
        ['dataset'],
        sort_columns=['dataset'],
    )
    upsert_csv_by_keys(
        summary_df,
        method_output_dir / 'diagnostic_summary.csv',
        ['dataset'],
        sort_columns=['dataset'],
    )
    upsert_csv_by_keys(
        transition_df,
        method_output_dir / 'transitions.csv',
        ['dataset', 'P'],
        sort_columns=['dataset', 'P'],
    )
    upsert_csv_by_keys(
        transition_df,
        method_output_dir / 'transition_diagnostics.csv',
        ['dataset', 'P'],
        sort_columns=['dataset', 'P'],
    )


def upsert_summary_json(
    summary: dict[str, Any],
    summary_json_path: Path,
    *,
    key: str = 'dataset',
) -> None:
    """Update method-level ``summary.json`` list with one recording summary."""
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_json_path.exists():
        payload = json.loads(summary_json_path.read_text(encoding='utf-8'))
        if isinstance(payload, dict):
            records = [payload]
        elif isinstance(payload, list):
            records = payload
        else:
            raise TypeError(f"{summary_json_path} must contain a dict or list of dicts.")
    else:
        records = []

    records = [record for record in records if record.get(key) != summary.get(key)]
    records.append(summary)
    records = sorted(records, key=lambda record: str(record.get(key, '')))
    summary_json_path.write_text(json.dumps(records, indent=2), encoding='utf-8')


# ============================================================================
# Restartable adjacency persistence
# ============================================================================


def load_adjacency_checkpoint(connectivity_pkl: Path) -> dict[int, np.ndarray]:
    """Load a restartable adjacency dict keyed by integer conditioning depth."""
    if not connectivity_pkl.exists():
        return {}

    raw = pd.read_pickle(connectivity_pkl)
    if not isinstance(raw, dict):
        raise TypeError(
            f"{connectivity_pkl} must contain a dict keyed by p_value, got {type(raw)!r}."
        )

    adjacencies: dict[int, np.ndarray] = {}
    for key, value in raw.items():
        adjacency = np.asarray(value)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(
                f"{connectivity_pkl} has invalid adjacency for P={key}: "
                f"expected square matrix, got {adjacency.shape}."
            )
        adjacencies[int(key)] = adjacency
    return adjacencies


def save_adjacency_checkpoint(
    connectivity_pkl: Path,
    adjacencies: dict[int, np.ndarray],
) -> None:
    """Atomically save adjacency matrices keyed by integer conditioning depth."""
    connectivity_pkl.parent.mkdir(parents=True, exist_ok=True)
    normalized = {
        int(p_value): np.asarray(adjacency)
        for p_value, adjacency in sorted(adjacencies.items())
    }
    tmp_path = connectivity_pkl.with_name(f".{connectivity_pkl.name}.tmp")
    pd.to_pickle(normalized, tmp_path)
    tmp_path.replace(connectivity_pkl)


def completed_p_values(
    adjacencies: dict[int, np.ndarray],
    p_values: list[int],
) -> list[int]:
    """Return requested p-values already present in an adjacency checkpoint."""
    requested = [int(p_value) for p_value in p_values]
    return [p_value for p_value in requested if p_value in adjacencies]


def binarize_adjacencies(
    adjacencies: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """Convert adjacency matrices to binary no-self-loop matrices."""
    binary: dict[int, np.ndarray] = {}
    for p_value, adjacency in adjacencies.items():
        matrix = (np.asarray(adjacency) != 0).astype(int)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"Adjacency for P={p_value} must be square, got {matrix.shape}."
            )
        np.fill_diagonal(matrix, 0)
        binary[int(p_value)] = matrix
    return dict(sorted(binary.items()))


def _safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float('nan')


def _argmax_key(values: dict[int, float]) -> int | None:
    return max(values, key=values.get) if values else None


def build_v2a_stability_summary(
    dataset_name: str,
    path: Path,
    adjacencies: dict[int, np.ndarray],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build one v2a-RSN stability summary row from graph metrics."""
    depths = sorted(int(p_value) for p_value in adjacencies)
    if not depths:
        raise ValueError("At least one adjacency is required to build a summary.")

    n_nodes = int(next(iter(adjacencies.values())).shape[0])
    n_possible_edges = n_nodes * (n_nodes - 1)
    edge_counts = {int(k): int(v) for k, v in metrics['edge_counts'].items()}
    d_p = {int(k): float(v) for k, v in metrics['D_p'].items()}
    d_parts = {
        int(k): {'D_minus': float(v['D_minus']), 'D_plus': float(v['D_plus'])}
        for k, v in metrics['D_parts'].items()
    }

    min_edge_depth = min(edge_counts, key=edge_counts.get)
    min_edge_count = edge_counts[min_edge_depth]
    first_depth = depths[0]
    final_depth = depths[-1]
    first_transition_depth = depths[1] if len(depths) > 1 else first_depth
    t_obs_depth = _argmax_key(d_p)
    t_minus_depth = _argmax_key({p: parts['D_minus'] for p, parts in d_parts.items()})
    t_plus_depth = _argmax_key({p: parts['D_plus'] for p, parts in d_parts.items()})

    first_edge_count = edge_counts[first_depth]
    first_step_drop_count = edge_counts[first_depth] - edge_counts[first_transition_depth]
    drop_to_min_count = edge_counts[first_depth] - min_edge_count
    final_drop_count = edge_counts[first_depth] - edge_counts[final_depth]
    post_first_d = {p: value for p, value in d_p.items() if p > first_transition_depth}

    return {
        'dataset': dataset_name,
        'file': path.name,
        'n_nodes': n_nodes,
        'n_possible_edges': n_possible_edges,
        'n_p_values': len(adjacencies),
        'T_obs': float(metrics['T_obs']),
        'T_obs_depth': int(t_obs_depth) if t_obs_depth is not None else None,
        'T_minus_obs': float(d_parts[t_minus_depth]['D_minus']) if t_minus_depth is not None else 0.0,
        'T_minus_depth': int(t_minus_depth) if t_minus_depth is not None else None,
        'T_plus_obs': float(d_parts[t_plus_depth]['D_plus']) if t_plus_depth is not None else 0.0,
        'T_plus_depth': int(t_plus_depth) if t_plus_depth is not None else None,
        'edge_count_p1': int(edge_counts[first_depth]),
        'edge_count_pmax': int(edge_counts[final_depth]),
        'min_edge_count': int(min_edge_count),
        'min_edge_depth': int(min_edge_depth),
        'first_step_drop_count': int(first_step_drop_count),
        'first_step_drop_fraction': _safe_fraction(first_step_drop_count, first_edge_count),
        'drop_to_min_count': int(drop_to_min_count),
        'drop_to_min_fraction': _safe_fraction(drop_to_min_count, first_edge_count),
        'final_drop_count': int(final_drop_count),
        'final_drop_fraction': _safe_fraction(final_drop_count, first_edge_count),
        'cumulative_D_p': float(sum(d_p.values())),
        'cumulative_D_minus': float(sum(parts['D_minus'] for parts in d_parts.values())),
        'cumulative_D_plus': float(sum(parts['D_plus'] for parts in d_parts.values())),
        'mean_post_first_D_p': float(np.mean(list(post_first_d.values()))) if post_first_d else float('nan'),
        'max_post_first_D_p': float(max(post_first_d.values())) if post_first_d else float('nan'),
        'edge_counts': edge_counts,
        'D_p': d_p,
        'D_parts': d_parts,
    }


def build_v2a_transition_df(
    dataset_name: str,
    file_name: str,
    adjacencies: dict[int, np.ndarray],
    metrics: dict[str, Any],
) -> pd.DataFrame:
    """Build per-depth transition metrics for one v2a-RSN recording."""
    depths = sorted(int(p_value) for p_value in adjacencies)
    if not depths:
        raise ValueError("At least one adjacency is required to build transitions.")

    n_nodes = int(next(iter(adjacencies.values())).shape[0])
    n_possible_edges = n_nodes * (n_nodes - 1)
    edge_counts = {int(k): int(v) for k, v in metrics['edge_counts'].items()}
    d_p = {int(k): float(v) for k, v in metrics['D_p'].items()}
    d_parts = {
        int(k): {'D_minus': float(v['D_minus']), 'D_plus': float(v['D_plus'])}
        for k, v in metrics['D_parts'].items()
    }

    rows: list[dict[str, Any]] = []
    for index, p_value in enumerate(depths):
        row: dict[str, Any] = {
            'dataset': dataset_name,
            'file': file_name,
            'P': p_value,
            'edge_count': int(edge_counts[p_value]),
        }
        if index > 0 and p_value in d_p:
            previous_p = depths[index - 1]
            previous_count = edge_counts[previous_p]
            d_minus = d_parts[p_value]['D_minus']
            d_plus = d_parts[p_value]['D_plus']
            d_value = d_p[p_value]
            deletion_count = int(round(d_minus * n_possible_edges))
            addition_count = int(round(d_plus * n_possible_edges))
            edge_delta = edge_counts[p_value] - previous_count
            row.update({
                'previous_P': previous_p,
                'previous_edge_count': int(previous_count),
                'edge_delta': int(edge_delta),
                'net_edge_loss_count': int(-edge_delta),
                'D_p': float(d_value),
                'D_minus': float(d_minus),
                'D_plus': float(d_plus),
                'edge_deletions': deletion_count,
                'edge_additions': addition_count,
                'net_deletion_count': int(deletion_count - addition_count),
                'deletion_share_of_changes': _safe_fraction(d_minus, d_value),
                'addition_share_of_changes': _safe_fraction(d_plus, d_value),
                'net_deletion_fraction_possible': float(d_minus - d_plus),
            })
        else:
            row.update({
                'previous_P': None,
                'previous_edge_count': None,
                'edge_delta': None,
                'net_edge_loss_count': None,
                'D_p': None,
                'D_minus': None,
                'D_plus': None,
                'edge_deletions': None,
                'edge_additions': None,
                'net_deletion_count': None,
                'deletion_share_of_changes': None,
                'addition_share_of_changes': None,
                'net_deletion_fraction_possible': None,
            })
        rows.append(row)

    return pd.DataFrame(rows)


def v2a_summary_json_payload(summary: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable summary payload preserving nested metrics."""
    payload = {
        key: value
        for key, value in summary.items()
        if key not in {'edge_counts', 'D_p', 'D_parts'}
    }
    payload.update({key: summary[key] for key in ['edge_counts', 'D_p', 'D_parts']})
    return payload


# ============================================================================
# V2a-RSN Metadata: Neuron Selection & Bad Frame Filtering
# ============================================================================


def get_v2a_selected_cell_indices(recording_dir: Path, recording_id: str) -> np.ndarray:
    """Load emitter and receiver cell indices for v2a-RSN recording.

    Returns ordered unique indices of identified neurons.
    """
    emitter_path = _best_recording_file(recording_dir, recording_id, 'emitter_cells')
    receiver_path = _best_recording_file(recording_dir, recording_id, 'receiver_cells')

    if emitter_path is None or receiver_path is None:
        raise FileNotFoundError(
            f"Missing emitter/receiver cell index files in {recording_dir}. "
            f"Expected *emitter_cells.npy and *receiver_cells.npy"
        )

    emitter = np.asarray(
        np.load(emitter_path, allow_pickle=False), dtype=int
    ).reshape(-1)
    receiver = np.asarray(
        np.load(receiver_path, allow_pickle=False), dtype=int
    ).reshape(-1)

    # Return ordered unique indices (emitter first, then new receiver indices)
    ordered_unique = list(dict.fromkeys(np.concatenate([emitter, receiver]).tolist()))
    return np.asarray(ordered_unique, dtype=int)


def get_v2a_bad_frame_indices(recording_dir: Path) -> np.ndarray:
    """Load bad frame indices from *_analysis_info.json metadata."""
    info_path = _analysis_info_path(recording_dir)

    if info_path is None:
        logger.warning(f"No *_analysis_info.json found in {recording_dir}")
        return np.empty(0, dtype=int)

    try:
        payload = json.loads(info_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Could not read {info_path}: {e}")
        return np.empty(0, dtype=int)

    bad_frames = payload.get('bad_frames', [])
    if bad_frames is None:
        return np.empty(0, dtype=int)

    return np.asarray(bad_frames, dtype=int).reshape(-1)


def subset_v2a_cells(
    traces: np.ndarray,
    recording_dir: Path,
    recording_id: str,
) -> np.ndarray:
    """Select identified neurons (emitter + receiver cells) from traces."""
    selected = get_v2a_selected_cell_indices(recording_dir, recording_id)

    if selected.size == 0:
        raise ValueError(f"No identified cell indices found in {recording_dir}")

    if np.any(selected < 0) or np.any(selected >= traces.shape[0]):
        raise ValueError(
            f"Cell indices {selected} outside trace bounds 0..{traces.shape[0] - 1}"
        )

    logger.info(f"Selecting {len(selected)} identified neurons from {traces.shape[0]} total")
    return traces[selected, :]


def drop_v2a_bad_frames(
    traces: np.ndarray,
    recording_dir: Path,
) -> np.ndarray:
    """Remove bad frames indicated in metadata."""
    bad_frames = get_v2a_bad_frame_indices(recording_dir)

    if bad_frames.size == 0:
        logger.info("No bad frames to drop")
        return traces

    if np.any(bad_frames < 0) or np.any(bad_frames >= traces.shape[1]):
        raise ValueError(
            f"Bad frame indices {bad_frames} outside bounds 0..{traces.shape[1] - 1}"
        )

    keep_mask = np.ones(traces.shape[1], dtype=bool)
    keep_mask[bad_frames] = False
    n_dropped = int(np.sum(~keep_mask))
    logger.info(f"Dropping {n_dropped} bad frames")
    return traces[:, keep_mask]


def load_and_filter_traces(
    recording_dir: Path,
    recording_id: str,
) -> np.ndarray:
    """Load raw fluorescence traces and apply neuron selection and frame filtering.

    Returns filtered traces (selected_neurons × remaining_frames).
    """
    # Load raw fluorescence signals
    trace_files = list(recording_dir.glob('*cells_fluorescence_signals.npy'))
    if not trace_files:
        raise FileNotFoundError(
            f'No fluorescence signal files in {recording_dir}. '
            f'Expected *cells_fluorescence_signals.npy'
        )

    trace_file = trace_files[0]
    X = np.load(trace_file, allow_pickle=False).astype(float)
    logger.info(f'Loaded traces from {trace_file.name}: shape {X.shape}')

    # Apply neuron selection (emitter + receiver cells)
    X = subset_v2a_cells(X, recording_dir, recording_id)

    # Apply bad frame filtering
    X = drop_v2a_bad_frames(X, recording_dir)

    return X


def load_and_filter_adjacencies(
    pkl_path: Path,
    recording_dir: Path,
    recording_id: str,
) -> dict[int, np.ndarray]:
    """Load adjacencies from pickle and apply neuron selection and frame dropping.

    This accounts for:
    1. Subset to identified neurons (emitter + receiver cells)
    2. Dropping bad frames

    Returns the filtered adjacency dict.
    """
    raw_data = pd.read_pickle(pkl_path)
    if not isinstance(raw_data, dict):
        raise TypeError(
            f'{pkl_path.name} must contain a dict keyed by P values, got {type(raw_data)!r}'
        )

    adjacencies: dict[int, np.ndarray] = {}
    for key, value in raw_data.items():
        adjacency = np.asarray(value).astype(int)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(
                f'{pkl_path.name} has invalid adjacency for P={key}: '
                f'expected square matrix, got {adjacency.shape}'
            )
        adjacency = (adjacency != 0).astype(int)
        np.fill_diagonal(adjacency, 0)
        adjacencies[int(key)] = adjacency

    return adjacencies


# ============================================================================
# Helper Functions
# ============================================================================


def _best_recording_file(
    run_dir: Path,
    recording_id: str,
    token: str,
) -> Path | None:
    """Find the best matching file in recording directory.

    Prefers files whose names contain recording_id tokens.
    """
    candidates = sorted(run_dir.glob(f'*{token}*.npy'))
    if not candidates:
        return None

    recording_tokens = tuple(part.lower() for part in recording_id.split('_'))

    def score(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        token_matches = sum(part in name for part in recording_tokens)
        return token_matches, -len(path.name), path.name

    return max(candidates, key=score)


def _analysis_info_path(run_dir: Path) -> Path | None:
    """Find *_analysis_info.json in recording directory."""
    matches = sorted(run_dir.glob('*_analysis_info.json'))
    if not matches:
        return None
    return matches[0]


# ============================================================================
# Path & Setup Utilities
# ============================================================================


def setup_recording_paths(
    project_root: Path,
    recording_name: str,
    method_dir: str,  # 'c-GC' or 'c-GC-star'
) -> dict[str, Path]:
    """Set up all required paths for a recording-specific notebook run.

    Returns dict with keys: data_dir, output_dir, checkpoint_file, result_log_file
    """
    data_dir = project_root / 'data' / 'v2a-RSNs' / recording_name
    output_dir = project_root / 'outputs' / 'v2a-RSNs' / method_dir / recording_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        'data_dir': data_dir,
        'output_dir': output_dir,
        'checkpoint_file': output_dir / '.checkpoint.json',
        'result_log_file': output_dir / 'processing_log.jsonl',
        'recording_dir': data_dir,
    }


def find_pkl_file(data_dir: Path) -> Path:
    """Find the single pickle file in data directory."""
    pkl_files = sorted(data_dir.glob('*.pkl'))

    if not pkl_files:
        raise FileNotFoundError(f'No pickle files found in {data_dir}')

    if len(pkl_files) != 1:
        logger.warning(
            f'Expected 1 pickle file in {data_dir}, found {len(pkl_files)}. '
            f'Using first: {pkl_files[0].name}'
        )

    return pkl_files[0]


__all__ = [
    'load_checkpoint',
    'save_checkpoint',
    'log_processing_run',
    'append_csv_safe',
    'append_multiple_csvs',
    'upsert_csv_by_keys',
    'upsert_method_outputs',
    'upsert_summary_json',
    'load_adjacency_checkpoint',
    'save_adjacency_checkpoint',
    'completed_p_values',
    'binarize_adjacencies',
    'build_v2a_stability_summary',
    'build_v2a_transition_df',
    'v2a_summary_json_payload',
    'get_v2a_selected_cell_indices',
    'get_v2a_bad_frame_indices',
    'subset_v2a_cells',
    'drop_v2a_bad_frames',
    'load_and_filter_traces',
    'load_and_filter_adjacencies',
    'setup_recording_paths',
    'find_pkl_file',
]
