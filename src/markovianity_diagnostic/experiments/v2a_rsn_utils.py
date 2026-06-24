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
    'get_v2a_selected_cell_indices',
    'get_v2a_bad_frame_indices',
    'subset_v2a_cells',
    'drop_v2a_bad_frames',
    'load_and_filter_traces',
    'load_and_filter_adjacencies',
    'setup_recording_paths',
    'find_pkl_file',
]
