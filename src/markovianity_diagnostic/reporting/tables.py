"""Functions for exporting analysis results as formatted tables."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def export_depth_selection_table(depth_results_path: str) -> pd.DataFrame:
    """Export depth selection results as formatted DataFrame.

    Load depth selection JSON and convert to a DataFrame with columns:
    fish, method, dataset, p_selected_relative, max_D_p, instability_pattern.

    Parameters
    ----------
    depth_results_path : str
        Path to depth_selection.json output file.

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame with one row per analysis record.
        Returns empty DataFrame with correct schema if file not found.
    """
    schema = {
        "fish": [],
        "method": [],
        "dataset": [],
        "p_selected_relative": [],
        "max_D_p": [],
        "instability_pattern": [],
    }

    path = Path(depth_results_path)
    if not path.exists():
        logger.warning(f"Depth selection file not found: {depth_results_path}")
        return pd.DataFrame(schema)

    try:
        data = json.loads(path.read_text())
    except Exception as e:
        logger.error(f"Failed to load depth selection JSON: {e}")
        return pd.DataFrame(schema)

    records = []
    for result in data.get("results", []):
        fish = result.get("fish", "")
        method = result.get("method", "")
        dataset = result.get("dataset", "")

        # Extract relative rule selection
        selected = result.get("selected", {})
        p_selected_relative = selected.get("relative")

        # Compute max instability
        d_p = result.get("D_p", {})
        d_values = [float(v) for v in d_p.values() if v is not None]
        max_d_p = max(d_values) if d_values else 0.0

        # Determine instability pattern from warnings
        warnings = result.get("warnings", [])
        pattern = "stable"
        if any("largest instability at first transition" in w for w in warnings):
            pattern = "decreasing"
        if any("no stable depth found" in w for w in warnings):
            pattern = "high_throughout"

        records.append(
            {
                "fish": fish,
                "method": method,
                "dataset": dataset,
                "p_selected_relative": p_selected_relative,
                "max_D_p": max_d_p,
                "instability_pattern": pattern,
            }
        )

    return pd.DataFrame(records) if records else pd.DataFrame(schema)


def export_calibration_table(calibration_results_path: str) -> pd.DataFrame:
    """Export calibration results as formatted DataFrame.

    Load calibration outputs and convert to a DataFrame with columns:
    null_model, n_bootstrap, critical_90, critical_95, critical_99, p_value.

    Parameters
    ----------
    calibration_results_path : str
        Path to calibration results directory or JSON file.

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame with one row per null model configuration.
        Returns empty DataFrame with correct schema if file not found.
    """
    schema = {
        "null_model": [],
        "n_bootstrap": [],
        "critical_90": [],
        "critical_95": [],
        "critical_99": [],
        "p_value": [],
    }

    path = Path(calibration_results_path)
    if not path.exists():
        logger.warning(f"Calibration results path not found: {calibration_results_path}")
        return pd.DataFrame(schema)

    records = []

    # Try to load as JSON file
    if path.is_file() and path.suffix == ".json":
        try:
            data = json.loads(path.read_text())
            calibration_data = data.get("bootstrap", {})
            records.append(
                {
                    "null_model": data.get("null_model", "unknown"),
                    "n_bootstrap": len(calibration_data.get("T_boot", [])),
                    "critical_90": calibration_data.get("critical_90"),
                    "critical_95": calibration_data.get("critical_95"),
                    "critical_99": calibration_data.get("critical_99"),
                    "p_value": calibration_data.get("p_value"),
                }
            )
        except Exception as e:
            logger.error(f"Failed to load calibration JSON: {e}")
            return pd.DataFrame(schema)
    else:
        # Try to load from directory with multiple JSON files
        if path.is_dir():
            json_files = list(path.glob("*.json"))
            for json_file in json_files:
                try:
                    data = json.loads(json_file.read_text())
                    if "bootstrap" in data:
                        calibration_data = data["bootstrap"]
                        records.append(
                            {
                                "null_model": json_file.stem,
                                "n_bootstrap": len(calibration_data.get("T_boot", [])),
                                "critical_90": calibration_data.get("critical_90"),
                                "critical_95": calibration_data.get("critical_95"),
                                "critical_99": calibration_data.get("critical_99"),
                                "p_value": calibration_data.get("p_value"),
                            }
                        )
                except Exception as e:
                    logger.debug(f"Could not parse {json_file}: {e}")

    return pd.DataFrame(records) if records else pd.DataFrame(schema)


def export_comparison_table(comparison_results_path: str) -> pd.DataFrame:
    """Export comparison results as formatted DataFrame.

    Load comparison JSON and convert to a DataFrame with columns:
    scenario, method, edge_overlap_with_instability, bidirected_marks, unique_edges.

    Parameters
    ----------
    comparison_results_path : str
        Path to comparison_results.json output file.

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame with one row per scenario-method combination.
        Returns empty DataFrame with correct schema if file not found.
    """
    schema = {
        "scenario": [],
        "method": [],
        "edge_overlap_with_instability": [],
        "bidirected_marks": [],
        "unique_edges": [],
    }

    path = Path(comparison_results_path)
    if not path.exists():
        logger.warning(f"Comparison results file not found: {comparison_results_path}")
        return pd.DataFrame(schema)

    try:
        data = json.loads(path.read_text())
    except Exception as e:
        logger.error(f"Failed to load comparison JSON: {e}")
        return pd.DataFrame(schema)

    records = []
    for scenario_name, scenario_data in data.items():
        scenario = scenario_data.get("scenario", scenario_name)

        # Extract metrics for each method
        for method_name, method_data in scenario_data.items():
            if method_name in ("scenario", "n_vars", "n_timepoints", "p_values"):
                continue

            # Try to extract comparison metrics
            overlap = method_data.get("edge_overlap_with_instability")
            if overlap is None and isinstance(method_data, dict):
                # Look for nested metrics
                if "metrics" in method_data:
                    overlap = method_data["metrics"].get("edge_overlap_with_instability")

            bidirected = method_data.get("bidirected_marks", 0)
            unique = method_data.get("unique_edges", 0)

            records.append(
                {
                    "scenario": scenario,
                    "method": method_name,
                    "edge_overlap_with_instability": overlap,
                    "bidirected_marks": bidirected,
                    "unique_edges": unique,
                }
            )

    return pd.DataFrame(records) if records else pd.DataFrame(schema)
