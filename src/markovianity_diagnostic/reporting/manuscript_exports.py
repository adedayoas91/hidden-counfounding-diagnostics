"""Functions for generating manuscript figures and tables."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .figure_specs import FIGURE_SPECS
from .tables import (
    export_calibration_table,
    export_comparison_table,
    export_depth_selection_table,
)

logger = logging.getLogger(__name__)


def generate_all_figures(
    output_base_dir: str,
    figure_specs_dict: Optional[dict[str, dict]] = None,
) -> list[dict[str, Any]]:
    """Generate all figures specified in FIGURE_SPECS.

    For each figure, checks if source data exists, regenerates figure if necessary,
    and saves to high resolution (dpi=300). If source data is missing, logs warning
    and skips that figure without error.

    Parameters
    ----------
    output_base_dir : str
        Base output directory (typically project root).
    figure_specs_dict : dict[str, dict], optional
        Figure specifications dict (defaults to FIGURE_SPECS from figure_specs.py).
        Each entry has keys: name, source_data_paths, output_path, caption, fig_type.

    Returns
    -------
    list[dict[str, Any]]
        List of result dicts with keys:
        {name, output_path, source_files, success, error_msg}
    """
    if figure_specs_dict is None:
        figure_specs_dict = FIGURE_SPECS

    results = []
    base_path = Path(output_base_dir)

    for fig_key, fig_spec in figure_specs_dict.items():
        result = {
            "name": fig_spec.get("name", fig_key),
            "output_path": fig_spec.get("output_path", ""),
            "source_files": fig_spec.get("source_data_paths", []),
            "success": False,
            "error_msg": None,
        }

        source_paths = fig_spec.get("source_data_paths", [])
        fig_type = fig_spec.get("fig_type", "unknown")

        # Check if any source data exists
        existing_sources = []
        for src_path_str in source_paths:
            src_path = base_path / src_path_str
            if src_path.exists():
                existing_sources.append(src_path)

        if not existing_sources:
            logger.warning(
                f"No source data found for {fig_spec.get('name')} "
                f"(expected: {source_paths}). Skipping."
            )
            result["error_msg"] = "No source data found"
            results.append(result)
            continue

        try:
            # For now, try to copy existing PNG if available
            png_sources = [p for p in existing_sources if p.suffix == ".png"]
            if png_sources:
                output_path = base_path / fig_spec.get("output_path", "")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                # Copy the first PNG source found
                shutil.copy2(png_sources[0], output_path)
                result["success"] = True
                result["output_path"] = str(output_path)
                logger.info(f"Generated {fig_spec.get('name')} from existing PNG.")
            else:
                # For JSON sources, would need to regenerate from data
                logger.info(
                    f"Figure {fig_spec.get('name')} requires regeneration from data "
                    f"(fig_type={fig_type}). Implement plotting logic as needed."
                )
                result["error_msg"] = "Requires data regeneration (not yet implemented)"

        except Exception as e:
            result["error_msg"] = str(e)
            logger.error(f"Failed to generate {fig_spec.get('name')}: {e}")

        results.append(result)

    return results


def generate_all_tables(output_base_dir: str) -> list[dict[str, Any]]:
    """Generate all tables from analysis outputs.

    Calls export functions for depth selection, calibration, and comparison tables.
    Exports each as CSV and LaTeX format to nature_methods/tables/.

    Parameters
    ----------
    output_base_dir : str
        Base output directory (typically project root).

    Returns
    -------
    list[dict[str, Any]]
        List of result dicts with keys:
        {name, output_path_csv, output_path_tex, success, error_msg}
    """
    results = []
    base_path = Path(output_base_dir)
    tables_dir = base_path / "nature_methods" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Table 1: Depth Selection
    table_name = "Table1_DepthSelection"
    try:
        df = export_depth_selection_table(
            str(base_path / "outputs/v2a-RSNs/depth_selection/depth_selection.json")
        )
        if not df.empty:
            csv_path = tables_dir / f"{table_name}.csv"
            tex_path = tables_dir / f"{table_name}.tex"

            df.to_csv(csv_path, index=False)
            df.to_latex(tex_path, index=False)

            results.append(
                {
                    "name": table_name,
                    "output_path_csv": str(csv_path),
                    "output_path_tex": str(tex_path),
                    "success": True,
                    "error_msg": None,
                }
            )
            logger.info(f"Generated {table_name}")
        else:
            results.append(
                {
                    "name": table_name,
                    "output_path_csv": "",
                    "output_path_tex": "",
                    "success": False,
                    "error_msg": "No data available",
                }
            )
    except Exception as e:
        logger.error(f"Failed to generate {table_name}: {e}")
        results.append(
            {
                "name": table_name,
                "output_path_csv": "",
                "output_path_tex": "",
                "success": False,
                "error_msg": str(e),
            }
        )

    # Table 2: Calibration
    table_name = "Table2_Calibration"
    try:
        calibration_path = base_path / "outputs/calibration/v2a"
        df = export_calibration_table(str(calibration_path))
        if not df.empty:
            csv_path = tables_dir / f"{table_name}.csv"
            tex_path = tables_dir / f"{table_name}.tex"

            df.to_csv(csv_path, index=False)
            df.to_latex(tex_path, index=False)

            results.append(
                {
                    "name": table_name,
                    "output_path_csv": str(csv_path),
                    "output_path_tex": str(tex_path),
                    "success": True,
                    "error_msg": None,
                }
            )
            logger.info(f"Generated {table_name}")
        else:
            logger.warning(f"No calibration data found; {table_name} skipped.")
            results.append(
                {
                    "name": table_name,
                    "output_path_csv": "",
                    "output_path_tex": "",
                    "success": False,
                    "error_msg": "No calibration data available",
                }
            )
    except Exception as e:
        logger.error(f"Failed to generate {table_name}: {e}")
        results.append(
            {
                "name": table_name,
                "output_path_csv": "",
                "output_path_tex": "",
                "success": False,
                "error_msg": str(e),
            }
        )

    # Table 3: Comparison
    table_name = "Table3_Comparison"
    try:
        df = export_comparison_table(
            str(base_path / "outputs/comparisons/lpcmci_svarfci/comparison_results.json")
        )
        if not df.empty:
            csv_path = tables_dir / f"{table_name}.csv"
            tex_path = tables_dir / f"{table_name}.tex"

            df.to_csv(csv_path, index=False)
            df.to_latex(tex_path, index=False)

            results.append(
                {
                    "name": table_name,
                    "output_path_csv": str(csv_path),
                    "output_path_tex": str(tex_path),
                    "success": True,
                    "error_msg": None,
                }
            )
            logger.info(f"Generated {table_name}")
        else:
            results.append(
                {
                    "name": table_name,
                    "output_path_csv": "",
                    "output_path_tex": "",
                    "success": False,
                    "error_msg": "No data available",
                }
            )
    except Exception as e:
        logger.error(f"Failed to generate {table_name}: {e}")
        results.append(
            {
                "name": table_name,
                "output_path_csv": "",
                "output_path_tex": "",
                "success": False,
                "error_msg": str(e),
            }
        )

    return results


def create_manuscript_manifest(
    figures_list: list[dict[str, Any]],
    tables_list: list[dict[str, Any]],
    output_dir: str = "outputs/reporting",
) -> dict[str, Any]:
    """Create manuscript manifest with figure and table metadata.

    Synthesizes figure and table generation results into a single manifest JSON.

    Parameters
    ----------
    figures_list : list[dict[str, Any]]
        Result list from generate_all_figures().
    tables_list : list[dict[str, Any]]
        Result list from generate_all_tables().
    output_dir : str, optional
        Output directory for manifest (default "outputs/reporting").

    Returns
    -------
    dict[str, Any]
        Manifest dict with keys:
        {created_at, analysis, figures, tables, summary}
    """
    manifest = {
        "created_at": datetime.now().isoformat(),
        "analysis": "Nature Methods Manuscript Exports (Phase 11)",
        "figures": [],
        "tables": [],
        "summary": {
            "n_figures_generated": 0,
            "n_figures_failed": 0,
            "n_tables_generated": 0,
            "n_tables_failed": 0,
        },
    }

    # Add figures
    specs_by_name = {
        spec.get("name", key): spec for key, spec in FIGURE_SPECS.items()
    }
    for fig in figures_list:
        fig_entry = {
            "name": fig.get("name"),
            "source_data": fig.get("source_files", []),
            "output_path": fig.get("output_path", ""),
            "code_path": "src/markovianity_diagnostic/reporting/figure_specs.py",
            "caption": specs_by_name.get(fig.get("name"), {}).get("caption", ""),
            "success": fig.get("success", False),
        }
        manifest["figures"].append(fig_entry)

        if fig.get("success"):
            manifest["summary"]["n_figures_generated"] += 1
        else:
            manifest["summary"]["n_figures_failed"] += 1

    # Add tables
    for tbl in tables_list:
        tbl_entry = {
            "name": tbl.get("name"),
            "output_path_csv": tbl.get("output_path_csv", ""),
            "output_path_tex": tbl.get("output_path_tex", ""),
            "code_path": "src/markovianity_diagnostic/reporting/tables.py",
            "success": tbl.get("success", False),
        }
        manifest["tables"].append(tbl_entry)

        if tbl.get("success"):
            manifest["summary"]["n_tables_generated"] += 1
        else:
            manifest["summary"]["n_tables_failed"] += 1

    # Write manifest to file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "nature_methods_figure_manifest.json"

    try:
        manifest_path.write_text(json.dumps(manifest, indent=2))
        logger.info(f"Manuscript manifest written to {manifest_path}")
    except Exception as e:
        logger.error(f"Failed to write manifest: {e}")

    return manifest


__all__ = [
    "generate_all_figures",
    "generate_all_tables",
    "create_manuscript_manifest",
]
