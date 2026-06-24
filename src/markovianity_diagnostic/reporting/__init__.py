"""Reporting and manuscript generation utilities."""

from .figure_specs import FIGURE_SPECS
from .manifest import Manifest, get_git_commit, get_software_versions
from .manuscript_exports import (
    create_manuscript_manifest,
    generate_all_figures,
    generate_all_tables,
)
from .tables import (
    export_calibration_table,
    export_comparison_table,
    export_depth_selection_table,
)

__all__ = [
    "FIGURE_SPECS",
    "Manifest",
    "get_git_commit",
    "get_software_versions",
    "export_depth_selection_table",
    "export_calibration_table",
    "export_comparison_table",
    "generate_all_figures",
    "generate_all_tables",
    "create_manuscript_manifest",
]
