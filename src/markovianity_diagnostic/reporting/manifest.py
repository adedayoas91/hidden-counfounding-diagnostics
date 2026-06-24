"""Manifest generation for standardized output schema.

Every analysis output directory should contain a manifest.json file
with the following structure:
- created_at: ISO timestamp
- git_commit: short commit hash
- analysis: phase name / analysis type
- input_paths: list of input files/directories
- output_paths: list of output files created
- method: method name (if applicable)
- method_params: dict of method parameters
- p_values: list of conditioning depths tested (if applicable)
- random_seed: seed used for reproducibility (if applicable)
- software_versions: dict with python, tigramite, numpy versions
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def get_git_commit() -> str:
    """Get short git commit hash of current repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        logger.warning(f"Could not get git commit: {e}")
    return "unknown"


def get_software_versions() -> dict[str, str]:
    """Get versions of key dependencies."""
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }

    # Try to get tigramite version
    try:
        import tigramite
        versions["tigramite"] = tigramite.__version__
    except (ImportError, AttributeError):
        pass

    # Try to get scipy version
    try:
        import scipy
        versions["scipy"] = scipy.__version__
    except ImportError:
        pass

    # Try to get pandas version
    try:
        import pandas
        versions["pandas"] = pandas.__version__
    except ImportError:
        pass

    return versions


class Manifest:
    """Manifest for documenting analysis outputs."""

    def __init__(
        self,
        analysis: str,
        input_paths: list[str | Path] | None = None,
        output_paths: list[str | Path] | None = None,
        method: str | None = None,
        method_params: dict[str, Any] | None = None,
        p_values: list[int] | None = None,
        random_seed: int | None = None,
    ):
        """Initialize manifest.

        Args:
            analysis: Phase name or analysis type (e.g., "v2a-RSN c-GC")
            input_paths: List of input files/directories
            output_paths: List of output files created
            method: Method name (e.g., "c-GC", "c-GC-star")
            method_params: Dict of method parameters
            p_values: List of conditioning depths tested
            random_seed: Seed used for reproducibility
        """
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.git_commit = get_git_commit()
        self.analysis = analysis
        self.input_paths = [str(p) for p in (input_paths or [])]
        self.output_paths = [str(p) for p in (output_paths or [])]
        self.method = method
        self.method_params = method_params or {}
        self.p_values = p_values or []
        self.random_seed = random_seed
        self.software_versions = get_software_versions()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "created_at": self.created_at,
            "git_commit": self.git_commit,
            "analysis": self.analysis,
            "input_paths": self.input_paths,
            "output_paths": self.output_paths,
            "software_versions": self.software_versions,
        }

        if self.method:
            data["method"] = self.method
        if self.method_params:
            data["method_params"] = self.method_params
        if self.p_values:
            data["p_values"] = self.p_values
        if self.random_seed is not None:
            data["random_seed"] = self.random_seed

        return data

    def to_json(self, path: str | Path) -> None:
        """Save manifest to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Manifest saved to {path}")

    @staticmethod
    def from_json(path: str | Path) -> Manifest:
        """Load manifest from JSON file."""
        path = Path(path)

        with open(path, "r") as f:
            data = json.load(f)

        manifest = Manifest(
            analysis=data.get("analysis", "unknown"),
            input_paths=data.get("input_paths", []),
            output_paths=data.get("output_paths", []),
            method=data.get("method"),
            method_params=data.get("method_params"),
            p_values=data.get("p_values"),
            random_seed=data.get("random_seed"),
        )

        # Restore the original created_at and git_commit
        manifest.created_at = data.get("created_at", manifest.created_at)
        manifest.git_commit = data.get("git_commit", manifest.git_commit)
        manifest.software_versions = data.get("software_versions", manifest.software_versions)

        return manifest


__all__ = ["Manifest", "get_git_commit", "get_software_versions"]
