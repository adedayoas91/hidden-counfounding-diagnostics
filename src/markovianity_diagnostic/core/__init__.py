"""Core estimator implementations."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .utils import adj_mtx, continuous_noise_fun


def _load_gcstar() -> type:
    """Load ``GcStar`` from the causalised module file."""

    module_path = Path(__file__).resolve().with_name("causalised-GC.py")
    spec = importlib.util.spec_from_file_location(
        "markovianity_diagnostic.core.causalised_gc",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load GcStar from {module_path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.GcStar


GcStar = _load_gcstar()

from .fast_causalised_GC import FastGcStar  # noqa: E402  (requires GcStar above)

__all__ = ["GcStar", "FastGcStar", "adj_mtx", "continuous_noise_fun"]