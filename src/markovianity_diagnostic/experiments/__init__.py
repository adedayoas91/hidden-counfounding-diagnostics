"""Experiment harness utilities."""

from .adapters import METHODS, make_gcstar_analyzer
from .benchmark_data import (
    DEFAULT_LENGTH_RANGE,
    DEFAULT_N_VARS_RANGE,
    DEFAULT_SEED,
    VALID_SCENARIOS,
    TrialData,
    generate_trial_data,
)
from .parallel import (
    parallel_bootstrap,
    parallel_over_fish,
    parallel_simulation_grid,
)

__all__ = [
    "METHODS",
    "make_gcstar_analyzer",
    "parallel_bootstrap",
    "parallel_over_fish",
    "parallel_simulation_grid",
    "TrialData",
    "generate_trial_data",
    "DEFAULT_SEED",
    "DEFAULT_N_VARS_RANGE",
    "DEFAULT_LENGTH_RANGE",
    "VALID_SCENARIOS",
]
