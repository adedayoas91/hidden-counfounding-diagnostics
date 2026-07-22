"""Regression tests for extension-plan requirements missed by initial tests."""

from __future__ import annotations

import json

import numpy as np

from markovianity_diagnostic.experiments.calibration import (
    StationaryBootstrapNull,
)
from markovianity_diagnostic.experiments.plotting import (
    plot_D_p_with_bands,
    plot_T_boot_histogram,
    plot_edge_count_trajectory,
)
from markovianity_diagnostic.experiments.nonlinear_tests import (
    NonlinearResidualCorrelationAdapter,
)
from markovianity_diagnostic.reporting.tables import export_calibration_table


def test_stationary_bootstrap_is_reproducible(synthetic_markov_data):
    model = StationaryBootstrapNull(block_length=5).fit(
        synthetic_markov_data, p0=1
    )
    first = model.sample(100, seed=7)
    second = model.sample(100, seed=7)
    np.testing.assert_array_equal(first, second)
    assert model.metadata["model"] == "StationaryBootstrap"


def test_calibration_table_reads_calibration_result_schema(tmp_path):
    path = tmp_path / "bootstrap.json"
    path.write_text(
        json.dumps(
            {
                "observed": {"T_obs": 0.2},
                "null": {
                    "T_boot": [0.1, 0.2],
                    "critical_90": 0.15,
                    "critical_95": 0.18,
                    "critical_99": 0.2,
                    "p_value": 0.5,
                },
                "diagnosis": {"reject_global_95": False},
            }
        )
    )
    table = export_calibration_table(str(path))
    assert len(table) == 1
    assert table.iloc[0]["n_bootstrap"] == 2
    assert table.iloc[0]["critical_95"] == 0.18


def test_calibration_plotting_functions_write_pngs(tmp_path):
    histogram = plot_T_boot_histogram(
        [0.1, 0.2, 0.3],
        0.25,
        {0.95: 0.29},
        tmp_path / "hist.png",
    )
    bands = plot_D_p_with_bands(
        {2: 0.2, 3: 0.1},
        {
            2: {"lower": 0.05, "upper": 0.25},
            3: {"lower": 0.02, "upper": 0.15},
        },
        [2, 3],
        tmp_path / "bands.png",
    )
    counts = plot_edge_count_trajectory(
        {1: 2, 2: 3},
        {1: [1, 2, 3], 2: [2, 3, 4]},
        tmp_path / "counts.png",
    )
    for path in (histogram, bands, counts):
        assert path.read_bytes().startswith(b"\x89PNG")


def test_dependency_light_nonlinear_ci_test_runs():
    rng = np.random.default_rng(42)
    z = rng.normal(size=120)
    x = z**2 + rng.normal(scale=0.1, size=120)
    y = z**2 + rng.normal(scale=0.1, size=120)
    array = np.vstack([x, y, z])
    xyz = np.array([0, 1, 2])
    test = NonlinearResidualCorrelationAdapter().build()
    value = test.get_dependence_measure(array, xyz)
    assert np.isfinite(value)
