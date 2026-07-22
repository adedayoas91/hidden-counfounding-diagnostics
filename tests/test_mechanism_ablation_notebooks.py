from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


NOTEBOOK_SCENARIOS = {
    "mechanism_ablations_hidden_state.ipynb": (
        "latent_confounder",
        "hidden_nodes",
        "omitted_lag_order",
    ),
    "mechanism_ablations_nonstationarity.ipynb": (
        "time_varying_coefficients",
        "regime_shift",
    ),
    "mechanism_ablations_observation_artifacts.ipynb": (
        "measurement_noise",
        "undersampling",
    ),
}


def _source(cell: dict[str, object]) -> str:
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else str(source)


def _load_notebook(name: str) -> dict[str, object]:
    root = Path(__file__).parents[1]
    path = root / "notebooks" / "simulations" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _fake_scenario(**kwargs: object) -> SimpleNamespace:
    del kwargs
    return SimpleNamespace(
        X=np.zeros((20, 3)),
        ground_truth_compact=np.zeros((3, 3), dtype=int),
        metadata={"fake": True},
    )


def _fake_summary(
    adjacencies: dict[int, np.ndarray],
    truth: np.ndarray,
) -> dict[str, object]:
    del truth
    depths = sorted(adjacencies)
    return {
        "T_obs": 0.0,
        "D_p": {depth: 0.0 for depth in depths[1:]},
    }


def _namespace(
    tmp_path: Path,
    scenarios: tuple[str, ...],
    *,
    outputs_exist: bool,
) -> dict[str, object]:
    namespace: dict[str, object] = {
        "outputs_exist": outputs_exist,
        "logger": logging.getLogger("mechanism-notebook-test"),
        "time": __import__("time"),
        "np": np,
        "pd": pd,
        "plt": plt,
        "json": json,
        "datetime": datetime,
        "timezone": timezone,
        "OUTPUT_DIR": tmp_path,
        "FIGURES_DIR": tmp_path / "figures",
        "P_VALUES": [1, 2],
        "REPEATS": 1,
        "SEED": 42,
        "METHODS_TO_TEST": ["gcstar_cgc", "gcstar_fcgc"],
        "SCENARIOS": [(scenario, {}) for scenario in scenarios],
        "METHODS": {
            "gcstar_cgc": lambda X, p: {
                depth: np.zeros((X.shape[1], X.shape[1]), dtype=int)
                for depth in p
            },
            "gcstar_fcgc": lambda X, p: {
                depth: np.zeros((X.shape[1], X.shape[1]), dtype=int)
                for depth in p
            },
        },
        "summarize_run": _fake_summary,
    }
    namespace["FIGURES_DIR"].mkdir(parents=True, exist_ok=True)
    function_names = {
        "latent_confounder": "scenario_latent_common_driver",
        "hidden_nodes": "scenario_hidden_nodes",
        "omitted_lag_order": "scenario_omitted_lag_order",
        "time_varying_coefficients": "scenario_time_varying_coefficients",
        "regime_shift": "scenario_regime_shift",
        "measurement_noise": "scenario_measurement_noise",
        "undersampling": "scenario_undersampled_markov",
    }
    for scenario in scenarios:
        namespace[function_names[scenario]] = _fake_scenario
    return namespace


def test_mechanism_notebooks_have_executable_fresh_and_cached_paths(tmp_path):
    for name, scenarios in NOTEBOOK_SCENARIOS.items():
        notebook = _load_notebook(name)
        cells = notebook["cells"]
        assert cells[5]["cell_type"] == "code"
        assert cells[8]["cell_type"] == "code"

        code = []
        for index, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            source = _source(cell)
            ast.parse(source, filename=f"{name}:cell-{index}")
            code.append(source)

        joined = "\n".join(code)
        assert "P_VALUES = [1, 2, 3, 4, 5, 6, 7]" in joined
        assert "ground_truth_compact=" not in joined
        assert "p_values=P_VALUES" not in joined
        assert ".get(str(p_val)" not in joined
        assert joined.count("summary_df.to_csv") == 1

        output = tmp_path / name.removesuffix(".ipynb")
        output.mkdir()
        fresh = _namespace(output, scenarios, outputs_exist=False)
        for index in (5, 6, 7, 8, 9, 10, 12):
            exec(_source(cells[index]), fresh)
        assert (output / "results.csv").exists()
        assert (output / "manifest.json").exists()
        assert (output / "figures" / "dp_trajectories.png").exists()

        cached = _namespace(output, scenarios, outputs_exist=True)
        for index in (5, 6, 7, 8, 9, 10, 12):
            exec(_source(cells[index]), cached)
        assert isinstance(cached["summary_df"], pd.DataFrame)
