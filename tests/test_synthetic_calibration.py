"""Contracts for resumable synthetic bootstrap calibration."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from markovianity_diagnostic.experiments import synthetic_calibration


def test_synthetic_calibration_reuses_compatible_replicates(
    tmp_path,
    monkeypatch,
):
    def scenario(*, T, d, seed):
        rng = np.random.default_rng(seed)
        return SimpleNamespace(
            X=rng.normal(size=(T, d)),
            metadata={"scenario": "test"},
        )

    def analyzer(X, p_values):
        output = {}
        for depth in p_values:
            adjacency = np.zeros((X.shape[1], X.shape[1]), dtype=int)
            adjacency[0, 1] = int(depth % 2 == 0)
            output[int(depth)] = adjacency
        return output

    monkeypatch.setitem(
        synthetic_calibration.SYNTHETIC_SCENARIOS,
        "order1_unconfounded",
        scenario,
    )
    monkeypatch.setitem(
        synthetic_calibration.ELIGIBLE_METHODS,
        "c-GC",
        "test_cgc",
    )
    monkeypatch.setitem(synthetic_calibration.METHODS, "test_cgc", analyzer)
    monkeypatch.setitem(
        synthetic_calibration.METHOD_METADATA,
        "test_cgc",
        {"implementation": "test"},
    )

    common = {
        "project_root": tmp_path,
        "output_dir": tmp_path / "calibration",
        "method_names": ["c-GC"],
        "scenario_names": ["order1_unconfounded"],
        "p_values": [1, 2, 3],
        "p0": 1,
        "block_length": 1,
        "seed": 9,
        "T": 30,
        "d": 3,
        "n_jobs": 1,
        "show_progress": False,
    }
    first = synthetic_calibration.run_synthetic_calibration(B=2, **common)
    second = synthetic_calibration.run_synthetic_calibration(B=3, **common)

    assert first["summary"].iloc[0]["reused_replicates"] == 0
    assert second["summary"].iloc[0]["reused_replicates"] == 2
    assert len(
        list(
            (
                tmp_path
                / "calibration"
                / "checkpoints"
                / "c-GC"
                / "order1_unconfounded"
            ).glob("replicate_*.pkl")
        )
    ) == 3
    assert set(second["summary"]["B"]) == {3}
    assert second["summary_path"] == (
        tmp_path / "calibration" / "c-GC" / "bootstrap_summary.csv"
    )


@pytest.mark.parametrize(
    ("filename", "method_name"),
    [
        ("bootstrap_null_synthetic_c-GC.ipynb", "c-GC"),
        ("bootstrap_null_synthetic_c-GC-star.ipynb", "c-GC-star"),
    ],
)
def test_synthetic_bootstrap_notebook_uses_matched_resumable_contract(
    filename,
    method_name,
):
    notebook_path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "calibration"
        / filename
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        if isinstance(cell["source"], list)
        else cell["source"]
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    markdown = "\n".join(
        "".join(cell["source"])
        if isinstance(cell["source"], list)
        else cell["source"]
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )

    assert "B = 5" in code
    assert "P_VALUES = [1, 2, 3, 4, 5, 6, 7]" in code
    assert f"METHODS_TO_RUN = ['{method_name}']" in code
    assert "run_synthetic_calibration" in code
    assert "SHOW_PROGRESS = True" in code
    assert "show_progress=SHOW_PROGRESS" in code
    assert "pcmciplus" not in code.lower()
    assert "target cumulative replicate count" in markdown.lower()
    assert "bootstrap replicates" in markdown
    assert "active conditioning depths" in markdown
    assert "not prerequisites" in markdown

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = (
            "".join(cell["source"])
            if isinstance(cell["source"], list)
            else cell["source"]
        )
        ast.parse(source, filename=f"{notebook_path.name}:cell-{index}")
        assert cell["execution_count"] is None
        assert cell["outputs"] == []


def test_combined_synthetic_bootstrap_notebook_is_removed():
    calibration_dir = Path(__file__).parents[1] / "notebooks" / "calibration"

    assert not (calibration_dir / "bootstrap_null_synthetic.ipynb").exists()
