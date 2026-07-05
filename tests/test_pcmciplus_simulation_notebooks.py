"""Contracts for the retained exploratory PCMCI+ simulation notebooks."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


SCENARIOS = (
    "singleLag-Markovian",
    "singleLag-NonMarkovian",
    "varLags-Markovian",
    "varLags-NonMarkovian",
)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_pcmciplus_simulation_notebook_uses_conditioning_depth(scenario):
    notebook_path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "simulations"
        / "pcmciplus"
        / f"pcmci_plus_{scenario}.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_source = "\n".join(
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    markdown_source = "\n".join(
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )

    assert "P_VALUES = [1, 2, 3, 4, 5, 6, 7]" in code_source
    assert "FIXED_LAG = 1" in code_source
    assert "analyze_with_pcmciplus" in code_source
    assert "tau_max=p" not in code_source
    assert "TAU_RANGE" not in code_source
    assert "'p': p_value" in code_source
    assert "maximum conditioning-set size" in markdown_source
    assert "maximum lag" not in markdown_source

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = (
            "".join(cell["source"])
            if isinstance(cell["source"], list)
            else cell["source"]
        )
        ast.parse(source, filename=f"{notebook_path.name}:cell-{index}")


def test_jpcmciplus_simulation_notebooks_are_removed():
    simulations_root = Path(__file__).parents[1] / "notebooks" / "simulations"

    assert not (simulations_root / "jpcmciplus").exists()
