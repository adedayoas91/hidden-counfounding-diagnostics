"""Contracts for fixed-lag v2a-RSN PCMCI+ recording notebooks."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


RECORDINGS = (
    "220119_F2_run11",
    "220127_F4_run2",
    "220210_F1_run6",
    "220210_F2_run5",
)


@pytest.mark.parametrize("recording", RECORDINGS)
def test_v2a_pcmciplus_notebook_uses_shared_profile(recording):
    method_dir = "pcmciplus"
    notebook_path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "v2a-RSNs"
        / method_dir
        / f"{recording}.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_source = "\n".join(
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert f"RECORDING_NAME = '{recording}'" in code_source
    assert f"METHOD_DIR = '{method_dir}'" in code_source
    assert "ANALYZER = analyze_with_pcmciplus" in code_source
    assert "P_VALUES = [1, 2, 3, 4, 5, 6, 7]" in code_source
    assert "'tau_min': 1" in code_source
    assert "'tau_max': 1" in code_source
    assert "'depth_parameter': 'maximum_conditioning_set_size'" in code_source
    assert "'max_conds_dim'" in code_source
    assert "'max_conds_py'" in code_source
    assert "'max_conds_px'" in code_source
    assert "'max_conds_px_lagged'" in code_source
    assert "analysis_profile=V2A_ANALYSIS_PROFILE" in code_source
    assert "load_and_filter_traces(recording_dir, RECORDING_NAME)" in code_source
    assert "get_v2a_selection_metadata(recording_dir, RECORDING_NAME)" in code_source
    assert "X_time_by_neurons = X_neurons_by_frames.T" in code_source
    assert "ANALYZER(X_time_by_neurons, [p_value], pc_alpha=PC_ALPHA)" in code_source
    assert "'analysis_profile': V2A_ANALYSIS_PROFILE" in code_source
    assert "'trace_selection': trace_selection" in code_source
    assert "Existing checkpoint trace selection does not match" in code_source
    assert "Existing checkpoint Tigramite parameters do not match" in code_source

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


def test_v2a_jpcmciplus_notebooks_are_removed():
    notebooks_root = Path(__file__).parents[1] / "notebooks" / "v2a-RSNs"

    assert not (notebooks_root / "jpcmciplus").exists()
