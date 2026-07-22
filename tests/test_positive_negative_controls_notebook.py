from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np


def _source(cell: dict[str, object]) -> str:
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else str(source)


def test_positive_negative_controls_notebook_is_executable_and_current():
    root = Path(__file__).parents[1]
    path = root / "notebooks" / "controls" / "positive_negative_controls.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))

    code_sources = []
    for index, cell in enumerate(notebook["cells"]):
        source = _source(cell)
        if cell["cell_type"] == "code":
            ast.parse(source, filename=f"{path.name}:cell-{index}")
            code_sources.append(source)

    code = "\n".join(code_sources)
    assert code.count(
        "# Run synthetic positive controls (latent confounder and hidden nodes)"
    ) == 1
    assert code.count("# Load v2a-RSN biological data (c-GC method)") == 1
    assert "p_grid_max" not in code
    assert '"p_values": [1, 2, 3, 4, 5, 6, 7]' in code
    assert "load_and_filter_traces(recording_dir, RECORDING_NAME).T" in code
    assert "Using synthetic baseline" not in code
    assert "ANALYSIS_METHOD = 'fast_gcstar_cgc'" in code

    namespace: dict[str, object] = {}
    exec(code_sources[0], namespace)
    namespace["METHODS"] = {
        "fast_gcstar_cgc": lambda X, depths: {
            depth: np.zeros((X.shape[1], X.shape[1]), dtype=int)
            for depth in depths
        }
    }
    result = namespace["run_analysis_on_data"](
        np.zeros((40, 3)),
        [1, 2, 3, 4, 5, 6, 7],
        "smoke",
    )
    assert result["success"] is True
    assert result["T_obs"] == 0.0
    assert sorted(result["edge_counts"]) == [1, 2, 3, 4, 5, 6, 7]
