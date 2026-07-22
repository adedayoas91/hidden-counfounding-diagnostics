"""Contract checks for the comparable three-method v2a-RSN notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path


def test_compare_all_methods_notebook_contract():
    notebooks_root = Path(__file__).parents[1] / "notebooks" / "v2a-RSNs"
    notebook_path = notebooks_root / "compare_all_methods.ipynb"

    assert notebook_path.exists()
    assert not (notebooks_root / "compare_c-GC_methods.ipynb").exists()

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_source = "\n".join(
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    for method_dir in ("c-GC", "c-GC-star", "pcmciplus"):
        assert f"'directory': '{method_dir}'" in code_source
    assert "jpcmciplus" not in code_source.lower()

    assert "P_VALUES = [1, 2, 3, 4, 5, 6, 7]" in code_source
    assert "EXPECTED_RECORDINGS" in code_source
    assert "All methods must contain the same recordings" in code_source
    assert "All methods must contain P=1,...,7" in code_source
    assert "'conditioning-set depth'" in code_source
    assert "'maximum conditioning-set size at fixed lag 1'" in code_source
    assert "method_comparison_summary.csv" in code_source
    assert "edge_count_comparison.png" in code_source
    assert "instability_comparison.png" in code_source
    assert "manifest.json" in code_source

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
