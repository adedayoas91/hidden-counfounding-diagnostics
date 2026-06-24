"""
LPCMCI and SVAR-FCI Comparison Notebook

Compares LPCMCI and c-GC methods on synthetic and reduced real data scenarios.
Evaluates edge overlap, latent confounder marks, and method agreement.

Output directory: outputs/comparisons/lpcmci_svarfci/
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from markovianity_diagnostic.experiments.adapters import make_gcstar_analyzer
from markovianity_diagnostic.experiments.simulations import (
    scenario_order1_unconfounded,
    scenario_latent_common_driver,
)
from markovianity_diagnostic.methods.lpcmci_adapter import LPCMCIAdapter


def setup_output_dir() -> Path:
    """Create and return output directory."""
    output_dir = Path("outputs/comparisons/lpcmci_svarfci")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def compare_methods_on_scenario(
    scenario_name: str,
    X: np.ndarray,
    p_values: list[int],
) -> dict:
    """Run c-GC, c-GC*, and LPCMCI on a scenario and compare."""

    results = {
        "scenario": scenario_name,
        "n_vars": X.shape[1],
        "n_timepoints": X.shape[0],
        "p_values": p_values,
    }

    # Run c-GC
    try:
        cgc_analyzer = make_gcstar_analyzer("cgc", n_perm=100)
        cgc_results = cgc_analyzer(X, p_values)
        results["cgc"] = {
            "success": True,
            "adjacency_by_p": {p: cgc_results[p].tolist() for p in p_values},
            "edge_counts": {p: int(np.sum(cgc_results[p])) for p in p_values},
        }
    except Exception as e:
        results["cgc"] = {
            "success": False,
            "error": str(e),
        }

    # Run c-GC*
    try:
        fcgc_analyzer = make_gcstar_analyzer("fcgc", n_perm=100)
        fcgc_results = fcgc_analyzer(X, p_values)
        results["fcgc"] = {
            "success": True,
            "adjacency_by_p": {p: fcgc_results[p].tolist() for p in p_values},
            "edge_counts": {p: int(np.sum(fcgc_results[p])) for p in p_values},
        }
    except Exception as e:
        results["fcgc"] = {
            "success": False,
            "error": str(e),
        }

    # Run LPCMCI
    try:
        lpcmci_adapter = LPCMCIAdapter(tau_min=1, tau_max=5, pc_alpha=0.05)
        lpcmci_results = lpcmci_adapter.fit(X, p_values)
        results["lpcmci"] = {
            "success": True,
            "adjacency_by_p": {
                p: lpcmci_results[p]["adjacency"].tolist() for p in p_values
            },
            "edge_marks_by_p": {
                p: {str(k): v for k, v in lpcmci_results[p]["edge_marks"].items()}
                for p in p_values
            },
            "edge_counts": {p: int(np.sum(lpcmci_results[p]["adjacency"])) for p in p_values},
            "metadata": {p: lpcmci_results[p]["metadata"] for p in p_values},
        }
    except Exception as e:
        results["lpcmci"] = {
            "success": False,
            "error": str(e),
        }

    return results


def compute_edge_overlap(adj1: np.ndarray, adj2: np.ndarray) -> dict:
    """Compute Jaccard overlap between two adjacency matrices."""
    union = np.logical_or(adj1, adj2).sum()
    intersection = np.logical_and(adj1, adj2).sum()

    if union == 0:
        jaccard = 1.0 if intersection == 0 else 0.0
    else:
        jaccard = intersection / union

    return {
        "jaccard": float(jaccard),
        "intersection": int(intersection),
        "union": int(union),
    }


def create_comparison_table(results: dict) -> pd.DataFrame:
    """Create summary table comparing methods."""

    rows = []

    for scenario_name, scenario_results in results.items():
        for p_value in scenario_results.get("p_values", []):
            row = {
                "scenario": scenario_name,
                "p_value": p_value,
            }

            # CGC metrics
            if scenario_results["cgc"]["success"]:
                cgc_adj = np.array(
                    scenario_results["cgc"]["adjacency_by_p"][p_value],
                    dtype=int,
                )
                row["cgc_edges"] = int(np.sum(cgc_adj))
            else:
                row["cgc_edges"] = None

            # FCGC metrics
            if scenario_results["fcgc"]["success"]:
                fcgc_adj = np.array(
                    scenario_results["fcgc"]["adjacency_by_p"][p_value],
                    dtype=int,
                )
                row["fcgc_edges"] = int(np.sum(fcgc_adj))
            else:
                row["fcgc_edges"] = None

            # LPCMCI metrics
            if scenario_results["lpcmci"]["success"]:
                lpcmci_adj = np.array(
                    scenario_results["lpcmci"]["adjacency_by_p"][p_value],
                    dtype=int,
                )
                row["lpcmci_edges"] = int(np.sum(lpcmci_adj))

                # Compute overlaps
                if scenario_results["cgc"]["success"]:
                    overlap_cgc_lpcmci = compute_edge_overlap(cgc_adj, lpcmci_adj)
                    row["cgc_lpcmci_jaccard"] = overlap_cgc_lpcmci["jaccard"]

                if scenario_results["fcgc"]["success"]:
                    overlap_fcgc_lpcmci = compute_edge_overlap(fcgc_adj, lpcmci_adj)
                    row["fcgc_lpcmci_jaccard"] = overlap_fcgc_lpcmci["jaccard"]
            else:
                row["lpcmci_edges"] = None

            rows.append(row)

    return pd.DataFrame(rows)


def main():
    """Run comparison analysis."""

    # Setup
    output_dir = setup_output_dir()
    print(f"Output directory: {output_dir}")

    # Configuration
    p_values = [1, 2, 3, 4, 5]
    np.random.seed(42)

    # Scenarios to compare
    scenarios = {}

    print("Generating scenarios...")
    # Scenario 1: Order-1 Markov (null)
    try:
        scenario_null = scenario_order1_unconfounded(T=500, d=5, seed=42)
        scenarios["order1_unconfounded"] = scenario_null.X
        print("✓ order1_unconfounded")
    except Exception as e:
        print(f"✗ order1_unconfounded: {e}")

    # Scenario 2: Latent common driver (positive control)
    try:
        scenario_latent = scenario_latent_common_driver(T=500, d=5, seed=42)
        scenarios["latent_common_driver"] = scenario_latent.X
        print("✓ latent_common_driver")
    except Exception as e:
        print(f"✗ latent_common_driver: {e}")

    # Run comparisons
    print("\nRunning method comparisons...")
    all_results = {}

    for scenario_name, X in scenarios.items():
        print(f"  {scenario_name}... ", end="", flush=True)
        try:
            result = compare_methods_on_scenario(scenario_name, X, p_values)
            all_results[scenario_name] = result
            print("done")
        except Exception as e:
            print(f"error: {e}")

    # Create summary table
    print("\nCreating summary table...")
    summary_df = create_comparison_table(all_results)
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    # Save detailed results
    results_path = output_dir / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: {results_path}")

    # Create manifest
    manifest = {
        "created_at": datetime.now().isoformat(),
        "analysis": "LPCMCI and SVAR-FCI comparison",
        "scenarios": list(scenarios.keys()),
        "p_values": p_values,
        "methods": ["cgc", "fcgc", "lpcmci"],
        "output_paths": [str(summary_path), str(results_path)],
        "sample_size": 500,
        "n_variables": 5,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved: {manifest_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    results = main()
    print("\nComparison complete!")
