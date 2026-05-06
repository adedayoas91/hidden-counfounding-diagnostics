"""Command-line runner for Monte Carlo experiment sweeps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .adapters import METHODS, load_external_method
from .metrics import summarize_run
from .simulations import SCENARIOS


def aggregate_results(results: list[dict[str, Any]], p_values: list[int]) -> dict[str, Any]:
    """Aggregate per-repeat summaries into manuscript-facing means."""

    t_obs_values = [float(result["summary"]["T_obs"]) for result in results]
    aggregate: dict[str, Any] = {
        "n_runs": len(results),
        "mean_T_obs": mean(t_obs_values) if t_obs_values else 0.0,
        "max_T_obs": max(t_obs_values) if t_obs_values else 0.0,
    }

    d_summary: dict[int, float] = {}
    for p_value in p_values[1:]:
        values = []
        for result in results:
            d_value = result["summary"]["D_p"].get(
                str(p_value),
                result["summary"]["D_p"].get(p_value),
            )
            if d_value is not None:
                values.append(float(d_value))
        d_summary[p_value] = mean(values) if values else 0.0
    aggregate["mean_D_p"] = d_summary

    if results and "metrics_by_p" in results[0]["summary"]:
        metric_names = ["accuracy", "precision", "recall", "fpr"]
        metrics_agg: dict[int, dict[str, float]] = {}
        for p_value in p_values:
            metrics_agg[p_value] = {}
            for metric in metric_names:
                values = []
                for result in results:
                    record = result["summary"]["metrics_by_p"].get(
                        str(p_value),
                        result["summary"]["metrics_by_p"].get(p_value),
                    )
                    if record is not None:
                        values.append(float(record[metric]))
                metrics_agg[p_value][metric] = mean(values) if values else 0.0
        aggregate["mean_metrics_by_p"] = metrics_agg

    return aggregate


def write_summary_csv(path: Path, results: list[dict[str, Any]], p_values: list[int]) -> None:
    """Write one row per repeat plus depth-specific instability summaries."""

    fieldnames = ["repeat", "scenario", "T_obs"]
    for p_value in p_values:
        fieldnames.append(f"edge_count_p{p_value}")
    for p_value in p_values[1:]:
        fieldnames.append(f"D_p{p_value}")

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "repeat": result["repeat"],
                "scenario": result["metadata"]["scenario"],
                "T_obs": result["summary"]["T_obs"],
            }
            for p_value in p_values:
                row[f"edge_count_p{p_value}"] = result["summary"]["edge_counts"].get(
                    str(p_value),
                    result["summary"]["edge_counts"].get(p_value),
                )
            for p_value in p_values[1:]:
                row[f"D_p{p_value}"] = result["summary"]["D_p"].get(
                    str(p_value),
                    result["summary"]["D_p"].get(p_value),
                )
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for experiment sweeps."""

    parser = argparse.ArgumentParser(description="Run synthetic experiment sweeps.")
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    parser.add_argument("--method", default="gcstar_cgc")
    parser.add_argument(
        "--user-method",
        default=None,
        help="Optional external method formatted as 'module:function'. Overrides --method.",
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--T", type=int, default=2000)
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--p-values", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conf-strength", type=float, default=0.35)
    parser.add_argument("--latent-ar", type=float, default=0.80)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--d-total", type=int, default=16)
    parser.add_argument("--d-observed", type=int, default=10)
    return parser.parse_args()


def scenario_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Build scenario-specific keyword arguments from parsed CLI options."""

    if args.scenario == "latent_common_driver":
        return {
            "T": args.T,
            "d": args.d,
            "conf_strength": args.conf_strength,
            "latent_ar": args.latent_ar,
            "noise_scale": args.noise_scale,
        }
    if args.scenario == "hidden_nodes":
        return {
            "T": args.T,
            "d_total": args.d_total,
            "d_observed": args.d_observed,
            "noise_scale": args.noise_scale,
        }
    return {"T": args.T, "d": args.d, "noise_scale": args.noise_scale}


def main() -> None:
    """Run a full experiment sweep and write JSON/CSV outputs."""

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_fn = SCENARIOS[args.scenario]
    if args.user_method:
        analyze_fn = load_external_method(args.user_method)
        method_name = args.user_method
    else:
        if args.method not in METHODS:
            raise ValueError(f"Unknown --method {args.method!r}. Known methods: {sorted(METHODS)}")
        analyze_fn = METHODS[args.method]
        method_name = args.method

    kwargs = scenario_kwargs(args)
    results: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        sample = scenario_fn(seed=repeat, **kwargs)
        adjacencies = analyze_fn(sample.X, args.p_values)
        summary = summarize_run(adjacencies, sample.ground_truth_compact)
        results.append(
            {
                "repeat": repeat,
                "metadata": sample.metadata,
                "summary": summary,
            }
        )

    aggregate = aggregate_results(results, args.p_values)
    config = {
        "scenario": args.scenario,
        "method": method_name,
        "repeats": args.repeats,
        "T": args.T,
        "d": args.d,
        "p_values": args.p_values,
        "scenario_kwargs": kwargs,
    }

    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    (output_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    write_summary_csv(output_dir / "summary.csv", results, args.p_values)

    print(f"Wrote outputs to {output_dir}")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
