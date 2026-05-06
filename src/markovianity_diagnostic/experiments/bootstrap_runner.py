"""Command-line runner for surrogate-null bootstrap calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import METHODS, load_external_method
from .bootstrap import bootstrap_global_test
from .simulations import SCENARIOS


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for bootstrap calibration runs."""

    parser = argparse.ArgumentParser(description="Run surrogate-null bootstrap calibration.")
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    parser.add_argument("--method", default="gcstar_cgc")
    parser.add_argument("--user-method", default=None, help="Optional 'module:function' analyzer.")
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--p-values", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--p0", type=int, default=1)
    parser.add_argument("--B", type=int, default=100)
    parser.add_argument("--block-length", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conf-strength", type=float, default=0.35)
    parser.add_argument("--latent-ar", type=float, default=0.80)
    return parser.parse_args()


def scenario_kwargs(args: argparse.Namespace) -> dict:
    """Build scenario kwargs for bootstrap runs."""

    if args.scenario == "latent_common_driver":
        return {
            "T": args.T,
            "d": args.d,
            "conf_strength": args.conf_strength,
            "latent_ar": args.latent_ar,
            "seed": args.seed,
        }
    return {"T": args.T, "d": args.d, "seed": args.seed}


def main() -> None:
    """Run bootstrap calibration and write a JSON payload to disk."""

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.user_method:
        analyze_fn = load_external_method(args.user_method)
        method_name = args.user_method
    else:
        if args.method not in METHODS:
            raise ValueError(f"Unknown method {args.method!r}. Known methods: {sorted(METHODS)}")
        analyze_fn = METHODS[args.method]
        method_name = args.method

    sample = SCENARIOS[args.scenario](**scenario_kwargs(args))
    result = bootstrap_global_test(
        sample.X,
        analyze_fn=analyze_fn,
        p_values=args.p_values,
        p0=args.p0,
        B=args.B,
        block_length=args.block_length,
        seed=args.seed + 1000,
    )
    payload = {
        "config": {
            "scenario": args.scenario,
            "method": method_name,
            "T": args.T,
            "d": args.d,
            "p_values": args.p_values,
            "p0": args.p0,
            "B": args.B,
            "block_length": args.block_length,
            "seed": args.seed,
        },
        "scenario_metadata": sample.metadata,
        "bootstrap": result,
    }
    (output_dir / "bootstrap.json").write_text(json.dumps(payload, indent=2))

    print(f"Wrote bootstrap output to {output_dir / 'bootstrap.json'}")
    print(
        json.dumps(
            {key: result[key] for key in ["T_obs", "p_value", "critical_95"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
