"""Notebook-facing runner for extension-compatible simulation artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def project_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "src" / "markovianity_diagnostic").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=[
            "order1_unconfounded",
            "order3_unconfounded",
            "latent_common_driver",
            "variable_lag_unconfounded",
        ],
    )
    parser.add_argument("--p-values", nargs="+", type=int, default=list(range(1, 8)))
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--T", type=int, default=2000)
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    sys.path.insert(0, str(root / "src"))
    from markovianity_diagnostic.experiments.simulation_extensions import (
        run_extension_simulations,
    )

    result = run_extension_simulations(
        method=args.method,
        scenario_names=args.scenarios,
        p_values=args.p_values,
        n_repeats=args.n_repeats,
        T=args.T,
        d=args.d,
        seed=args.seed,
        output_dir=(
            root / "outputs" / "simulations" / "extension_metrics" / args.method
        ),
    )
    print(result)


if __name__ == "__main__":
    main()
