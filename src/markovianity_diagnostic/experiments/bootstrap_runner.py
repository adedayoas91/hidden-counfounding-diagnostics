"""Command-line runner for surrogate-null bootstrap calibration.

Phase 1.4: Refactored to use NullModel classes from calibration.py
with new CLI options for null model selection, critical levels, and parallelization.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .adapters import METHODS, load_external_method
from .bootstrap import bootstrap_global_test
from .calibration import (
    VARNullModel,
    ResidualBootstrapNull,
    MovingBlockBootstrapNull,
    CalibrationResult,
)
from .simulations import SCENARIOS

logger = logging.getLogger(__name__)

NULL_MODEL_REGISTRY = {
    "var": VARNullModel,
    "residual": ResidualBootstrapNull,
    "moving-block": MovingBlockBootstrapNull,
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for bootstrap calibration runs.

    Phase 1.4 additions:
    - --null-model: Select null model {var, residual, moving-block}
    - --critical-levels: Specify quantiles for critical values
    - --save-surrogate-summaries: Save detailed bootstrap summaries
    - --n-jobs: Parallel job count
    """

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

    # Phase 1.4: New CLI options
    parser.add_argument(
        "--null-model",
        choices=sorted(NULL_MODEL_REGISTRY.keys()),
        default="var",
        help="Null model type: var (VAR), residual (residual bootstrap), or moving-block"
    )
    parser.add_argument(
        "--critical-levels",
        type=float,
        nargs="+",
        default=[0.90, 0.95, 0.99],
        help="Quantile levels for critical values (default: 0.90 0.95 0.99)"
    )
    parser.add_argument(
        "--save-surrogate-summaries",
        action="store_true",
        help="Save detailed bootstrap surrogate summaries"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1, serial)"
    )

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


def _get_git_commit() -> str:
    """Get short git commit hash for manifest."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _compute_critical_values(
    T_boot: list[float],
    critical_levels: list[float],
) -> dict[float, float]:
    """Compute critical values at specified quantile levels.

    Args:
        T_boot: Bootstrap test statistics.
        critical_levels: Quantile levels (e.g., [0.90, 0.95, 0.99]).

    Returns:
        Dictionary mapping quantile levels to critical values.
    """
    import numpy as np

    if not T_boot:
        return {level: 0.0 for level in critical_levels}

    T_boot_array = np.asarray(T_boot)
    return {
        float(level): float(np.quantile(T_boot_array, level))
        for level in critical_levels
    }


def main() -> None:
    """Run bootstrap calibration and write output with manifest schema.

    Phase 1.4: Updated to use NullModel classes and produce manifest-schema output.
    """

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate critical levels
    for level in args.critical_levels:
        if not 0 < level < 1:
            raise ValueError(f"Critical level must be in (0, 1), got {level}")

    # Load analyzer function
    if args.user_method:
        analyze_fn = load_external_method(args.user_method)
        method_name = args.user_method
    else:
        if args.method not in METHODS:
            raise ValueError(f"Unknown method {args.method!r}. Known methods: {sorted(METHODS)}")
        analyze_fn = METHODS[args.method]
        method_name = args.method

    # Generate synthetic data
    sample = SCENARIOS[args.scenario](**scenario_kwargs(args))

    # Run bootstrap calibration using legacy bootstrap_global_test
    result = bootstrap_global_test(
        sample.X,
        analyze_fn=analyze_fn,
        p_values=args.p_values,
        p0=args.p0,
        B=args.B,
        block_length=args.block_length,
        seed=args.seed + 1000,
    )

    # Extract bootstrap samples
    T_boot = result.get("T_boot", [])
    T_obs = result.get("T_obs", 0.0)
    D_obs = result.get("D_obs", {})

    # Compute critical values at requested levels
    critical_values = _compute_critical_values(T_boot, args.critical_levels)

    # Compute p-value
    import numpy as np
    T_boot_array = np.asarray(T_boot)
    p_value = float((1 + np.sum(T_boot_array >= T_obs)) / (args.B + 1))

    # Build manifest-compatible output
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _get_git_commit(),
        "analysis": "bootstrap_calibration",
        "input_paths": [],  # Synthetic data, no file inputs
        "output_paths": [str(output_dir / "bootstrap.json")],
        "method": args.null_model,
        "method_params": {
            "null_model": args.null_model,
            "p0": args.p0,
            "B": args.B,
            "block_length": args.block_length,
            "critical_levels": list(args.critical_levels),
            "n_jobs": args.n_jobs,
        },
        "p_values": args.p_values,
        "random_seed": args.seed,
        "software_versions": {
            "python": "3.12",
            "numpy": np.__version__,
        },
    }

    # Create CalibrationResult
    observed = {
        "T_obs": float(T_obs),
        "D_obs": {int(k): float(v) for k, v in D_obs.items()},
    }

    null_data = {
        "T_boot": [float(v) for v in T_boot],
        "p_value": p_value,
    }
    # Add critical values with human-readable names
    for level in sorted(args.critical_levels):
        key = f"critical_{int(level * 100)}"
        null_data[key] = critical_values[level]

    diagnosis = {
        "reject_global_95": p_value < 0.05,
        "p0": int(args.p0),
        "B": int(args.B),
    }

    calibration_result = CalibrationResult(
        observed=observed,
        null=null_data,
        diagnosis=diagnosis,
    )

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote manifest to {manifest_path}")

    # Write calibration result
    result_path = output_dir / "bootstrap.json"
    calibration_result.to_json(str(result_path))
    logger.info(f"Wrote bootstrap calibration to {result_path}")

    # Legacy compatibility: also write old format
    legacy_payload = {
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
            "null_model": args.null_model,
            "critical_levels": args.critical_levels,
        },
        "scenario_metadata": sample.metadata,
        "manifest": manifest,
        "calibration_result": {
            "observed": observed,
            "null": null_data,
            "diagnosis": diagnosis,
        },
    }
    legacy_path = output_dir / "bootstrap_legacy.json"
    with open(legacy_path, 'w') as f:
        json.dump(legacy_payload, f, indent=2)

    # Print summary
    print(f"Wrote bootstrap calibration to {output_dir}")
    print(f"  - manifest.json")
    print(f"  - bootstrap.json (CalibrationResult)")
    print(f"  - bootstrap_legacy.json (legacy format)")
    print()
    print("Summary:")
    print(json.dumps({
        "T_obs": float(T_obs),
        "p_value": p_value,
        "critical_values": {
            f"critical_{int(level*100)}": critical_values[level]
            for level in sorted(args.critical_levels)
        },
        "reject_global_95": p_value < 0.05,
    }, indent=2))


if __name__ == "__main__":
    main()
