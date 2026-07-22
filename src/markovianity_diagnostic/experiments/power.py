"""Power analysis for sample size and parameter variation effects.

This module implements power analysis that studies how detection performance
varies across different sample sizes (T), dimensions (d), and other parameters
like edge density and confounder strength.

Key output metrics:
- TPR (True Positive Rate): Proportion of true edges correctly detected
- FPR (False Positive Rate): Proportion of null edges incorrectly detected
- p_star: Selected conditioning depth where instability stabilizes
- runtime_sec: Wall-clock execution time
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import numpy as np

from .adapters import METHODS
from .graph_metrics import compute_graph_instability

logger = logging.getLogger(__name__)


@dataclass
class PowerGrid:
    """Specification of parameter grid for power analysis.

    Parameters
    ----------
    T_values : list[int]
        Sample sizes to test.
    d_values : list[int]
        Dimensionalities to test.
    edge_densities : list[float], optional
        Edge probability for sparse matrix generation. Default: [0.12]
    confounder_strengths : list[float], optional
        Strength of latent confounding. Default: [0.35]
    latent_autocorrs : list[float], optional
        Autocorrelation of latent AR process. Default: [0.80]
    noise_scales : list[float], optional
        Noise standard deviation. Default: [1.0]
    """

    T_values: list[int]
    d_values: list[int]
    edge_densities: list[float] = field(default_factory=lambda: [0.12])
    confounder_strengths: list[float] = field(default_factory=lambda: [0.35])
    latent_autocorrs: list[float] = field(default_factory=lambda: [0.80])
    noise_scales: list[float] = field(default_factory=lambda: [1.0])

    def total_combinations(self) -> int:
        """Return total number of grid combinations."""
        return (
            len(self.T_values)
            * len(self.d_values)
            * len(self.edge_densities)
            * len(self.confounder_strengths)
            * len(self.latent_autocorrs)
            * len(self.noise_scales)
        )


@dataclass
class PowerResult:
    """Single power analysis result (one grid point, one repeat).

    Parameters
    ----------
    T : int
        Sample size.
    d : int
        Dimensionality.
    edge_density : float
        Edge probability used.
    confounder_strength : float
        Confounder strength used.
    latent_autocorr : float
        Latent autocorr used.
    noise_scale : float
        Noise scale used.
    repeat : int
        Repeat index.
    tpr : float
        True positive rate (0 to 1).
    fpr : float
        False positive rate (0 to 1).
    p_star : int
        Selected conditioning depth (≥1).
    runtime_sec : float
        Wall-clock execution time in seconds.
    """

    T: int
    d: int
    edge_density: float
    confounder_strength: float
    latent_autocorr: float
    noise_scale: float
    repeat: int
    tpr: float
    fpr: float
    p_star: int
    runtime_sec: float


@dataclass
class MetricAggregation:
    """Aggregated metrics over repeats for a single grid point.

    Parameters
    ----------
    T : int
        Sample size.
    d : int
        Dimensionality.
    edge_density : float
        Edge probability used.
    confounder_strength : float
        Confounder strength used.
    latent_autocorr : float
        Latent autocorr used.
    noise_scale : float
        Noise scale used.
    n_repeats : int
        Number of repeats aggregated.
    mean_tpr : float
        Mean TPR over repeats.
    std_tpr : float
        Standard deviation of TPR.
    mean_fpr : float
        Mean FPR over repeats.
    std_fpr : float
        Standard deviation of FPR.
    mean_p_star : float
        Mean selected depth over repeats.
    std_p_star : float
        Standard deviation of selected depth.
    mean_runtime : float
        Mean runtime over repeats.
    """

    T: int
    d: int
    edge_density: float
    confounder_strength: float
    latent_autocorr: float
    noise_scale: float
    n_repeats: int
    mean_tpr: float
    std_tpr: float
    mean_fpr: float
    std_fpr: float
    mean_p_star: float
    std_p_star: float
    mean_runtime: float


class PowerAnalyzer:
    """Orchestrate power analysis across parameter grids.

    Parameters
    ----------
    grid : PowerGrid
        Parameter grid specification.
    method : str
        Causal inference method (e.g., "gcstar_cgc").
    repeats : int, optional
        Number of repeats per grid point. Default: 10
    p_values : list[int], optional
        Conditioning depths to evaluate. Default: [1,2,3,4,5,6]
    verbose : bool, optional
        Enable progress logging. Default: False
    seed : int, optional
        Random seed for reproducibility. Default: None
    """

    def __init__(
        self,
        grid: PowerGrid,
        method: str,
        repeats: int = 10,
        p_values: list[int] | None = None,
        verbose: bool = False,
        seed: int | None = None,
        method_fn: Callable[[np.ndarray, list[int]], dict[int, np.ndarray]] | None = None,
        detection_threshold: float = 0.1,
    ):
        """Initialize power analyzer."""
        self.grid = grid
        self.method = method
        self.repeats = repeats
        self.p_values = p_values or [1, 2, 3, 4, 5, 6]
        self.verbose = verbose
        self.seed = seed
        self.method_fn = method_fn
        if detection_threshold < 0:
            raise ValueError("detection_threshold must be non-negative")
        self.detection_threshold = detection_threshold
        self._scenario_fn = self._get_scenario_fn()

    def _get_scenario_fn(self):
        """Get scenario function for latent confounder scenario."""
        from .simulations import scenario_latent_common_driver

        return scenario_latent_common_driver

    def run(self) -> list[PowerResult]:
        """Execute power analysis across grid.

        Returns
        -------
        list[PowerResult]
            Results for each (grid_point, repeat) combination.
        """
        results = []
        method_fn = self.method_fn or METHODS[self.method]
        total = self.grid.total_combinations() * self.repeats
        completed = 0

        for T in self.grid.T_values:
            for d in self.grid.d_values:
                for edge_density in self.grid.edge_densities:
                    for conf_strength in self.grid.confounder_strengths:
                        for latent_ar in self.grid.latent_autocorrs:
                            for noise_scale in self.grid.noise_scales:
                                for repeat in range(self.repeats):
                                    # Use seed offset for reproducibility
                                    seed_material = (
                                        f"{self.seed or 0}|{T}|{d}|{edge_density}|"
                                        f"{conf_strength}|{latent_ar}|{noise_scale}|{repeat}"
                                    ).encode()
                                    local_seed = int.from_bytes(
                                        hashlib.sha256(seed_material).digest()[:4],
                                        "little",
                                    )

                                    # Generate synthetic data
                                    start_time = time.time()
                                    scenario = self._scenario_fn(
                                        T=T,
                                        d=d,
                                        edge_prob=edge_density,
                                        conf_strength=conf_strength,
                                        latent_ar=latent_ar,
                                        noise_scale=noise_scale,
                                        seed=local_seed,
                                    )

                                    from .simulations import scenario_order1_unconfounded

                                    null_scenario = scenario_order1_unconfounded(
                                        T=T,
                                        d=d,
                                        edge_prob=edge_density,
                                        noise_scale=noise_scale,
                                        seed=local_seed + 1,
                                    )

                                    positive_adjacencies = method_fn(
                                        scenario.X, self.p_values
                                    )
                                    null_adjacencies = method_fn(
                                        null_scenario.X, self.p_values
                                    )

                                    positive_instability = compute_graph_instability(
                                        positive_adjacencies
                                    )
                                    null_instability = compute_graph_instability(
                                        null_adjacencies
                                    )
                                    p_star = self._select_p_star(positive_instability)
                                    tpr = float(
                                        max(positive_instability.values(), default=0.0)
                                        > self.detection_threshold
                                    )
                                    fpr = float(
                                        max(null_instability.values(), default=0.0)
                                        > self.detection_threshold
                                    )
                                    runtime_sec = time.time() - start_time

                                    results.append(
                                        PowerResult(
                                            T=T,
                                            d=d,
                                            edge_density=edge_density,
                                            confounder_strength=conf_strength,
                                            latent_autocorr=latent_ar,
                                            noise_scale=noise_scale,
                                            repeat=repeat,
                                            tpr=tpr,
                                            fpr=fpr,
                                            p_star=p_star,
                                            runtime_sec=runtime_sec,
                                        )
                                    )

                                    completed += 1
                                    if self.verbose:
                                        logger.info(f"Completed {completed}/{total}")

        return results

    def _select_p_star(self, instability: dict[int, float]) -> int:
        """Select conditioning depth where instability stabilizes.

        Uses the first local minimum as p_star, or defaults to max depth.

        Parameters
        ----------
        instability : dict[int, float]
            D_p values per conditioning depth.

        Returns
        -------
        int
            Selected p_star.
        """
        if not instability:
            return 1

        sorted_depths = sorted(instability.keys())
        if len(sorted_depths) < 2:
            return sorted_depths[0]

        # Find first local minimum
        for i in range(len(sorted_depths) - 1):
            if instability[sorted_depths[i]] < instability[sorted_depths[i + 1]]:
                return sorted_depths[i]

        # Fallback: return second-largest depth
        return sorted_depths[-2] if len(sorted_depths) > 1 else sorted_depths[0]

    def _compute_metrics(
        self, predicted: np.ndarray, ground_truth: np.ndarray
    ) -> tuple[float, float]:
        """Compute TPR and FPR for edge detection.

        Parameters
        ----------
        predicted : np.ndarray
            Predicted binary adjacency (d, d).
        ground_truth : np.ndarray
            Ground truth binary adjacency (d, d).

        Returns
        -------
        tuple[float, float]
            (TPR, FPR) both in [0, 1].
        """
        # Threshold predicted at 0.5 if continuous
        pred_binary = (predicted > 0.5).astype(int)
        np.fill_diagonal(pred_binary, 0)
        np.fill_diagonal(ground_truth, 0)

        # TP, FP, TN, FN
        tp = np.sum((pred_binary == 1) & (ground_truth == 1))
        fp = np.sum((pred_binary == 1) & (ground_truth == 0))
        fn = np.sum((pred_binary == 0) & (ground_truth == 1))
        tn = np.sum((pred_binary == 0) & (ground_truth == 0))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return float(tpr), float(fpr)

    def aggregate(self, results: list[PowerResult]) -> list[MetricAggregation]:
        """Aggregate results over repeats for each grid point.

        Parameters
        ----------
        results : list[PowerResult]
            Results from run().

        Returns
        -------
        list[MetricAggregation]
            One aggregation per grid point.
        """
        # Group by grid point
        grid_points = {}
        for result in results:
            key = (
                result.T,
                result.d,
                result.edge_density,
                result.confounder_strength,
                result.latent_autocorr,
                result.noise_scale,
            )
            if key not in grid_points:
                grid_points[key] = []
            grid_points[key].append(result)

        # Aggregate each group
        aggregated = []
        for (T, d, edge_density, conf_strength, latent_ar, noise_scale), group in grid_points.items():
            tprs = [r.tpr for r in group]
            fprs = [r.fpr for r in group]
            p_stars = [r.p_star for r in group]
            runtimes = [r.runtime_sec for r in group]

            agg = MetricAggregation(
                T=T,
                d=d,
                edge_density=edge_density,
                confounder_strength=conf_strength,
                latent_autocorr=latent_ar,
                noise_scale=noise_scale,
                n_repeats=len(group),
                mean_tpr=float(np.mean(tprs)),
                std_tpr=float(np.std(tprs)),
                mean_fpr=float(np.mean(fprs)),
                std_fpr=float(np.std(fprs)),
                mean_p_star=float(np.mean(p_stars)),
                std_p_star=float(np.std(p_stars)),
                mean_runtime=float(np.mean(runtimes)),
            )
            aggregated.append(agg)

        return aggregated

    def export(
        self,
        results: list[PowerResult],
        aggregated: list[MetricAggregation],
        output_dir: Path,
    ) -> None:
        """Export results to JSON and CSV files.

        Parameters
        ----------
        results : list[PowerResult]
            Detailed results per repeat.
        aggregated : list[MetricAggregation]
            Aggregated results per grid point.
        output_dir : Path
            Output directory for results.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export detailed results as JSON
        results_data = [asdict(r) for r in results]
        (output_dir / "results.json").write_text(
            json.dumps(results_data, indent=2)
        )

        # Export aggregated results as JSON
        aggregated_data = [asdict(a) for a in aggregated]
        (output_dir / "aggregated.json").write_text(
            json.dumps(aggregated_data, indent=2)
        )

        # Export summary as CSV
        csv_path = output_dir / "summary.csv"
        fieldnames = [
            "T",
            "d",
            "edge_density",
            "confounder_strength",
            "latent_autocorr",
            "noise_scale",
            "n_repeats",
            "mean_tpr",
            "std_tpr",
            "mean_fpr",
            "std_fpr",
            "mean_p_star",
            "std_p_star",
            "mean_runtime",
        ]
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for agg in aggregated:
                writer.writerow(asdict(agg))

        logger.info(f"Exported power analysis to {output_dir}")
