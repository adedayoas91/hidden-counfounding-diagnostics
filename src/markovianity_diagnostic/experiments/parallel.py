"""Parallelization infrastructure for simulation and analysis grids."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed, dump, load

logger = logging.getLogger(__name__)


def _hash_inputs(*args, **kwargs) -> str:
    """Generate a deterministic SHA256 hash of input data and parameters."""

    def _serialize(obj: Any) -> str:
        """Convert object to a hashable string representation."""
        if isinstance(obj, np.ndarray):
            return hashlib.sha256(obj.tobytes()).hexdigest()
        elif isinstance(obj, dict):
            sorted_items = sorted(obj.items())
            return hashlib.sha256(str(sorted_items).encode()).hexdigest()
        elif isinstance(obj, (list, tuple)):
            return hashlib.sha256(str(obj).encode()).hexdigest()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return hashlib.sha256(str(obj).encode()).hexdigest()
        else:
            return hashlib.sha256(str(obj).encode()).hexdigest()

    combined_hash = hashlib.sha256()
    for arg in args:
        combined_hash.update(_serialize(arg).encode())
    for key, value in sorted(kwargs.items()):
        combined_hash.update(_serialize(value).encode())

    return combined_hash.hexdigest()


def _get_cache_path(cache_dir: Path | str, cache_key: str) -> Path:
    """Construct the cache file path for a given cache key."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cache_key}.pkl"


def _load_resume_file(cache_dir: Path | str) -> dict[str, bool]:
    """Load the resume tracking file from cache directory."""
    cache_dir = Path(cache_dir)
    resume_file = cache_dir / "resume.json"
    if resume_file.exists():
        with open(resume_file) as f:
            return json.load(f)
    return {}


def _save_resume_file(cache_dir: Path | str, resume_tracker: dict[str, bool]) -> None:
    """Save the resume tracking file to cache directory."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    resume_file = cache_dir / "resume.json"
    with open(resume_file, "w") as f:
        json.dump(resume_tracker, f, indent=2)


def parallel_simulation_grid(
    scenarios: dict[str, Callable[..., Any]],
    methods: dict[str, Callable[[np.ndarray, list[int]], dict[int, np.ndarray]]],
    p_values: list[int],
    n_repeats: int,
    n_jobs: int = 4,
    cache_dir: Path | str | None = None,
    resume: bool = False,
    force: bool = False,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Run multiple scenario + method combinations in parallel.

    Parameters
    ----------
    scenarios : dict[str, Callable]
        Dictionary mapping scenario names to functions that return ScenarioResult.
    methods : dict[str, Callable]
        Dictionary mapping method names to analysis functions that take
        (X, p_values) and return dict[int, np.ndarray].
    p_values : list[int]
        List of depths to search over.
    n_repeats : int
        Number of repeats per (scenario, method) combination.
    n_jobs : int, optional
        Number of parallel jobs. Default is 4.
    cache_dir : Path | str | None, optional
        Directory for caching results. If None, no caching is performed.
    resume : bool, optional
        If True, skip cached (scenario, method, repeat) tuples. Default is False.
    force : bool, optional
        If True, bypass cache and recompute all results. Default is False.

    Returns
    -------
    dict[tuple[str, str, int], dict[str, Any]]
        Dictionary with keys (scenario, method, repeat) and values containing
        results dict with 'adjacency', 'ground_truth', 'metadata'.
    """

    resume_tracker = {}
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        if resume and not force:
            resume_tracker = _load_resume_file(cache_dir)

    tasks = []
    task_mapping = {}

    for scenario_name, scenario_func in scenarios.items():
        for method_name, method_func in methods.items():
            for repeat in range(n_repeats):
                task_key = (scenario_name, method_name, repeat)

                tasks.append(
                    delayed(_run_single_simulation)(
                        scenario_name=scenario_name,
                        scenario_func=scenario_func,
                        method_name=method_name,
                        method_func=method_func,
                        p_values=p_values,
                        repeat=repeat,
                        cache_dir=cache_dir if not force else None,
                    )
                )
                task_mapping[len(tasks) - 1] = task_key

    logger.info(
        f"Running {len(tasks)} tasks with n_jobs={n_jobs} "
        f"(cache_dir={cache_dir}, force={force})"
    )

    results_list = Parallel(n_jobs=n_jobs, prefer="threads")(tasks)

    results = {}
    for task_idx, result in enumerate(results_list):
        if result is not None:
            task_key = task_mapping[task_idx]
            results[task_key] = result

            if cache_dir is not None:
                cache_key = result.get("_cache_key")
                if cache_key:
                    resume_tracker[cache_key] = True

    if cache_dir is not None and resume_tracker:
        _save_resume_file(cache_dir, resume_tracker)

    logger.info(f"Completed {len(results)} simulation tasks")
    return results


def _run_single_simulation(
    scenario_name: str,
    scenario_func: Callable[..., Any],
    method_name: str,
    method_func: Callable[[np.ndarray, list[int]], dict[int, np.ndarray]],
    p_values: list[int],
    repeat: int,
    cache_dir: Path | str | None = None,
) -> dict[str, Any] | None:
    """Run a single simulation scenario with a given method.

    Parameters
    ----------
    scenario_name : str
        Name of the scenario.
    scenario_func : Callable
        Function that generates ScenarioResult.
    method_name : str
        Name of the analysis method.
    method_func : Callable
        Analysis function that takes (X, p_values) and returns dict[int, np.ndarray].
    p_values : list[int]
        Depths to analyze.
    repeat : int
        Repeat number (used in caching).
    cache_dir : Path | str | None
        Cache directory. If None, no caching.

    Returns
    -------
    dict[str, Any] | None
        Result dictionary with 'adjacency', 'ground_truth', 'metadata',
        or None if loaded from cache.
    """

    cache_key = _hash_inputs(scenario_name, method_name, repeat, p_values)

    if cache_dir is not None:
        cache_path = _get_cache_path(cache_dir, cache_key)
        if cache_path.exists():
            try:
                cached_result = load(cache_path)
                logger.debug(f"Loaded cached result for {cache_key[:8]}...")
                return cached_result
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key[:8]}...: {e}")

    try:
        scenario_result = scenario_func()
        X = scenario_result.X
        ground_truth = scenario_result.ground_truth_compact
        metadata = scenario_result.metadata

        analysis_results = method_func(X, p_values)

        result = {
            "scenario": scenario_name,
            "method": method_name,
            "repeat": repeat,
            "adjacency": analysis_results,
            "ground_truth": ground_truth,
            "metadata": metadata,
            "X_shape": X.shape,
            "_cache_key": cache_key,
        }

        if cache_dir is not None:
            cache_path = _get_cache_path(cache_dir, cache_key)
            try:
                dump(result, cache_path)
                logger.debug(f"Cached result to {cache_key[:8]}...")
            except Exception as e:
                logger.warning(f"Failed to cache result {cache_key[:8]}...: {e}")

        return result

    except Exception as e:
        logger.error(
            f"Failed to run simulation {scenario_name}/{method_name}/repeat={repeat}: {e}"
        )
        return None


def parallel_bootstrap(
    X: np.ndarray,
    null_model_fit: dict[str, Any],
    B: int,
    analysis_func: Callable[[np.ndarray], float],
    n_jobs: int = 4,
    seed: int = 42,
    cache_dir: Path | str | None = None,
) -> np.ndarray:
    """Run B bootstrap replicates in parallel.

    Parameters
    ----------
    X : np.ndarray
        Original data array of shape (T, d).
    null_model_fit : dict[str, Any]
        Fitted null model from fit_null_var() containing coef, residuals, p0, d, init.
    B : int
        Number of bootstrap replicates.
    analysis_func : Callable[[np.ndarray], float]
        Function that takes synthetic data and returns a scalar test statistic.
    n_jobs : int, optional
        Number of parallel jobs. Default is 4.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    cache_dir : Path | str | None, optional
        Directory for caching. If None, no caching.

    Returns
    -------
    np.ndarray
        Array of shape (B,) containing bootstrap test statistics.
    """

    cache_key = _hash_inputs(X, null_model_fit.get("p0"), seed, B)

    if cache_dir is not None:
        cache_path = _get_cache_path(cache_dir, cache_key)
        if cache_path.exists():
            try:
                cached_result = load(cache_path)
                logger.info(f"Loaded cached bootstrap for {cache_key[:8]}...")
                return cached_result
            except Exception as e:
                logger.warning(f"Failed to load bootstrap cache {cache_key[:8]}...: {e}")

    T = X.shape[0]

    tasks = [
        delayed(_run_single_bootstrap)(
            null_model_fit=null_model_fit,
            T=T,
            analysis_func=analysis_func,
            seed=seed + b,
        )
        for b in range(B)
    ]

    logger.info(f"Running {B} bootstrap replicates with n_jobs={n_jobs}")
    T_boot = np.array(Parallel(n_jobs=n_jobs, prefer="threads")(tasks))

    if cache_dir is not None:
        cache_path = _get_cache_path(cache_dir, cache_key)
        try:
            dump(T_boot, cache_path)
            logger.debug(f"Cached bootstrap results to {cache_key[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to cache bootstrap {cache_key[:8]}...: {e}")

    return T_boot


def _run_single_bootstrap(
    null_model_fit: dict[str, Any],
    T: int,
    analysis_func: Callable[[np.ndarray], float],
    seed: int,
) -> float:
    """Run a single bootstrap replicate.

    Parameters
    ----------
    null_model_fit : dict[str, Any]
        Fitted null model.
    T : int
        Time series length.
    analysis_func : Callable
        Function to apply to synthetic data.
    seed : int
        Random seed for this replicate.

    Returns
    -------
    float
        Test statistic for this bootstrap replicate.
    """

    from .bootstrap import simulate_from_null

    X_boot = simulate_from_null(null_model_fit, T, seed=seed)
    return float(analysis_func(X_boot))


def parallel_over_fish(
    fish_list: list[np.ndarray],
    analysis_func: Callable[[np.ndarray], Any],
    n_jobs: int = 4,
    cache_dir: Path | str | None = None,
) -> list[Any]:
    """Run analysis function on each fish in parallel.

    Parameters
    ----------
    fish_list : list[np.ndarray]
        List of fish data arrays.
    analysis_func : Callable[[np.ndarray], Any]
        Function that takes a single fish array and returns a result.
    n_jobs : int, optional
        Number of parallel jobs. Default is 4.
    cache_dir : Path | str | None, optional
        Directory for caching results by fish index. If None, no caching.

    Returns
    -------
    list[Any]
        List of results, one per fish.
    """

    resume_tracker = {}
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        resume_tracker = _load_resume_file(cache_dir)

    tasks = []
    task_mapping = {}

    for fish_idx, fish_data in enumerate(fish_list):
        cache_key = _hash_inputs("fish", fish_idx, fish_data)

        tasks.append(
            delayed(_run_fish_analysis)(
                fish_idx=fish_idx,
                fish_data=fish_data,
                analysis_func=analysis_func,
                cache_dir=cache_dir,
                cache_key=cache_key,
            )
        )
        task_mapping[len(tasks) - 1] = fish_idx

    logger.info(f"Running analysis on {len(tasks)} fish with n_jobs={n_jobs}")
    results_list = Parallel(n_jobs=n_jobs, prefer="threads")(tasks)

    results = [None] * len(fish_list)
    for task_idx, result in enumerate(results_list):
        if result is not None:
            fish_idx = task_mapping[task_idx]
            results[fish_idx] = result

            if cache_dir is not None:
                cache_key = result.get("_cache_key")
                if cache_key:
                    resume_tracker[cache_key] = True

    if cache_dir is not None and resume_tracker:
        _save_resume_file(cache_dir, resume_tracker)

    logger.info(f"Completed analysis on {len([r for r in results if r is not None])} fish")
    return results


def _run_fish_analysis(
    fish_idx: int,
    fish_data: np.ndarray,
    analysis_func: Callable[[np.ndarray], Any],
    cache_dir: Path | str | None = None,
    cache_key: str | None = None,
) -> dict[str, Any] | None:
    """Run analysis on a single fish.

    Parameters
    ----------
    fish_idx : int
        Index of the fish.
    fish_data : np.ndarray
        Data for this fish.
    analysis_func : Callable
        Analysis function.
    cache_dir : Path | str | None
        Cache directory.
    cache_key : str | None
        Cache key for this fish.

    Returns
    -------
    dict[str, Any] | None
        Result dictionary with analysis output, or None if loaded from cache.
    """

    if cache_dir is not None and cache_key is not None:
        cache_path = _get_cache_path(cache_dir, cache_key)
        if cache_path.exists():
            try:
                cached_result = load(cache_path)
                logger.debug(f"Loaded cached fish {fish_idx}")
                return cached_result
            except Exception as e:
                logger.warning(f"Failed to load fish cache {fish_idx}: {e}")

    try:
        analysis_result = analysis_func(fish_data)

        result = {
            "fish_idx": fish_idx,
            "fish_shape": fish_data.shape,
            "analysis_result": analysis_result,
            "_cache_key": cache_key,
        }

        if cache_dir is not None and cache_key is not None:
            cache_path = _get_cache_path(cache_dir, cache_key)
            try:
                dump(result, cache_path)
                logger.debug(f"Cached fish {fish_idx}")
            except Exception as e:
                logger.warning(f"Failed to cache fish {fish_idx}: {e}")

        return result

    except Exception as e:
        logger.error(f"Failed to analyze fish {fish_idx}: {e}")
        return None
