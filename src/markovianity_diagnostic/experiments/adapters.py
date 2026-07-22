"""Adapters that connect estimators to the experiment harness."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable
from pathlib import Path
import sys

import numpy as np


def _load_gcstar_class() -> type:
    """Load GcStar from the notebook-friendly causalised module."""

    module_path = Path(__file__).resolve().parents[1] / "core" / "causalised-GC.py"
    spec = importlib.util.spec_from_file_location(
        "markovianity_diagnostic.core.causalised_gc",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load GcStar from {module_path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.GcStar


GcStar = _load_gcstar_class()


def _load_fast_gcstar_class() -> type:
    """Load the vectorized ``FastGcStar`` drop-in from the core package."""

    from ..core import FastGcStar

    return FastGcStar


FastGcStar = _load_fast_gcstar_class()

PCMCI_FIXED_LAG = 1
PCMCI_CONDITIONING_PARAMETERS = (
    "max_conds_dim",
    "max_conds_py",
    "max_conds_px",
    "max_conds_px_lagged",
)


def _design_matrix(X: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a stacked autoregressive design matrix from ``X``."""

    T, d = X.shape
    Y = X[p:]
    rows = []
    for t in range(p, T):
        rows.append(np.concatenate([X[t - lag] for lag in range(1, p + 1)]))
    Z = np.asarray(rows)
    return Z, Y


def _binary_adjacency_from_lstsq(
    X: np.ndarray,
    p: int,
    *,
    threshold: float = 0.08,
) -> np.ndarray:
    """Baseline least-squares adjacency estimator used for smoke tests."""

    _, d = X.shape
    Z, Y = _design_matrix(X, p)
    beta, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    beta = beta.reshape(p, d, d)

    adjacency = np.zeros((d, d), dtype=int)
    for lag in range(p):
        adjacency = np.logical_or(adjacency, np.abs(beta[lag]) > threshold)

    adjacency = adjacency.astype(int)
    np.fill_diagonal(adjacency, 0)
    return adjacency


def analyze_with_baseline_lstsq(
    X: np.ndarray,
    p_values: list[int],
) -> dict[int, np.ndarray]:
    """Run the baseline least-squares estimator for each depth in ``p_values``."""

    return {
        int(p_value): _binary_adjacency_from_lstsq(X, int(p_value))
        for p_value in p_values
    }


def _run_gcstar_single_depth(
    X: np.ndarray,
    p: int,
    *,
    method: str,
    alpha: float,
    beta: float,
    n_perm: int,
    n_lags: int,
    temporal: bool,
    verbose: int,
    simulation: bool,
    estimator_cls: type = GcStar,
) -> np.ndarray:
    """Run a ``GcStar``-compatible estimator for one depth and return a binary graph."""

    estimator = estimator_cls(
        n_perm=n_perm,
        n_pasts=int(p),
        n_lags=n_lags,
        temporal=temporal,
        method=method,
    )
    estimator.fit(np.asarray(X).T, verbose=verbose)
    connectivity = estimator.get_connectivity_matrix(
        simulation=simulation,
        alpha=alpha,
        beta=beta,
    )
    adjacency = (np.asarray(connectivity) != 0).astype(int)
    np.fill_diagonal(adjacency, 0)
    return adjacency


def make_gcstar_analyzer(
    method: str,
    *,
    alpha: float = 0.01,
    beta: float = 0.001,
    n_perm: int = 200,
    n_lags: int = 1,
    temporal: bool = True,
    verbose: int = 0,
    simulation: bool = True,
    estimator_cls: type = GcStar,
) -> Callable[[np.ndarray, list[int]], dict[int, np.ndarray]]:
    """Create an analyzer function backed by a ``GcStar``-compatible estimator."""

    if method not in {"cgc", "fcgc"}:
        raise ValueError("method must be 'cgc' or 'fcgc'.")

    def analyze(X: np.ndarray, p_values: list[int]) -> dict[int, np.ndarray]:
        return {
            int(p_value): _run_gcstar_single_depth(
                X,
                int(p_value),
                method=method,
                alpha=alpha,
                beta=beta,
                n_perm=n_perm,
                n_lags=n_lags,
                temporal=temporal,
                verbose=verbose,
                simulation=simulation,
                estimator_cls=estimator_cls,
            )
            for p_value in p_values
        }

    return analyze


def analyze_with_gcstar_cgc(
    X: np.ndarray,
    p_values: list[int],
) -> dict[int, np.ndarray]:
    """Run ``GcStar`` with the conventional c-GC conditioning set."""

    return make_gcstar_analyzer("cgc")(X, p_values)


def analyze_with_gcstar_fcgc(
    X: np.ndarray,
    p_values: list[int],
) -> dict[int, np.ndarray]:
    """Run ``GcStar`` with the full-conditioning fcGC variant."""

    return make_gcstar_analyzer("fcgc")(X, p_values)


def analyze_with_fast_gcstar_cgc(
    X: np.ndarray,
    p_values: list[int],
) -> dict[int, np.ndarray]:
    """Run the vectorized ``FastGcStar`` with the c-GC conditioning set."""

    return make_gcstar_analyzer("cgc", estimator_cls=FastGcStar)(X, p_values)


def analyze_with_fast_gcstar_fcgc(
    X: np.ndarray,
    p_values: list[int],
) -> dict[int, np.ndarray]:
    """Run the vectorized ``FastGcStar`` with the full-conditioning fcGC variant."""

    return make_gcstar_analyzer("fcgc", estimator_cls=FastGcStar)(X, p_values)


def _collapse_tigramite_graph(
    graph: np.ndarray,
    *,
    directed_only: bool,
) -> np.ndarray:
    """Collapse Tigramite lag marks to the package's target-by-source schema."""
    n_variables = graph.shape[0]
    adjacency = np.zeros((n_variables, n_variables), dtype=int)
    for source in range(graph.shape[0]):
        for target in range(graph.shape[1]):
            if source == target:
                continue
            marks = [str(mark).strip() for mark in np.ravel(graph[source, target])]
            if directed_only:
                present = any(">" in mark and "<" not in mark for mark in marks)
            else:
                present = any(mark for mark in marks)
            adjacency[target, source] = int(present)
    np.fill_diagonal(adjacency, 0)
    return adjacency


def _pcmciplus_run_kwargs(
    conditioning_depth: int,
    *,
    pc_alpha: float,
) -> dict[str, int | float]:
    """Map the shared depth index to PCMCI+ conditioning limits at lag one."""

    depth = int(conditioning_depth)
    if depth < 1:
        raise ValueError("PCMCI+ conditioning depth must be at least 1.")
    kwargs: dict[str, int | float] = {
        "tau_min": PCMCI_FIXED_LAG,
        "tau_max": PCMCI_FIXED_LAG,
        "pc_alpha": float(pc_alpha),
    }
    kwargs.update(
        {parameter: depth for parameter in PCMCI_CONDITIONING_PARAMETERS}
    )
    return kwargs


def _analyze_with_tigramite(
    X: np.ndarray,
    p_values: list[int],
    *,
    algorithm: str,
    pc_alpha: float = 0.05,
) -> dict[int, np.ndarray]:
    """Run one Tigramite algorithm over the shared conditioning-depth grid."""
    from tigramite import data_processing as pp

    output: dict[int, np.ndarray] = {}
    for depth in p_values:
        dataframe = pp.DataFrame(np.asarray(X, dtype=float))
        if algorithm == "pcmciplus":
            from tigramite.independence_tests.parcorr import ParCorr
            from tigramite.pcmci import PCMCI

            learner = PCMCI(
                dataframe=dataframe,
                cond_ind_test=ParCorr(),
                verbosity=0,
            )
            result = learner.run_pcmciplus(
                **_pcmciplus_run_kwargs(depth, pc_alpha=pc_alpha)
            )
            directed_only = True
        elif algorithm == "fullci":
            from tigramite.independence_tests.parcorr import ParCorr
            from tigramite.pcmci import PCMCI

            learner = PCMCI(
                dataframe=dataframe,
                cond_ind_test=ParCorr(),
                verbosity=0,
            )
            result = learner.run_fullci(tau_max=int(depth))
            directed_only = True
        elif algorithm == "lpcmci":
            from tigramite.independence_tests.parcorr import ParCorr
            from tigramite.lpcmci import LPCMCI

            learner = LPCMCI(
                dataframe=dataframe,
                cond_ind_test=ParCorr(significance="analytic"),
                verbosity=0,
            )
            result = learner.run_lpcmci(
                tau_min=0,
                tau_max=int(depth),
                pc_alpha=pc_alpha,
            )
            directed_only = False
        else:  # pragma: no cover - private helper guards this
            raise ValueError(f"Unknown Tigramite algorithm: {algorithm}")

        output[int(depth)] = _collapse_tigramite_graph(
            result["graph"],
            directed_only=directed_only,
        )
    return output


def analyze_with_pcmciplus(
    X: np.ndarray,
    p_values: list[int],
    *,
    pc_alpha: float = 0.05,
) -> dict[int, np.ndarray]:
    """Run fixed-lag PCMCI+ over maximum conditioning-set sizes."""
    return _analyze_with_tigramite(
        X,
        p_values,
        algorithm="pcmciplus",
        pc_alpha=pc_alpha,
    )


def analyze_with_fullci(X: np.ndarray, p_values: list[int]) -> dict[int, np.ndarray]:
    return _analyze_with_tigramite(X, p_values, algorithm="fullci")


def analyze_with_lpcmci(X: np.ndarray, p_values: list[int]) -> dict[int, np.ndarray]:
    return _analyze_with_tigramite(X, p_values, algorithm="lpcmci")


def analyze_with_user_method(
    X: np.ndarray, p_values: list[int]
) -> dict[int, np.ndarray]:
    """Placeholder for user-supplied methods.

    Replace this function body if you want to keep the external-method hook
    inside the package instead of using ``--user-method module:function``.
    """

    raise NotImplementedError(
        "Replace analyze_with_user_method with your custom analyzer or use --user-method."
    )


def normalize_adjacency_output(
    raw: object,
    p_values: list[int],
) -> dict[int, np.ndarray]:
    """Normalize user-method outputs into ``dict[int, adjacency]`` form."""

    if isinstance(raw, dict):
        output = {int(key): np.asarray(value).astype(int) for key, value in raw.items()}
    elif isinstance(raw, (list, tuple)):
        if len(raw) != len(p_values):
            raise ValueError("List/tuple method output must match len(p_values).")
        output = {
            int(p_value): np.asarray(value).astype(int)
            for p_value, value in zip(p_values, raw, strict=True)
        }
    else:
        raise TypeError(
            "User method must return dict[p, adjacency] or a list of adjacency matrices."
        )

    required = {int(p_value) for p_value in p_values}
    missing = sorted(required.difference(output))
    if missing:
        raise ValueError(f"User method output is missing p-values: {missing}")

    d = next(iter(output.values())).shape[0]
    for p_value, adjacency in output.items():
        if adjacency.shape != (d, d):
            raise ValueError(
                f"Adjacency for p={p_value} has shape {adjacency.shape}, expected {(d, d)}."
            )
        adjacency = (adjacency != 0).astype(int)
        np.fill_diagonal(adjacency, 0)
        output[p_value] = adjacency
    return output


def load_external_method(
    spec: str,
) -> Callable[[np.ndarray, list[int]], dict[int, np.ndarray]]:
    """Load an analyzer from ``module:function`` notation."""

    if ":" not in spec:
        raise ValueError("--user-method must be formatted as 'module:function'.")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)

    def wrapped(X: np.ndarray, p_values: list[int]) -> dict[int, np.ndarray]:
        return normalize_adjacency_output(function(X, p_values), p_values)

    return wrapped


METHODS = {
    "baseline_lstsq": analyze_with_baseline_lstsq,
    "gcstar_cgc": analyze_with_fast_gcstar_cgc,
    "gcstar_fcgc": analyze_with_fast_gcstar_fcgc,
    "fast_gcstar_cgc": analyze_with_fast_gcstar_cgc,
    "fast_gcstar_fcgc": analyze_with_fast_gcstar_fcgc,
    "legacy_gcstar_cgc": analyze_with_gcstar_cgc,
    "legacy_gcstar_fcgc": analyze_with_gcstar_fcgc,
    "pcmciplus": analyze_with_pcmciplus,
    "fullci": analyze_with_fullci,
    "lpcmci": analyze_with_lpcmci,
    "user_method": analyze_with_user_method,
}

METHOD_METADATA: dict[str, dict[str, object]] = {
    "gcstar_cgc": {
        "implementation": "FastGcStar",
        "variant": "cgc",
        "exact_circular_shift_null": True,
        "ridge": 1e-6,
        "screen_alpha": 0.2,
    },
    "gcstar_fcgc": {
        "implementation": "FastGcStar",
        "variant": "fcgc",
        "exact_circular_shift_null": True,
        "ridge": 1e-6,
        "screen_alpha": 0.2,
    },
    "fast_gcstar_cgc": {
        "implementation": "FastGcStar",
        "variant": "cgc",
        "exact_circular_shift_null": True,
        "ridge": 1e-6,
        "screen_alpha": 0.2,
    },
    "fast_gcstar_fcgc": {
        "implementation": "FastGcStar",
        "variant": "fcgc",
        "exact_circular_shift_null": True,
        "ridge": 1e-6,
        "screen_alpha": 0.2,
    },
    "legacy_gcstar_cgc": {"implementation": "GcStar", "variant": "cgc"},
    "legacy_gcstar_fcgc": {"implementation": "GcStar", "variant": "fcgc"},
    "pcmciplus": {
        "implementation": "Tigramite PCMCI+",
        "pc_alpha": 0.05,
        "fixed_lag": PCMCI_FIXED_LAG,
        "depth_parameter": "maximum_conditioning_set_size",
        "conditioning_caps": list(PCMCI_CONDITIONING_PARAMETERS),
    },
    "fullci": {"implementation": "Tigramite FullCI"},
    "lpcmci": {
        "implementation": "Tigramite LPCMCI",
        "pc_alpha": 0.05,
        "collapse": "all nonempty PAG marks",
    },
}
