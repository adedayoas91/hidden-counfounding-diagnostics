"""Conditional-independence test comparison for Tigramite PCMCI+."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CITestRun:
    """Normalized result for one conditional-independence test family."""

    adjacency: dict[int, np.ndarray]
    runtime_seconds: float
    status: str
    error: str | None = None


class _TigramiteTestAdapter:
    import_path: str
    class_name: str

    def build(self) -> Any:
        module_name, _, _ = self.import_path.rpartition(".")
        module = __import__(module_name, fromlist=[self.class_name])
        return getattr(module, self.class_name)()


class ParCorrAdapter(_TigramiteTestAdapter):
    import_path = "tigramite.independence_tests.parcorr.ParCorr"
    class_name = "ParCorr"


class GPDCAdapter(_TigramiteTestAdapter):
    import_path = "tigramite.independence_tests.gpdc.GPDC"
    class_name = "GPDC"


class CMIknnAdapter(_TigramiteTestAdapter):
    import_path = "tigramite.independence_tests.cmiknn.CMIknn"
    class_name = "CMIknn"


class NonlinearResidualCorrelationAdapter:
    """Dependency-light nonlinear CI test based on tree residualization."""

    def build(self) -> Any:
        from scipy.stats import spearmanr
        from sklearn.ensemble import ExtraTreesRegressor
        from tigramite.independence_tests.independence_tests_base import CondIndTest

        class NonlinearResidualCorrelation(CondIndTest):
            def __init__(self) -> None:
                self._measure = "nonlinear_residual_spearman"
                self.two_sided = True
                self.residual_based = True
                super().__init__(
                    significance="shuffle_test",
                    sig_samples=100,
                    seed=42,
                )

            @property
            def measure(self) -> str:
                return self._measure

            def get_dependence_measure(self, array, xyz, data_type=None):
                x = np.asarray(array[np.flatnonzero(xyz == 0)[0]], dtype=float)
                y = np.asarray(array[np.flatnonzero(xyz == 1)[0]], dtype=float)
                z_rows = np.flatnonzero(xyz == 2)
                if z_rows.size:
                    z = np.asarray(array[z_rows].T, dtype=float)

                    def residualize(target: np.ndarray) -> np.ndarray:
                        model = ExtraTreesRegressor(
                            n_estimators=40,
                            min_samples_leaf=5,
                            random_state=42,
                            n_jobs=1,
                        )
                        return target - model.fit(z, target).predict(z)

                    x = residualize(x)
                    y = residualize(y)
                value = spearmanr(x, y).statistic
                return 0.0 if not np.isfinite(value) else float(value)

        return NonlinearResidualCorrelation()


CI_TESTS: dict[str, type[_TigramiteTestAdapter]] = {
    "parcorr": ParCorrAdapter,
    "gpdc": GPDCAdapter,
    "cmiknn": CMIknnAdapter,
    "nonlinear-residual": NonlinearResidualCorrelationAdapter,
}


def _collapse_tigramite_graph(graph: np.ndarray) -> np.ndarray:
    """Collapse non-empty lag marks to a binary directed adjacency."""
    d = graph.shape[0]
    adjacency = np.zeros((d, d), dtype=int)
    for source in range(d):
        for target in range(d):
            if source == target:
                continue
            adjacency[source, target] = int(
                any(str(mark).strip() for mark in graph[source, target].ravel())
            )
    return adjacency


def run_ci_test_comparison(
    X: np.ndarray,
    ci_test_names: list[str],
    p_values: list[int],
    method: str = "pcmciplus",
    *,
    pc_alpha: float = 0.05,
) -> dict[str, CITestRun]:
    """Run PCMCI+ across depths for each available CI-test family.

    c-GC uses its own permutation-test implementation and cannot accept
    Tigramite CI-test objects. Rejecting that combination prevents a comparison
    from being labelled as c-GC when PCMCI+ was actually executed.
    """
    if method.lower() not in {"pcmciplus", "pcmci+"}:
        raise ValueError("CI-test substitution is currently supported only for PCMCI+")
    if X.ndim != 2 or X.shape[0] < 3:
        raise ValueError("X must have shape (time, variables) with at least 3 rows")
    if not p_values or any(int(p) < 1 for p in p_values):
        raise ValueError("p_values must contain positive depths")

    from tigramite.data_processing import DataFrame
    from tigramite.pcmci import PCMCI

    outputs: dict[str, CITestRun] = {}
    for requested_name in ci_test_names:
        name = requested_name.lower()
        if name not in CI_TESTS:
            raise ValueError(f"Unknown CI test {requested_name!r}; choose {sorted(CI_TESTS)}")

        started = time.perf_counter()
        try:
            ci_test = CI_TESTS[name]().build()
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning("Skipping unavailable CI test %s: %s", name, exc)
            outputs[name] = CITestRun(
                adjacency={},
                runtime_seconds=time.perf_counter() - started,
                status="skipped",
                error=str(exc),
            )
            continue

        adjacency: dict[int, np.ndarray] = {}
        try:
            for depth in p_values:
                pcmci = PCMCI(
                    dataframe=DataFrame(np.asarray(X, dtype=float)),
                    cond_ind_test=ci_test,
                    verbosity=0,
                )
                result = pcmci.run_pcmciplus(
                    tau_min=1,
                    tau_max=int(depth),
                    pc_alpha=pc_alpha,
                )
                adjacency[int(depth)] = _collapse_tigramite_graph(result["graph"])
        except Exception as exc:
            logger.warning("CI test %s failed: %s", name, exc)
            outputs[name] = CITestRun(
                adjacency={},
                runtime_seconds=time.perf_counter() - started,
                status="failed",
                error=str(exc),
            )
            continue

        outputs[name] = CITestRun(
            adjacency=adjacency,
            runtime_seconds=time.perf_counter() - started,
            status="ok",
        )

    return outputs


__all__ = [
    "CITestRun",
    "CMIknnAdapter",
    "GPDCAdapter",
    "NonlinearResidualCorrelationAdapter",
    "ParCorrAdapter",
    "run_ci_test_comparison",
]
