from __future__ import annotations

import numpy as np

from markovianity_diagnostic.core.fast_causalised_GC import (
    FastGcStar,
    _fft_cross_null,
)


def test_fft_cross_null_matches_brute_force_circular_shifts():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(3, 64))
    y = rng.normal(size=(3, 64))
    observed, p_values = _fft_cross_null(x, y)

    brute_observed = []
    brute_p_values = []
    for x_row, y_row in zip(x, y, strict=True):
        correlations = np.array(
            [
                abs(np.corrcoef(np.roll(x_row, shift), y_row)[0, 1])
                for shift in range(len(x_row))
            ]
        )
        brute_observed.append(correlations[0])
        brute_p_values.append(
            (1 + np.sum(correlations[1:] >= correlations[0] - 1e-12))
            / len(x_row)
        )

    np.testing.assert_allclose(observed, brute_observed, atol=1e-12)
    np.testing.assert_allclose(p_values, brute_p_values, atol=1e-12)


def test_fast_gcstar_runs_both_conditioning_variants():
    rng = np.random.default_rng(7)
    data = rng.normal(size=(3, 100))
    for method in ("cgc", "fcgc"):
        first = FastGcStar(n_pasts=2, n_lags=1, method=method).fit(data)
        second = FastGcStar(n_pasts=2, n_lags=1, method=method).fit(data)
        first_graph = first.get_connectivity_matrix(alpha=0.05, beta=0.05)
        second_graph = second.get_connectivity_matrix(alpha=0.05, beta=0.05)
        assert first_graph.shape == (3, 3)
        np.testing.assert_array_equal(first_graph, second_graph)
