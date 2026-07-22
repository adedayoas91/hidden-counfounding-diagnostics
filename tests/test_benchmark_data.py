"""Tests for deterministic cross-method benchmark data generation."""

from __future__ import annotations

import numpy as np
import pytest

from markovianity_diagnostic.experiments.benchmark_data import (
    DEFAULT_N_VARS_RANGE,
    VALID_SCENARIOS,
    generate_trial_data,
)


class TestDeterminism:
    """Same (scenario, trial_id, seed) must yield byte-identical data."""

    @pytest.mark.parametrize("scenario", VALID_SCENARIOS)
    def test_repeatable(self, scenario: str) -> None:
        a = generate_trial_data(scenario, trial_id=3, seed=42)
        b = generate_trial_data(scenario, trial_id=3, seed=42)
        assert np.array_equal(a.X, b.X)
        assert np.array_equal(a.A, b.A)
        assert a.n_vars == b.n_vars
        assert a.length == b.length

    @pytest.mark.parametrize("scenario", VALID_SCENARIOS)
    def test_isolated_from_global_rng(self, scenario: str) -> None:
        """Algorithm RNG use between trials must not change the data."""
        baseline = generate_trial_data(scenario, trial_id=5, seed=42)
        # Simulate an inference algorithm consuming the global RNG.
        np.random.seed(999)
        _ = np.random.normal(size=10_000)
        after = generate_trial_data(scenario, trial_id=5, seed=42)
        assert np.array_equal(baseline.X, after.X)
        assert np.array_equal(baseline.A, after.A)


class TestCrossNotebookConsistency:
    """n_vars and length must match across all scenarios for a given trial."""

    @pytest.mark.parametrize("trial_id", range(5))
    def test_n_vars_and_length_match_across_scenarios(self, trial_id: int) -> None:
        trials = [generate_trial_data(s, trial_id=trial_id, seed=7) for s in VALID_SCENARIOS]
        n_vars = {t.n_vars for t in trials}
        lengths = {t.length for t in trials}
        assert len(n_vars) == 1
        assert len(lengths) == 1


class TestScenarioSemantics:
    """Markovian and NonMarkovian must produce different realisations."""

    def test_markovian_differs_from_nonmarkovian(self) -> None:
        mark = generate_trial_data("singleLag-Markovian", trial_id=0, seed=1)
        nonmark = generate_trial_data("singleLag-NonMarkovian", trial_id=0, seed=1)
        # Same n_vars/length (first two draws) but different series.
        assert mark.n_vars == nonmark.n_vars
        assert mark.length == nonmark.length
        assert not np.array_equal(mark.X, nonmark.X)

    def test_singlelag_differs_from_varlags(self) -> None:
        single = generate_trial_data("singleLag-Markovian", trial_id=0, seed=1)
        var = generate_trial_data("varLags-Markovian", trial_id=0, seed=1)
        assert not np.array_equal(single.X, var.X)


class TestRangeBounds:
    """Variable count must respect the inclusive range."""

    @pytest.mark.parametrize("scenario", VALID_SCENARIOS)
    def test_n_vars_within_inclusive_range(self, scenario: str) -> None:
        low, high = DEFAULT_N_VARS_RANGE
        for trial_id in range(50):
            t = generate_trial_data(scenario, trial_id=trial_id, seed=0)
            assert low <= t.n_vars <= high
            assert t.X.shape[0] == t.n_vars

    def test_custom_range_inclusive_upper(self) -> None:
        seen = {
            generate_trial_data(
                "singleLag-Markovian", trial_id=i, seed=0, n_vars_range=(20, 20)
            ).n_vars
            for i in range(10)
        }
        assert seen == {20}

    def test_invalid_scenario_raises(self) -> None:
        with pytest.raises(ValueError):
            generate_trial_data("bogus", trial_id=0)
