"""Tests for parallelization infrastructure."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.markovianity_diagnostic.experiments.parallel import (
    _get_cache_path,
    _hash_inputs,
    _load_resume_file,
    _save_resume_file,
    parallel_bootstrap,
    parallel_over_fish,
    parallel_simulation_grid,
)
from src.markovianity_diagnostic.experiments.simulations import (
    ScenarioResult,
)


@pytest.fixture
def sample_scenarios() -> dict[str, callable]:
    """Create sample scenario functions for testing."""

    def scenario_a(*, T: int = 200, d: int = 5, seed: int = 0) -> ScenarioResult:
        """Simple test scenario A."""
        X = np.random.RandomState(seed).randn(T, d)
        ground_truth = np.eye(d, dtype=int)
        return ScenarioResult(
            X=X,
            ground_truth_compact=ground_truth,
            metadata={"scenario": "test_a", "T": T, "d": d},
        )

    def scenario_b(*, T: int = 200, d: int = 5, seed: int = 0) -> ScenarioResult:
        """Simple test scenario B."""
        X = np.random.RandomState(seed + 100).randn(T, d)
        ground_truth = np.zeros((d, d), dtype=int)
        return ScenarioResult(
            X=X,
            ground_truth_compact=ground_truth,
            metadata={"scenario": "test_b", "T": T, "d": d},
        )

    return {"scenario_a": scenario_a, "scenario_b": scenario_b}


@pytest.fixture
def sample_methods() -> dict[str, callable]:
    """Create sample analysis methods for testing."""

    def method_simple(X: np.ndarray, p_values: list[int]) -> dict[int, np.ndarray]:
        """Simple baseline method."""
        d = X.shape[1]
        return {
            p: np.random.RandomState(p).randint(0, 2, (d, d)) for p in p_values
        }

    def method_random_seed_dependent(
        X: np.ndarray, p_values: list[int]
    ) -> dict[int, np.ndarray]:
        """Method whose results depend on a fixed seed."""
        d = X.shape[1]
        rng = np.random.RandomState(42)  # Fixed seed for determinism
        return {
            p: rng.randint(0, 2, (d, d)) for p in p_values
        }

    return {
        "method_simple": method_simple,
        "method_seed_dependent": method_random_seed_dependent,
    }


class TestHashInputs:
    """Test the hash input function."""

    def test_identical_inputs_produce_same_hash(self) -> None:
        """Identical inputs should hash identically."""
        hash1 = _hash_inputs(42, "test", {"a": 1})
        hash2 = _hash_inputs(42, "test", {"a": 1})
        assert hash1 == hash2

    def test_different_inputs_produce_different_hashes(self) -> None:
        """Different inputs should produce different hashes."""
        hash1 = _hash_inputs(42, "test")
        hash2 = _hash_inputs(42, "test2")
        assert hash1 != hash2

    def test_arrays_hashed_correctly(self) -> None:
        """Arrays should be hashed by their content."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])
        arr3 = np.array([1, 2, 4])
        hash1 = _hash_inputs(arr1)
        hash2 = _hash_inputs(arr2)
        hash3 = _hash_inputs(arr3)
        assert hash1 == hash2
        assert hash1 != hash3

    def test_hash_is_hexadecimal_string(self) -> None:
        """Hash should be a valid hexadecimal string."""
        hash_result = _hash_inputs(123, "test")
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA256 hex is 64 chars
        int(hash_result, 16)  # Should not raise


class TestCachePaths:
    """Test cache path generation."""

    def test_cache_path_creation(self, tmp_path: Path) -> None:
        """Cache paths should be correctly generated."""
        cache_path = _get_cache_path(tmp_path, "test_hash_abc123")
        assert cache_path.parent == tmp_path
        assert cache_path.name == "test_hash_abc123.pkl"
        assert cache_path.parent.exists()

    def test_cache_dir_created_if_not_exists(self, tmp_path: Path) -> None:
        """Cache directory should be created if it doesn't exist."""
        cache_dir = tmp_path / "nonexistent" / "cache"
        assert not cache_dir.exists()
        cache_path = _get_cache_path(cache_dir, "test_hash")
        assert cache_path.parent.exists()


class TestResumeTracking:
    """Test resume file tracking."""

    def test_load_nonexistent_resume_file_returns_empty_dict(
        self, tmp_path: Path
    ) -> None:
        """Loading from nonexistent resume file should return empty dict."""
        result = _load_resume_file(tmp_path)
        assert result == {}

    def test_save_and_load_resume_file(self, tmp_path: Path) -> None:
        """Should save and load resume tracking correctly."""
        tracker = {"key1": True, "key2": False}
        _save_resume_file(tmp_path, tracker)
        loaded = _load_resume_file(tmp_path)
        assert loaded == tracker

    def test_resume_file_is_json(self, tmp_path: Path) -> None:
        """Resume file should be valid JSON."""
        tracker = {"key1": True}
        _save_resume_file(tmp_path, tracker)
        with open(tmp_path / "resume.json") as f:
            content = json.load(f)
        assert content == tracker


class TestParallelSimulationGrid:
    """Test parallel simulation grid execution."""

    def test_runs_all_combinations(
        self, sample_scenarios, sample_methods
    ) -> None:
        """Should run all scenario x method x repeat combinations."""
        results = parallel_simulation_grid(
            scenarios=sample_scenarios,
            methods=sample_methods,
            p_values=[1, 2],
            n_repeats=2,
            n_jobs=1,
        )

        expected_combinations = (
            len(sample_scenarios) * len(sample_methods) * 2
        )
        assert len(results) == expected_combinations

    def test_results_have_correct_structure(
        self, sample_scenarios, sample_methods
    ) -> None:
        """Results should have correct keys and structure."""
        results = parallel_simulation_grid(
            scenarios=sample_scenarios,
            methods=sample_methods,
            p_values=[1, 2],
            n_repeats=1,
            n_jobs=1,
        )

        for key, result in results.items():
            assert isinstance(key, tuple)
            assert len(key) == 3
            assert isinstance(result, dict)
            assert "adjacency" in result
            assert "ground_truth" in result
            assert "metadata" in result
            assert "X_shape" in result

    def test_determinism_with_fixed_seed_njobs_1_vs_njobs_4(
        self, sample_scenarios, sample_methods
    ) -> None:
        """Results should be identical for n_jobs=1 and n_jobs=4 with fixed seeds."""
        results_serial = parallel_simulation_grid(
            scenarios=sample_scenarios,
            methods=sample_methods,
            p_values=[1, 2],
            n_repeats=2,
            n_jobs=1,
        )

        results_parallel = parallel_simulation_grid(
            scenarios=sample_scenarios,
            methods=sample_methods,
            p_values=[1, 2],
            n_repeats=2,
            n_jobs=4,
        )

        assert results_serial.keys() == results_parallel.keys()
        for key in results_serial.keys():
            for p_value, adjacency_serial in results_serial[key][
                "adjacency"
            ].items():
                adjacency_parallel = results_parallel[key]["adjacency"][p_value]
                np.testing.assert_array_equal(
                    adjacency_serial, adjacency_parallel
                )

    def test_caching_creates_files(
        self, sample_scenarios, sample_methods, tmp_path
    ) -> None:
        """Caching should create cache files."""
        cache_dir = tmp_path / "cache"
        parallel_simulation_grid(
            scenarios=sample_scenarios,
            methods=sample_methods,
            p_values=[1, 2],
            n_repeats=1,
            n_jobs=1,
            cache_dir=cache_dir,
        )

        assert len(list(cache_dir.glob("*.pkl"))) > 0

    def test_resume_skips_cached_tasks(
        self, sample_scenarios, sample_methods, tmp_path
    ) -> None:
        """Resume mode should skip already cached tasks."""
        cache_dir = tmp_path / "cache"

        results1 = parallel_simulation_grid(
            scenarios=sample_scenarios,
            methods=sample_methods,
            p_values=[1, 2],
            n_repeats=2,
            n_jobs=1,
            cache_dir=cache_dir,
        )
        results2 = parallel_simulation_grid(
            scenarios=sample_scenarios,
            methods=sample_methods,
            p_values=[1, 2],
            n_repeats=2,
            n_jobs=1,
            cache_dir=cache_dir,
            resume=True,
        )

        assert results2.keys() == results1.keys()

    def test_force_bypasses_cache(
        self, sample_scenarios, sample_methods, tmp_path
    ) -> None:
        """Force mode should bypass cache and rerun all."""
        cache_dir = tmp_path / "cache"

        results1 = parallel_simulation_grid(
            scenarios=sample_scenarios,
            methods=sample_methods,
            p_values=[1, 2],
            n_repeats=1,
            n_jobs=1,
            cache_dir=cache_dir,
        )

        results2 = parallel_simulation_grid(
            scenarios=sample_scenarios,
            methods=sample_methods,
            p_values=[1, 2],
            n_repeats=1,
            n_jobs=1,
            cache_dir=cache_dir,
            force=True,
        )

        assert len(results1) > 0
        assert len(results2) > 0


class TestParallelBootstrap:
    """Test parallel bootstrap execution."""

    @pytest.fixture
    def null_model_fit(self) -> dict:
        """Create a fitted null model for testing."""
        T, d = 200, 5
        p0 = 1
        coef = np.random.RandomState(42).randn(p0 * d, d) * 0.3
        residuals = np.random.RandomState(42).randn(T - p0, d)
        X_init = np.random.RandomState(42).randn(p0, d)

        return {
            "coef": coef,
            "residuals": residuals,
            "p0": p0,
            "d": d,
            "init": X_init,
        }

    def test_returns_correct_shape(self, null_model_fit) -> None:
        """Bootstrap results should have shape (B,)."""

        def dummy_analysis(X: np.ndarray) -> float:
            return float(np.sum(X))

        T_boot = parallel_bootstrap(
            X=np.random.randn(200, 5),
            null_model_fit=null_model_fit,
            B=10,
            analysis_func=dummy_analysis,
            n_jobs=1,
            seed=42,
        )

        assert T_boot.shape == (10,)
        assert T_boot.dtype == np.float64

    def test_determinism_with_fixed_seed(self, null_model_fit) -> None:
        """Bootstrap with same seed should produce identical results."""

        def dummy_analysis(X: np.ndarray) -> float:
            return float(np.mean(X))

        X = np.random.RandomState(42).randn(200, 5)

        T_boot1 = parallel_bootstrap(
            X=X,
            null_model_fit=null_model_fit,
            B=5,
            analysis_func=dummy_analysis,
            n_jobs=1,
            seed=42,
        )

        T_boot2 = parallel_bootstrap(
            X=X,
            null_model_fit=null_model_fit,
            B=5,
            analysis_func=dummy_analysis,
            n_jobs=1,
            seed=42,
        )

        np.testing.assert_array_equal(T_boot1, T_boot2)

    def test_caching_works(self, null_model_fit, tmp_path) -> None:
        """Bootstrap caching should work."""

        def dummy_analysis(X: np.ndarray) -> float:
            return float(np.sum(X))

        X = np.random.RandomState(42).randn(200, 5)

        parallel_bootstrap(
            X=X,
            null_model_fit=null_model_fit,
            B=5,
            analysis_func=dummy_analysis,
            n_jobs=1,
            seed=42,
            cache_dir=tmp_path,
        )

        cache_files_before = list(tmp_path.glob("*.pkl"))
        assert len(cache_files_before) > 0


class TestParallelOverFish:
    """Test parallel fish analysis."""

    @pytest.fixture
    def fish_data(self) -> list[np.ndarray]:
        """Create sample fish data."""
        return [
            np.random.RandomState(i).randn(100, 3) for i in range(3)
        ]

    def test_runs_on_all_fish(self, fish_data) -> None:
        """Should process all fish."""

        def dummy_analysis(X: np.ndarray) -> float:
            return float(np.mean(X))

        results = parallel_over_fish(
            fish_list=fish_data,
            analysis_func=dummy_analysis,
            n_jobs=1,
        )

        assert len(results) == len(fish_data)

    def test_results_are_ordered(self, fish_data) -> None:
        """Results should be in the same order as input fish."""

        def dummy_analysis(X: np.ndarray) -> float:
            return float(X.shape[0])

        results = parallel_over_fish(
            fish_list=fish_data,
            analysis_func=dummy_analysis,
            n_jobs=1,
        )

        for i, result in enumerate(results):
            if result is not None:
                assert result["fish_idx"] == i

    def test_caching_skips_repeated_calls(
        self, fish_data, tmp_path
    ) -> None:
        """Caching should reuse values without dropping result positions."""

        calls = 0
        def dummy_analysis(X: np.ndarray) -> float:
            nonlocal calls
            calls += 1
            return float(np.sum(X))

        results1 = parallel_over_fish(
            fish_list=fish_data,
            analysis_func=dummy_analysis,
            n_jobs=1,
            cache_dir=tmp_path,
        )

        results2 = parallel_over_fish(
            fish_list=fish_data,
            analysis_func=dummy_analysis,
            n_jobs=1,
            cache_dir=tmp_path,
        )

        assert calls == len(fish_data)
        assert results2 == results1
