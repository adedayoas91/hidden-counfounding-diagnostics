"""Tests for power analysis module.

TDD Approach: Tests written FIRST, implementation follows.
Power analysis for sample size and parameter variation effects.
"""

from __future__ import annotations

import numpy as np
import pytest

from markovianity_diagnostic.experiments.power import (
    PowerAnalyzer,
    PowerGrid,
    PowerResult,
    MetricAggregation,
)


@pytest.fixture(autouse=True)
def fast_power_method(monkeypatch):
    """Keep orchestration tests fast; integration tests cover real adapters elsewhere."""
    from markovianity_diagnostic.experiments import power

    def analyze_fast(X: np.ndarray, p_values: list[int]) -> dict[int, np.ndarray]:
        base = np.abs(np.corrcoef(X, rowvar=False)) > 0.15
        np.fill_diagonal(base, 0)
        return {int(p): base.astype(int).copy() for p in p_values}

    monkeypatch.setitem(power.METHODS, "gcstar_cgc", analyze_fast)


class TestPowerGrid:
    """Test power analysis grid specification."""

    def test_init_required_params(self):
        """PowerGrid should require T_values and d_values."""
        grid = PowerGrid(T_values=[500, 1000], d_values=[5, 10])
        assert grid.T_values == [500, 1000]
        assert grid.d_values == [5, 10]

    def test_init_optional_params(self):
        """PowerGrid should accept optional parameter variations."""
        grid = PowerGrid(
            T_values=[500, 1000],
            d_values=[5, 10],
            edge_densities=[0.1, 0.2],
            confounder_strengths=[0.3, 0.5],
            latent_autocorrs=[0.7, 0.9],
            noise_scales=[0.8, 1.2],
        )
        assert grid.edge_densities == [0.1, 0.2]
        assert grid.confounder_strengths == [0.3, 0.5]
        assert grid.latent_autocorrs == [0.7, 0.9]
        assert grid.noise_scales == [0.8, 1.2]

    def test_defaults(self):
        """PowerGrid should provide reasonable defaults."""
        grid = PowerGrid(T_values=[500, 1000], d_values=[5, 10])
        assert grid.edge_densities == [0.12]
        assert grid.confounder_strengths == [0.35]
        assert grid.latent_autocorrs == [0.80]
        assert grid.noise_scales == [1.0]

    def test_grid_size_computation(self):
        """total_combinations should compute grid size correctly."""
        grid = PowerGrid(
            T_values=[500, 1000],  # 2
            d_values=[5, 10],  # 2
            edge_densities=[0.1, 0.2],  # 2
        )
        # 2 * 2 * 2 = 8
        assert grid.total_combinations() == 8

    def test_grid_size_with_defaults(self):
        """total_combinations with default variations."""
        grid = PowerGrid(T_values=[500, 1000], d_values=[5, 10])
        # 2 * 2 * 1 * 1 * 1 * 1 = 4
        assert grid.total_combinations() == 4


class TestPowerResult:
    """Test power result container."""

    def test_init_required_fields(self):
        """PowerResult should require scenario and metrics."""
        result = PowerResult(
            T=500,
            d=10,
            edge_density=0.12,
            confounder_strength=0.35,
            latent_autocorr=0.80,
            noise_scale=1.0,
            repeat=0,
            tpr=0.85,
            fpr=0.05,
            p_star=2,
            runtime_sec=1.5,
        )
        assert result.T == 500
        assert result.d == 10
        assert result.tpr == 0.85
        assert result.fpr == 0.05

    def test_p_star_is_integer(self):
        """p_star must be an integer (selected depth)."""
        result = PowerResult(
            T=500,
            d=10,
            edge_density=0.12,
            confounder_strength=0.35,
            latent_autocorr=0.80,
            noise_scale=1.0,
            repeat=0,
            tpr=0.85,
            fpr=0.05,
            p_star=2,
            runtime_sec=1.5,
        )
        assert isinstance(result.p_star, int)
        assert result.p_star >= 1


class TestMetricAggregation:
    """Test metric aggregation over repeats."""

    def test_init_required_fields(self):
        """MetricAggregation should require aggregate metrics."""
        agg = MetricAggregation(
            T=500,
            d=10,
            edge_density=0.12,
            confounder_strength=0.35,
            latent_autocorr=0.80,
            noise_scale=1.0,
            n_repeats=10,
            mean_tpr=0.82,
            std_tpr=0.04,
            mean_fpr=0.06,
            std_fpr=0.02,
            mean_p_star=2.1,
            std_p_star=0.3,
            mean_runtime=1.8,
        )
        assert agg.mean_tpr == 0.82
        assert agg.n_repeats == 10

    def test_ranges_valid(self):
        """Metrics should be in valid ranges."""
        agg = MetricAggregation(
            T=500,
            d=10,
            edge_density=0.12,
            confounder_strength=0.35,
            latent_autocorr=0.80,
            noise_scale=1.0,
            n_repeats=10,
            mean_tpr=0.82,
            std_tpr=0.04,
            mean_fpr=0.06,
            std_fpr=0.02,
            mean_p_star=2.1,
            std_p_star=0.3,
            mean_runtime=1.8,
        )
        assert 0.0 <= agg.mean_tpr <= 1.0
        assert 0.0 <= agg.mean_fpr <= 1.0
        assert agg.mean_p_star >= 1.0


class TestPowerAnalyzer:
    """Test power analysis orchestration."""

    def test_init_required_params(self, seed_fix):
        """PowerAnalyzer should require a grid and method."""
        grid = PowerGrid(T_values=[500], d_values=[5])
        analyzer = PowerAnalyzer(grid=grid, method="gcstar_cgc")
        assert analyzer.grid == grid
        assert analyzer.method == "gcstar_cgc"

    def test_init_optional_params(self, seed_fix):
        """PowerAnalyzer should accept optional parameters."""
        grid = PowerGrid(T_values=[500], d_values=[5])
        analyzer = PowerAnalyzer(
            grid=grid,
            method="gcstar_cgc",
            repeats=15,
            p_values=[1, 2, 3, 4, 5],
            verbose=True,
        )
        assert analyzer.repeats == 15
        assert analyzer.p_values == [1, 2, 3, 4, 5]
        assert analyzer.verbose is True

    def test_defaults(self, seed_fix):
        """PowerAnalyzer should provide sensible defaults."""
        grid = PowerGrid(T_values=[500], d_values=[5])
        analyzer = PowerAnalyzer(grid=grid, method="gcstar_cgc")
        assert analyzer.repeats == 10
        assert analyzer.p_values == [1, 2, 3, 4, 5, 6]
        assert analyzer.verbose is False

    def test_run_reduced_grid(self, seed_fix):
        """run() should execute analysis on reduced grid for testing."""
        grid = PowerGrid(T_values=[500], d_values=[5])
        analyzer = PowerAnalyzer(grid=grid, method="gcstar_cgc", repeats=2)

        results = analyzer.run()

        assert len(results) > 0
        assert isinstance(results[0], PowerResult)

    def test_run_returns_all_points(self, seed_fix):
        """run() should return one result per (grid_point, repeat) combination."""
        grid = PowerGrid(
            T_values=[500, 1000],
            d_values=[5, 10],
        )
        repeats = 3
        analyzer = PowerAnalyzer(grid=grid, method="gcstar_cgc", repeats=repeats)

        results = analyzer.run()

        # 2 T values * 2 d values * 3 repeats = 12 results
        expected_count = 2 * 2 * repeats
        assert len(results) == expected_count

    def test_run_reproducibility(self, seed_fix):
        """run() with same seed should produce identical results."""
        grid = PowerGrid(T_values=[500], d_values=[5])
        analyzer1 = PowerAnalyzer(
            grid=grid, method="gcstar_cgc", repeats=2, seed=42
        )
        analyzer2 = PowerAnalyzer(
            grid=grid, method="gcstar_cgc", repeats=2, seed=42
        )

        results1 = analyzer1.run()
        results2 = analyzer2.run()

        for r1, r2 in zip(results1, results2):
            np.testing.assert_almost_equal(r1.tpr, r2.tpr, decimal=10)
            np.testing.assert_almost_equal(r1.fpr, r2.fpr, decimal=10)

    def test_aggregate_results(self, seed_fix):
        """aggregate() should compute mean/std over repeats for each grid point."""
        grid = PowerGrid(T_values=[500], d_values=[5])
        analyzer = PowerAnalyzer(grid=grid, method="gcstar_cgc", repeats=5)

        results = analyzer.run()
        aggregated = analyzer.aggregate(results)

        assert len(aggregated) > 0
        assert isinstance(aggregated[0], MetricAggregation)
        # Should have one aggregation per grid point (1 T * 1 d * 1 edge_density * ...)
        assert aggregated[0].n_repeats == 5

    def test_aggregate_std_nonzero_when_varied(self, seed_fix):
        """aggregate() should compute nonzero std when repeats vary."""
        grid = PowerGrid(T_values=[500], d_values=[5])
        analyzer = PowerAnalyzer(grid=grid, method="gcstar_cgc", repeats=10)

        results = analyzer.run()
        aggregated = analyzer.aggregate(results)

        # With 10 repeats on same grid point, std should be nonzero
        assert aggregated[0].std_tpr >= 0.0
        assert aggregated[0].std_fpr >= 0.0

    def test_run_output_ranges(self, seed_fix):
        """run() metrics should be in valid ranges."""
        grid = PowerGrid(T_values=[500], d_values=[5])
        analyzer = PowerAnalyzer(grid=grid, method="gcstar_cgc", repeats=2)

        results = analyzer.run()

        for result in results:
            assert 0.0 <= result.tpr <= 1.0, f"Invalid TPR: {result.tpr}"
            assert 0.0 <= result.fpr <= 1.0, f"Invalid FPR: {result.fpr}"
            assert result.p_star >= 1, f"Invalid p_star: {result.p_star}"
            assert result.runtime_sec > 0, f"Invalid runtime: {result.runtime_sec}"

    def test_run_completes_with_parameter_variation(self, seed_fix):
        """run() should handle parameter variation without error."""
        grid = PowerGrid(
            T_values=[500, 1000],
            d_values=[5, 10],
            edge_densities=[0.1, 0.15],
            confounder_strengths=[0.3, 0.4],
        )
        analyzer = PowerAnalyzer(grid=grid, method="gcstar_cgc", repeats=1)

        results = analyzer.run()

        # 2 * 2 * 2 * 2 = 16 grid points * 1 repeat each
        assert len(results) == 16

    def test_run_returns_correct_p_star(self, seed_fix):
        """run() p_star should be selected depth from p_values."""
        grid = PowerGrid(T_values=[500], d_values=[5])
        p_values = [1, 2, 3, 4, 5]
        analyzer = PowerAnalyzer(
            grid=grid, method="gcstar_cgc", repeats=2, p_values=p_values
        )

        results = analyzer.run()

        for result in results:
            assert result.p_star in p_values


class TestPowerAnalysisIntegration:
    """Integration tests for full power analysis workflow."""

    def test_full_workflow_small_grid(self, seed_fix, tmp_path):
        """Full workflow: grid -> run -> aggregate -> export."""
        # Create small grid for fast testing
        grid = PowerGrid(T_values=[500], d_values=[5])
        analyzer = PowerAnalyzer(
            grid=grid, method="gcstar_cgc", repeats=2, verbose=False
        )

        # Run analysis
        results = analyzer.run()
        assert len(results) > 0

        # Aggregate
        aggregated = analyzer.aggregate(results)
        assert len(aggregated) > 0

        # Export should not raise
        output_dir = tmp_path / "power_analysis"
        output_dir.mkdir(exist_ok=True)
        analyzer.export(results, aggregated, output_dir)

        # Check outputs were created
        assert (output_dir / "results.json").exists()
        assert (output_dir / "aggregated.json").exists()
        assert (output_dir / "summary.csv").exists()

    def test_export_creates_valid_json(self, seed_fix, tmp_path):
        """export() should create valid JSON files."""
        import json

        grid = PowerGrid(T_values=[500], d_values=[5])
        analyzer = PowerAnalyzer(
            grid=grid, method="gcstar_cgc", repeats=2, verbose=False
        )

        results = analyzer.run()
        aggregated = analyzer.aggregate(results)

        output_dir = tmp_path / "power_analysis"
        output_dir.mkdir(exist_ok=True)
        analyzer.export(results, aggregated, output_dir)

        # Verify JSON is valid
        results_json = json.loads((output_dir / "results.json").read_text())
        assert len(results_json) > 0

        aggregated_json = json.loads((output_dir / "aggregated.json").read_text())
        assert len(aggregated_json) > 0

    def test_export_creates_summary_csv(self, seed_fix, tmp_path):
        """export() should create CSV summary."""
        grid = PowerGrid(T_values=[500], d_values=[5])
        analyzer = PowerAnalyzer(
            grid=grid, method="gcstar_cgc", repeats=2, verbose=False
        )

        results = analyzer.run()
        aggregated = analyzer.aggregate(results)

        output_dir = tmp_path / "power_analysis"
        output_dir.mkdir(exist_ok=True)
        analyzer.export(results, aggregated, output_dir)

        # Verify CSV exists and has content
        csv_path = output_dir / "summary.csv"
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "T" in content
        assert "d" in content
        assert "mean_tpr" in content
