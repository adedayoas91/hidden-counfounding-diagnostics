"""Tests for Phase 1.4: CLI refactoring with calibration.py integration.

TDD Approach: Tests written FIRST, then CLI implementation updated.
Tests verify:
1. --null-model CLI option works {var, residual, moving-block}
2. --critical-levels accepts multiple values (0.90, 0.95, 0.99)
3. --save-surrogate-summaries flag saves bootstrap summaries
4. --n-jobs parameter for parallelization
5. Output JSON matches manifest schema
6. Reproducibility with --seed
"""

import json
import sys
from pathlib import Path
import tempfile
import numpy as np
import pytest


class TestBootstrapCLIArgumentParsing:
    """Test that CLI arguments are parsed correctly."""

    def test_null_model_option_defaults_to_var(self):
        """--null-model should default to 'var'."""
        from markovianity_diagnostic.experiments.bootstrap_runner import parse_args

        old_argv = sys.argv
        try:
            sys.argv = ["test", "--scenario", "order1_unconfounded", "--output-dir", "/tmp"]
            args = parse_args()
            assert args.null_model == "var"
        finally:
            sys.argv = old_argv

    def test_null_model_option_accepts_var(self):
        """--null-model var should be accepted."""
        from markovianity_diagnostic.experiments.bootstrap_runner import parse_args

        old_argv = sys.argv
        try:
            sys.argv = [
                "test",
                "--scenario", "order1_unconfounded",
                "--output-dir", "/tmp",
                "--null-model", "var"
            ]
            args = parse_args()
            assert args.null_model == "var"
        finally:
            sys.argv = old_argv

    def test_null_model_option_accepts_residual(self):
        """--null-model residual should be accepted."""
        from markovianity_diagnostic.experiments.bootstrap_runner import parse_args

        old_argv = sys.argv
        try:
            sys.argv = [
                "test",
                "--scenario", "order1_unconfounded",
                "--output-dir", "/tmp",
                "--null-model", "residual"
            ]
            args = parse_args()
            assert args.null_model == "residual"
        finally:
            sys.argv = old_argv

    def test_null_model_option_accepts_moving_block(self):
        """--null-model moving-block should be accepted."""
        from markovianity_diagnostic.experiments.bootstrap_runner import parse_args

        old_argv = sys.argv
        try:
            sys.argv = [
                "test",
                "--scenario", "order1_unconfounded",
                "--output-dir", "/tmp",
                "--null-model", "moving-block"
            ]
            args = parse_args()
            assert args.null_model == "moving-block"
        finally:
            sys.argv = old_argv

    def test_critical_levels_option_defaults(self):
        """--critical-levels should default to [0.90, 0.95, 0.99]."""
        from markovianity_diagnostic.experiments.bootstrap_runner import parse_args

        old_argv = sys.argv
        try:
            sys.argv = ["test", "--scenario", "order1_unconfounded", "--output-dir", "/tmp"]
            args = parse_args()
            assert args.critical_levels == [0.90, 0.95, 0.99]
        finally:
            sys.argv = old_argv

    def test_critical_levels_option_accepts_custom_values(self):
        """--critical-levels should accept custom values."""
        from markovianity_diagnostic.experiments.bootstrap_runner import parse_args

        old_argv = sys.argv
        try:
            sys.argv = [
                "test",
                "--scenario", "order1_unconfounded",
                "--output-dir", "/tmp",
                "--critical-levels", "0.85", "0.95"
            ]
            args = parse_args()
            assert args.critical_levels == [0.85, 0.95]
        finally:
            sys.argv = old_argv

    def test_save_surrogate_summaries_defaults_to_false(self):
        """--save-surrogate-summaries should default to False."""
        from markovianity_diagnostic.experiments.bootstrap_runner import parse_args

        old_argv = sys.argv
        try:
            sys.argv = ["test", "--scenario", "order1_unconfounded", "--output-dir", "/tmp"]
            args = parse_args()
            assert args.save_surrogate_summaries is False
        finally:
            sys.argv = old_argv

    def test_save_surrogate_summaries_flag_sets_true(self):
        """--save-surrogate-summaries should set flag to True."""
        from markovianity_diagnostic.experiments.bootstrap_runner import parse_args

        old_argv = sys.argv
        try:
            sys.argv = [
                "test",
                "--scenario", "order1_unconfounded",
                "--output-dir", "/tmp",
                "--save-surrogate-summaries"
            ]
            args = parse_args()
            assert args.save_surrogate_summaries is True
        finally:
            sys.argv = old_argv

    def test_n_jobs_defaults_to_1(self):
        """--n-jobs should default to 1."""
        from markovianity_diagnostic.experiments.bootstrap_runner import parse_args

        old_argv = sys.argv
        try:
            sys.argv = ["test", "--scenario", "order1_unconfounded", "--output-dir", "/tmp"]
            args = parse_args()
            assert args.n_jobs == 1
        finally:
            sys.argv = old_argv

    def test_n_jobs_accepts_integer(self):
        """--n-jobs should accept integer values."""
        from markovianity_diagnostic.experiments.bootstrap_runner import parse_args

        old_argv = sys.argv
        try:
            sys.argv = [
                "test",
                "--scenario", "order1_unconfounded",
                "--output-dir", "/tmp",
                "--n-jobs", "4"
            ]
            args = parse_args()
            assert args.n_jobs == 4
        finally:
            sys.argv = old_argv


class TestBootstrapCLIComputeCriticalValues:
    """Test critical value computation function."""

    def test_compute_critical_values_basic(self):
        """Critical values should be computed correctly."""
        from markovianity_diagnostic.experiments.bootstrap_runner import _compute_critical_values

        T_boot = [1.0, 2.0, 3.0, 4.0, 5.0]
        critical_levels = [0.50, 0.95]

        result = _compute_critical_values(T_boot, critical_levels)

        assert 0.50 in result
        assert 0.95 in result
        assert result[0.50] <= result[0.95]

    def test_compute_critical_values_ordered(self):
        """Critical values should be ordered with quantile level."""
        from markovianity_diagnostic.experiments.bootstrap_runner import _compute_critical_values

        T_boot = list(range(100))
        critical_levels = [0.50, 0.75, 0.95]

        result = _compute_critical_values(T_boot, critical_levels)

        assert result[0.50] <= result[0.75] <= result[0.95]

    def test_compute_critical_values_empty_boots(self):
        """Empty boot list should return zero critical values."""
        from markovianity_diagnostic.experiments.bootstrap_runner import _compute_critical_values

        T_boot = []
        critical_levels = [0.90, 0.95, 0.99]

        result = _compute_critical_values(T_boot, critical_levels)

        assert all(v == 0.0 for v in result.values())


class TestBootstrapCLIMainFunction:
    """Test the main CLI function."""

    def test_main_creates_output_directory(self, tmp_path):
        """main() should create output directory if it doesn't exist."""
        from markovianity_diagnostic.experiments.bootstrap_runner import main

        output_dir = tmp_path / "new_output"
        assert not output_dir.exists()

        old_argv = sys.argv
        try:
            sys.argv = [
                "test",
                "--scenario", "order1_unconfounded",
                "--output-dir", str(output_dir),
                "--T", "100",
                "--B", "5",
                "--p-values", "1", "2"
            ]
            main()
        finally:
            sys.argv = old_argv

        assert output_dir.exists()

    def test_main_creates_manifest_json(self, tmp_path):
        """main() should create manifest.json."""
        from markovianity_diagnostic.experiments.bootstrap_runner import main

        output_dir = tmp_path / "output"

        old_argv = sys.argv
        try:
            sys.argv = [
                "test",
                "--scenario", "order1_unconfounded",
                "--output-dir", str(output_dir),
                "--T", "100",
                "--B", "5",
                "--p-values", "1", "2"
            ]
            main()
        finally:
            sys.argv = old_argv

        manifest_path = output_dir / "manifest.json"
        assert manifest_path.exists()

    def test_main_creates_bootstrap_json(self, tmp_path):
        """main() should create bootstrap.json (CalibrationResult)."""
        from markovianity_diagnostic.experiments.bootstrap_runner import main

        output_dir = tmp_path / "output"

        old_argv = sys.argv
        try:
            sys.argv = [
                "test",
                "--scenario", "order1_unconfounded",
                "--output-dir", str(output_dir),
                "--T", "100",
                "--B", "5",
                "--p-values", "1", "2"
            ]
            main()
        finally:
            sys.argv = old_argv

        bootstrap_path = output_dir / "bootstrap.json"
        assert bootstrap_path.exists()

    def test_manifest_contains_required_fields(self, tmp_path):
        """Manifest should contain all required fields."""
        from markovianity_diagnostic.experiments.bootstrap_runner import main

        output_dir = tmp_path / "output"

        old_argv = sys.argv
        try:
            sys.argv = [
                "test",
                "--scenario", "order1_unconfounded",
                "--output-dir", str(output_dir),
                "--T", "100",
                "--B", "5",
                "--p-values", "1", "2"
            ]
            main()
        finally:
            sys.argv = old_argv

        manifest_path = output_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        required_fields = {
            "created_at",
            "git_commit",
            "analysis",
            "input_paths",
            "output_paths",
            "method",
            "method_params",
            "p_values",
            "random_seed",
            "software_versions"
        }
        assert required_fields.issubset(set(manifest.keys()))

    def test_bootstrap_json_valid_format(self, tmp_path):
        """bootstrap.json should be valid JSON with CalibrationResult schema."""
        from markovianity_diagnostic.experiments.bootstrap_runner import main

        output_dir = tmp_path / "output"

        old_argv = sys.argv
        try:
            sys.argv = [
                "test",
                "--scenario", "order1_unconfounded",
                "--output-dir", str(output_dir),
                "--T", "100",
                "--B", "5",
                "--p-values", "1", "2"
            ]
            main()
        finally:
            sys.argv = old_argv

        bootstrap_path = output_dir / "bootstrap.json"
        with open(bootstrap_path) as f:
            data = json.load(f)

        # Check CalibrationResult schema
        assert "observed" in data
        assert "null" in data
        assert "diagnosis" in data

    def test_critical_values_in_output(self, tmp_path):
        """Output should include critical values from --critical-levels."""
        from markovianity_diagnostic.experiments.bootstrap_runner import main

        output_dir = tmp_path / "output"

        old_argv = sys.argv
        try:
            sys.argv = [
                "test",
                "--scenario", "order1_unconfounded",
                "--output-dir", str(output_dir),
                "--T", "100",
                "--B", "5",
                "--p-values", "1", "2",
                "--critical-levels", "0.90", "0.95", "0.99"
            ]
            main()
        finally:
            sys.argv = old_argv

        bootstrap_path = output_dir / "bootstrap.json"
        with open(bootstrap_path) as f:
            data = json.load(f)

        # Check critical values are present
        null_data = data["null"]
        assert "critical_90" in null_data
        assert "critical_95" in null_data
        assert "critical_99" in null_data

    def test_p_value_in_output_valid_range(self, tmp_path):
        """p_value in output should be in [0, 1]."""
        from markovianity_diagnostic.experiments.bootstrap_runner import main

        output_dir = tmp_path / "output"

        old_argv = sys.argv
        try:
            sys.argv = [
                "test",
                "--scenario", "order1_unconfounded",
                "--output-dir", str(output_dir),
                "--T", "100",
                "--B", "5",
                "--p-values", "1", "2"
            ]
            main()
        finally:
            sys.argv = old_argv

        bootstrap_path = output_dir / "bootstrap.json"
        with open(bootstrap_path) as f:
            data = json.load(f)

        p_value = data["null"]["p_value"]
        assert 0 <= p_value <= 1

    def test_main_with_custom_critical_levels(self, tmp_path):
        """main() should respect --critical-levels option."""
        from markovianity_diagnostic.experiments.bootstrap_runner import main

        output_dir = tmp_path / "output"

        old_argv = sys.argv
        try:
            sys.argv = [
                "test",
                "--scenario", "order1_unconfounded",
                "--output-dir", str(output_dir),
                "--T", "100",
                "--B", "5",
                "--p-values", "1", "2",
                "--critical-levels", "0.85", "0.95"
            ]
            main()
        finally:
            sys.argv = old_argv

        manifest_path = output_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["method_params"]["critical_levels"] == [0.85, 0.95]

    def test_main_with_null_model_option(self, tmp_path):
        """main() should record --null-model in manifest."""
        from markovianity_diagnostic.experiments.bootstrap_runner import main

        output_dir = tmp_path / "output"

        old_argv = sys.argv
        try:
            sys.argv = [
                "test",
                "--scenario", "order1_unconfounded",
                "--output-dir", str(output_dir),
                "--T", "100",
                "--B", "5",
                "--p-values", "1", "2",
                "--null-model", "var"
            ]
            main()
        finally:
            sys.argv = old_argv

        manifest_path = output_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["method"] == "var"
        assert manifest["method_params"]["null_model"] == "var"

    def test_main_reproducible_with_seed(self, tmp_path):
        """Two runs with same seed should produce identical T_obs."""
        from markovianity_diagnostic.experiments.bootstrap_runner import main

        def run_with_seed(seed):
            output_dir = tmp_path / f"output_{seed}"
            old_argv = sys.argv
            try:
                sys.argv = [
                    "test",
                    "--scenario", "order1_unconfounded",
                    "--output-dir", str(output_dir),
                    "--T", "100",
                    "--B", "5",
                    "--p-values", "1", "2",
                    "--seed", str(seed)
                ]
                main()
            finally:
                sys.argv = old_argv

            bootstrap_path = output_dir / "bootstrap.json"
            with open(bootstrap_path) as f:
                return json.load(f)

        result1 = run_with_seed(42)
        result2 = run_with_seed(42)

        assert result1["observed"]["T_obs"] == result2["observed"]["T_obs"]


class TestBootstrapCLIErrorHandling:
    """Test error handling in CLI."""

    def test_invalid_critical_level_raises_error(self, tmp_path):
        """Critical level outside (0, 1) should raise ValueError."""
        from markovianity_diagnostic.experiments.bootstrap_runner import main

        output_dir = tmp_path / "output"

        old_argv = sys.argv
        try:
            sys.argv = [
                "test",
                "--scenario", "order1_unconfounded",
                "--output-dir", str(output_dir),
                "--T", "100",
                "--B", "5",
                "--p-values", "1", "2",
                "--critical-levels", "1.5"
            ]
            with pytest.raises(ValueError):
                main()
        finally:
            sys.argv = old_argv

    def test_invalid_null_model_raises_error(self, tmp_path):
        """Invalid null model choice should raise error."""
        from markovianity_diagnostic.experiments.bootstrap_runner import parse_args

        old_argv = sys.argv
        try:
            sys.argv = [
                "test",
                "--scenario", "order1_unconfounded",
                "--output-dir", str(tmp_path),
                "--null-model", "invalid_model"
            ]
            with pytest.raises(SystemExit):  # argparse exits with error
                parse_args()
        finally:
            sys.argv = old_argv
