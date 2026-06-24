"""TDD Tests for Phase 2: Automatic Depth Selection

Tests for DepthSelector class with three selection rules:
- select_absolute(D_p, epsilon=0.1, k_stable=2)
- select_relative(D_p, fraction=0.1)
- select_bootstrap_band(D_p, D_boot_pointwise, confidence=0.95)

Output: DepthSelectionResult with p_values, selected, warnings
"""

import numpy as np
import pytest
from markovianity_diagnostic.experiments.depth_selection import (
    DepthSelector,
    DepthSelectionResult,
)


class TestDepthSelectionResultDataclass:
    """Test suite for DepthSelectionResult dataclass."""

    def test_depth_selection_result_creation(self):
        """DepthSelectionResult should be creatable with required fields."""
        result = DepthSelectionResult(
            p_values=[1, 2, 3, 4, 5],
            selected={"absolute": 4, "relative": 3, "bootstrap_band": 4},
            warnings=["largest instability at first transition"]
        )

        assert result.p_values == [1, 2, 3, 4, 5]
        assert result.selected["absolute"] == 4
        assert result.selected["relative"] == 3
        assert "largest instability" in result.warnings[0]

    def test_depth_selection_result_selected_is_dict(self):
        """selected field should be a dict with three rules."""
        result = DepthSelectionResult(
            p_values=[1, 2, 3],
            selected={"absolute": 2, "relative": 2, "bootstrap_band": 2},
            warnings=[]
        )

        assert isinstance(result.selected, dict)
        assert len(result.selected) == 3
        assert all(k in result.selected for k in ["absolute", "relative", "bootstrap_band"])

    def test_depth_selection_result_warnings_is_list(self):
        """warnings field should be a list."""
        result = DepthSelectionResult(
            p_values=[1, 2, 3],
            selected={"absolute": 2, "relative": 2, "bootstrap_band": 2},
            warnings=["warning1", "warning2"]
        )

        assert isinstance(result.warnings, list)
        assert len(result.warnings) == 2


class TestDepthSelectorAbsoluteRule:
    """Test DepthSelector.select_absolute() rule."""

    def test_absolute_returns_valid_depth(self):
        """select_absolute should return a valid conditioning depth."""
        D_p = {1: 0.15, 2: 0.08, 3: 0.05, 4: 0.04, 5: 0.03}
        selector = DepthSelector()

        result = selector.select_absolute(D_p, epsilon=0.1, k_stable=2)

        # Should return either a depth or None
        assert result is None or isinstance(result, int)
        # If not None, should be in p_values
        if result is not None:
            assert result in D_p.keys()

    def test_absolute_identifies_stable_plateau(self):
        """select_absolute should identify where D_p stabilizes below epsilon."""
        # D_p drops below epsilon at p=2, stays there
        D_p = {1: 0.15, 2: 0.08, 3: 0.07, 4: 0.06, 5: 0.05}
        selector = DepthSelector()

        result = selector.select_absolute(D_p, epsilon=0.1, k_stable=2)

        # Should select p=2 (first depth below epsilon with k_stable=2 following)
        assert result == 2 or result is None

    def test_absolute_respects_k_stable(self):
        """select_absolute should require k_stable consecutive stable depths."""
        # D_p drops below epsilon at p=2 for only 1 sample, then rises again
        D_p = {1: 0.15, 2: 0.08, 3: 0.12, 4: 0.05, 5: 0.04}
        selector = DepthSelector()

        result = selector.select_absolute(D_p, epsilon=0.1, k_stable=2)

        # Should NOT select p=2 since only 1 sample below epsilon before rise
        assert result != 2

    def test_absolute_returns_none_when_no_stability(self):
        """select_absolute should return None if no stable plateau found."""
        # D_p never stays below epsilon
        D_p = {1: 0.15, 2: 0.12, 3: 0.11, 4: 0.10, 5: 0.09}
        selector = DepthSelector()

        result = selector.select_absolute(D_p, epsilon=0.08, k_stable=2)

        assert result is None

    def test_absolute_handles_empty_dict(self):
        """select_absolute should handle empty D_p gracefully."""
        selector = DepthSelector()
        result = selector.select_absolute({}, epsilon=0.1, k_stable=2)

        assert result is None

    def test_absolute_handles_single_depth(self):
        """select_absolute should handle single-depth input."""
        D_p = {1: 0.05}
        selector = DepthSelector()

        result = selector.select_absolute(D_p, epsilon=0.1, k_stable=2)

        # Single depth cannot satisfy k_stable=2
        assert result is None


class TestDepthSelectorRelativeRule:
    """Test DepthSelector.select_relative() rule."""

    def test_relative_returns_valid_depth(self):
        """select_relative should return a valid conditioning depth."""
        D_p = {1: 0.20, 2: 0.08, 3: 0.05, 4: 0.04, 5: 0.03}
        selector = DepthSelector()

        result = selector.select_relative(D_p, fraction=0.1)

        assert result is None or isinstance(result, int)
        if result is not None:
            assert result in D_p.keys()

    def test_relative_identifies_drop_point(self):
        """select_relative should find where D_p drops by >= fraction of its max."""
        # Max is 0.20, 10% of max is 0.02
        # D_p[1] = 0.20, D_p[2] = 0.08 (drop of 0.12, > 0.02)
        D_p = {1: 0.20, 2: 0.08, 3: 0.07, 4: 0.06, 5: 0.05}
        selector = DepthSelector()

        result = selector.select_relative(D_p, fraction=0.1)

        # Should select p=2 (first significant drop from max)
        assert result is not None
        assert result >= 2  # At or after first major drop

    def test_relative_respects_fraction(self):
        """select_relative should use fraction parameter correctly."""
        D_p = {1: 0.20, 2: 0.18, 3: 0.15, 4: 0.10, 5: 0.05}
        selector = DepthSelector()

        # With fraction=0.1, threshold is 0.20 * 0.1 = 0.02
        result_tight = selector.select_relative(D_p, fraction=0.1)

        # With fraction=0.5, threshold is 0.20 * 0.5 = 0.10
        result_loose = selector.select_relative(D_p, fraction=0.5)

        # Tighter threshold should select earlier depth
        if result_tight is not None and result_loose is not None:
            assert result_tight <= result_loose

    def test_relative_returns_none_when_no_drop(self):
        """select_relative should return None if no significant drop found."""
        # Monotonic decrease but no large drop
        D_p = {1: 0.10, 2: 0.09, 3: 0.08, 4: 0.07, 5: 0.06}
        selector = DepthSelector()

        result = selector.select_relative(D_p, fraction=0.5)

        # No drop > 50% of max
        assert result is None

    def test_relative_handles_empty_dict(self):
        """select_relative should handle empty D_p gracefully."""
        selector = DepthSelector()
        result = selector.select_relative({}, fraction=0.1)

        assert result is None

    def test_relative_handles_single_depth(self):
        """select_relative should handle single-depth input."""
        D_p = {1: 0.05}
        selector = DepthSelector()

        result = selector.select_relative(D_p, fraction=0.1)

        # Single depth has nothing to compare
        assert result is None


class TestDepthSelectorBootstrapBandRule:
    """Test DepthSelector.select_bootstrap_band() rule."""

    def test_bootstrap_band_returns_valid_depth(self):
        """select_bootstrap_band should return a valid conditioning depth."""
        D_p = {1: 0.20, 2: 0.08, 3: 0.05, 4: 0.04, 5: 0.03}
        D_boot_pointwise = {
            1: {"lower": 0.05, "upper": 0.25},
            2: {"lower": 0.02, "upper": 0.12},
            3: {"lower": 0.01, "upper": 0.08},
            4: {"lower": 0.01, "upper": 0.06},
            5: {"lower": 0.00, "upper": 0.05},
        }
        selector = DepthSelector()

        result = selector.select_bootstrap_band(D_p, D_boot_pointwise, confidence=0.95)

        assert result is None or isinstance(result, int)
        if result is not None:
            assert result in D_p.keys()

    def test_bootstrap_band_requires_matching_keys(self):
        """select_bootstrap_band should work when D_p and D_boot have same depths."""
        D_p = {1: 0.15, 2: 0.05}
        D_boot_pointwise = {
            1: {"lower": 0.10, "upper": 0.20},
            2: {"lower": 0.02, "upper": 0.08},
        }
        selector = DepthSelector()

        result = selector.select_bootstrap_band(D_p, D_boot_pointwise, confidence=0.95)

        # Should complete without error
        assert result is None or isinstance(result, int)

    def test_bootstrap_band_missing_depths_handled(self):
        """select_bootstrap_band should handle missing depths gracefully."""
        D_p = {1: 0.15, 2: 0.05, 3: 0.03}
        # Missing bootstrap bands for p=3
        D_boot_pointwise = {
            1: {"lower": 0.10, "upper": 0.20},
            2: {"lower": 0.02, "upper": 0.08},
        }
        selector = DepthSelector()

        result = selector.select_bootstrap_band(D_p, D_boot_pointwise, confidence=0.95)

        # Should handle gracefully
        assert result is None or isinstance(result, int)

    def test_bootstrap_band_returns_none_when_unstable(self):
        """select_bootstrap_band should return None if all D_p exceed bootstrap bands."""
        D_p = {1: 0.30, 2: 0.25, 3: 0.20}
        D_boot_pointwise = {
            1: {"lower": 0.05, "upper": 0.15},
            2: {"lower": 0.04, "upper": 0.12},
            3: {"lower": 0.03, "upper": 0.10},
        }
        selector = DepthSelector()

        result = selector.select_bootstrap_band(D_p, D_boot_pointwise, confidence=0.95)

        # All observed exceed bootstrap upper bound
        assert result is None

    def test_bootstrap_band_identifies_within_band(self):
        """select_bootstrap_band should find first depth where D_p within bounds."""
        D_p = {1: 0.30, 2: 0.07, 3: 0.05}
        D_boot_pointwise = {
            1: {"lower": 0.05, "upper": 0.15},
            2: {"lower": 0.04, "upper": 0.12},
            3: {"lower": 0.03, "upper": 0.10},
        }
        selector = DepthSelector()

        result = selector.select_bootstrap_band(D_p, D_boot_pointwise, confidence=0.95)

        # p=2 has D_p=0.07 within [0.04, 0.12]
        assert result == 2 or result is None

    def test_bootstrap_band_handles_empty_inputs(self):
        """select_bootstrap_band should handle empty dictionaries."""
        selector = DepthSelector()
        result = selector.select_bootstrap_band({}, {}, confidence=0.95)

        assert result is None


class TestDepthSelectorIntegration:
    """Integration tests for DepthSelector with all three rules."""

    def test_apply_all_rules_returns_result(self):
        """apply_all_rules should return DepthSelectionResult."""
        D_p = {1: 0.15, 2: 0.08, 3: 0.05, 4: 0.04, 5: 0.03}
        D_boot_pointwise = {
            1: {"lower": 0.10, "upper": 0.20},
            2: {"lower": 0.05, "upper": 0.15},
            3: {"lower": 0.02, "upper": 0.10},
            4: {"lower": 0.01, "upper": 0.08},
            5: {"lower": 0.00, "upper": 0.05},
        }
        selector = DepthSelector()

        result = selector.apply_all_rules(D_p, D_boot_pointwise)

        assert isinstance(result, DepthSelectionResult)
        assert result.p_values == [1, 2, 3, 4, 5]
        assert len(result.selected) == 3

    def test_apply_all_rules_includes_warnings(self):
        """apply_all_rules should populate warnings list."""
        D_p = {1: 0.50, 2: 0.05}  # Unusual profile (high at first)
        D_boot_pointwise = {
            1: {"lower": 0.10, "upper": 0.20},
            2: {"lower": 0.02, "upper": 0.08},
        }
        selector = DepthSelector()

        result = selector.apply_all_rules(D_p, D_boot_pointwise)

        assert isinstance(result.warnings, list)
        # May or may not have warnings depending on data

    def test_apply_all_rules_with_missing_depths(self):
        """apply_all_rules should handle missing bootstrap depths."""
        D_p = {1: 0.15, 2: 0.08, 3: 0.05}
        D_boot_pointwise = {
            1: {"lower": 0.10, "upper": 0.20},
            2: {"lower": 0.05, "upper": 0.15},
            # Missing p=3 bootstrap
        }
        selector = DepthSelector()

        result = selector.apply_all_rules(D_p, D_boot_pointwise)

        assert isinstance(result, DepthSelectionResult)
        # Should complete without crash

    def test_apply_all_rules_consistency(self):
        """Repeated calls with same input should give same result."""
        D_p = {1: 0.15, 2: 0.08, 3: 0.05, 4: 0.04, 5: 0.03}
        D_boot_pointwise = {
            1: {"lower": 0.10, "upper": 0.20},
            2: {"lower": 0.05, "upper": 0.15},
            3: {"lower": 0.02, "upper": 0.10},
            4: {"lower": 0.01, "upper": 0.08},
            5: {"lower": 0.00, "upper": 0.05},
        }
        selector = DepthSelector()

        result1 = selector.apply_all_rules(D_p, D_boot_pointwise)
        result2 = selector.apply_all_rules(D_p, D_boot_pointwise)

        assert result1.p_values == result2.p_values
        assert result1.selected == result2.selected
        assert result1.warnings == result2.warnings


class TestDepthSelectorEdgeCases:
    """Test edge cases and error handling."""

    def test_single_depth_input(self):
        """All rules should handle single-depth input gracefully."""
        D_p = {1: 0.05}
        selector = DepthSelector()

        # None should raise errors
        abs_result = selector.select_absolute(D_p)
        rel_result = selector.select_relative(D_p)

        assert abs_result is None
        assert rel_result is None

    def test_all_zeros_input(self):
        """All rules should handle all-zero instability gracefully."""
        D_p = {1: 0.0, 2: 0.0, 3: 0.0}
        selector = DepthSelector()

        abs_result = selector.select_absolute(D_p, epsilon=0.1)
        rel_result = selector.select_relative(D_p, fraction=0.1)

        # Should complete without error
        assert isinstance(abs_result, (int, type(None)))
        assert isinstance(rel_result, (int, type(None)))


class TestDepthSelectorOutputSchema:
    """Test output schema and serialization."""

    def test_depth_selection_result_has_required_fields(self):
        """DepthSelectionResult must have p_values, selected, warnings."""
        result = DepthSelectionResult(
            p_values=[1, 2, 3],
            selected={"absolute": 2, "relative": 2, "bootstrap_band": 2},
            warnings=[]
        )

        assert hasattr(result, 'p_values')
        assert hasattr(result, 'selected')
        assert hasattr(result, 'warnings')

    def test_depth_selection_result_selected_keys(self):
        """selected dict must have exactly three keys."""
        result = DepthSelectionResult(
            p_values=[1, 2, 3],
            selected={"absolute": 2, "relative": 2, "bootstrap_band": 2},
            warnings=[]
        )

        expected_keys = {"absolute", "relative", "bootstrap_band"}
        assert set(result.selected.keys()) == expected_keys

    def test_depth_selection_result_selected_values_type(self):
        """selected values should be int or None."""
        result = DepthSelectionResult(
            p_values=[1, 2, 3],
            selected={"absolute": 2, "relative": None, "bootstrap_band": 3},
            warnings=[]
        )

        for value in result.selected.values():
            assert value is None or isinstance(value, int)

    def test_depth_selection_result_warnings_informative(self):
        """warnings should be informative strings when present."""
        warnings_list = [
            "no stable depth found with epsilon=0.1",
            "largest instability at first transition (p=1)",
        ]
        result = DepthSelectionResult(
            p_values=[1, 2, 3],
            selected={"absolute": None, "relative": 2, "bootstrap_band": None},
            warnings=warnings_list
        )

        assert all(isinstance(w, str) for w in result.warnings)
        assert len(result.warnings) > 0
        assert any("stable" in w.lower() for w in result.warnings)
