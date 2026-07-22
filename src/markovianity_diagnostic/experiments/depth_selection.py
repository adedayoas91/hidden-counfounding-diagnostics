"""Automatic conditioning-depth selection (Phase 2).

Provides three rules for selecting an appropriate conditioning depth p_star:
1. Absolute rule: First stable plateau below epsilon
2. Relative rule: First major drop from maximum instability
3. Bootstrap-band rule: First depth within bootstrap confidence band
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DepthSelectionResult:
    """Result of depth selection analysis.

    Attributes
    ----------
    p_values : list[int]
        Conditioning depths examined.
    selected : dict[str, int | None]
        Selected depths for each rule: {"absolute": p*, "relative": p*, "bootstrap_band": p*}.
    warnings : list[str]
        Informative warnings about selection reliability.
    """

    p_values: list[int]
    selected: dict[str, Optional[int]]
    warnings: list[str] = field(default_factory=list)


class DepthSelector:
    """Automatic conditioning-depth selection using three heuristic rules."""

    def __init__(self):
        """Initialize DepthSelector."""
        self._last_warnings: list[str] = []

    def select_absolute(
        self,
        D_p: dict[int, float],
        epsilon: float = 0.1,
        k_stable: int = 2,
    ) -> Optional[int]:
        """Select depth by absolute stability criterion.

        Find the first depth p where D_p < epsilon and remains below epsilon
        for the next k_stable depths.

        Parameters
        ----------
        D_p : dict[int, float]
            Dictionary mapping conditioning depth to instability value.
        epsilon : float, optional
            Threshold for stability (default 0.1).
        k_stable : int, optional
            Number of consecutive stable depths required (default 2).

        Returns
        -------
        int or None
            Selected depth, or None if no stable plateau found.
        """
        self._last_warnings = []

        if not D_p:
            return None

        p_values = sorted(D_p.keys())

        if len(p_values) < k_stable + 1:
            return None

        # Check for maximum at first transition (unusual pattern)
        if len(p_values) > 1:
            if D_p[p_values[0]] == max(D_p.values()):
                self._last_warnings.append(
                    f"largest instability at first transition (p={p_values[0]})"
                )

        # Find first depth below epsilon with k_stable consecutive below threshold
        for i, p in enumerate(p_values):
            if D_p[p] < epsilon:
                # Check if next k_stable depths are also below epsilon
                remaining = p_values[i : i + k_stable + 1]
                if len(remaining) > k_stable:
                    if all(D_p[q] < epsilon for q in remaining[: k_stable + 1]):
                        return p

        return None

    def select_relative(
        self,
        D_p: dict[int, float],
        fraction: float = 0.1,
    ) -> Optional[int]:
        """Select depth by relative drop from maximum instability.

        Find the first depth p where D_p drops by >= fraction * max(D_p)
        from the maximum.

        Parameters
        ----------
        D_p : dict[int, float]
            Dictionary mapping conditioning depth to instability value.
        fraction : float, optional
            Fraction of maximum drop required (default 0.1).

        Returns
        -------
        int or None
            Selected depth, or None if no significant drop found.
        """
        if not D_p or len(D_p) < 2:
            return None

        p_values = sorted(D_p.keys())
        max_val = max(D_p.values())
        threshold = fraction * max_val

        # Find first depth where drop from max >= threshold
        for i, p in enumerate(p_values):
            drop = max_val - D_p[p]
            if drop >= threshold:
                return p

        return None

    def select_bootstrap_band(
        self,
        D_p: dict[int, float],
        D_boot_pointwise: dict[int, dict[str, float]],
        confidence: float = 0.95,
    ) -> Optional[int]:
        """Select depth by bootstrap confidence band criterion.

        Find the first depth p where D_p falls within the bootstrap
        pointwise confidence band.

        Parameters
        ----------
        D_p : dict[int, float]
            Dictionary mapping conditioning depth to instability value.
        D_boot_pointwise : dict[int, dict[str, float]]
            Pointwise bootstrap bands: {p: {"lower": val, "upper": val}}.
        confidence : float, optional
            Confidence level (for documentation; not used directly).

        Returns
        -------
        int or None
            Selected depth, or None if no depth within band found.
        """
        if not D_p or not D_boot_pointwise:
            return None

        p_values = sorted(D_p.keys())

        # Find first depth where D_p is within bootstrap band
        for p in p_values:
            if p not in D_boot_pointwise:
                # Skip depths without bootstrap bands
                continue

            band = D_boot_pointwise[p]
            lower = band.get("lower", -np.inf)
            upper = band.get("upper", np.inf)

            if lower <= D_p[p] <= upper:
                return p

        return None

    def apply_all_rules(
        self,
        D_p: dict[int, float],
        D_boot_pointwise: dict[int, dict[str, float]],
        epsilon: float = 0.1,
        k_stable: int = 2,
        fraction: float = 0.1,
        confidence: float = 0.95,
    ) -> DepthSelectionResult:
        """Apply all three selection rules and return unified result.

        Parameters
        ----------
        D_p : dict[int, float]
            Dictionary mapping conditioning depth to instability value.
        D_boot_pointwise : dict[int, dict[str, float]]
            Pointwise bootstrap bands.
        epsilon : float, optional
            Threshold for absolute rule (default 0.1).
        k_stable : int, optional
            Consecutive stable depths for absolute rule (default 2).
        fraction : float, optional
            Fraction drop for relative rule (default 0.1).
        confidence : float, optional
            Confidence level for bootstrap rule (default 0.95).

        Returns
        -------
        DepthSelectionResult
            Result with selected depths for all three rules and warnings.
        """
        p_values = sorted(D_p.keys()) if D_p else []
        warnings = []

        # Apply all three rules
        abs_selected = self.select_absolute(D_p, epsilon=epsilon, k_stable=k_stable)
        rel_selected = self.select_relative(D_p, fraction=fraction)
        boot_selected = self.select_bootstrap_band(
            D_p, D_boot_pointwise, confidence=confidence
        )

        # Collect warnings
        if self._last_warnings:
            warnings.extend(self._last_warnings)

        if abs_selected is None:
            warnings.append(f"no stable depth found with epsilon={epsilon}")

        if rel_selected is None:
            warnings.append(f"no significant drop found with fraction={fraction}")

        if boot_selected is None:
            warnings.append(
                f"no depth within bootstrap band at confidence={confidence}"
            )

        return DepthSelectionResult(
            p_values=p_values,
            selected={
                "absolute": abs_selected,
                "relative": rel_selected,
                "bootstrap_band": boot_selected,
            },
            warnings=warnings,
        )
