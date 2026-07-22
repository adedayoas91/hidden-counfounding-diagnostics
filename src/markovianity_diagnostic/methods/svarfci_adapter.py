"""Placeholder for SVAR-FCI (Structural Vector Autoregression Fully Causal approach).

IMPLEMENTATION STATUS: DEFERRED

This module is a placeholder for SVAR-FCI, which handles latent confounding in
time series by combining structural VAR constraints with FCI-like logic.

RATIONALE FOR DEFERRAL:
- SVAR-FCI availability and stability across platforms needs verification
- LPCMCI provides sufficient latent-confounder handling for initial submission
- Can be added in future phases once validation complete

FUTURE IMPLEMENTATION:
When SVAR-FCI becomes available, this module should follow the same pattern as
LPCMCIAdapter: normalize outputs to the standard schema {adjacency, edge_marks,
raw_graph, metadata}.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class SVARFCIAdapter:
    """Placeholder for SVAR-FCI adapter (deferred implementation).

    Raises
    ------
    NotImplementedError
        Always, since SVAR-FCI implementation is deferred.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize SVAR-FCI adapter (deferred)."""
        raise NotImplementedError(
            "SVAR-FCI adapter is deferred. "
            "LPCMCI is sufficient for current submission. "
            "See svarfci_adapter.py module docstring for rationale."
        )

    def fit(self, X: np.ndarray, p_values_to_test: list[int]) -> dict[int, dict[str, Any]]:
        """Run SVAR-FCI (deferred).

        Parameters
        ----------
        X : np.ndarray
            Observed data.
        p_values_to_test : list[int]
            Depths to test.

        Raises
        ------
        NotImplementedError
            Always, since implementation is deferred.
        """
        raise NotImplementedError(
            "SVAR-FCI adapter is deferred. "
            "LPCMCI is sufficient for current submission. "
            "See svarfci_adapter.py module docstring for rationale."
        )
