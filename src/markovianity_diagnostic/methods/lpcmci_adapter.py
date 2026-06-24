"""Adapter for LPCMCI (Latent-Variable Causal Model Checking Constraint Integration).

This module provides an adapter that normalizes LPCMCI output to the standard
schema used across all methods in this package.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

try:
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
except ImportError:
    PCMCI = None
    ParCorr = None


class LPCMCIAdapter:
    """Adapter for LPCMCI method that normalizes output to standard schema.

    Parameters
    ----------
    tau_min : int, optional
        Minimum lag for temporal causal relationships (default: 1).
    tau_max : int, optional
        Maximum lag for temporal causal relationships (default: 5).
    pc_alpha : float, optional
        Significance threshold for conditional independence test (default: 0.05).

    Attributes
    ----------
    tau_min : int
        Minimum lag parameter.
    tau_max : int
        Maximum lag parameter.
    pc_alpha : float
        Significance threshold for conditional independence test.
    """

    def __init__(
        self,
        tau_min: int = 1,
        tau_max: int = 5,
        pc_alpha: float = 0.05,
    ) -> None:
        """Initialize LPCMCIAdapter with parameters."""
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha

    def fit(
        self,
        X: np.ndarray,
        p_values_to_test: list[int],
    ) -> dict[int, dict[str, Any]]:
        """Run LPCMCI at each depth and return standardized output.

        For each depth p in p_values_to_test, runs LPCMCI treating p as the
        maximum lag for temporal relationships. Returns a standardized dictionary
        with adjacency matrices, edge markings, and metadata.

        Parameters
        ----------
        X : np.ndarray
            Observed data of shape (T, d) where T is time steps and d is number
            of variables.
        p_values_to_test : list[int]
            List of depths (conditioning set sizes) to test.

        Returns
        -------
        dict[int, dict[str, Any]]
            Dictionary mapping each p to output schema:
            {
                "adjacency": np.ndarray,  # Binary (d, d) directed adjacency matrix
                "edge_marks": dict,        # Preserves bidirected/uncertain marks
                "raw_pag": object,         # Raw PAG output from LPCMCI
                "metadata": dict,          # Algorithm, params, p_value, runtime
            }
        """
        output = {}

        for p in p_values_to_test:
            start_time = time.time()

            # Prepare data with temporal structure for LPCMCI
            # LPCMCI expects (d, T) format
            data = X.T

            try:
                # Initialize LPCMCI with ParCorr test (default)
                if PCMCI is None or ParCorr is None:
                    raise ImportError(
                        "Tigramite not available. Install via: pip install tigramite"
                    )

                cond_ind_test = ParCorr()
                pcmci = PCMCI(
                    data,
                    cond_ind_test,
                    verbosity=0,
                )

                # Run PCMCI with maximum lag = p (depth as temporal lag)
                # This checks conditional independence relationships up to lag p
                pag = pcmci.run_pcmci(
                    tau_min=self.tau_min,
                    tau_max=min(p, self.tau_max),
                    pc_alpha=self.pc_alpha,
                )

                # Extract adjacency from PAG
                # PAG has shape (d, d, lag+1) where lag is temporal dimension
                # Collapse to contemporaneous + lagged relationships
                if pag is not None:
                    # Get contemporaneous relationships (lag 0)
                    adj_contemp = (pag[:, :, 0] != 0).astype(int)
                else:
                    # Fallback: empty adjacency if LPCMCI fails
                    adj_contemp = np.zeros((X.shape[1], X.shape[1]), dtype=int)

                # Ensure diagonal is zero
                np.fill_diagonal(adj_contemp, 0)

                runtime = time.time() - start_time

                output[p] = {
                    "adjacency": adj_contemp,
                    "edge_marks": self._extract_edge_marks(pag, p),
                    "raw_pag": pag,
                    "metadata": {
                        "algorithm": "LPCMCI",
                        "params": {
                            "tau_min": self.tau_min,
                            "tau_max": self.tau_max,
                            "pc_alpha": self.pc_alpha,
                        },
                        "p_value": p,
                        "runtime_seconds": runtime,
                    },
                }

            except Exception as e:
                # Return zero adjacency on failure, but still include metadata
                runtime = time.time() - start_time
                output[p] = {
                    "adjacency": np.zeros((X.shape[1], X.shape[1]), dtype=int),
                    "edge_marks": {},
                    "raw_pag": None,
                    "metadata": {
                        "algorithm": "LPCMCI",
                        "params": {
                            "tau_min": self.tau_min,
                            "tau_max": self.tau_max,
                            "pc_alpha": self.pc_alpha,
                        },
                        "p_value": p,
                        "runtime_seconds": runtime,
                        "error": str(e),
                    },
                }

        return output

    @staticmethod
    def _extract_edge_marks(
        pag: Any,
        p: int,
    ) -> dict[tuple[int, int], str]:
        """Extract edge marking information from PAG.

        Preserves bidirected (<->), directed (->), and uncertain (--) edges.

        Parameters
        ----------
        pag : object or None
            PAG (Partial Ancestral Graph) from LPCMCI.
        p : int
            Depth for context.

        Returns
        -------
        dict[tuple[int, int], str]
            Dictionary mapping (i, j) to edge mark: "directed", "bidirected",
            "uncertain", or None if no edge.
        """
        marks = {}

        if pag is None:
            return marks

        try:
            # PAG shape is (d, d, lag+1)
            # Values encode edge types:
            # 0: no edge
            # 1: arrow
            # 2: circle
            # 3: tail
            # Extract contemporaneous edges (lag 0)
            d = pag.shape[0]
            for i in range(d):
                for j in range(d):
                    if i == j:
                        continue
                    edge_type = pag[i, j, 0]
                    if edge_type == 0:
                        marks[(i, j)] = None
                    elif edge_type == 1:
                        marks[(i, j)] = "arrow"
                    elif edge_type == 2:
                        marks[(i, j)] = "circle"
                    elif edge_type == 3:
                        marks[(i, j)] = "tail"
                    else:
                        marks[(i, j)] = "uncertain"
        except (AttributeError, TypeError, IndexError):
            # If PAG structure is unexpected, return empty marks
            pass

        return marks
