"""Bootstrap calibration module for markovianity diagnostics.

Provides NullModel base class and null hypothesis implementations for
calibrating the markovianity test statistic T_obs against bootstrap
null distributions.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict


class NullModel(ABC):
    """Abstract base class for null hypothesis models.

    A NullModel represents a surrogate null distribution for the markovianity
    test. Subclasses implement specific null hypotheses (e.g., VAR, residual
    bootstrap, moving-block bootstrap).

    Attributes:
        None (to be defined by subclasses)
    """

    @abstractmethod
    def fit(self, X: np.ndarray, p0: int) -> "NullModel":
        """Fit the null model to observed data.

        Args:
            X: Observed data array of shape (T, d) where T is time steps
               and d is number of variables.
            p0: Conditioning depth (lag order to fit).

        Returns:
            self: For method chaining.

        Raises:
            ValueError: If X shape is invalid or p0 is invalid.
        """
        pass

    @abstractmethod
    def sample(self, T: int, seed: int) -> np.ndarray:
        """Sample from the fitted null model.

        Args:
            T: Number of time steps to sample.
            seed: Random seed for reproducibility.

        Returns:
            Sampled data array of shape (T, d) where d matches the fitted model.

        Raises:
            ValueError: If model has not been fitted.
        """
        pass

    @property
    @abstractmethod
    def metadata(self) -> Dict:
        """Return metadata about the fitted model.

        Returns:
            Dictionary with keys:
                - 'fitted' (bool): Whether model has been fitted
                - 'n_samples' (int): Number of samples in fit data
                - 'n_features' (int): Number of variables (dimensions)
                - Additional model-specific metadata (subclass-dependent)
        """
        pass
