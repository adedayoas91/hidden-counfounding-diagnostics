"""Bootstrap calibration module for markovianity diagnostics.

Provides NullModel base class and null hypothesis implementations for
calibrating the markovianity test statistic T_obs against bootstrap
null distributions.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Optional


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


class VARNullModel(NullModel):
    """Fit VAR(p0) and sample from it.
    
    Implements a Vector AutoRegression null model. Fits a VAR(p0) process
    to observed data via OLS and generates samples from the fitted model.
    """

    def __init__(self, p0: int = 1):
        """Initialize VARNullModel.
        
        Args:
            p0: Lag order for VAR model (default 1).
        """
        self.p0 = p0
        self._A: Optional[np.ndarray] = None
        self._cov: Optional[np.ndarray] = None
        self._fitted = False
        self._X_fit: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, p0: int) -> "VARNullModel":
        """Fit VAR(p0) to observed data using OLS.
        
        Args:
            X: Data array of shape (T, d).
            p0: Lag order (overrides __init__ p0).
            
        Returns:
            self
            
        Raises:
            ValueError: If X is invalid or p0 >= T.
        """
        # Validate input
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        
        T, d = X.shape
        if p0 >= T:
            raise ValueError(f"p0={p0} must be < T={T}")
        
        self.p0 = p0
        self._X_fit = X.copy()
        
        # Build design matrix for VAR: stack X[p0:] as target, X[p0-1:-1] as predictor
        # For VAR(p0): X_t = A @ X_{t-1} + epsilon_t
        Y = X[p0:, :].T  # Shape: (d, T-p0)
        Z = X[p0-1:-1, :].T  # Shape: (d, T-p0)
        
        # OLS: A = Y @ Z^T @ (Z @ Z^T)^{-1}
        ZZt = Z @ Z.T  # Shape: (d, d)
        YZt = Y @ Z.T  # Shape: (d, d)
        
        try:
            self._A = YZt @ np.linalg.inv(ZZt)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            self._A = YZt @ np.linalg.pinv(ZZt)
        
        # Compute residuals and covariance
        residuals = X[p0:, :] - X[p0-1:-1, :] @ self._A.T
        self._cov = np.cov(residuals.T, ddof=1)
        
        # Ensure covariance is 2D
        if self._cov.ndim == 0:
            self._cov = np.array([[self._cov]])
        elif self._cov.ndim == 1:
            self._cov = np.diag(self._cov)
        
        self._fitted = True
        return self

    def sample(self, T: int, seed: int) -> np.ndarray:
        """Sample T timesteps from fitted VAR(p0).
        
        Args:
            T: Number of timesteps to sample.
            seed: Random seed for reproducibility.
            
        Returns:
            Array of shape (T, d).
            
        Raises:
            ValueError: If model not fitted.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before sampling")
        
        np.random.seed(seed)
        d = self._X_fit.shape[1]
        
        # Initialize with random starting point
        X_sample = np.zeros((T, d))
        X_sample[0, :] = np.random.randn(d)
        
        # Generate trajectory
        for t in range(1, T):
            noise = np.random.multivariate_normal(np.zeros(d), self._cov)
            X_sample[t, :] = X_sample[t-1, :] @ self._A.T + noise
        
        return X_sample

    @property
    def metadata(self) -> Dict:
        """Return model metadata."""
        return {
            "model": "VAR",
            "p0": self.p0,
            "fitted": self._fitted,
            "n_samples": self._X_fit.shape[0] if self._fitted else None,
            "n_features": self._X_fit.shape[1] if self._fitted else None,
        }


class ResidualBootstrapNull(NullModel):
    """Fit VAR(p0), resample residuals with replacement.
    
    Implements residual bootstrap null model. Fits a VAR(p0) process,
    then generates samples by resampling residuals with replacement
    and reconstructing trajectories.
    """

    def __init__(self, p0: int = 1):
        """Initialize ResidualBootstrapNull.
        
        Args:
            p0: Lag order for VAR model (default 1).
        """
        self.p0 = p0
        self._A: Optional[np.ndarray] = None
        self._residuals: Optional[np.ndarray] = None
        self._fitted = False
        self._X_fit: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, p0: int) -> "ResidualBootstrapNull":
        """Fit VAR(p0) and store residuals.
        
        Args:
            X: Data array of shape (T, d).
            p0: Lag order (overrides __init__ p0).
            
        Returns:
            self
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        
        T, d = X.shape
        if p0 >= T:
            raise ValueError(f"p0={p0} must be < T={T}")
        
        self.p0 = p0
        self._X_fit = X.copy()
        
        # Build design matrix for VAR
        Y = X[p0:, :].T  # Shape: (d, T-p0)
        Z = X[p0-1:-1, :].T  # Shape: (d, T-p0)
        
        # OLS: A = Y @ Z^T @ (Z @ Z^T)^{-1}
        ZZt = Z @ Z.T
        YZt = Y @ Z.T
        
        try:
            self._A = YZt @ np.linalg.inv(ZZt)
        except np.linalg.LinAlgError:
            self._A = YZt @ np.linalg.pinv(ZZt)
        
        # Compute and store residuals
        self._residuals = X[p0:, :] - X[p0-1:-1, :] @ self._A.T
        
        self._fitted = True
        return self

    def sample(self, T: int, seed: int) -> np.ndarray:
        """Sample by resampling residuals with replacement.
        
        Args:
            T: Number of timesteps to sample.
            seed: Random seed for reproducibility.
            
        Returns:
            Array of shape (T, d).
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before sampling")
        
        np.random.seed(seed)
        d = self._X_fit.shape[1]
        n_residuals = self._residuals.shape[0]
        
        # Initialize with random starting point
        X_sample = np.zeros((T, d))
        X_sample[0, :] = np.random.randn(d)
        
        # Generate trajectory by resampling residuals
        for t in range(1, T):
            # Resample one residual with replacement
            idx = np.random.randint(0, n_residuals)
            noise = self._residuals[idx, :]
            X_sample[t, :] = X_sample[t-1, :] @ self._A.T + noise
        
        return X_sample

    @property
    def metadata(self) -> Dict:
        """Return model metadata."""
        return {
            "model": "ResidualBootstrap",
            "p0": self.p0,
            "fitted": self._fitted,
            "n_samples": self._X_fit.shape[0] if self._fitted else None,
            "n_features": self._X_fit.shape[1] if self._fitted else None,
        }


class MovingBlockBootstrapNull(NullModel):
    """Fit VAR(p0), resample blocks of residuals.
    
    Implements moving-block bootstrap null model. Fits a VAR(p0) process,
    then generates samples by resampling contiguous blocks of residuals
    with replacement.
    """

    def __init__(self, p0: int = 1, block_length: int = 20):
        """Initialize MovingBlockBootstrapNull.
        
        Args:
            p0: Lag order for VAR model (default 1).
            block_length: Length of blocks to resample (default 20).
        """
        self.p0 = p0
        self.block_length = block_length
        self._A: Optional[np.ndarray] = None
        self._residuals: Optional[np.ndarray] = None
        self._fitted = False
        self._X_fit: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, p0: int) -> "MovingBlockBootstrapNull":
        """Fit VAR(p0) and store residuals for block resampling.
        
        Args:
            X: Data array of shape (T, d).
            p0: Lag order (overrides __init__ p0).
            
        Returns:
            self
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        
        T, d = X.shape
        if p0 >= T:
            raise ValueError(f"p0={p0} must be < T={T}")
        
        self.p0 = p0
        self._X_fit = X.copy()
        
        # Build design matrix for VAR
        Y = X[p0:, :].T  # Shape: (d, T-p0)
        Z = X[p0-1:-1, :].T  # Shape: (d, T-p0)
        
        # OLS: A = Y @ Z^T @ (Z @ Z^T)^{-1}
        ZZt = Z @ Z.T
        YZt = Y @ Z.T
        
        try:
            self._A = YZt @ np.linalg.inv(ZZt)
        except np.linalg.LinAlgError:
            self._A = YZt @ np.linalg.pinv(ZZt)
        
        # Compute and store residuals
        self._residuals = X[p0:, :] - X[p0-1:-1, :] @ self._A.T
        
        self._fitted = True
        return self

    def sample(self, T: int, seed: int) -> np.ndarray:
        """Sample by resampling blocks of residuals with replacement.
        
        Args:
            T: Number of timesteps to sample.
            seed: Random seed for reproducibility.
            
        Returns:
            Array of shape (T, d).
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before sampling")
        
        np.random.seed(seed)
        d = self._X_fit.shape[1]
        n_residuals = self._residuals.shape[0]
        
        # Determine number of blocks available
        n_blocks = max(1, n_residuals - self.block_length + 1)
        
        # Initialize with random starting point
        X_sample = np.zeros((T, d))
        X_sample[0, :] = np.random.randn(d)
        
        # Generate trajectory by resampling blocks
        t = 1
        while t < T:
            # Resample one block start position with replacement
            block_start = np.random.randint(0, n_blocks)
            block_end = min(block_start + self.block_length, n_residuals)
            block_residuals = self._residuals[block_start:block_end, :]
            
            # Apply block to trajectory
            for offset, residual in enumerate(block_residuals):
                if t >= T:
                    break
                X_sample[t, :] = X_sample[t-1, :] @ self._A.T + residual
                t += 1
        
        return X_sample

    @property
    def metadata(self) -> Dict:
        """Return model metadata."""
        return {
            "model": "MovingBlockBootstrap",
            "p0": self.p0,
            "block_length": self.block_length,
            "fitted": self._fitted,
            "n_samples": self._X_fit.shape[0] if self._fitted else None,
            "n_features": self._X_fit.shape[1] if self._fitted else None,
        }
