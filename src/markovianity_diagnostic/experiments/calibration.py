"""Bootstrap calibration module for markovianity diagnostics.

Provides NullModel base class and null hypothesis implementations for
calibrating the markovianity test statistic T_obs against bootstrap
null distributions.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _validate_var_input(X: np.ndarray, p0: int) -> tuple[int, int]:
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    T, d = X.shape
    if p0 < 1:
        raise ValueError(f"p0={p0} must be >= 1")
    if p0 >= T:
        raise ValueError(f"p0={p0} must be < T={T}")
    return T, d


def _var_design_matrix(X: np.ndarray, p0: int) -> tuple[np.ndarray, np.ndarray]:
    """Return targets and stacked lag predictors for a VAR(p0) fit."""
    T, _ = X.shape
    Y = X[p0:, :]
    Z = np.asarray([
        np.concatenate([X[t - lag, :] for lag in range(1, p0 + 1)])
        for t in range(p0, T)
    ])
    return Y, Z


def _fit_var_coefficients(X: np.ndarray, p0: int) -> tuple[np.ndarray, np.ndarray]:
    Y, Z = _var_design_matrix(X, p0)
    coefficients, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    residuals = Y - Z @ coefficients
    return coefficients, residuals


def _covariance_matrix(residuals: np.ndarray) -> np.ndarray:
    covariance = np.cov(residuals.T, ddof=1)
    if covariance.ndim == 0:
        covariance = np.array([[covariance]])
    elif covariance.ndim == 1:
        covariance = np.diag(covariance)
    return covariance


def _initial_conditions(X_fit: np.ndarray, T: int, p0: int) -> np.ndarray:
    d = X_fit.shape[1]
    X_sample = np.zeros((T, d))
    n_init = min(T, p0)
    X_sample[:n_init, :] = X_fit[:n_init, :]
    return X_sample


def _predict_next(history: np.ndarray, coefficients: np.ndarray, p0: int) -> np.ndarray:
    lag_vector = np.concatenate([history[-lag, :] for lag in range(1, p0 + 1)])
    return lag_vector @ coefficients


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
    def metadata(self) -> dict[str, Any]:
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
        _validate_var_input(X, p0)
        
        self.p0 = p0
        self._X_fit = X.copy()
        
        self._A, residuals = _fit_var_coefficients(X, p0)
        self._cov = _covariance_matrix(residuals)
        
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
        
        rng = np.random.default_rng(seed)
        d = self._X_fit.shape[1]
        
        X_sample = _initial_conditions(self._X_fit, T, self.p0)
        
        for t in range(self.p0, T):
            noise = rng.multivariate_normal(np.zeros(d), self._cov)
            X_sample[t, :] = _predict_next(X_sample[:t, :], self._A, self.p0) + noise
        
        return X_sample

    @property
    def metadata(self) -> dict[str, Any]:
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
        _validate_var_input(X, p0)
        
        self.p0 = p0
        self._X_fit = X.copy()
        
        self._A, self._residuals = _fit_var_coefficients(X, p0)
        
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
        
        rng = np.random.default_rng(seed)
        n_residuals = self._residuals.shape[0]
        
        X_sample = _initial_conditions(self._X_fit, T, self.p0)
        
        for t in range(self.p0, T):
            idx = rng.integers(0, n_residuals)
            noise = self._residuals[idx, :]
            X_sample[t, :] = _predict_next(X_sample[:t, :], self._A, self.p0) + noise
        
        return X_sample

    @property
    def metadata(self) -> dict[str, Any]:
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
        if block_length < 1:
            raise ValueError("block_length must be >= 1")
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
        _validate_var_input(X, p0)
        
        self.p0 = p0
        self._X_fit = X.copy()
        
        self._A, self._residuals = _fit_var_coefficients(X, p0)
        
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
        
        rng = np.random.default_rng(seed)
        n_residuals = self._residuals.shape[0]
        
        # Determine number of blocks available
        n_blocks = max(1, n_residuals - self.block_length + 1)
        
        # Initialize with random starting point
        X_sample = _initial_conditions(self._X_fit, T, self.p0)
        
        # Generate trajectory by resampling blocks
        t = self.p0
        while t < T:
            block_start = rng.integers(0, n_blocks)
            block_end = min(block_start + self.block_length, n_residuals)
            block_residuals = self._residuals[block_start:block_end, :]
            
            # Apply block to trajectory
            for offset, residual in enumerate(block_residuals):
                if t >= T:
                    break
                X_sample[t, :] = _predict_next(X_sample[:t, :], self._A, self.p0) + residual
                t += 1
        
        return X_sample

    @property
    def metadata(self) -> dict[str, Any]:
        """Return model metadata."""
        return {
            "model": "MovingBlockBootstrap",
            "p0": self.p0,
            "block_length": self.block_length,
            "fitted": self._fitted,
            "n_samples": self._X_fit.shape[0] if self._fitted else None,
            "n_features": self._X_fit.shape[1] if self._fitted else None,
        }


class StationaryBootstrapNull(MovingBlockBootstrapNull):
    """Resample VAR residuals using geometrically distributed blocks.

    ``block_length`` is the expected block length. This preserves local
    residual dependence while avoiding fixed block boundaries.
    """

    def sample(self, T: int, seed: int) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Model must be fitted before sampling")
        if T < 1:
            raise ValueError("T must be >= 1")

        rng = np.random.default_rng(seed)
        n_residuals = self._residuals.shape[0]
        residual_index = int(rng.integers(0, n_residuals))
        restart_probability = 1.0 / self.block_length
        X_sample = _initial_conditions(self._X_fit, T, self.p0)

        for t in range(self.p0, T):
            if t > self.p0 and rng.random() < restart_probability:
                residual_index = int(rng.integers(0, n_residuals))
            else:
                residual_index = (residual_index + 1) % n_residuals
            X_sample[t] = (
                _predict_next(X_sample[:t], self._A, self.p0)
                + self._residuals[residual_index]
            )

        return X_sample

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata
        metadata["model"] = "StationaryBootstrap"
        metadata["expected_block_length"] = metadata.pop("block_length")
        return metadata


def _json_serializer(obj):
    """JSON serializer for numpy and other non-JSON-serializable types."""
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


@dataclass(frozen=True)
class CalibrationResult:
    """Result of calibration experiment with null distribution and diagnosis.
    
    Attributes:
        observed: Dictionary containing observed test statistics and diagnostics.
                 Keys: T_obs, D_obs, D_parts_obs, edge_counts (optional)
        null: Dictionary containing null distribution statistics.
              Keys: T_boot, D_boot (optional), critical_90, critical_95, critical_99, p_value
        diagnosis: Dictionary containing diagnostic results.
                  Keys: reject_global_95, first_exceedance_depth (optional)
    """
    observed: dict[str, Any]
    null: dict[str, Any]
    diagnosis: dict[str, Any]
    
    def to_json(self, path: str) -> None:
        """Serialize CalibrationResult to JSON file.
        
        Args:
            path: File path where JSON should be written.
            
        Raises:
            IOError: If file cannot be written.
            TypeError: If data contains non-serializable types.
        """
        try:
            data = {
                "observed": self.observed,
                "null": self.null,
                "diagnosis": self.diagnosis,
            }
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=_json_serializer)
            logger.info(f"Saved CalibrationResult to {path}")
        except (IOError, TypeError) as e:
            logger.error(f"Failed to serialize CalibrationResult to {path}: {e}")
            raise
    
    @classmethod
    def from_json(cls, path: str) -> "CalibrationResult":
        """Deserialize CalibrationResult from JSON file.
        
        Args:
            path: File path to load JSON from.
            
        Returns:
            CalibrationResult instance.
            
        Raises:
            FileNotFoundError: If file does not exist.
            json.JSONDecodeError: If JSON is malformed.
            KeyError: If required keys are missing.
            ValueError: If schema validation fails.
        """
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"CalibrationResult file not found: {path}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in CalibrationResult file {path}: {e}") from e
        
        # Validate required keys
        required_keys = {"observed", "null", "diagnosis"}
        if not required_keys.issubset(set(data.keys())):
            missing = required_keys - set(data.keys())
            raise ValueError(f"CalibrationResult JSON missing required keys: {missing}")
        
        logger.info(f"Loaded CalibrationResult from {path}")
        return cls(
            observed=data["observed"],
            null=data["null"],
            diagnosis=data["diagnosis"],
        )
