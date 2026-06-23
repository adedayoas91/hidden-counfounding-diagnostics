"""Pytest configuration and shared fixtures for markovianity_diagnostic tests."""

import numpy as np
import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def seed_fix():
    """Fixture to set random seeds for reproducibility."""
    np.random.seed(42)


@pytest.fixture
def tmp_outputs(tmp_path):
    """Fixture providing a temporary directory for test outputs."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def synthetic_markov_data(seed_fix):
    """
    Fixture providing order-1 Markov data.
    Returns (T=200, d=5) VAR(1) process.
    """
    T, d = 200, 5
    X = np.zeros((T, d))
    A = np.random.randn(d, d) * 0.3
    np.fill_diagonal(A, 0)  # No self-loops

    for t in range(1, T):
        X[t] = A @ X[t - 1] + np.random.randn(d) * 0.5

    return X


@pytest.fixture
def synthetic_data_with_confounder(seed_fix):
    """
    Fixture providing VAR(1) data with a latent confounder.
    Observed d=5 nodes + 1 hidden confounder.
    Returns (T=200, d=5) data affected by latent driver.
    """
    T, d = 200, 5

    # Latent confounder: independent AR(1) process
    C = np.zeros(T)
    C[0] = np.random.randn()
    for t in range(1, T):
        C[t] = 0.7 * C[t - 1] + np.random.randn() * 0.5

    # Observed nodes affected by confounder
    X = np.zeros((T, d))
    A = np.random.randn(d, d) * 0.2
    np.fill_diagonal(A, 0)

    for t in range(1, T):
        X[t] = A @ X[t - 1] + 0.4 * C[t] * np.ones(d) + np.random.randn(d) * 0.3

    return X


@pytest.fixture
def simple_adjacency_dict():
    """Fixture providing a simple adjacency matrix dictionary across depths."""
    # Simulate outputs from markov-exp: {p_value: adjacency_matrix}
    return {
        0: np.array([[0, 1, 0, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0]]),
        1: np.array([[0, 1, 0, 0, 0],
                     [1, 0, 1, 0, 0],
                     [0, 0, 0, 1, 0],
                     [0, 0, 1, 0, 1],
                     [0, 0, 0, 0, 0]]),
        2: np.array([[0, 0, 0, 0, 0],
                     [1, 0, 1, 0, 0],
                     [0, 0, 0, 1, 0],
                     [0, 0, 1, 0, 1],
                     [0, 0, 0, 0, 0]]),
    }
