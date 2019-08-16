"""Configuration and fixtures for test suite."""
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def random_seed():
    """Reset numpy random seed generator."""
    np.random.seed(0)
