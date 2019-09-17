"""Configuration and fixtures for test suite."""
import numpy as np
import pytest
import scipy.stats as st

from couplings import metropolis_hastings
from couplings.utils import mixture_of_gaussians


@pytest.fixture(autouse=True)
def random_seed():
    """Reset numpy random seed generator."""
    np.random.seed(0)


@pytest.fixture(scope="module")
def mh_samples():
    rv = mixture_of_gaussians(((3, 1), (-3, 1)), (0.5, 0.5))
    log_prob = rv.logpdf

    data = metropolis_hastings(
        log_prob=log_prob, proposal_cov=0.5, init_x=2, init_y=-2, lag=10, iters=100, chains=20
    )
    return data


@pytest.fixture(scope="module")
def mh_samples_nd():
    dim = 2
    mean = np.zeros(dim)
    cov = 0.1 * np.eye(dim) + 0.9 * np.ones((dim, dim))
    rv = st.multivariate_normal(mean, cov)
    chains = 8
    data = metropolis_hastings(
        log_prob=rv.logpdf,
        proposal_cov=0.1 * np.eye(dim),
        init_x=4 * np.ones(dim),
        init_y=-4 * np.ones(dim),
        lag=1,
        iters=500,
        chains=chains,
    )
    return data
