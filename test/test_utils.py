"""Test couplings/utils.py"""
import numpy as np
import pytest
import scipy.stats as st
from scipy.special import logsumexp

from couplings.utils import mixture_of_gaussians, total_variation, plot_coupled_chains, wasserstein


@pytest.fixture
def mog():
    return mixture_of_gaussians([(-4, 1), (4, 1)], [0.5, 0.5])


def test_mog_rvs(mog):
    assert mog.rvs().shape == (1,)
    assert mog.rvs(size=100).shape == (100,)


@pytest.mark.parametrize("point", np.arange(-6, 6))
def test_mog_pdfs(mog, point):
    neg_normal = st.norm(-4, 1)
    pos_normal = st.norm(4, 1)

    expected = 0.5 * (neg_normal.pdf(point) + pos_normal.pdf(point))
    assert mog.pdf(point) == expected
    expected = logsumexp(
        [np.log(0.5) + neg_normal.logpdf(point), np.log(0.5) + pos_normal.logpdf(point)]
    )
    assert mog.logpdf(point) == expected


def test_total_variation(mh_samples):
    assert total_variation(mh_samples).shape == (mh_samples.iters,)


def test_wasserstein(mh_samples):
    assert wasserstein(mh_samples).shape == (mh_samples.iters,)


def test_plot_coupled_chains(mh_samples):
    chains = 8
    axes = plot_coupled_chains(mh_samples, max_chains=chains)
    assert axes.size == chains


def test_plot_coupled_chains_nd(mh_samples_nd):
    chains = 8
    axes = plot_coupled_chains(mh_samples_nd, max_chains=chains)
    assert axes.size == chains
