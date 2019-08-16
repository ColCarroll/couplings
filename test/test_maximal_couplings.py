"""Tests for maximal_couplings.py"""
import numpy as np
import scipy.stats as st

from couplings import (
    maximal_coupling_reference,
    maximal_coupling,
    ReflectionMaximalCoupling,
    reflection_maximal_coupling,
)


def test_maximal_coupling_reference_same():
    """Every draw from the same distributions should be the same."""
    rv1 = st.norm()
    rv2 = st.norm()
    x, y, cost = maximal_coupling_reference(rv1, rv2)
    assert x == y
    assert cost == 1


def test_maximal_coupling_reference_disjoint():
    """Every draw from disjoint distributions should be different."""
    rv1 = st.uniform(0, 1)
    rv2 = st.uniform(1, 2)
    x, y, cost = maximal_coupling_reference(rv1, rv2)
    assert x != y
    assert cost == 2


def test_maximal_coupling_reference_high_cost():
    """I can't find any 1d distributions that cost more than 2 on average."""
    rv1 = st.uniform(0, 1)
    rv2 = st.uniform(0.0, 0.99)
    costs = []

    # Note this depends on the random seed!
    for _ in range(200):
        *_, cost = maximal_coupling_reference(rv1, rv2)
        costs.append(cost)
    assert max(costs) > 2
    assert min(costs) == 1


def test_maximal_coupling_same():
    rv1 = st.norm()
    rv2 = st.norm()
    samples, cost = maximal_coupling(rv1, rv2)
    assert (samples[:, 0] == samples[:, 1]).all()
    assert (cost == 1).all()


def test_maximal_coupling_disjoint():
    """Every draw from disjoint distributions should be different."""
    rv1 = st.uniform(0, 1)
    rv2 = st.uniform(1, 2)
    samples, cost = maximal_coupling(rv1, rv2)
    assert (samples[:, 0] != samples[:, 1]).any()
    assert (cost == 2).all()


def test_maximal_coupling_high_cost():
    """I can't find any 1d distributions that cost more than 2 on average."""
    rv1 = st.uniform(0, 1)
    rv2 = st.uniform(0.0, 0.99)
    costs = []

    # Note this depends on the random seed!
    _, costs = maximal_coupling(rv1, rv2)
    assert max(costs) > 2
    assert min(costs) == 1


def test_ReflectionMaximalCoupling():
    rmc = ReflectionMaximalCoupling(st.norm(), 1)

    # make sure different means give separated points, and
    # that the class can be reused
    x, y = rmc(-4, 4)
    assert x < y
    x, y = rmc(4, -4)
    assert x > y


def test_ReflectionMaximalCoupling_same():
    rmc = ReflectionMaximalCoupling(st.norm(), 1)
    x, y = rmc(4, 4)
    assert x == y


def test_ReflectionMaximalCoupling_3d():
    # Make sure this works with high dimensional distributions
    rmc = ReflectionMaximalCoupling(st.multivariate_normal(np.zeros(3), np.eye(3)), np.eye(3))

    # make sure different means give separated points, and
    # that the class can be reused
    x, y = rmc(-4 * np.ones(3), 4 * np.ones(3))
    assert (x < y).all()

    x, y = rmc(4 * np.ones(3), -4 * np.ones(3))
    assert (x > y).all()


def test_reflection_maximal_coupling():
    # make sure different means give separated points, and
    # that the class can be reused
    x, y = reflection_maximal_coupling(st.norm(), 1, -4, 4)
    assert x < y
    x, y = reflection_maximal_coupling(st.norm(), 1, 4, -4)
    assert x > y


def test_reflection_maximal_coupling_same():
    x, y = reflection_maximal_coupling(st.norm(), 1, 4, 4)
    assert x == y


def test_reflection_maximal_coupling_3d():
    # Make sure this works with high dimensional distributions
    base_distribution = st.multivariate_normal(np.zeros(3), np.eye(3))
    proposal_cov = np.eye(3)
    # make sure different means give separated points, and
    # that the class can be reused
    x, y = reflection_maximal_coupling(
        base_distribution, proposal_cov, -4 * np.ones(3), 4 * np.ones(3)
    )
    assert (x < y).all()

    x, y = reflection_maximal_coupling(
        base_distribution, proposal_cov, 4 * np.ones(3), -4 * np.ones(3)
    )
    assert (x > y).all()
