"""Tests for metropolis_hastings.py"""
import numpy as np
import scipy.stats as st

from couplings.metropolis_hastings import (
    _metropolis_accept,
    metropolis_hastings,
    unbiased_estimator,
)


def test__metropolis_accept():
    logpdf = st.norm().logpdf

    current = 1
    current_log_prob = logpdf(current)

    # Deterministic accept
    new, new_log_prob, accept = _metropolis_accept(
        logpdf, current, current, current_log_prob, log_unif=np.log(0.99)
    )
    assert accept
    assert new == current
    assert new_log_prob == current_log_prob

    # Deterministic reject
    new, new_log_prob, accept = _metropolis_accept(
        logpdf, current, current, current_log_prob, log_unif=np.log(1.01)
    )
    assert not accept
    assert new == current
    assert new_log_prob == current_log_prob

    # Always accept a higher probability point
    new, new_log_prob, accept = _metropolis_accept(logpdf, 0, current, current_log_prob)
    assert accept
    assert new == 0.0
    assert new_log_prob != current_log_prob


def test_metropolis_hastings_scalar():
    rv = st.norm()
    log_prob = rv.logpdf
    init_x, init_y = 0.5, 0.5
    data = metropolis_hastings(log_prob, 10, init_x, init_y, lag=3, iters=20)
    assert (data.x[0] == init_x).all()
    assert (data.y[0] == init_y).all()

    assert data.x.shape[0] == 20
    assert data.y.shape[0] == 20 - 3

    assert data.meeting_time < 20
    assert 0 <= data.x_accept.mean() <= 1.0
    assert 0 <= data.y_accept.mean() <= 1.0


def test_metropolis_hastings_vec():
    dim = 10
    rv = st.multivariate_normal(np.zeros(dim), np.eye(dim))

    log_prob = rv.logpdf
    init_x, init_y = rv.rvs(size=2)
    data = metropolis_hastings(log_prob, 10 * np.eye(dim), init_x, init_y, lag=1, iters=20)
    assert (data.x[0] == init_x).all()
    assert (data.y[0] == init_y).all()

    assert data.x.shape[0] == 20
    assert data.y.shape[0] == 20 - 1

    assert data.meeting_time < 20
    assert 0 <= data.x_accept.mean() <= 1.0
    assert 0 <= data.y_accept.mean() <= 1.0


def test_metropolis_hastings_short_circuit():
    rv = st.norm()
    log_prob = rv.logpdf
    init_x, init_y = 0.5, 0.5
    data = metropolis_hastings(log_prob, 10, init_x, init_y, iters=200, short_circuit=True)
    assert (data.x[0] == init_x).all()
    assert (data.y[0] == init_y).all()

    assert data.x.shape[0] == data.meeting_time
    assert data.y.shape[0] == data.meeting_time - 1

    assert data.meeting_time < 20
    assert 0 <= data.x_accept.mean() <= 1.0
    assert 0 <= data.y_accept.mean() <= 1.0


def test_unbiased_estimator_vec():
    dim = 10
    rv = st.multivariate_normal(np.zeros(dim), np.eye(dim))
    log_prob = rv.logpdf
    init_x, init_y = rv.rvs(size=2)
    data = metropolis_hastings(log_prob, 10 * np.eye(dim), init_x, init_y, iters=20)
    mcmc_estimate, bias_correction = unbiased_estimator(data, lambda x: x, burn_in=10)
    assert mcmc_estimate.shape == (dim,)
    assert bias_correction.shape == (dim,)
    estimate = mcmc_estimate + bias_correction
    assert (-3 < estimate).all()
    assert (estimate < 3).all()


def test_unbiased_estimator_scalar():
    rv = st.norm()
    log_prob = rv.logpdf
    init_x, init_y = 0.5, 0.5
    data = metropolis_hastings(log_prob, 10, init_x, init_y, lag=3, iters=20)
    mcmc_estimate, bias_correction = unbiased_estimator(data, lambda x: x, burn_in=0)
    assert mcmc_estimate.shape == (1,)
    assert bias_correction.shape == (1,)
    estimate = mcmc_estimate + bias_correction
    assert (-3 < estimate).all()
    assert (estimate < 3).all()
