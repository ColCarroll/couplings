"""Coupled Metropolis-Hastings implementation."""
from dataclasses import dataclass

import numpy as np
import scipy.stats as st

from .maximal_couplings import ReflectionMaximalCoupling


__all__ = ["CoupledData", "metropolis_hastings", "unbiased_estimator"]


@dataclass
class CoupledData:
    """Data store for MCMC sampling.

    This stores samples, heuristics, diagnostics, etc, and
    should be passed to other functions in the library.
    """

    x: np.ndarray
    y: np.ndarray
    cost: np.ndarray
    x_accept: np.ndarray
    y_accept: np.ndarray
    meeting_time: int
    lag: int


def _metropolis_accept(log_prob, proposal, current, current_log_prob, log_unif=None):
    """Handle metropolis acceptance step.

    Note it accepts a uniform variable, since we reuse one for the coupled steps.
    """
    if log_unif is None:
        log_unif = np.log(np.random.rand())
    proposal_log_prob = log_prob(proposal)
    if log_unif < proposal_log_prob - current_log_prob:
        return proposal, proposal_log_prob, True
    else:
        return current, current_log_prob, False


def metropolis_hastings(
    log_prob, proposal_cov, init_x, init_y, lag=1, iters=1000, short_circuit=False
):
    """Sample from a density function using coupled Metropolis-Hastings.

    Reference section 4.2 of "Unbiased Markov chain Monte Carlo with couplings."

    Implementation notes:
        - The user provides the initial points for the two chains. This helps
        with experiments.
        - The lag is adjustable, but the paper uses a lag of 1. This was a
        suggestion from the author, pointing to his own implementation
        (https://github.com/pierrejacob/unbiasedmcmc) and to Niloy Biswas'
        work with him (https://arxiv.org/abs/1905.09971).

    TODO:
        - Tuning (could adjust proposal covariance to a target, or as in PyMC3,
          and adjust the number of iterations to some percentile of the
          meeting time)
        - This could probably be "more general", but for experimenting it is
          probably best to not reuse too much machinery from elsewhere.

    Parameters
    ----------
    log_prob : callable
        Log probability to sample from
    proposal_cov : np.ndarray
        Covariance matrix to use for proposals
    init_x : np.ndarray
        Where to initialize one chain
    init_y : np.ndarray
        Where to initialize the other chain
    lag : int
        How many steps the first chain runs before adding the coupled chain
    iters : int
        How many iterations the *first* chain will have (the second has iters - lag)
    short_circuit : bool
        Set to True to return immediately after the chains meet. Note that `iters`
        is still respected to avoid an unterminating loop, so set it very high
        to simulate a true `while` loop!

    Returns
    -------
    CoupledData

    """
    proposal_cov = np.atleast_2d(proposal_cov)
    dim = proposal_cov.shape[0]
    data = CoupledData(
        x=np.empty((iters, dim)),
        y=np.empty((iters - lag, dim)),
        cost=np.zeros(iters),
        x_accept=np.zeros(iters, dtype=bool),
        y_accept=np.zeros(iters - lag, dtype=bool),
        meeting_time=-1,
        lag=lag,
    )
    data.x[0], data.y[0] = init_x, init_y
    x_log_prob, y_log_prob = log_prob(init_x), log_prob(init_y)

    # Run for the first `lag` steps
    # Vectorize the RNG
    samples = np.random.multivariate_normal(np.zeros(dim), proposal_cov, size=lag)
    for idx, sample in enumerate(samples, 1):
        x_proposal = sample + data.x[idx - 1]
        data.x[idx], x_log_prob, data.x_accept[idx] = _metropolis_accept(
            log_prob, x_proposal, data.x[idx - 1], x_log_prob
        )

    # Coupled sampling
    base_distribution = st.multivariate_normal(np.zeros(dim), np.eye(dim))
    rmc = ReflectionMaximalCoupling(base_distribution, proposal_cov)
    met = False

    # Vectorize the RNG
    log_unifs = np.log(np.random.rand(iters - lag - 1))
    for t, log_unif in enumerate(log_unifs, lag + 1):
        x_proposal, y_proposal = rmc(data.x[t - 1], data.y[t - lag - 1])
        data.cost[t] = 1

        data.x[t], x_log_prob, data.x_accept[t] = _metropolis_accept(
            log_prob, x_proposal, data.x[t - 1], x_log_prob, log_unif
        )
        data.y[t - lag], y_log_prob, data.y_accept[t - lag] = _metropolis_accept(
            log_prob, y_proposal, data.y[t - lag - 1], y_log_prob, log_unif
        )
        if not met and np.isclose(data.x[t], data.y[t - lag]).all():
            data.meeting_time = t + 1
            met = True
            if short_circuit:
                data.x = data.x[: t + 1]
                data.y = data.y[: t - lag + 1]
                data.cost = data.cost[: t + 1]
                data.x_accept = data.x_accept[: t + 1]
                data.y_accept = data.y_accept[: t - lag + 1]
                return data
    return data


def unbiased_estimator(data, fn, burn_in):
    """Compute an unbiased estimator of a function using coupled data.

    This is an implementation of Equation 2.1

    Note the return is both the mcmc_average and the bias correction,
    for debugging. Add them to get the return value from Equation 2.1.

    Parameters
    ----------
    data : CoupledData
        Results from a coupled MCMC experiment
    fn : callable
        Should accept an array and return an array with the same shape
    burn_in : int
        This is k in the paper, and are discarded samples from the start
        of the experiment

    Returns
    -------
    mcmc_average, bias_correction
        Two arrays with shape data.x.shape[1:]. Adding them gives an
        unbiased estimate of the function.

    """
    if data.meeting_time <= burn_in + 1:
        bias_correction = np.zeros(data.x.shape[1:])
    else:
        split_idxs = np.arange(burn_in + 1, data.meeting_time)
        bias_correction = np.mean(
            np.minimum(1, (split_idxs - burn_in).reshape(-1, 1) / (len(data.x) - burn_in + 1))
            * (fn(data.x[split_idxs]) - fn(data.y[split_idxs - data.lag])),
            axis=0,
        )
    mcmc_average = fn(data.x[burn_in:]).mean(axis=0)
    return mcmc_average, bias_correction
