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
    x_accept: np.ndarray
    y_accept: np.ndarray
    meeting_time: np.ndarray
    lag: int


def _metropolis_accept(log_prob, proposal, current, current_log_prob, log_unif=None):
    """Handle metropolis acceptance step."""
    proposal_log_prob = np.atleast_1d(log_prob(proposal))
    unif_shape = proposal_log_prob.shape
    if log_unif is None:
        log_unif = np.log(np.random.rand(*unif_shape))
    else:
        log_unif = log_unif.reshape(unif_shape)

    return_val = current.copy()
    new_log_prob = current_log_prob.copy()

    accept = log_unif < proposal_log_prob - current_log_prob
    return_val[accept] = proposal[accept]
    new_log_prob[accept] = proposal_log_prob[accept]

    # print(current_log_prob.shape)
    # print(return_val.shape, new_log_prob.shape, accept.shape)
    return return_val, new_log_prob, accept.squeeze()


def metropolis_hastings(
    log_prob, proposal_cov, init_x, init_y, lag=1, iters=1000, chains=128, short_circuit=False
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
        x=np.empty((iters, chains, dim)),
        y=np.empty((iters - lag, chains, dim)),
        x_accept=np.zeros((iters, chains), dtype=bool),
        y_accept=np.zeros((iters - lag, chains), dtype=bool),
        meeting_time=-1 * np.ones(chains, dtype=int),
        lag=lag,
    )
    if not hasattr(init_x, "shape") or init_x.shape == (dim,):
        init_x = np.tile(init_x, (chains, 1))
    if not hasattr(init_y, "shape") or init_y.shape == (dim,):
        init_y = np.tile(init_y, (chains, 1))
    data.x[0], data.y[0] = init_x, init_y
    x_log_prob = np.atleast_1d(log_prob(init_x))
    y_log_prob = np.atleast_1d(log_prob(init_y))

    # Run for the first `lag` steps
    # Vectorize the RNG
    samples = np.random.multivariate_normal(np.zeros(dim), proposal_cov, size=(lag, chains))
    for idx, sample in enumerate(samples, 1):
        x_proposal = sample + data.x[idx - 1]
        data.x[idx], x_log_prob, data.x_accept[idx] = _metropolis_accept(
            log_prob, x_proposal, data.x[idx - 1], x_log_prob
        )
    # Coupled sampling
    base_distribution = st.multivariate_normal(np.zeros(dim), np.eye(dim))
    rmc = ReflectionMaximalCoupling(base_distribution, proposal_cov)

    # Vectorize the RNG
    log_unifs = np.log(np.random.rand(iters - lag - 1, chains))
    for t, log_unif in enumerate(log_unifs, lag + 1):
        x_proposal, y_proposal = rmc(data.x[t - 1], data.y[t - lag - 1], chains)

        data.x[t], x_log_prob, data.x_accept[t] = _metropolis_accept(
            log_prob, x_proposal, data.x[t - 1], x_log_prob, log_unif
        )
        data.y[t - lag], y_log_prob, data.y_accept[t - lag] = _metropolis_accept(
            log_prob, y_proposal, data.y[t - lag - 1], y_log_prob, log_unif
        )
        met = np.isclose(data.x[t], data.y[t - lag])
        met = met.reshape((met.shape[0], -1)).all(axis=1)
        data.meeting_time[met * (data.meeting_time < 0)] = t + 1
        if short_circuit and met.all():
            data.x = data.x[: t + 1]
            data.y = data.y[: t - lag + 1]
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
    shape = data.x.shape
    max_idx = shape[0] if data.meeting_time.min() == -1 else data.meeting_time.max() + 1
    slicer = np.arange(burn_in + 1, max_idx)
    split_idxs = np.tile(slicer, (shape[1], 1)).T
    ratio = np.minimum(1, (split_idxs - burn_in) / (len(data.x) - burn_in + 1))
    mult = fn(data.x[slicer]) - fn(data.y[slicer - data.lag])
    mult[split_idxs > data.meeting_time] = 0
    bias_correction = np.mean(np.expand_dims(ratio, -1) * mult, axis=0)
    mcmc_average = fn(data.x[burn_in:]).mean(axis=0)
    return mcmc_average, bias_correction
