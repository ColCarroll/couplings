"""Miscellaneous utilities."""
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from scipy.special import logsumexp

from .coupled_data import CoupledData


__all__ = ["total_variation", "plot_coupled_chains", "wasserstein", "mixture_of_gaussians"]


class mixture_of_gaussians:  # pylint: disable=invalid-name
    """Mixture of Gaussians with the given parameters."""

    def __init__(self, params: List[Tuple[float]], probs: List[float]):
        """Construct normal distributions.

        Parameters
        ----------
        params : list of tuples
            loc and scale passed to each st.norm
        probs : list of floats between 0 and 1
            probability of each mixture component

        """
        self._probs = np.array(probs)
        self._logp = np.log(self._probs).reshape((1, len(self._probs)))
        self._rvs = [st.norm(*param) for param in params]

    def rvs(self, size: int = 1) -> np.ndarray:
        """Sample from the random variable."""
        vals = np.concatenate(
            [
                rv.rvs(size=size_)
                for rv, size_ in zip(self._rvs, np.random.multinomial(size, self._probs))
            ]
        )
        np.random.shuffle(vals)
        return vals

    def pdf(self, point: np.ndarray) -> float:
        """Compute the pdf of the distribution at a point."""
        return self._probs.dot([rv.pdf(point) for rv in self._rvs])

    def logpdf(self, point: np.ndarray) -> float:
        """Compute the log probability of the distribution at a point."""
        point = np.array(point)
        if point.size > 1:
            point = point.reshape((2, -1))
            parts = self._logp.T + np.reshape([rv.logpdf(point) for rv in self._rvs], (2, -1))
            return logsumexp(parts, axis=0)
        parts = self._logp + np.array([rv.logpdf(point) for rv in self._rvs])
        return logsumexp(parts)


def _tv_pointwise(data):
    """Compute pointwise total variation."""
    return np.maximum(
        0,
        np.ceil(
            (np.expand_dims(data.meeting_time - data.lag, -1) - np.arange(data.x.shape[0]))
            / data.lag
        ),
    )


def wasserstein(data):
    r"""Compute the Wasserstein distance to the stationary distribution.

    See Definition 2.1 and Equation (4) from

    Biswas, Niloy, and Pierre E. Jacob. “Estimating Convergence of Markov Chains
    with L-Lag Couplings.” ArXiv:1905.09971 [Stat], May 23, 2019.
    http://arxiv.org/abs/1905.09971.


    Parameters
    ----------
    data : CoupledData
        Results of sampling. Lots of chains is good, since this is an expectation.

    Returns
    -------
    np.ndarray
        An array with the number of iterations from the input data, where each
    entry t is the Wasserstein distance $d_{W}(\pi_t, \pi)

    """
    tv_pointwise = _tv_pointwise(data).T.astype(int)
    wass = np.empty(tv_pointwise.shape[:1])
    for idx, row in enumerate(tv_pointwise):
        expect = np.zeros(data.x.shape[1])
        for j in range(1, row.max() + 1):
            non_empty = row >= j
            expect[non_empty] += np.abs(
                data.x[idx + j * data.lag, non_empty] - data.y[idx + (j - 1) * data.lag, non_empty]
            ).sum(axis=-1)
        wass[idx] = expect.mean()
    return wass


def total_variation(data) -> np.ndarray:
    r"""Compute the total variation distance to the stationary distribution.

    See Definition 2.1 and Equation (3) from

    Biswas, Niloy, and Pierre E. Jacob. “Estimating Convergence of Markov Chains
    with L-Lag Couplings.” ArXiv:1905.09971 [Stat], May 23, 2019.
    http://arxiv.org/abs/1905.09971.


    Parameters
    ----------
    data : CoupledData
        Results of sampling. Lots of chains is good, since this is an expectation.

    Returns
    -------
    np.ndarray
        An array with the number of iterations from the input data, where each
    entry t is the total variation $d_{TV}(\pi_t, \pi)

    """
    return _tv_pointwise(data).mean(axis=0)


def plot_coupled_chains(data: CoupledData, *, max_chains=8):  # pylint: disable=too-many-locals
    """Plot coupled chains in 1 or more dimensions.

    Parameters
    ----------
    data : CoupledData
        Results of sampling
    max_chains : int
        There is one chain per axis. You probably do not want more than, like, 20 axes.

    Returns
    -------
    axes
        A single matplotlib axis, or an array of axes

    """
    chains = min(max_chains, data.chains)
    dim = data.dim
    ncols = 2 if dim == 1 else 4
    _, axes = plt.subplots(
        nrows=chains // ncols,
        ncols=ncols,
        figsize=(20, chains),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for chain_idx, axis in enumerate(axes.ravel()):
        x_chain, y_chain = data.x[:, chain_idx, :], data.y[:, chain_idx, :]
        met = data.meeting_time[chain_idx] > 0
        iters = data.iters

        if met:
            meeting_time = data.meeting_time[chain_idx]
        else:
            meeting_time = iters

        after_meet = np.array([[], []])
        if dim == 1:
            y_line = np.vstack(
                (
                    np.arange(data.lag, min(meeting_time + data.lag, iters)),
                    y_chain[:meeting_time].flatten(),
                )
            ).T
            x_line = np.vstack(
                (np.arange(min(meeting_time, iters)), x_chain[:meeting_time].flatten())
            ).T
            if met:
                after_meet = np.vstack(
                    (np.arange(meeting_time, iters), x_chain[meeting_time:].flatten())
                ).T
                axis.plot(meeting_time - 1, x_chain[meeting_time - 1, -1], "rx")
        else:
            x_line = x_chain[: meeting_time + data.lag, :2]
            y_line = y_chain[:meeting_time, :2]

        for pts in (x_line, y_line, after_meet):
            axis.plot(*pts.T, "-", lw=2)
    return axes
