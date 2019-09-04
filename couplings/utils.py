"""Miscellaneous utilities."""
import matplotlib.pyplot as plt
import numpy as np


__all__ = ["total_variation", "plot_coupled_chains", "wasserstein"]


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


def total_variation(data):
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


def plot_coupled_chains(data, *, max_chains=8):  # pylint: disable=too-many-locals
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
    chains = min(max_chains, data.x.shape[1])
    _, axes = plt.subplots(
        nrows=chains // 2,
        ncols=2,
        figsize=(20, chains),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for chain_idx, axis in enumerate(axes.ravel()):
        x_chain, y_chain = data.x[:, chain_idx, :], data.y[:, chain_idx, :]
        met = data.meeting_time[chain_idx] > 0
        iters = len(x_chain)

        if met:
            meeting_time = data.meeting_time[chain_idx]
        else:
            meeting_time = iters

        after_meet = np.array([[], []])
        if data.x.shape[2] == 1:
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
