"""Code for generating maximal couplings."""
import numpy as np

__all__ = [
    "maximal_coupling_reference",
    "maximal_coupling",
    "ReflectionMaximalCoupling",
    "reflection_maximal_coupling",
]


def maximal_coupling_reference(rv1, rv2):
    """Sample from a maximal coupling between distributions rv1 and rv2.

    Matches algorithm 2 from the paper quite closely.
    """
    rv1_draw = rv1.rvs()
    cost = 1  # 1 draw, 2 pdf evals
    if np.random.uniform(0, rv1.pdf(rv1_draw)) < rv2.pdf(rv1_draw):
        return rv1_draw, rv1_draw, cost
    rv2_draw = rv2.rvs()
    cost += 1
    while np.random.uniform(0, rv2.pdf(rv2_draw)) < rv1.pdf(rv2_draw):
        rv2_draw = rv2.rvs()
        cost += 1
    return rv1_draw, rv2_draw, cost


def maximal_coupling(p, q, size=1000):
    """Vectorized implementation of a maximal coupling between q and p.

    ~500-1000x speedup over the reference implementation
    """
    x = p.rvs(size=size)
    y = x.copy()
    cost = np.ones_like(x)

    resample = np.random.uniform(0, p.pdf(x)) > q.pdf(x)
    n_resample = resample.sum()

    while n_resample:
        cost[resample] += 1  # pylint: disable=unsupported-assignment-operation
        y[resample] = q.rvs(size=n_resample)
        resample[resample] = np.random.uniform(0, q.pdf(y[resample])) < p.pdf(y[resample])
        n_resample = resample.sum()
    return np.vstack((x, y)).T, cost


class ReflectionMaximalCoupling:
    """Generates samples from a reflection maximal coupling.

    We use a `base_distribution` (s in the paper), and a `proposal_cov`
    (Σ in the paper) to produce a maximal coupling where X and Y are
    correlated even when they are not equal.

    Generally,

        base_distribution ~ N(0, I),

    so that

        X ~ N(mu1, Σ)
        Y ~ N(mu2, Σ)

    Somehow the performance of this is not much better than the function one.
    """

    def __init__(self, base_distribution, proposal_cov):
        """Compute and store Cholesky decomposition of the covariance matrix."""
        self.base_distribution = base_distribution
        # np.atleast_2d protects against 1d case
        self.cholesky = np.linalg.cholesky(np.atleast_2d(proposal_cov))
        self.dim = self.cholesky.shape[0]

    def __call__(self, mu1, mu2, chains):
        """Produce a single sample from the coupling."""
        if not hasattr(mu1, "shape") or mu1.shape == (self.dim,):
            mu1 = np.tile(mu1, (chains, 1))
        if not hasattr(mu2, "shape") or mu2.shape == (self.dim,):
            mu2 = np.tile(mu2, (chains, 1))

        transformed_diff = np.linalg.solve(self.cholesky, np.atleast_2d(mu1 - mu2).T).T
        base_draw_x = self.base_distribution.rvs(size=chains).reshape((chains, self.dim))

        base_draw_y = base_draw_x + transformed_diff
        # np.atleast_1d protects against 1d case
        ratio = (
            self.base_distribution.logpdf(base_draw_x + transformed_diff)
            - self.base_distribution.logpdf(base_draw_x)
        ).reshape((chains,))
        couple = np.log(np.random.rand(chains)) < ratio
        if not couple.all():
            unit_diff = transformed_diff[~couple] / np.expand_dims(
                np.linalg.norm(transformed_diff[~couple], axis=-1), -1
            )
            base_draw_y[~couple] = (
                base_draw_x[~couple]
                - 2
                * np.expand_dims((unit_diff * base_draw_x[~couple]).sum(axis=-1), -1)
                * unit_diff
            )
        return mu1 + self.cholesky.dot(base_draw_x.T).T, mu2 + self.cholesky.dot(base_draw_y.T).T


def reflection_maximal_coupling(base_distribution, proposal_cov, mu1, mu2):
    """Generate samples from a reflection maximal coupling.

    We use a `base_distribution` (s in the paper), and a `proposal_cov`
    (Σ in the paper) to produce a maximal coupling where X and Y are
    correlated even when they are not equal.

    Generally,

        base_distribution ~ N(0, I),

    so that

        X ~ N(mu1, Σ)
        Y ~ N(mu2, Σ)

    Somehow the performance of this is not much better than the function one.
    """
    base_draw_x = base_distribution.rvs()
    cholesky = np.linalg.cholesky(np.atleast_2d(proposal_cov))
    transformed_diff = np.linalg.solve(cholesky, np.atleast_1d(mu1 - mu2))
    ratio = base_distribution.pdf(base_draw_x + transformed_diff) / base_distribution.pdf(
        base_draw_x
    )
    if np.random.rand() < ratio:
        base_draw_y = base_draw_x + transformed_diff
    else:
        unit_diff = transformed_diff / np.linalg.norm(transformed_diff)
        base_draw_y = base_draw_x - 2 * np.dot(unit_diff.T, base_draw_x) * unit_diff
    return mu1 + cholesky.dot(base_draw_x), mu2 + cholesky.dot(base_draw_y)
