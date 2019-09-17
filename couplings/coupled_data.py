"""Data class for returning samples."""
from dataclasses import dataclass
import numpy as np

__all__ = ["CoupledData"]


@dataclass
class CoupledData:
    """Data store for MCMC sampling.

    This stores samples, heuristics, diagnostics, etc, and
    should be passed to other functions in the library.
    """

    x: np.ndarray  # pylint: disable=invalid-name
    y: np.ndarray
    x_accept: np.ndarray
    y_accept: np.ndarray
    meeting_time: np.ndarray
    lag: int
    iters: int
    dim: int
    chains: int
