import abc

from ..custom_types import Array, Scalar
from ..path import AbstractPath


class AbstractBrownianPath(AbstractPath):
    "Abstract base class for all Brownian paths."

    @abc.abstractmethod
    def evaluate(self, t0: Scalar, t1: Scalar) -> Array:
        r"""Samples a Brownian increment $w(t_1) - w(t_0)$.

        Each increment has distribution $\mathcal{N}(0, t_1 - t_0)$.

        **Arguments:**

        - `t0`: Start of interval
        - `t1`: End of interval

        **Returns:**

        A JAX array corresponding to the increment $w(t_1) - w(t_0)$.
        """
