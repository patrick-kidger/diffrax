import abc
from typing import Optional

from jaxtyping import Array, PyTree

from .._custom_types import RealScalarLike
from .._path import AbstractPath


class AbstractBrownianPath(AbstractPath):
    "Abstract base class for all Brownian paths."

    @abc.abstractmethod
    def evaluate(
        self, t0: RealScalarLike, t1: Optional[RealScalarLike] = None, left: bool = True
    ) -> PyTree[Array]:
        r"""Samples a Brownian increment $w(t_1) - w(t_0)$.

        Each increment has distribution $\mathcal{N}(0, t_1 - t_0)$.

        **Arguments:**

        - `t0`: Start of interval.
        - `t1`: End of interval.
        - `left`: Ignored. (This determines whether to treat the path as
            left-continuous or right-continuous at any jump points, but Brownian
            motion has no jump points.)

        **Returns:**

        A pytree of JAX arrays corresponding to the increment $w(t_1) - w(t_0)$.

        Some subclasses may allow `t1=None`, in which case just the value $w(t_0)$ is
        returned.
        """