import abc
from typing import Optional, Union

from ..custom_types import Array, LevyVal, PyTree, Scalar
from ..path import AbstractPath


class AbstractBrownianPath(AbstractPath):
    """Abstract base class for all Brownian paths."""

    @abc.abstractmethod
    def evaluate(
        self,
        t0: Scalar,
        t1: Optional[Scalar] = None,
        left: bool = True,
        use_levy: bool = False,
    ) -> Union[PyTree[Array], LevyVal]:
        r"""Samples a Brownian increment $w(t_1) - w(t_0)$.

        Each increment has distribution $\mathcal{N}(0, t_1 - t_0)$.

        **Arguments:**

        - `t0`: Start of interval.
        - `t1`: End of interval.
        - `left`: Ignored. (This determines whether to treat the path as
            left-continuous or right-continuous at any jump points, but Brownian
            motion has no jump points.)
        - `use_levy`: If True, the return type will be a `LevyVal`, which contains
            PyTrees of Brownian increments and their Levy areas.

        **Returns:**

        A pytree of JAX arrays corresponding to the increment $w(t_1) - w(t_0)$.

        Some subclasses may allow `t1=None`, in which case just the value $w(t_0)$ is
        returned.
        """
