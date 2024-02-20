import abc
from typing import Generic, Optional, TYPE_CHECKING, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp


if TYPE_CHECKING:
    from typing import ClassVar as AbstractVar
else:
    from equinox import AbstractVar

from ._custom_types import Control, RealScalarLike


_Control = TypeVar("_Control", bound=Control)


class AbstractPath(eqx.Module, Generic[_Control]):
    """Abstract base class for all paths.

    Every path has a start point `t0` and an end point `t1`. In between these values
    it is possible to `evaluate` the path, or (if it is differentiable, e.g. not
    Brownian motion) calculate its `derivative`.

    !!! example

        ```python
        class QuadraticPath(AbstractPath):
            @property
            def t0(self):
                return 0

            @property
            def t1(self):
                return 3

            def evaluate(self, t0, t1=None, left=True):
                del left
                if t1 is not None:
                    return self.evaluate(t1) - self.evaluate(t0)
                return t0 ** 2
        ```
    """

    t0: AbstractVar[RealScalarLike]
    t1: AbstractVar[RealScalarLike]

    @abc.abstractmethod
    def evaluate(
        self, t0: RealScalarLike, t1: Optional[RealScalarLike] = None, left: bool = True
    ) -> _Control:
        r"""Evaluate the path at any point in the interval $[t_0, t_1]$.

        **Arguments:**

        - `t0`: Any point in $[t_0, t_1]$ to evaluate the path at.
        - `t1`: If passed, then the increment from `t1` to `t0` is evaluated instead.
        - `left`: Across jump points: whether to treat the path as left-continuous
            or right-continuous.

        !!! faq "FAQ"

            Note that we use $t_0$ and $t_1$ to refer to the overall interval, as
            obtained via `instance.t0` and `instance.t1`. We use `t0` and `t1` to refer
            to some subinterval of $[t_0, t_1]$. This is an API that is used for
            consistency with the rest of the package, and just happens to be a little
            confusing here.

        **Returns:**

        If `t1` is not passed:

        The value of the path at `t0`.

        If `t1` is passed:

        The increment of the path between `t0` and `t1`.
        """

    def derivative(self, t: RealScalarLike, left: bool = True) -> _Control:
        r"""Evaluate the derivative of the path. Essentially equivalent
        to `jax.jvp(self.evaluate, (t,), (jnp.ones_like(t),))` (and indeed this is its
        default implementation if no other is specified).


        **Arguments:**

        - `t`: Any point in $[t_0, t_1]$ to evaluate the derivative at.
        - `left`: Whether to obtain the left-derivative or right-derivative at that
            point.

        **Returns:**

        The derivative of the path.
        """

        _, deriv = jax.jvp(
            lambda _t: self.evaluate(_t, left=left), (t,), (jnp.ones_like(t),)
        )
        return deriv
