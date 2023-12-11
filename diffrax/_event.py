import abc
from collections.abc import Callable
from typing import Optional

import equinox as eqx
import optimistix as optx
from jaxtyping import Array, PyTree

from ._custom_types import BoolScalarLike, RealScalarLike
from ._step_size_controller import AbstractAdaptiveStepSizeController


class AbstractDiscreteTerminatingEvent(eqx.Module):
    """Evaluated at the end of each integration step. If true then the solve is stopped
    at that time.
    """

    @abc.abstractmethod
    def __call__(self, state, **kwargs) -> BoolScalarLike:
        """**Arguments:**

        - `state`: a dataclass of the evolving state of the system, including in
            particular the solution `state.y` at time `state.tprev`.
        - `**kwargs`: the integration options held constant throughout the solve
            are passed as keyword arguments: `terms`, `solver`, `args`. etc.

        **Returns**

        A boolean. If true then the solve is terminated.
        """


class DiscreteTerminatingEvent(AbstractDiscreteTerminatingEvent):
    """Terminates the solve if its condition is ever active."""

    cond_fn: Callable[..., BoolScalarLike]

    def __call__(self, state, **kwargs):
        return self.cond_fn(state, **kwargs)


DiscreteTerminatingEvent.__init__.__doc__ = """**Arguments:**

- `cond_fn`: A function `(state, **kwargs) -> bool` that is evaluated on every step of
    the differential equation solve. If it returns `True` then the solve is finished at
    that timestep. `state` is a dataclass of the evolving state of the system,
    including in particular the solution `state.y` at time `state.tprev`. Passed as
    keyword arguments are the `terms`, `solver`, `args` etc. that are constant
    throughout the solve.
"""


class SteadyStateEvent(AbstractDiscreteTerminatingEvent):
    """Terminates the solve once it reaches a steady state."""

    rtol: Optional[float] = None
    atol: Optional[float] = None
    norm: Callable[[PyTree[Array]], RealScalarLike] = optx.rms_norm

    def __call__(self, state, *, terms, args, solver, stepsize_controller, **kwargs):
        del kwargs
        msg = (
            "The `rtol` and `atol` tolerances for `SteadyStateEvent` default "
            "to the `rtol` and `atol` used with an adaptive step size "
            "controller (such as `diffrax.PIDController`). Either use an "
            "adaptive step size controller, or specify these tolerances "
            "manually."
        )
        if self.rtol is None:
            if isinstance(stepsize_controller, AbstractAdaptiveStepSizeController):
                _rtol = stepsize_controller.rtol
            else:
                raise ValueError(msg)
        else:
            _rtol = self.rtol
        if self.atol is None:
            if isinstance(stepsize_controller, AbstractAdaptiveStepSizeController):
                _atol = stepsize_controller.atol
            else:
                raise ValueError(msg)
        else:
            _atol = self.atol

        # TODO: this makes an additional function evaluation that in practice has
        # probably already been made by the solver.
        vf = solver.func(terms, state.tprev, state.y, args)
        return self.norm(vf) < _atol + _rtol * self.norm(state.y)


SteadyStateEvent.__init__.__doc__ = """**Arguments:**

- `rtol`: The relative tolerance for determining convergence. Defaults to the
    same `rtol` as passed to an adaptive step controller if one is used.
- `atol`: The absolute tolerance for determining convergence. Defaults to the
    same `atol` as passed to an adaptive step controller if one is used.
- `norm`: A function `PyTree -> Scalar`, which is called to determine whether
    the vector field is close to zero.
"""
