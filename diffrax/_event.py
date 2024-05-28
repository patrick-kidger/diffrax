import abc
from collections.abc import Callable
from typing import Optional, Union

import equinox as eqx
import optimistix as optx
from jaxtyping import Array, PyTree

from ._custom_types import BoolScalarLike, FloatScalarLike, RealScalarLike
from ._step_size_controller import AbstractAdaptiveStepSizeController


class Event(eqx.Module):
    """Can be used to terminate the solve early if one of multiple conditions
    is triggered. It allows for both continuous and boolean condition functions. In the
    former case, a root finder can be used to find the exact time of the event.

    Instances of this class should be passed as the `event` argument of
    [`diffrax.diffeqsolve`][].
    """

    cond_fn: PyTree[Callable[..., Union[BoolScalarLike, RealScalarLike]]]
    root_finder: Optional[optx.AbstractRootFinder] = None


Event.__init__.__doc__ = """**Arguments:**

- `cond_fn`: A PyTree of functions `f(t, y, args, **kwargs) -> c` returning a boolean or
    a real number. If the return value is a boolean, the solve will terminate when `c`
    is `True`. If the return value is a real number, the solve will terminate when `c`
    changes sign.
- `root_finder`: An optional root finder to use for finding the exact time of the event.
    If the triggered condition function is boolean,  the returned time will be the right
    endpoint of the last successful step.

"""


def steady_state_event(
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    norm: Optional[Callable[[PyTree[Array]], RealScalarLike]] = None,
):
    """Create a condition function that terminates the solve once a steady state is
    achieved. The returned function should be passed as the `cond_fn` argument of
    [`diffrax.Event`][].

    **Arguments:**

    - `rtol`, `atol`, `norm`: the solve will terminate once
        `norm(f) < atol + rtol * norm(y)`, where `f` is the result of evaluating the
        vector field. Will default to the values used in the `stepsize_controller` if
        they are not specified here.

    **Returns:**

    A function `f(t, y, args, **kwargs)`, that can be passed to
    `diffrax.Evetnt(cond_fn=..., ...)`.
    """

    def _cond_fn(t, y, args, *, terms, solver, stepsize_controller, **kwargs):
        del kwargs
        msg = (
            "The `rtol`, `atol`, and `norm` for `steady_state_event` default to the "
            "values used with an adaptive step size controller (such as "
            "`diffrax.PIDController`). Either use an adaptive step size controller, or "
            "specify these tolerances manually."
        )
        if rtol is None:
            if isinstance(stepsize_controller, AbstractAdaptiveStepSizeController):
                _rtol = stepsize_controller.rtol
            else:
                raise ValueError(msg)
        else:
            _rtol = rtol
        if atol is None:
            if isinstance(stepsize_controller, AbstractAdaptiveStepSizeController):
                _atol = stepsize_controller.atol
            else:
                raise ValueError(msg)
        else:
            _atol = atol
        if norm is None:
            if isinstance(stepsize_controller, AbstractAdaptiveStepSizeController):
                _norm = stepsize_controller.norm
            else:
                raise ValueError(msg)
        else:
            _norm = norm

        # TODO: this makes an additional function evaluation that in practice has
        # probably already been made by the solver.
        vf = solver.func(terms, t, y, args)
        return _norm(vf) < _atol + _rtol * _norm(y)

    return _cond_fn


#
# Backward compatibility: continue to support `AbstractDiscreteTerminatingEvent`.
# TODO: eventually remove everything below this line.
#


class AbstractDiscreteTerminatingEvent(eqx.Module):
    @abc.abstractmethod
    def __call__(self, state, **kwargs) -> BoolScalarLike:
        pass


class DiscreteTerminatingEvent(AbstractDiscreteTerminatingEvent):
    cond_fn: Callable[..., BoolScalarLike]

    def __call__(self, state, **kwargs):
        return self.cond_fn(state, **kwargs)


class SteadyStateEvent(AbstractDiscreteTerminatingEvent):
    rtol: Optional[float] = None
    atol: Optional[float] = None
    norm: Callable[[PyTree[Array]], RealScalarLike] = optx.rms_norm

    def __call__(self, state, *, args, **kwargs):
        return steady_state_event(self.rtol, self.atol, self.norm)(
            state.tprev, state.y, args, **kwargs
        )


class _StateCompat(eqx.Module):
    tprev: FloatScalarLike
    y: PyTree[Array]


class DiscreteTerminatingEventToCondFn(eqx.Module):
    event: AbstractDiscreteTerminatingEvent

    def __call__(self, t, y, args, **kwargs):
        return self.event(_StateCompat(tprev=t, y=y), args=args, **kwargs)
