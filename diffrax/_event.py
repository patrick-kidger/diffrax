import abc
from collections.abc import Callable
from typing import Optional, Union

import equinox as eqx
import optimistix as optx
from jaxtyping import Array, PyTree

from ._custom_types import BoolScalarLike, FloatScalarLike, RealScalarLike
from ._step_size_controller import AbstractAdaptiveStepSizeController


class Event(eqx.Module):
    """Can be used to terminate the solve early if a condition, or one of multiple
    conditions, is triggered. It allows for both boolean and continuous condition
    functions. In the latter case, a root finder can be used to find the exact time of
    the event. Boolean and continuous conditions can be used together.

    Instances of this class should be passed as the `event` argument of
    [`diffrax.diffeqsolve`][].
    """

    cond_fn: PyTree[Callable[..., Union[BoolScalarLike, RealScalarLike]]]
    root_finder: Optional[optx.AbstractRootFinder] = None


Event.__init__.__doc__ = """**Arguments:**

- `cond_fn`: A function or PyTree of functions `f(t, y, args, **kwargs) -> c` each
    returning either a boolean or a real number. If the return value is a boolean, then
    the solve will terminate on the first step on which `c` becomes `True`. If the
    return value is a real number, then the solve will terminate on the step when `c`
    changes sign.

- `root_finder`: An optional [root finder](../nonlinear_solver/) to use for finding
    the exact time of the event. If the triggered condition function returns a real
    number, then the final time will be the time at which that real number equals zero.
    (If the triggered condition function returns a boolean, then the returned time will
    just be the end of the step on which it becomes `True`.) 
    [`optimistix.Newton`](https://docs.kidger.site/optimistix/api/root_find/#optimistix.Newton)
    would be a typical choice here.

!!! Example

    Consider a bouncing ball dropped from some intial height $x_0$. We can model 
    the ball by a 2-dimensional ODE

    $\\frac{dx_t}{dt} = v_t, \\quad \\frac{dv_t}{dt} = -g,$

    where $x_t$ represents the height of the ball, $v_t$ its velocity, 
    and $g$ is the gravitational constant. With $g=8$, this corresponds to the
    vector field:

    ```python
    def vector_field(t, y, args):
        _, v = y
        return jnp.array([v, -8.0])
    ```

    Figuring out exactly when the ball hits the ground amounts to 
    solving the ODE until the event $x_t=0$ is triggered. This can be done by using 
    the real-valued condition function:

    ```python
    def cond_fn(t, y, args, **kwargs):
        x, _ = y
        return x
    ```

    With $x_0=10$, this would yield:

    ```python
    y0 = jnp.array([10.0, 0.0])
    t0 = 0
    t1 = jnp.inf
    dt0 = 0.1
    term = diffrax.ODETerm(vector_field)
    root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
    event = diffrax.Event(cond_fn, root_finder)
    solver = diffrax.Tsit5()
    sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, event=event)
    print(f"Event time: {sol.ts[0]}") # Event time: 1.58...
    print(f"Velocity at event time: {sol.ys[0, 1]}") # Velocity at event time: -12.64...
    ```
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
    `diffrax.Event(cond_fn=..., ...)`.
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
