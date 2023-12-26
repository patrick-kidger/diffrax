from collections.abc import Callable, Sequence
from typing import Optional, Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PyTree, Real

from ._custom_types import RealScalarLike


def save_y(t, y, args):
    return y


def _convert_ts(
    ts: Union[None, Sequence[RealScalarLike], Real[Array, " times"]],
) -> Optional[Real[Array, " times"]]:
    if ts is None or len(ts) == 0:
        return None
    else:
        return jnp.asarray(ts)


class SubSaveAt(eqx.Module):
    """Used for finer-grained control over what is saved. A PyTree of these should be
    passed to `SaveAt(subs=...)`.

    See [`diffrax.SaveAt`][] for more details on how this is used. (This is a
    relatively niche feature and most users will probably not need to use `SubSaveAt`.)
    """

    t0: bool = False
    t1: bool = False
    ts: Optional[Real[Array, " times"]] = eqx.field(default=None, converter=_convert_ts)
    steps: bool = False
    fn: Callable = save_y

    def __check_init__(self):
        if not self.t0 and not self.t1 and self.ts is None and not self.steps:
            raise ValueError("Empty saveat -- nothing will be saved.")


SubSaveAt.__init__.__doc__ = """**Arguments:**

- `t0`: If `True`, save the initial input `y0`.
- `t1`: If `True`, save the output at `t1`.
- `ts`: Some array of times at which to save the output.
- `steps`: If `True`, save the output at every step of the numerical solver.
- `fn`: A function `fn(t, y, args)` which specifies what to save into `sol.ys` when
    using `t0`, `t1`, `ts` or `steps`. Defaults to `fn(t, y, args) -> y`, so that the
    evolving solution is saved. This can be useful to save only statistics of your
    solution, so as to reduce memory usage.
"""


class SaveAt(eqx.Module):
    """Determines what to save as output from the differential equation solve.

    Instances of this class should be passed as the `saveat` argument of
    [`diffrax.diffeqsolve`][].
    """

    subs: PyTree[SubSaveAt] = None
    dense: bool = False
    solver_state: bool = False
    controller_state: bool = False
    made_jump: bool = False

    def __init__(
        self,
        *,
        t0: bool = False,
        t1: bool = False,
        ts: Union[None, Sequence[RealScalarLike], Real[Array, " times"]] = None,
        steps: bool = False,
        fn: Callable = save_y,
        subs: PyTree[SubSaveAt] = None,
        dense: bool = False,
        solver_state: bool = False,
        controller_state: bool = False,
        made_jump: bool = False,
    ):
        if subs is None:
            if t0 or t1 or (ts is not None) or steps:
                subs = SubSaveAt(t0=t0, t1=t1, ts=ts, steps=steps, fn=fn)
        else:
            if t0 or t1 or (ts is not None) or steps:
                raise ValueError(
                    "Cannot pass both `subs` and any of `t0`, `t1`, `ts`, `steps` to "
                    "`SaveAt`."
                )
        self.subs = subs
        self.dense = dense
        self.solver_state = solver_state
        self.controller_state = controller_state
        self.made_jump = made_jump


SaveAt.__init__.__doc__ = """**Main Arguments:**

- `t0`: If `True`, save the initial input `y0`.
- `t1`: If `True`, save the output at `t1`.
- `ts`: Some array of times at which to save the output.
- `steps`: If `True`, save the output at every step of the numerical solver.
- `dense`: If `True`, save dense output, that can later be evaluated at any part of
    the interval $[t_0, t_1]$ via `sol = diffeqsolve(...); sol.evaluate(...)`.

**Other Arguments:**

These arguments are used less frequently.

- `fn`: A function `fn(t, y, args)` which specifies what to save into `sol.ys` when
    using `t0`, `t1`, `ts` or `steps`. Defaults to `fn(t, y, args) -> y`, so that the
    evolving solution is saved. For example this can be useful to save only statistics
    of your solution, so as to reduce memory usage.

- `subs`: Some PyTree of [`diffrax.SubSaveAt`][], which allows for finer-grained control
    over what is saved. Each `SubSaveAt` specifies a combination of a function `fn` and
    some times `t0`, `t1`, `ts`, `steps` at which to evaluate it. `sol.ts` and `sol.ys`
    will then be PyTrees of the same structure as `subs`, with each leaf of the PyTree
    saving what the corresponding `SubSaveAt` specifies. The arguments
    `SaveAt(t0=..., t1=..., ts=..., steps=..., fn=...)` are actually just a convenience
    for passing a single `SubSaveAt` as
    `SaveAt(subs=SubSaveAt(t0=..., t1=..., ts=..., steps=..., fn=...))`. This
    functionality can be useful when you need different functions of the output saved
    at different times; see the examples below.

- `solver_state`: If `True`, save the internal state of the numerical solver at
    `t1`; accessible as `sol.solver_state`.

- `controller_state`: If `True`, save the internal state of the step size
    controller at `t1`; accessible as `sol.controller_state`.

- `made_jump`: If `True`, save the internal state of the jump tracker at `t1`;
    accessible as `sol.made_jump`.


!!! Example

    When solving a large PDE system, it may be the case that saving the full output
    `y` at all timesteps is too memory-intensive. Instead, we may prefer to save only
    the full final value, and only save statistics of the evolving solution. We can do
    this by:
    ```python
    t0 = 0
    t1 = 100
    ts = jnp.linspace(t0, t1, 1000)

    def statistics(t, y, args):
        return jnp.mean(y), jnp.std(y)

    final_subsaveat = diffrax.SubSaveAt(t1=True)
    evolving_subsaveat = diffrax.SubSaveAt(ts=ts, fn=statistics)
    saveat = diffrax.SaveAt(subs=[final_subsaveat, evolving_subsaveat])

    sol = diffrax.diffeqsolve(..., t0=t0, t1=t1, saveat=saveat)
    (y1, evolving_stats) = sol.ys  # PyTree of the save structure as `SaveAt(subs=...)`.
    evolving_means, evolving_stds = evolving_stats
    ```

    As another example, it may be the case that you are solving a 2-dimensional
    ODE, and want to save each component of its solution at different times. (Perhaps
    because you are comparing your model against data, and each dimension has data
    observed at different times.) This can be done through:
    ```python
    y0 = (y0_a, y0_b)
    ts_a = ...
    ts_b = ...
    subsaveat_a = diffrax.SubSaveAt(ts=ts_a, fn=lambda t, y, args: y[0])
    subsaveat_b = diffrax.SubSaveAt(ts=ts_b, fn=lambda t, y, args: y[1])
    saveat = diffrax.SaveAt(subs=[subsaveat_a, subsaveat_b])
    sol = diffrax.diffeqsolve(..., y0=y0, saveat=saveat)
    y_a, y_b = sol.ys  # PyTree of the same structure as `SaveAt(subs=...)`.
    # `sol.ts` will equal `(ts_a, ts_b)`.
    ```
"""
