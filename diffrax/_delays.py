from typing import Any, Callable, Optional

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import unvmap_any, ω
from jaxtyping import Array, Bool, PyTree, Scalar
from optimistix import (
    AbstractFixedPointSolver,
    fixed_point,
    Newton,
    RecursiveCheckpointAdjoint,
    RESULTS,
    rms_norm,
    root_find,
)
from optimistix._custom_types import Aux, Fn, Y

from ._custom_types import BoolScalarLike, IntScalarLike, RealScalarLike
from ._global_interpolation import DenseInterpolation
from ._local_interpolation import AbstractLocalInterpolation
from ._term import VectorFieldWrapper


class _FixedPointState(eqx.Module, strict=True):
    relative_error: Array
    steps: Array


class ModifiedFixedPointIteration(AbstractFixedPointSolver):
    rtol: float
    atol: float
    implicit_step: BoolScalarLike
    max_steps: int = eqx.field(static=True)
    norm: Callable[[PyTree], Scalar] = rms_norm

    def init(
        self,
        fn: Fn[Y, Y, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _FixedPointState:
        del fn, y, args, options, f_struct, aux_struct
        return _FixedPointState(jnp.array(jnp.inf), jnp.array(0))

    def step(
        self,
        fn: Fn[Y, Y, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _FixedPointState,
        tags: frozenset[object],
    ) -> tuple[Y, _FixedPointState, Aux]:
        new_y, aux = fn(y, args)
        error = (y**ω - new_y**ω).ω
        scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
        new_state = _FixedPointState(
            self.norm((error**ω / scale**ω).ω), state.steps + 1
        )
        return new_y, new_state, aux

    def terminate(
        self,
        fn: Fn[Y, Y, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _FixedPointState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return (
            ((state.relative_error < 1) | jnp.invert(self.implicit_step))
            & (state.steps > 0)
        ), RESULTS.successful

    def postprocess(
        self,
        fn: Fn[Y, Y, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _FixedPointState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}


class Delays(eqx.Module):
    """Module that incorporates all the information needed for integrating DDEs"""

    delays: PyTree[Callable]
    initial_discontinuities: Optional[Array] = jnp.array([0.0])
    max_discontinuities: IntScalarLike = 100
    recurrent_checking: bool = False
    sub_intervals: IntScalarLike = 10
    max_steps: IntScalarLike = 10
    rtol: RealScalarLike = 10e-3
    atol: RealScalarLike = 10e-3


class HistoryVectorField(eqx.Module):
    """VectorField equivalent for a DDE solver that incorporates former
    estimated values of y(t).

    **Arguments:**
        - `vector_field`: vector field of the delayed differential equation.
        - `t0`: global integration start time
        - `tprev`: start time of current integration step
        - `tnext`: end time of current integration step
        - `dense_info` : dense_info from current integration step
        - `y0_history` : DDE's history function
        - `delays` : DDE's different deviated arguments
    """

    vector_field: Callable[..., PyTree]
    t0: RealScalarLike
    tprev: RealScalarLike
    tnext: RealScalarLike
    dense_info: PyTree[Array]
    dense_interp: Optional[DenseInterpolation]
    interpolation_cls: Callable[..., AbstractLocalInterpolation]
    y0_history: Callable
    delays: PyTree[Callable]

    def __call__(self, t, y, args):
        history_vals = []
        delays, treedef = jtu.tree_flatten(self.delays)
        if self.dense_interp is None:
            assert self.dense_info is None
            for delay in delays:
                delay_val = delay(t, y, args)
                alpha_val = t - delay_val
                y0_val = self.y0_history(alpha_val)
                history_vals.append(y0_val)
        else:
            assert self.dense_info is not None
            for delay in delays:
                delay_val = delay(t, y, args)
                alpha_val = t - delay_val

                is_before_t0 = alpha_val < self.t0
                is_before_tprev = alpha_val < self.tprev
                at_most_t0 = jnp.where(alpha_val < self.t0, alpha_val, self.t0)
                t0_to_tprev = jnp.clip(alpha_val, self.t0, self.tprev)
                at_least_tprev = jnp.maximum(self.tprev, alpha_val)
                step_interpolation = self.interpolation_cls(
                    t0=self.tprev, t1=self.tnext, **self.dense_info
                )
                switch = jnp.where(is_before_t0, 0, jnp.where(is_before_tprev, 1, 2))
                history_val = lax.switch(
                    switch,
                    [
                        lambda: self.y0_history(at_most_t0),
                        lambda: self.dense_interp.evaluate(t0_to_tprev),  # pyright: ignore
                        lambda: step_interpolation.evaluate(at_least_tprev),
                    ],
                )
                history_vals.append(history_val)

        history_vals = jtu.tree_unflatten(treedef, history_vals)
        history_vals = tuple(history_vals)
        return self.vector_field(t, y, args, history=history_vals)


def bind_history(
    terms,
    delays,
    dense_info,
    dense_interp,
    solver,
    direction,
    t0,
    tprev,
    tnext,
    y0_history,
):
    delays_fn = jtu.tree_map(
        lambda x: (lambda t, y, args: x(t, y, args) * direction), delays.delays
    )

    is_vf_wrapper = lambda x: isinstance(x, VectorFieldWrapper)

    def _apply_history(
        x,
    ):
        if is_vf_wrapper(x):
            vector_field = HistoryVectorField(
                x.vector_field,
                t0,
                tprev,
                tnext,
                dense_info,
                dense_interp,
                solver.interpolation_cls,
                y0_history,
                delays_fn,
            )
            return VectorFieldWrapper(vector_field)
        else:
            return x

    terms_ = jtu.tree_map(_apply_history, terms, is_leaf=is_vf_wrapper)
    return terms_


def history_extrapolation_implicit(
    implicit_step,
    terms,
    dense_interp,
    init_guess,
    solver,
    delays,
    t0,
    y0_history,
    state,
    args,
):
    def fn(dense_info, args):
        (
            terms,
            _,
            dense_interp,
            solver,
            delays,
            t0,
            y0_history,
            state,
            vf_args,
        ) = args
        terms_ = bind_history(
            terms,
            delays,
            dense_info,
            dense_interp,
            solver,
            1,
            t0,
            state.tprev,
            state.tnext,
            y0_history,
        )
        (y, y_error, new_dense_info, solver_state, solver_result) = solver.step(
            terms_,
            state.tprev,
            state.tnext,
            state.y,
            vf_args,
            state.solver_state,
            state.made_jump,
        )

        return new_dense_info, (y, y_error, solver_state, solver_result)

    solv = ModifiedFixedPointIteration(
        rtol=delays.rtol,
        atol=delays.atol,
        norm=rms_norm,
        implicit_step=implicit_step,
        max_steps=delays.max_steps,
    )

    nonlinear_args = (
        terms,
        implicit_step,
        dense_interp,
        solver,
        delays,
        t0,
        y0_history,
        state,
        args,
    )
    sol = fixed_point(
        fn,
        solv,
        init_guess,
        nonlinear_args,
        has_aux=True,
        throw=False,
        adjoint=RecursiveCheckpointAdjoint(),
    )

    dense_info, (y, y_error, solver_state, solver_result) = sol.value, sol.aux

    return y, y_error, dense_info, solver_state, solver_result


def maybe_find_discontinuity(
    tprev,
    tnext,
    dense_info,
    state,
    delays,
    solver,
    args,
    keep_step,
    sub_tprev,
    sub_tnext,
):
    dense_discont = solver.interpolation_cls(t0=tprev, t1=tnext, **dense_info)
    flat_delays = jtu.tree_leaves(delays.delays)
    _gs = []

    def make_g(delay):
        # Creating the artifical event functions g that is used to
        # detect future breaking points.
        # http://www.cs.toronto.edu/pub/reports/na/hzpEnrightNA09Preprint.pdf
        # page 7
        def g(t):
            return (
                t
                - delay(t, dense_discont.evaluate(t), args)
                - state.discontinuities[...]
            )

        return g

    for delay in flat_delays:
        _gs.append(make_g(delay))

    def _find_discontinuity():
        # Start by doing a cheap bisection search to reduce
        # over the stored-discontinuity dimension.

        def _cond_fun(_val):
            _, _, _pred, _ = _val
            return _pred

        def _body_fun(_val):
            _ta, _tb, _, _step = _val
            _step = _step + 1
            _tmid = _ta + 0.5 * (_tb - _ta)
            _gas = jnp.stack([jnp.sign(g(_ta)) for g in _gs])
            _gmids = jnp.stack([jnp.sign(g(_tmid)) for g in _gs])
            _gbs = jnp.stack([jnp.sign(g(_tb)) for g in _gs])
            _any_left = jnp.any(_gas != _gmids)
            _next_ta = jnp.where(_any_left, _ta, _tmid)
            _next_tb = jnp.where(_any_left, _tmid, _tb)
            _pred = (
                jnp.any(jnp.sum(_gas != _gbs, axis=1) > 1) | _step > delays.max_steps
            )
            return _next_ta, _next_tb, _pred, _step

        _init_val = (sub_tprev, sub_tnext, True, 0)
        _final_val = lax.while_loop(_cond_fun, _body_fun, _init_val)
        _ta, _tb, _, _ = _final_val

        # Then do a more expensive Newton search
        # to find the first discontinuity.
        # _discont_solver = NewtonNonlinearSolver(rtol=delays.rtol, atol=delays.atol)
        _discont_solver = Newton(rtol=delays.rtol, atol=delays.atol)
        _disconts = []
        for g, delay in zip(_gs, flat_delays):
            changed_sign = jnp.sign(g(_ta)) != jnp.sign(g(_tb))
            _i = jnp.argmax(changed_sign)
            _d = state.discontinuities[_i]
            _h = (
                lambda t, args, delay=delay, _d=_d: t
                - delay(t, dense_discont.evaluate(t), args)
                - _d
            )
            _discont = root_find(_h, _discont_solver, _tb, args)
            # _discont = _discont_solver(_h, _tb, args).root
            _disconts.append(_discont.value)
        _disconts = jnp.stack(_disconts)

        best_candidate = jnp.where(
            (_disconts > sub_tprev) & (_disconts < sub_tnext),
            _disconts,
            jnp.inf,
        )
        best_candidate = jnp.min(best_candidate)
        discont_update = jnp.where(
            jnp.isinf(best_candidate),
            False,
            True,
        )
        return best_candidate, discont_update

    def _find_discontinuity_wrapper():
        return lax.cond(
            jnp.any(init_discont & jnp.invert(keep_step)),
            _find_discontinuity,
            lambda: (sub_tnext, False),
        )

    init_discont = jnp.stack(
        [jnp.sign(g(sub_tprev)) != jnp.sign(g(sub_tnext)) for g in _gs]
    )
    # We might have rejected the step for normal reasons;
    # skip looking for a discontinuity if so.
    return lax.cond(
        unvmap_any((init_discont & jnp.invert(keep_step))),
        _find_discontinuity_wrapper,
        lambda: (sub_tnext, False),
    )


Delays.__init__.__doc__ = """
**Arguments:**

- `delays`: A `PyTree` where the leaves are the DDE's different scalar 
  deviated arguments.
- `initial_discontinuities`: Discontinuities given by the initial point time
and history function.
- `max_discontinuities`: Array length that tracks the discontinuity jumps 
during integration (only relevant when `recurrent_checking` is True). If 
`recurrent checking` is set to `True`, the computation quits unconditionally
when the total number of discontinuities detected is larger 
than `max_discontinuities`.
- `recurrent_checking` : If `True`, there will be a systematic check at 
integration step for potential discontinuities (this involves nonlinear solves 
hence expensive). If `False`, discontinuities will only be checked when a step 
is rejected. This allows to integrate faster but can also impact 
the accuracy of the DDE solution.
- `sub_intervals` : Number of subintervals of the integration step where 
discontinuity tracking is done.
- `rtol` : Relative  tolerance for the nonlinear solver for the DDE's 
implicit stepping and dichotomy for detecting breaking points.
- `atol` : Absolute tolerance for the nonlinear solver for the DDE's
implicit stepping and dichotomy for detecting breaking points.
- `max_steps` : Max iteration of the dichotomy algorithm to 
find a discontinuity.

!!! example
    To integrate `y'(t) = - y(t-1)`, we need to define the vector 
    field and the `Delays` object. 
    ```py
    y0 = lambda t: 1.2
    def vector_field(t, y, args, history):
        return - history[0]

    delays = Delays(
        delays=[lambda t, y, args: 1.0],
        initial_discontinuities=jnp.array([0.0])     
    )
    t0, t1 = 0.0, 50.0
    ts = jnp.linspace(t0, t1, 500)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0 = y0_history,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays
        )
    ```
"""
