import functools as ft
import typing
import warnings
from collections.abc import Callable
from typing import (
    Any,
    get_args,
    get_origin,
    Optional,
    Tuple,
    TYPE_CHECKING,
)

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.core
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax.internal as lxi
from jaxtyping import Array, ArrayLike, Float, Inexact, PyTree, Real

from ._adjoint import AbstractAdjoint, RecursiveCheckpointAdjoint
from ._custom_types import (
    BoolScalarLike,
    BufferDenseInfos,
    FloatScalarLike,
    IntScalarLike,
    RealScalarLike,
)
from ._event import AbstractDiscreteTerminatingEvent
from ._global_interpolation import DenseInterpolation
from ._heuristics import is_sde, is_unsafe_sde
from ._misc import linear_rescale, static_select
from ._progress_meter import (
    AbstractProgressMeter,
    NoProgressMeter,
)
from ._root_finder import use_stepsize_tol
from ._saveat import SaveAt, SubSaveAt
from ._solution import is_okay, is_successful, RESULTS, Solution
from ._solver import (
    AbstractImplicitSolver,
    AbstractItoSolver,
    AbstractSolver,
    AbstractStratonovichSolver,
    Euler,
    EulerHeun,
    ItoMilstein,
    StratonovichMilstein,
)
from ._step_size_controller import (
    AbstractAdaptiveStepSizeController,
    AbstractStepSizeController,
    ConstantStepSize,
    StepTo,
)
from ._term import AbstractTerm, MultiTerm, ODETerm, WrapTerm
from ._typing import better_isinstance, get_args_of, get_origin_no_specials


class SaveState(eqx.Module):
    saveat_ts_index: IntScalarLike
    ts: eqxi.MaybeBuffer[Real[Array, " times"]]
    ys: PyTree[eqxi.MaybeBuffer[Inexact[Array, "times ..."]]]
    save_index: IntScalarLike


class State(eqx.Module):
    # Evolving state during the solve
    y: PyTree[Array]
    tprev: FloatScalarLike
    tnext: FloatScalarLike
    made_jump: BoolScalarLike
    solver_state: PyTree[ArrayLike]
    controller_state: PyTree[ArrayLike]
    result: RESULTS
    num_steps: IntScalarLike
    num_accepted_steps: IntScalarLike
    num_rejected_steps: IntScalarLike
    # Output that is .at[].set() updated during the solve (and their indices)
    save_state: PyTree[SaveState]
    dense_ts: Optional[eqxi.MaybeBuffer[Float[Array, " times_plus_1"]]]
    dense_infos: Optional[BufferDenseInfos]
    dense_save_index: Optional[IntScalarLike]
    progress_meter_state: PyTree[Array]


def _is_none(x: Any) -> bool:
    return x is None


def _term_compatible(
    y: PyTree[ArrayLike],
    args: PyTree[Any],
    terms: PyTree[AbstractTerm],
    term_structure: PyTree,
    contr_kwargs: PyTree[dict],
) -> bool:
    error_msg = "term_structure"

    def _check(term_cls, term, term_contr_kwargs, yi):
        if get_origin_no_specials(term_cls, error_msg) is MultiTerm:
            if isinstance(term, MultiTerm):
                [_tmp] = get_args(term_cls)
                assert get_origin(_tmp) in (tuple, Tuple), "Malformed term_structure"
                assert len(term.terms) == len(get_args(_tmp))
                assert type(term_contr_kwargs) is tuple
                assert len(term.terms) == len(term_contr_kwargs)
                for term, arg, term_contr_kwarg in zip(
                    term.terms, get_args(_tmp), term_contr_kwargs
                ):
                    if not _term_compatible(yi, args, term, arg, term_contr_kwarg):
                        raise ValueError
            else:
                raise ValueError
        else:
            # Check that `term` is an instance of `term_cls` (ignoring any generic
            # parameterization).
            origin_cls = get_origin_no_specials(term_cls, error_msg)
            if origin_cls is None:
                origin_cls = term_cls
            if not isinstance(term, origin_cls):
                raise ValueError

            # Now check the generic parametrization of `term_cls`; can be one of:
            # -----------------------------------------
            # `term_cls`                | `term_args`
            # --------------------------|--------------
            # AbstractTerm              | ()
            # AbstractTerm[VF, Control] | (VF, Control)
            # -----------------------------------------
            term_args = get_args_of(AbstractTerm, term_cls, error_msg)
            n_term_args = len(term_args)
            if n_term_args == 0:
                pass
            elif n_term_args == 2:
                vf_type_expected, control_type_expected = term_args
                vf_type = eqx.filter_eval_shape(term.vf, 0.0, yi, args)
                vf_type_compatible = eqx.filter_eval_shape(
                    better_isinstance, vf_type, vf_type_expected
                )
                if not vf_type_compatible:
                    raise ValueError

                contr = ft.partial(term.contr, **term_contr_kwargs)
                control_type = jax.eval_shape(contr, 0.0, 0.0)
                control_type_compatible = eqx.filter_eval_shape(
                    better_isinstance, control_type, control_type_expected
                )
                if not control_type_compatible:
                    raise ValueError
            else:
                assert False, "Malformed term structure"
            # If we've got to this point then the term is compatible

    try:
        with jax.numpy_dtype_promotion("standard"):
            jtu.tree_map(_check, term_structure, terms, contr_kwargs, y)
    except ValueError:
        # ValueError may also arise from mismatched tree structures
        return False
    return True


def _is_subsaveat(x: Any) -> bool:
    return isinstance(x, SubSaveAt)


def _inner_buffers(save_state):
    assert type(save_state) is SaveState
    return save_state.ts, save_state.ys


def _outer_buffers(state):
    assert type(state) is State
    is_save_state = lambda x: isinstance(x, SaveState)
    # state.save_state has type PyTree[SaveState]. In particular this may include some
    # `None`s, which may sometimes be treated as leaves (e.g.
    # `tree_at(_outer_buffers, ..., is_leaf=lambda x: x is None)`).
    # So we need to only get those leaves which really are a SaveState.
    save_states = jtu.tree_leaves(state.save_state, is_leaf=is_save_state)
    save_states = [x for x in save_states if is_save_state(x)]
    return (
        [s.ts for s in save_states]
        + [s.ys for s in save_states]
        + [state.dense_ts, state.dense_infos]
    )


def _save(
    t: FloatScalarLike,
    y: PyTree[Array],
    args: PyTree,
    fn: Callable,
    save_state: SaveState,
) -> SaveState:
    ts = save_state.ts
    ys = save_state.ys
    save_index = save_state.save_index

    ts = ts.at[save_index].set(t)
    ys = jtu.tree_map(lambda ys_, y_: ys_.at[save_index].set(y_), ys, fn(t, y, args))
    save_index = save_index + 1

    return eqx.tree_at(
        lambda s: [s.ts, s.ys, s.save_index], save_state, [ts, ys, save_index]
    )


def _clip_to_end(tprev, tnext, t1, keep_step):
    # The tolerance means that we don't end up with too-small intervals for
    # dense output, which then gives numerically unstable answers due to floating
    # point errors.
    if tnext.dtype is jnp.dtype("float64"):
        tol = 1e-10
    else:
        tol = 1e-6
    clip = tnext > t1 - tol
    tclip = jnp.where(keep_step, t1, tprev + 0.5 * (t1 - tprev))
    return jnp.where(clip, tclip, tnext)


def _maybe_static(static_x: Optional[ArrayLike], x: ArrayLike) -> ArrayLike:
    # Some values (made_jump and result) are not used in many common use-cases. If we
    # detect that they're unused then we make sure they're non-Array Python values, so
    # that we can special case on them at trace time and get a performance boost.
    if isinstance(static_x, (bool, int, float, complex)):
        return static_x
    elif static_x is None:
        return x
    elif type(jax.core.get_aval(static_x)) is jax.core.ConcreteArray:
        return static_x
    else:
        return x


_PRINT_STATIC = False  # used in tests


def loop(
    *,
    solver,
    stepsize_controller,
    discrete_terminating_event,
    saveat,
    t0,
    t1,
    dt0,
    max_steps,
    terms,
    args,
    init_state,
    inner_while_loop,
    outer_while_loop,
    progress_meter,
):
    if saveat.dense:
        dense_ts = init_state.dense_ts
        dense_ts = dense_ts.at[0].set(t0)
        init_state = eqx.tree_at(lambda s: s.dense_ts, init_state, dense_ts)

    def save_t0(subsaveat: SubSaveAt, save_state: SaveState) -> SaveState:
        if subsaveat.t0:
            save_state = _save(t0, init_state.y, args, subsaveat.fn, save_state)
        return save_state

    save_state = jtu.tree_map(
        save_t0, saveat.subs, init_state.save_state, is_leaf=_is_subsaveat
    )
    init_state = eqx.tree_at(
        lambda s: s.save_state, init_state, save_state, is_leaf=_is_none
    )

    def _handle_static(state):
        # We can improve runtime by resolving `result` at trace time if possible.
        # We can improve compiletime by resolving `made_jump` at trace time if possible.
        if static_result is None:
            result = state.result
        else:
            result = jtu.tree_map(_maybe_static, static_result, state.result)
        made_jump = _maybe_static(static_made_jump, state.made_jump)
        return eqx.tree_at(
            lambda s: (s.result, s.made_jump), state, (result, made_jump)
        )

    def cond_fun(state):
        if isinstance(stepsize_controller, StepTo):
            # Privileged optimisation.
            # This is a measurably cheaper check than the tprev < t1 check.
            out = state.num_steps < len(stepsize_controller.ts) - 1
        else:
            out = state.tprev < t1
        state = _handle_static(state)
        return out & is_successful(state.result)

    def body_fun_aux(state):
        state = _handle_static(state)

        #
        # Actually do some differential equation solving! Make numerical steps, adapt
        # step sizes, all that jazz.
        #

        (y, y_error, dense_info, solver_state, solver_result) = solver.step(
            terms,
            state.tprev,
            state.tnext,
            state.y,
            args,
            state.solver_state,
            state.made_jump,
        )

        # e.g. if someone has a sqrt(y) in the vector field, and dt0 is so large that
        # we get a negative value for y, and then get a NaN vector field. (And then
        # everything breaks.) See #143.
        y_error = jtu.tree_map(lambda x: jnp.where(jnp.isnan(x), jnp.inf, x), y_error)

        error_order = solver.error_order(terms)
        (
            keep_step,
            tprev,
            tnext,
            made_jump,
            controller_state,
            stepsize_controller_result,
        ) = stepsize_controller.adapt_step_size(
            state.tprev,
            state.tnext,
            state.y,
            y,
            args,
            y_error,
            error_order,
            state.controller_state,
        )
        assert jnp.result_type(keep_step) is jnp.dtype(bool)

        #
        # Do some book-keeping.
        #

        tprev = jnp.minimum(tprev, t1)
        tnext = _clip_to_end(tprev, tnext, t1, keep_step)

        progress_meter_state = progress_meter.step(
            state.progress_meter_state, linear_rescale(t0, tprev, t1)
        )

        # The other parts of the mutable state are kept/not-kept (based on whether the
        # step was accepted) by the stepsize controller. But it doesn't get access to
        # these parts, so we do them here.
        keep = lambda a, b: jnp.where(keep_step, a, b)
        y = jtu.tree_map(keep, y, state.y)
        solver_state = jtu.tree_map(keep, solver_state, state.solver_state)
        made_jump = static_select(keep_step, made_jump, state.made_jump)
        solver_result = RESULTS.where(keep_step, solver_result, RESULTS.successful)

        # TODO: if we ever support non-terminating events, then they should go in here.
        # In particular the thing to be careful about is in the `if saveat.steps`
        # branch below, where we want to make sure that it is the value of `y` at
        # `tprev` that is actually saved. (And not just the value of `y` at the
        # previous step's `tnext`, i.e. immediately before the jump.)

        # Store the first unsuccessful result we get whilst iterating (if any).
        result = RESULTS.where(is_okay(state.result), solver_result, state.result)
        result = RESULTS.where(is_okay(result), stepsize_controller_result, result)

        # Count the number of steps, just for statistical purposes.
        num_steps = state.num_steps + 1
        num_accepted_steps = state.num_accepted_steps + jnp.where(keep_step, 1, 0)
        # Not just ~keep_step, which does the wrong thing when keep_step is a non-array
        # bool True/False.
        num_rejected_steps = state.num_rejected_steps + jnp.where(keep_step, 0, 1)

        #
        # Store the output produced from this numerical step.
        #

        interpolator = solver.interpolation_cls(
            t0=state.tprev, t1=state.tnext, **dense_info
        )
        save_state = state.save_state
        dense_ts = state.dense_ts
        dense_infos = state.dense_infos
        dense_save_index = state.dense_save_index

        def save_ts(subsaveat: SubSaveAt, save_state: SaveState) -> SaveState:
            if subsaveat.ts is not None:
                save_state = save_ts_impl(subsaveat.ts, subsaveat.fn, save_state)
            return save_state

        def save_ts_impl(ts, fn, save_state: SaveState) -> SaveState:
            def _cond_fun(_save_state):
                return (
                    keep_step
                    & (ts[_save_state.saveat_ts_index] <= state.tnext)
                    & (_save_state.saveat_ts_index < len(ts))
                )

            def _body_fun(_save_state):
                _t = ts[_save_state.saveat_ts_index]
                _y = interpolator.evaluate(_t)
                _ts = _save_state.ts.at[_save_state.save_index].set(_t)
                _ys = jtu.tree_map(
                    lambda __y, __ys: __ys.at[_save_state.save_index].set(__y),
                    fn(_t, _y, args),
                    _save_state.ys,
                )
                return SaveState(
                    saveat_ts_index=_save_state.saveat_ts_index + 1,
                    ts=_ts,
                    ys=_ys,
                    save_index=_save_state.save_index + 1,
                )

            return inner_while_loop(
                _cond_fun,
                _body_fun,
                save_state,
                max_steps=len(ts),
                buffers=_inner_buffers,
                checkpoints=len(ts),
            )

        save_state = jtu.tree_map(
            save_ts, saveat.subs, save_state, is_leaf=_is_subsaveat
        )

        def maybe_inplace(i, u, x):
            return eqxi.buffer_at_set(x, i, u, pred=keep_step)

        def save_steps(subsaveat: SubSaveAt, save_state: SaveState) -> SaveState:
            if subsaveat.steps:
                ts = maybe_inplace(save_state.save_index, tprev, save_state.ts)
                ys = jtu.tree_map(
                    ft.partial(maybe_inplace, save_state.save_index),
                    subsaveat.fn(tprev, y, args),
                    save_state.ys,
                )
                save_index = save_state.save_index + jnp.where(keep_step, 1, 0)
                save_state = eqx.tree_at(
                    lambda s: [s.ts, s.ys, s.save_index],
                    save_state,
                    [ts, ys, save_index],
                )
            return save_state

        save_state = jtu.tree_map(
            save_steps, saveat.subs, save_state, is_leaf=_is_subsaveat
        )

        if saveat.dense:
            dense_ts = maybe_inplace(dense_save_index + 1, tprev, dense_ts)
            dense_infos = jtu.tree_map(
                ft.partial(maybe_inplace, dense_save_index),
                dense_info,
                dense_infos,
            )
            dense_save_index = dense_save_index + jnp.where(keep_step, 1, 0)

        new_state = State(
            y=y,
            tprev=tprev,
            tnext=tnext,
            made_jump=made_jump,  # pyright: ignore
            solver_state=solver_state,
            controller_state=controller_state,
            result=result,
            num_steps=num_steps,
            num_accepted_steps=num_accepted_steps,
            num_rejected_steps=num_rejected_steps,
            save_state=save_state,
            dense_ts=dense_ts,  # pyright: ignore
            dense_infos=dense_infos,
            dense_save_index=dense_save_index,
            progress_meter_state=progress_meter_state,
        )

        if discrete_terminating_event is not None:
            discrete_terminating_event_occurred = discrete_terminating_event(
                new_state,
                solver=solver,
                stepsize_controller=stepsize_controller,
                saveat=saveat,
                t0=t0,
                t1=t1,
                dt0=dt0,
                max_steps=max_steps,
                terms=terms,
                args=args,
            )
            result = RESULTS.where(
                discrete_terminating_event_occurred,
                RESULTS.discrete_terminating_event_occurred,
                result,
            )
            new_state = eqx.tree_at(lambda s: s.result, new_state, result)

        return (
            new_state,
            (type(new_state.made_jump) is not bool),
            new_state.result.is_traced(),
        )

    static_made_jump = init_state.made_jump
    static_result = init_state.result
    _, traced_jump, traced_result = eqx.filter_eval_shape(body_fun_aux, init_state)
    if traced_jump:
        static_made_jump = None
    if traced_result:
        static_result = None
    if traced_jump or traced_result:
        # In case changing one changes the other.
        _, traced_jump, traced_result = eqx.filter_eval_shape(body_fun_aux, init_state)
        if traced_jump:
            static_made_jump = None
        if traced_result:
            static_result = None
    if _PRINT_STATIC:
        print(f"{static_made_jump=} {static_result=}")

    def body_fun(state):
        new_state, _, _ = body_fun_aux(state)
        return new_state

    final_state = outer_while_loop(
        cond_fun, body_fun, init_state, max_steps=max_steps, buffers=_outer_buffers
    )

    def _save_t1(subsaveat, save_state):
        if subsaveat.t1 and not subsaveat.steps:
            # If subsaveat.steps then the final value is already saved.
            #
            # Use `tprev` instead of `t1` in case of an event terminating the solve
            # early. (And absent such an event then `tprev == t1`.)
            save_state = _save(
                final_state.tprev, final_state.y, args, subsaveat.fn, save_state
            )
        return save_state

    save_state = jtu.tree_map(
        _save_t1, saveat.subs, final_state.save_state, is_leaf=_is_subsaveat
    )
    final_state = eqx.tree_at(
        lambda s: s.save_state, final_state, save_state, is_leaf=_is_none
    )

    final_state = _handle_static(final_state)
    result = RESULTS.where(
        cond_fun(final_state), RESULTS.max_steps_reached, final_state.result
    )
    aux_stats = dict()
    return eqx.tree_at(lambda s: s.result, final_state, result), aux_stats


if not TYPE_CHECKING:
    if getattr(typing, "GENERATING_DOCUMENTATION", False):
        # Nicer documentation for the default `diffeqsolve(saveat=...)` argument.
        # Not using `eqxi.doc_repr` as some IDEs (Helix, at least) show the source code
        # of the default argument directly.
        class SaveAt(eqx.Module):  # noqa: F811
            t1: bool


@eqx.filter_jit
def diffeqsolve(
    terms: PyTree[AbstractTerm],
    solver: AbstractSolver,
    t0: RealScalarLike,
    t1: RealScalarLike,
    dt0: Optional[RealScalarLike],
    y0: PyTree[ArrayLike],
    args: PyTree[Any] = None,
    *,
    saveat: SaveAt = SaveAt(t1=True),
    stepsize_controller: AbstractStepSizeController = ConstantStepSize(),
    adjoint: AbstractAdjoint = RecursiveCheckpointAdjoint(),
    discrete_terminating_event: Optional[AbstractDiscreteTerminatingEvent] = None,
    max_steps: Optional[int] = 4096,
    throw: bool = True,
    progress_meter: AbstractProgressMeter = NoProgressMeter(),
    solver_state: Optional[PyTree[ArrayLike]] = None,
    controller_state: Optional[PyTree[ArrayLike]] = None,
    made_jump: Optional[BoolScalarLike] = None,
) -> Solution:
    """Solves a differential equation.

    This function is the main entry point for solving all kinds of initial value
    problems, whether they are ODEs, SDEs, or CDEs.

    The differential equation is integrated from `t0` to `t1`.

    See the [Getting started](../usage/getting-started.md) page for example usage.

    **Main arguments:**

    These are the arguments most commonly used day-to-day.

    - `terms`: The terms of the differential equation. This specifies the vector field.
        (For non-ordinary differential equations (SDEs, CDEs), this also specifies the
        Brownian motion or the control.)
    - `solver`: The solver for the differential equation. See the guide on [how to
        choose a solver](../usage/how-to-choose-a-solver.md).
    - `t0`: The start of the region of integration.
    - `t1`: The end of the region of integration.
    - `dt0`: The step size to use for the first step. If using fixed step sizes then
        this will also be the step size for all other steps. (Except the last one,
        which may be slightly smaller and clipped to `t1`.) If set as `None` then the
        initial step size will be determined automatically.
    - `y0`: The initial value. This can be any PyTree of JAX arrays. (Or types that
        can be coerced to JAX arrays, like Python floats.)
    - `args`: Any additional arguments to pass to the vector field.
    - `saveat`: What times to save the solution of the differential equation. See
        [`diffrax.SaveAt`][]. Defaults to just the last time `t1`. (Keyword-only
        argument.)
    - `stepsize_controller`: How to change the step size as the integration progresses.
        See the [list of stepsize controllers](../api/stepsize_controller.md).
        Defaults to using a fixed constant step size. (Keyword-only argument.)

    **Other arguments:**

    These arguments are less frequently used, and for most purposes you shouldn't need
    to understand these. All of these are keyword-only arguments.

    - `adjoint`: How to differentiate `diffeqsolve`. Defaults to
        discretise-then-optimise, which is usually the best option for most problems.
        See the page on [Adjoints](./adjoints.md) for more information.

    - `discrete_terminating_event`: A discrete event at which to terminate the solve
        early. See the page on [Events](./events.md) for more information.

    - `max_steps`: The maximum number of steps to take before quitting the computation
        unconditionally.

        Can also be set to `None` to allow an arbitrary number of steps, although this
        is incompatible with `saveat=SaveAt(steps=True)` or `saveat=SaveAt(dense=True)`.

    - `throw`: Whether to raise an exception if the integration fails for any reason.

        If `True` then an integration failure will raise an error. Note that the errors
        are only reliably raised on CPUs. If on GPUs then the error may only be
        printed to stderr, whilst on TPUs then the behaviour is undefined.

        If `False` then the returned solution object will have a `result` field
        indicating whether any failures occurred.

        Possible failures include for example hitting `max_steps`, or the problem
        becoming too stiff to integrate. (For most purposes these failures are
        unusual.)

        !!! note

            When `jax.vmap`-ing a differential equation solve, then
            `throw=True` means that an exception will be raised if any batch element
            fails. You may prefer to set `throw=False` and inspect the `result` field
            of the returned solution object, to determine which batch elements
            succeeded and which failed.

    - `progress_meter`: A progress meter to indicate how far through the solve has
        progressed. See [the progress meters page](./progress_meter.md).

    - `solver_state`: Some initial state for the solver. Generally obtained by
        `SaveAt(solver_state=True)` from a previous solve.

    - `controller_state`: Some initial state for the step size controller. Generally
        obtained by `SaveAt(controller_state=True)` from a previous solve.

    - `made_jump`: Whether a jump has just been made at `t0`. Used to update
        `solver_state` (if passed). Generally obtained by `SaveAt(made_jump=True)`
        from a previous solve.

    **Returns:**

    Returns a [`diffrax.Solution`][] object specifying the solution to the differential
    equation.

    **Raises:**

    - `ValueError` for bad inputs.
    - `RuntimeError` if `throw=True` and the integration fails (e.g. hitting the
        maximum number of steps).

    !!! note

        It is possible to have `t1 < t0`, in which case integration proceeds backwards
        in time.
    """

    #
    # Initial set-up
    #

    # Error checking
    if dt0 is not None:
        msg = (
            "Must have (t1 - t0) * dt0 >= 0, we instead got "
            f"t1 with value {t1} and type {type(t1)}, "
            f"t0 with value {t0} and type {type(t0)}, "
            f"dt0 with value {dt0} and type {type(dt0)}"
        )
        with jax.ensure_compile_time_eval(), jax.numpy_dtype_promotion("standard"):
            pred = (t1 - t0) * dt0 < 0
        dt0 = eqxi.error_if(jnp.array(dt0), pred, msg)

    # Error checking and warning for complex dtypes
    if any(
        eqx.is_array(xi) and jnp.iscomplexobj(xi)
        for xi in jtu.tree_leaves((terms, y0, args))
    ):
        warnings.warn(
            "Complex dtype support is work in progress, please read "
            "https://github.com/patrick-kidger/diffrax/pull/197 and proceed carefully.",
            stacklevel=2,
        )

    # Allow setting e.g. t0 as an int with dt0 as a float.
    timelikes = [t0, t1, dt0] + [
        s.ts for s in jtu.tree_leaves(saveat.subs, is_leaf=_is_subsaveat)
    ]
    timelikes = [x for x in timelikes if x is not None]
    with jax.numpy_dtype_promotion("standard"):
        time_dtype = jnp.result_type(*timelikes)
    if jnp.issubdtype(time_dtype, jnp.complexfloating):
        raise ValueError(
            "Cannot use complex dtype for `t0`, `t1`, `dt0`, or `SaveAt(ts=...)`."
        )
    elif jnp.issubdtype(time_dtype, jnp.floating):
        pass
    elif jnp.issubdtype(time_dtype, jnp.integer):
        time_dtype = lxi.default_floating_dtype()
    else:
        raise ValueError(f"Unrecognised time dtype {time_dtype}.")
    t0 = jnp.asarray(t0, dtype=time_dtype)
    t1 = jnp.asarray(t1, dtype=time_dtype)
    if dt0 is not None:
        dt0 = jnp.asarray(dt0, dtype=time_dtype)

    def _get_subsaveat_ts(saveat):
        out = [s.ts for s in jtu.tree_leaves(saveat.subs, is_leaf=_is_subsaveat)]
        return [x for x in out if x is not None]

    saveat = eqx.tree_at(
        _get_subsaveat_ts,
        saveat,
        replace_fn=lambda ts: ts.astype(time_dtype),  # noqa: F821
    )

    # Time will affect state, so need to promote the state dtype as well if necessary.
    # fixing issue with float64 and weak dtypes, see discussion at:
    # https://github.com/patrick-kidger/diffrax/pull/197#discussion_r1130173527
    def _promote(yi):
        with jax.numpy_dtype_promotion("standard"):
            _dtype = jnp.result_type(yi, time_dtype)  # noqa: F821
        return jnp.asarray(yi, dtype=_dtype)

    y0 = jtu.tree_map(_promote, y0)
    del timelikes

    # Backward compatibility
    if isinstance(
        solver, (EulerHeun, ItoMilstein, StratonovichMilstein)
    ) and _term_compatible(
        y0, args, terms, (ODETerm, AbstractTerm), solver.term_compatible_contr_kwargs
    ):
        warnings.warn(
            "Passing `terms=(ODETerm(...), SomeOtherTerm(...))` to "
            f"{solver.__class__.__name__} is deprecated in favour of "
            "`terms=MultiTerm(ODETerm(...), SomeOtherTerm(...))`. This means that "
            "the same terms can now be passed used for both general and SDE-specific "
            "solvers!",
            stacklevel=2,
        )
        terms = MultiTerm(*terms)

    # Error checking
    if not _term_compatible(
        y0, args, terms, solver.term_structure, solver.term_compatible_contr_kwargs
    ):
        raise ValueError(
            "`terms` must be a PyTree of `AbstractTerms` (such as `ODETerm`), with "
            f"structure {solver.term_structure}"
        )

    if is_sde(terms):
        if not isinstance(solver, (AbstractItoSolver, AbstractStratonovichSolver)):
            warnings.warn(
                f"`{type(solver).__name__}` is not marked as converging to either the "
                "Itô or the Stratonovich solution.",
                stacklevel=2,
            )
        if isinstance(stepsize_controller, AbstractAdaptiveStepSizeController):
            # Specific check to not work even if using HalfSolver(Euler())
            if isinstance(solver, Euler):
                raise ValueError(
                    "An SDE should not be solved with adaptive step sizes with Euler's "
                    "method, as it may not converge to the correct solution."
                )
    if is_unsafe_sde(terms):
        if isinstance(stepsize_controller, AbstractAdaptiveStepSizeController):
            raise ValueError(
                "`UnsafeBrownianPath` cannot be used with adaptive step sizes."
            )

    # Normalises time: if t0 > t1 then flip things around.
    direction = jnp.where(t0 < t1, 1, -1)
    t0 = t0 * direction
    t1 = t1 * direction
    if dt0 is not None:
        dt0 = dt0 * direction
    saveat = eqx.tree_at(
        _get_subsaveat_ts, saveat, replace_fn=lambda ts: ts * direction
    )
    stepsize_controller = stepsize_controller.wrap(direction)

    def _wrap(term):
        assert isinstance(term, AbstractTerm)
        assert not isinstance(term, MultiTerm)
        return WrapTerm(term, direction)

    terms = jtu.tree_map(
        _wrap,
        terms,
        is_leaf=lambda x: isinstance(x, AbstractTerm) and not isinstance(x, MultiTerm),
    )

    if isinstance(solver, AbstractImplicitSolver):

        def _get_tols(x):
            outs = []
            for attr in ("rtol", "atol", "norm"):
                if getattr(solver.root_finder, attr) is use_stepsize_tol:  # pyright: ignore
                    outs.append(getattr(x, attr))
            return tuple(outs)

        if isinstance(stepsize_controller, AbstractAdaptiveStepSizeController):
            solver = eqx.tree_at(
                lambda s: _get_tols(s.root_finder),
                solver,
                _get_tols(stepsize_controller),
            )
        else:
            if len(_get_tols(solver.root_finder)) > 0:
                raise ValueError(
                    "A fixed step size controller is being used alongside an implicit "
                    "solver, but the tolerances for the implicit solver have not been "
                    "specified. (Being unspecified is the default in Diffrax.)\n"
                    "The correct fix is almost always to use an adaptive step size "
                    "controller. For example "
                    "`diffrax.diffeqsolve(..., "
                    "stepsize_controller=diffrax.PIDController(rtol=..., atol=...))`. "
                    "In this case the same tolerances are used for the implicit "
                    "solver as are used to control the adaptive stepping.\n"
                    "(Note for advanced users: the tolerances for the implicit "
                    "solver can also be explicitly set instead. For example "
                    "`diffrax.diffeqsolve(..., solver=diffrax.Kvaerno5(root_finder="
                    "diffrax.VeryChord(rtol=..., atol=..., "
                    "norm=optimistix.max_norm)))`. In this case the norm must also be "
                    "explicitly specified.)\n"
                    "Adaptive step size controllers are the preferred solution, as "
                    "sometimes the implicit solver may fail to converge, and in this "
                    "case an adaptive step size controller can reject the step and try "
                    "a smaller one, whilst with a fixed step size controller the "
                    "overall differential equation solve will simply fail."
                )

    # Error checking
    def _check_subsaveat_ts(ts):
        ts = eqxi.error_if(
            ts,
            ts[1:] < ts[:-1],
            "saveat.ts must be increasing or decreasing.",
        )
        ts = eqxi.error_if(
            ts,
            (ts > t1) | (ts < t0),
            "saveat.ts must lie between t0 and t1.",
        )
        return ts

    saveat = eqx.tree_at(_get_subsaveat_ts, saveat, replace_fn=_check_subsaveat_ts)

    # Initialise states
    tprev = t0
    error_order = solver.error_order(terms)
    if controller_state is None:
        passed_controller_state = False
        (tnext, controller_state) = stepsize_controller.init(
            terms, t0, t1, y0, dt0, args, solver.func, error_order
        )
    else:
        passed_controller_state = True
        if dt0 is None:
            (tnext, _) = stepsize_controller.init(
                terms, t0, t1, y0, dt0, args, solver.func, error_order
            )
        else:
            tnext = t0 + dt0
    tnext = jnp.minimum(tnext, t1)
    if solver_state is None:
        passed_solver_state = False
        solver_state = solver.init(terms, t0, tnext, y0, args)
    else:
        passed_solver_state = True

    # Allocate memory to store output.
    def _allocate_output(subsaveat: SubSaveAt) -> SaveState:
        out_size = 0
        if subsaveat.t0:
            out_size += 1
        if subsaveat.ts is not None:
            out_size += len(subsaveat.ts)
        if subsaveat.steps:
            # We have no way of knowing how many steps we'll actually end up taking, and
            # XLA doesn't support dynamic shapes. So we just have to allocate the
            # maximum amount of steps we can possibly take.
            if max_steps is None:
                raise ValueError(
                    "`max_steps=None` is incompatible with saving at `steps=True`"
                )
            out_size += max_steps
        if subsaveat.t1 and not subsaveat.steps:
            out_size += 1
        saveat_ts_index = 0
        save_index = 0
        ts = jnp.full(out_size, jnp.inf, dtype=time_dtype)
        struct = eqx.filter_eval_shape(subsaveat.fn, t0, y0, args)
        ys = jtu.tree_map(
            lambda y: jnp.full((out_size,) + y.shape, jnp.inf, dtype=y.dtype), struct
        )
        return SaveState(
            ts=ts, ys=ys, save_index=save_index, saveat_ts_index=saveat_ts_index
        )

    save_state = jtu.tree_map(_allocate_output, saveat.subs, is_leaf=_is_subsaveat)
    num_steps = 0
    num_accepted_steps = 0
    num_rejected_steps = 0
    made_jump = False if made_jump is None else made_jump
    result = RESULTS.successful
    if saveat.dense:
        if max_steps is None:
            raise ValueError(
                "`max_steps=None` is incompatible with `saveat.dense=True`"
            )
        (
            _,
            _,
            dense_info,
            _,
            _,
        ) = eqx.filter_eval_shape(
            solver.step, terms, tprev, tnext, y0, args, solver_state, made_jump
        )
        dense_ts = jnp.full(max_steps + 1, jnp.inf, dtype=time_dtype)
        _make_full = lambda x: jnp.full(
            (max_steps,) + jnp.shape(x), jnp.inf, dtype=x.dtype
        )
        dense_infos = jtu.tree_map(_make_full, dense_info)
        dense_save_index = 0
    else:
        dense_ts = None
        dense_infos = None
        dense_save_index = None

    # Progress meter
    progress_meter_state = progress_meter.init()

    # Initialise state
    init_state = State(
        y=y0,
        tprev=tprev,
        tnext=tnext,
        made_jump=made_jump,
        solver_state=solver_state,
        controller_state=controller_state,
        result=result,
        num_steps=num_steps,
        num_accepted_steps=num_accepted_steps,
        num_rejected_steps=num_rejected_steps,
        save_state=save_state,
        dense_ts=dense_ts,
        dense_infos=dense_infos,
        dense_save_index=dense_save_index,
        progress_meter_state=progress_meter_state,
    )

    #
    # Main loop
    #

    final_state, aux_stats = adjoint.loop(
        args=args,
        terms=terms,
        solver=solver,
        stepsize_controller=stepsize_controller,
        discrete_terminating_event=discrete_terminating_event,
        saveat=saveat,
        t0=t0,
        t1=t1,
        dt0=dt0,
        max_steps=max_steps,
        init_state=init_state,
        throw=throw,
        passed_solver_state=passed_solver_state,
        passed_controller_state=passed_controller_state,
        progress_meter=progress_meter,
    )

    #
    # Finish up
    #

    progress_meter.close(final_state.progress_meter_state)

    is_save_state = lambda x: isinstance(x, SaveState)
    ts = jtu.tree_map(
        lambda s: s.ts * direction, final_state.save_state, is_leaf=is_save_state
    )
    ys = jtu.tree_map(lambda s: s.ys, final_state.save_state, is_leaf=is_save_state)
    # It's important that we don't do any further postprocessing on `ys` here, as
    # it is the `final_state` value that is used when backpropagating via
    # optimise-then-discretise.

    if saveat.controller_state:
        controller_state = final_state.controller_state
    else:
        controller_state = None
    if saveat.solver_state:
        solver_state = final_state.solver_state
    else:
        solver_state = None
    if saveat.made_jump:
        made_jump = final_state.made_jump
    else:
        made_jump = None
    if saveat.dense:
        interpolation = DenseInterpolation(
            ts=final_state.dense_ts,
            ts_size=final_state.dense_save_index + 1,
            infos=final_state.dense_infos,
            interpolation_cls=solver.interpolation_cls,
            direction=direction,
            t0_if_trivial=t0,
            y0_if_trivial=y0,
        )
    else:
        interpolation = None

    t0 = t0 * direction
    t1 = t1 * direction

    # Store metadata
    stats = {
        "num_steps": final_state.num_steps,
        "num_accepted_steps": final_state.num_accepted_steps,
        "num_rejected_steps": final_state.num_rejected_steps,
        "max_steps": max_steps,
    }
    result = final_state.result
    sol = Solution(
        t0=t0,
        t1=t1,
        ts=ts,
        ys=ys,
        interpolation=interpolation,
        stats=stats,
        result=result,
        solver_state=solver_state,
        controller_state=controller_state,
        made_jump=made_jump,
    )

    if throw:
        sol = result.error_if(sol, jnp.invert(is_okay(result)))
    return sol
