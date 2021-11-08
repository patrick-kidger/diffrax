import functools as ft
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from .custom_types import Array, PyTree, Scalar
from .global_interpolation import DenseInterpolation
from .misc import ravel_pytree, stack_pytrees, unvmap
from .saveat import SaveAt
from .solution import RESULTS, Solution
from .solver import AbstractSolver
from .step_size_controller import AbstractStepSizeController, ConstantStepSize


# TODO: this is making one step too many.
@jax.jit
def _not_done(tprev, t1, result):
    return (tprev < t1) & (result == RESULTS.successful)


def _step(
    tprev: Scalar,
    tnext: Scalar,
    y: Array["state"],  # noqa: F821
    solver_state: PyTree,
    controller_state: PyTree,
    made_jump: Array[(), bool],
    solver: AbstractSolver,
    stepsize_controller: AbstractStepSizeController,
    t1: Scalar,
    args: PyTree,
    result: RESULTS,
):

    (
        y_candidate,
        y_error,
        dense_info,
        solver_state_candidate,
        solver_result,
    ) = solver.step(tprev, tnext, y, args, solver_state, made_jump)

    (
        keep_step,
        tprev_candidate,
        tnext_candidate,
        made_jump_candidate,
        controller_state_candidate,
        stepsize_controller_result,
    ) = stepsize_controller.adapt_step_size(
        tprev,
        tnext,
        y,
        y_candidate,
        args,
        y_error,
        solver.order,
        controller_state,
    )

    # The 1e-6 tolerance means that we don't end up with too-small intervals for dense
    # output, which then gives numerically unstable answers due to floating point
    # errors.
    tnext_candidate = jnp.where(tnext_candidate > t1 - 1e-6, t1, tnext_candidate)
    tprev_candidate = jnp.minimum(tprev_candidate, t1)

    # We have different update rules for
    # - y and solver_state
    # - tprev, tnext, and controller_state
    # The solution and solver state only update if we keep the step.
    # It's up to the stepsize_controller to update tprev, tnext, and controller_state.
    keep = lambda a, b: jnp.where(keep_step, a, b)
    y_candidate = keep(y_candidate, y)
    made_jump_candidate = keep(made_jump_candidate, made_jump)
    solver_state_candidate = jax.tree_map(keep, solver_state_candidate, solver_state)
    solver_result = keep(solver_result, RESULTS.successful)

    # Next: we need to consider the fact that one batch element may have finished
    # integrating even whilst other batch elements are still going. In this case we
    # just have the "done" batch elements just stay constant (in every respect: time,
    # solution, controller state etc.) whilst we wait.
    not_done = _not_done(tprev, t1, result)
    keep = lambda a, b: jnp.where(not_done, a, b)
    tprev = keep(tprev_candidate, tprev)
    tnext = keep(tnext_candidate, tnext)
    y = keep(y_candidate, y)
    made_jump = keep(made_jump_candidate, made_jump)
    solver_state = jax.tree_map(keep, solver_state_candidate, solver_state)
    controller_state = jax.tree_map(keep, controller_state_candidate, controller_state)
    solver_result = keep(solver_result, RESULTS.successful)
    stepsize_controller_result = keep(stepsize_controller_result, RESULTS.successful)
    # TODO: is this necessary? If we're making zero-length steps then we're not going
    # anywhere.

    # The one exception to the above discussion is dense_info.
    # This is difficult to apply the same logic to: unlike the other pieces, it isn't
    # initialised.
    # If we reject the first step, then there's no previous step we can roll back to.
    # (And we still have to store *something* as other batch elements may have been
    # accepted.)
    # Fortunately, it doesn't matter. dense_info is stored against a sequence of
    # timestamps, so any time a dense interpolation is evaluated, repeated times (i.e.
    # rejected or after-batch-element-is-done steps) are never found. When evaluating
    # at some time t, then the dense interpolation routines seek some i for which
    # t_{i-1} < t <= t_i or t_{i-1} <= t < t_i (depending on mode). In particular this
    # implies t_{i-1} != t_i, so repeated timestamps are never found. (And the
    # corresponding junk data never used.)

    made_step = keep_step & not_done
    return (
        tprev,
        tnext,
        y,
        solver_state,
        controller_state,
        made_step,
        made_jump,
        not_done,
        dense_info,
        solver_result,
        stepsize_controller_result,
    )


# TODO: support custom filter functions?
# TODO: support donate_argnums if on the GPU.
_jit_step = eqx.filter_jit(_step)


@jax.jit
def _jit_neq(a, b):
    return a != b


@jax.jit
def _jit_any(x):
    return jnp.any(x)


@jax.jit
def _pre_save_interp(tnext_before, tinterp_index, saveat_ts):
    tinterp = saveat_ts[jnp.minimum(tinterp_index, len(saveat_ts) - 1)]
    interp_cond = (tinterp <= tnext_before) & (tinterp_index < len(saveat_ts))
    return tinterp, interp_cond


@ft.partial(jax.jit, static_argnums=0)
def _save_interp(
    interpolation_cls,
    tprev_before,
    tnext_before,
    dense_info,
    tinterp,
    tinterp_index,
    saveat_ts,
    interp_cond,
):
    interpolator = interpolation_cls(t0=tprev_before, t1=tnext_before, **dense_info)
    yinterp = interpolator.evaluate(tinterp)
    tinterp_index = tinterp_index + jnp.where(interp_cond, 1, 0)
    tinterp = saveat_ts[jnp.minimum(tinterp_index, len(saveat_ts) - 1)]
    interp_cond = (tinterp <= tnext_before) & (tinterp_index < len(saveat_ts))
    return tinterp, yinterp, interp_cond, tinterp_index


@ft.partial(eqx.filter_jit, filter_spec=eqx.is_array)
def _compress_output_constant(ts, ys, direction, unravel_y):
    ts = jnp.stack(ts)
    ts = jnp.where(direction == 1, ts, -ts[::-1])
    ys = jnp.stack(ys)
    ys = jax.vmap(unravel_y)(ys)
    return ts, ys


@ft.partial(eqx.filter_jit, filter_spec=eqx.is_array)
def _compress_output_adaptive(
    ts, ys, out_indices, out_len, has_minus_one, direction, unravel_y
):
    out_indices = jnp.stack(out_indices)
    if has_minus_one:
        out_indices = jnp.unique(out_indices, size=out_len + 1)[1:]
    else:
        out_indices = jnp.unique(out_indices, size=out_len)
    ts = jnp.stack(ts)
    ts = jnp.where(direction == 1, ts, -ts[::-1])
    ys = jnp.stack(ys)
    ys = jax.vmap(unravel_y)(ys)
    return ts, ys, out_indices


def diffeqsolve(
    solver: AbstractSolver,
    t0: Scalar,
    t1: Scalar,
    y0: PyTree,
    dt0: Optional[Scalar],
    args: Optional[PyTree] = None,
    *,
    saveat: SaveAt = SaveAt(t1=True),
    stepsize_controller: AbstractStepSizeController = ConstantStepSize(),
    solver_state: Optional[PyTree] = None,
    controller_state: Optional[PyTree] = None,
    max_steps: Scalar = 2 ** 31 - 1,
    jit: bool = True,
    throw: bool = True,
) -> Solution:
    """Solves a differential equation.

    This function is the main entry point for solving all kinds of initial value
    problems, whether they're ODEs, SDEs, or CDEs.

    The differential equation is integrated from `t0` to `t1`.

    **Main arguments:**

    These are the arguments most commonly used day-to-day.

    - `solver`: The solver for the differential equation. The vector field of the
        differential equation is specified when instantiating the solver.
    - `t0`: The start of the region of integration.
    - `t1`: The end of the region of integration.
    - `y0`: The initial value. This can be any PyTree of JAX arrays. (Or types that
        can be coerced to JAX arrays, like Python floats.)
    - `dt0`: The step size to use for the first step. If using fixed step sizes then
        this will also be the step size for all other steps. (Except the last one,
        which may be slightly smaller and clipped to `t1`.) If set as `None` then the
        initial step size will be determined automatically if possible.
    - `args`: Any additional arguments to pass to the vector field.
    - `saveat`: What times to save the solution of the differential equation. Defaults
        to just the last time `t1`.
    - `stepsize_controller`: How to change the step size as the integration progresses.
        Defaults to using a fixed constant step size.

    **Other arguments:**

    These arguments are infrequently used, and for most purposes you shouldn't need to
    understand these.

    - `solver_state`: Some initial state for the solver. Can be useful when for example
        using a reversible solver to recompute a solution.
    - `controller_state`: Some initial state for the step size controller.
    - `max_steps`: The maximum number of steps to take before quitting the computation
        unconditionally. The `result` of the returned solution object will have a flag
        set to indicate that this happened. Can be useful to bound the total amount of
        work expended on problems of unknown complexity.
    - `jit`: Whether to `jax.jit` anything using the vector field. The default is to do
        so. Not doing so will typically be much slower, but allows for operations other
        than pure JAX in the vector field.
    - `throw`: Whether to raise an exception if the integration fails for any reason.
        If `False` then the returned solution object will have a `result` field
        indicating whether any failures occurred.
        Possible failures include hitting `max_steps`, or the problem becoming too
        stiff to integrate. (For most purposes these failures are unusual.) Note that
        when `jax.vmap`-ing a differential equation solve, this means that an exception
        will be raised if any batch element fails. You may prefer to set it to `False`
        and inspect the `result` field of the returned solution object, to determine
        which batch elements succeeded and which failed.

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

    !!! note
        It is possible to use this function directly on Python types that can be
        coerced to JAX arrys, e.g. `diffeqsolve(..., t0=0, t1=1, y0=1, dt0=0.1)`.
    """

    if dt0 is not None and _jit_any(unvmap((t1 - t0) * dt0 <= 0)):
        raise ValueError("Must have (t1 - t0) * dt0 > 0")

    # Normalise state: ravel PyTree state down to just a flat Array.
    # Normalise time: if t0 > t1 then flip things around.
    direction = jnp.where(t0 < t1, 1, -1)
    y, unravel_y = ravel_pytree(y0)
    solver = solver.wrap(t0, y0, args, direction)
    stepsize_controller = stepsize_controller.wrap(unravel_y, direction)
    t0 = t0 * direction
    t1 = t1 * direction
    if dt0 is not None:
        dt0 = dt0 * direction
    if saveat.ts is not None:
        saveat = eqx.tree_at(lambda s: s.ts, saveat, saveat.ts * direction)

    if saveat.ts is not None:
        if _jit_any(unvmap(saveat.ts[1:] < saveat.ts[:-1])):
            raise ValueError("saveat.ts must be increasing or decreasing.")
        if _jit_any(unvmap((saveat.ts > t1) | (saveat.ts < t0))):
            raise ValueError("saveat.ts must lie between t0 and t1.")
        tinterp_index = 0

    tprev = t0
    if controller_state is None:
        (tnext, controller_state) = stepsize_controller.init(t0, y, dt0, args, solver)
    else:
        assert dt0 is not None
        tnext = t0 + dt0
    tnext = jnp.minimum(tnext, t1)
    tnext_before = t0

    if solver_state is None:
        solver_state = solver.init(t0, tnext, y, args)

    out_indices = []
    ts = []
    ys = []
    dense_ts = []
    dense_infos = []
    if saveat.t0:
        out_indices.append(0)
        ts.append(t0)
        ys.append(y)
    del y0, dt0

    made_jump = jnp.full_like(t0, fill_value=False, dtype=bool)

    if jit:
        step_maybe_jit = _jit_step
    else:
        step_maybe_jit = _step

    num_steps = 0
    raw_num_steps = 0
    has_minus_one = False
    result = jnp.full_like(t1, RESULTS.successful)
    not_done = _not_done(tprev, t1, result)
    save_intermediate = (saveat.ts is not None) or saveat.dense
    # We don't use lax.while_loop as it doesn't support reverse-mode autodiff
    while _jit_any(unvmap(not_done)):
        # We have to keep track of several different times -- tprev, tnext,
        # tprev_before, tnext_before, t1.
        #
        # tprev and tnext are the start and end points of the interval we're about to
        # step over. tprev_before and tnext_before are tprev and tnext from the
        # previous iteration. t1 is the terminal time.
        #
        # Note that t_after != tprev in general. If we have discontinuities in the
        # vector field then it may be the case that
        # tprev = nextafter(nextafter(tnext_before)).
        # (They should never differ by more than this though.)
        #
        # In particular this means that y technically corresponds to the value of the
        # solution at tnext_before.
        num_steps = num_steps + not_done
        raw_num_steps = raw_num_steps + 1
        y_before = y
        tnext_before_before = tnext_before
        tprev_before = tprev
        tnext_before = tnext
        (
            tprev,
            tnext,
            y,
            solver_state,
            controller_state,
            made_step,
            made_jump,
            not_done,
            dense_info,
            solver_result,
            stepsize_controller_result,
        ) = step_maybe_jit(
            tprev,
            tnext,
            y,
            solver_state,
            controller_state,
            made_jump,
            solver,
            stepsize_controller,
            t1,
            args,
            result,
        )

        # save_intermediate=False offers a fast path that avoids JAX operations
        if save_intermediate and _jit_any(unvmap(made_step)):
            if saveat.ts is not None:
                tinterp, interp_cond = _pre_save_interp(
                    tnext_before, tinterp_index, saveat.ts
                )
                while _jit_any(unvmap(interp_cond)):
                    ts.append(tinterp)
                    out_indices.append(jnp.where(interp_cond, len(ys), -1))
                    has_minus_one = (
                        has_minus_one or _jit_any(unvmap(~interp_cond)).item()
                    )
                    tinterp, yinterp, interp_cond, tinterp_index = _save_interp(
                        solver.interpolation_cls,
                        tprev_before,
                        tnext_before,
                        dense_info,
                        tinterp,
                        tinterp_index,
                        saveat.ts,
                        interp_cond,
                    )
                    ys.append(yinterp)
            if saveat.dense:
                dense_ts.append(tprev_before)
                dense_infos.append(dense_info)
        if saveat.steps:
            out_indices.append(len(ys))
            ts.append(jnp.where(made_step, tnext_before, tnext_before_before))
            ys.append(jnp.where(made_step, y, y_before))

        should_break = False

        _controller_unsuccessful = _jit_neq(
            stepsize_controller_result, RESULTS.successful
        )
        if _jit_any(unvmap(_controller_unsuccessful)):
            cond = jnp.where(
                result == RESULTS.successful, _controller_unsuccessful, False
            )
            result = jnp.where(cond, stepsize_controller_result, result)
            should_break = throw

        _solver_unsuccessful = _jit_neq(solver_result, RESULTS.successful)
        if _jit_any(unvmap(_solver_unsuccessful)):
            cond = jnp.where(result == RESULTS.successful, _solver_unsuccessful, False)
            result = jnp.where(cond, solver_result, result)
            should_break = throw

        _nan_tnext = jnp.isnan(tnext)
        if _jit_any(unvmap(_nan_tnext)):
            cond = jnp.where(result == RESULTS.successful, _nan_tnext, False)
            result = jnp.where(cond, RESULTS.nan_time, result)
            should_break = throw

        if raw_num_steps >= max_steps:
            result = jnp.where(not_done, RESULTS.max_steps_reached, result)
            should_break = True

        if should_break:
            break

    if throw and _jit_any(unvmap(_jit_neq(result, RESULTS.successful))):
        error = RESULTS[jnp.max(unvmap(result)).item()]
        raise RuntimeError(error)

    # saveat.steps will include the final timepoint anyway
    if saveat.t1 and not saveat.steps:
        out_indices.append(len(ys))
        ts.append(tnext_before)
        ys.append(y)

    if saveat.dense:
        dense_ts.append(t1)
        dense_ts = jnp.stack(dense_ts)
        if not len(dense_infos):
            assert jnp.all(unvmap(t0 == t1))
            raise ValueError("Cannot save dense output when t0 == t1")
        dense_infos = stack_pytrees(dense_infos)
        interpolation = DenseInterpolation(
            ts=dense_ts,
            interpolation_cls=solver.interpolation_cls,
            infos=dense_infos,
            unravel_y=unravel_y,
            direction=direction,
        )
    else:
        interpolation = None

    stats = {"num_steps": num_steps, "num_observations": len(ts)}

    t0 = t0 * direction
    t1 = t1 * direction
    if len(ts):
        assert len(ys) == len(ts)
        assert len(out_indices) == len(ts)
        out_len = 0
        if saveat.t0:
            out_len = out_len + 1
        if saveat.ts is not None:
            out_len = out_len + len(saveat.ts)
        if saveat.t1 and not saveat.steps:
            out_len = out_len + 1
        if saveat.steps:
            out_len = out_len + raw_num_steps

        if len(ts) == out_len:
            # Fast path for constant step size controllers. (And lucky adaptive ones.)
            ts, ys = _compress_output_constant(ts, ys, direction, unravel_y)
        else:
            # We pad ts, ys, out_indices out to a multiple of pad_to.
            # This means that when using adaptive controllers, we'll only re-JIT
            # _compress_output_adaptive if the number of steps rounds to a different
            # multiple of pad_to.
            # Note the use of _t=ts[-1], _y=ys[-1], _i=out_indices[-1], and not any
            # other dummy placeholder. These will have the correct Tracers wrapping
            # them.
            if len(ts) < 11:
                pad_to = 10
            else:
                pad_to = 20
            rem = len(ts) % pad_to
            if rem != 0:
                padding = pad_to - rem
                _t = ts[-1]
                _y = ys[-1]
                _i = out_indices[-1]
                ts.extend([_t for _ in range(padding)])
                ys.extend([_y for _ in range(padding)])
                out_indices.extend([_i for _ in range(padding)])
            ts, ys, out_indices = _compress_output_adaptive(
                ts, ys, out_indices, out_len, has_minus_one, direction, unravel_y
            )
            # These should _not_ be folded into the above _compress_output_adaptive
            # function. Jitting these lines together with the above results in very
            # long compile times (e.g. 70 seconds instead of 20 seconds) for
            # backpropagating through diffeqsolve.
            #
            # I have no idea why.
            ts = ts[out_indices]
            ys = ys[out_indices]
    else:
        assert not len(ys)
        assert not len(out_indices)
        ts = None
        ys = None
    if not saveat.controller_state:
        controller_state = None
    if not saveat.solver_state:
        solver_state = None

    return Solution(
        t0=t0,
        t1=t1,
        ts=ts,
        ys=ys,
        controller_state=controller_state,
        solver_state=solver_state,
        interpolation=interpolation,
        stats=stats,
        result=result,
    )
