import functools as ft
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from .custom_types import Array, PyTree, Scalar
from .global_interpolation import DenseInterpolation
from .misc import ravel_pytree, stack_pytrees, vmap_all, vmap_any
from .saveat import SaveAt
from .solution import RESULTS, Solution
from .solver import AbstractSolver
from .step_size_controller import AbstractStepSizeController, ConstantStepSize


def _step(
    tprev: Scalar,
    tnext: Scalar,
    y: Array["state"],  # noqa: F821
    solver_state: PyTree,
    controller_state: PyTree,
    solver: AbstractSolver,
    stepsize_controller: AbstractStepSizeController,
    t1: Scalar,
    args: PyTree,
):

    (y_candidate, y_error, dense_info, solver_state_candidate) = solver.step(
        tprev, tnext, y, args, solver_state
    )

    (
        keep_step,
        tprev_candidate,
        tnext_candidate,
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
    # We have different update rules for
    # - the solution y and the controller state
    # - time
    # The solution and controller state only update if we keep the step.
    # However the time step updates unconditionally. (tprev should probably remain the
    # the same, but tnext will be over a small step if the step is rejected)
    # This means we let stepsize_controller handle the updates to tprev and tnext,
    # and then handle the rest of it here.

    tnext_candidate = jnp.where(tnext_candidate > t1 - 1e-6, t1, tnext_candidate)
    tprev_candidate = jnp.minimum(tprev_candidate, t1)
    keep = lambda a, b: jnp.where(keep_step, a, b)
    y_candidate = keep(y_candidate, y)
    solver_state_candidate = jax.tree_map(keep, solver_state_candidate, solver_state)
    controller_state_candidate = jax.tree_map(
        keep, controller_state_candidate, controller_state
    )

    # Next: we need to consider the fact that one batch element may have finished
    # integrating even whilst other batch elements are still going. In this case we
    # just have the "done" batch elements just stay constant (in every respect: time,
    # solution, controller state etc.) whilst we wait.

    not_done = tprev < t1
    keep = lambda a, b: jnp.where(not_done, a, b)
    tprev = keep(tprev_candidate, tprev)
    tnext = keep(tnext_candidate, tnext)
    y = keep(y_candidate, y)
    solver_state = jax.tree_map(keep, solver_state_candidate, solver_state)
    controller_state = jax.tree_map(keep, controller_state_candidate, controller_state)
    stepsize_controller_result = keep(stepsize_controller_result, RESULTS.successful)

    # The one exception to the above discussion is dense_info.
    # This is difficult to apply the same logic to: unlike the others pieces, it isn't
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
        dense_info,
        stepsize_controller_result,
    )


# TODO: support custom filter functions?
# TODO: support donate_argnums if on the GPU.
_jit_step = eqx.jitf(_step, filter_fn=eqx.is_nonboolean_array_like)


@jax.jit
def _lt(a, b):
    return a < b


@jax.jit
def _neq(a, b):
    return a != b


@jax.jit
def _pre_save_interp(tnext_before, tinterp_index, saveat_t):
    tinterp = saveat_t[jnp.minimum(tinterp_index, len(saveat_t) - 1)]
    interp_cond = (tinterp <= tnext_before) & (tinterp_index < len(saveat_t))
    return tinterp, interp_cond


@ft.partial(jax.jit, static_argnums=0)
def _save_interp(
    interpolation_cls,
    tprev_before,
    tnext_before,
    dense_info,
    tinterp,
    tinterp_index,
    saveat_t,
    interp_cond,
):
    interpolator = interpolation_cls(t0=tprev_before, t1=tnext_before, **dense_info)
    yinterp = interpolator.evaluate(tinterp)
    tinterp_index = tinterp_index + jnp.where(interp_cond, 1, 0)
    tinterp = saveat_t[jnp.minimum(tinterp_index, len(saveat_t) - 1)]
    interp_cond = (tinterp <= tnext_before) & (tinterp_index < len(saveat_t))
    return tinterp, yinterp, interp_cond, tinterp_index


@ft.partial(eqx.jitf, static_argnums=4, filter_fn=eqx.is_nonboolean_array_like)
def _compress_output(ts, ys, out_indices, direction, out_len, unravel_y):
    out_indices = jnp.stack(out_indices)
    out_indices = jnp.unique(out_indices, size=out_len)
    ts = jnp.stack(ts)
    ts = ts[out_indices]
    ts = jnp.where(direction == 1, ts, -ts[::-1])
    ys = jnp.stack(ys)
    ys = ys[out_indices]
    ys = jax.vmap(unravel_y)(ys)
    return ts, ys


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

    if dt0 is not None and (t1 - t0) * dt0 <= 0:
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
    if saveat.t is not None:
        saveat = eqx.tree_at(lambda s: s.t, saveat, saveat.t * direction)

    if saveat.t is not None:
        if vmap_any(saveat.t[1:] < saveat.t[:-1]):
            raise ValueError("saveat.t must be strictly increasing or decreasing.")
        if vmap_any((saveat.t > t1) | (saveat.t < t0)):
            raise ValueError("saveat.t must lie between t0 and t1.")
        tinterp_index = 0

    tprev = t0
    if controller_state is None:
        (tnext, controller_state) = stepsize_controller.init(
            t0, y, dt0, args, solver.order, solver.func_for_init
        )
    else:
        assert dt0 is not None
        tnext = t0 + dt0
    tnext = jnp.minimum(tnext, t1)

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
        ys.append(y0)
    del y0, dt0

    if jit:
        step_maybe_jit = _jit_step
    else:
        step_maybe_jit = _step

    num_steps = 0
    result = jnp.full_like(t1, RESULTS.successful)
    # We don't use lax.while_loop as it doesn't support reverse-mode autodiff
    while vmap_any(_lt(tprev, t1)) and num_steps < max_steps:
        # We have to keep track of several different times -- tprev, tnext,
        # tprev_before, tnext_before, t1.
        #
        # tprev and tnext are the start and end points of the interval we're about to
        # step over. tprev_before and tnext_before are tprev and tnext from the
        # previous iteration. t1 is the terminal time.
        #
        # Note that t_after != tprev in general. If we have discontinuities in the
        # vector field then it may be the case that tprev = nextafter(tnext_before).
        # (They should never differ by more than this though.)
        #
        # In particular this means that y technically corresponds to the value of the
        # solution at tnext_before.
        num_steps = num_steps + 1
        tprev_before = tprev
        tnext_before = tnext
        (
            tprev,
            tnext,
            y,
            solver_state,
            controller_state,
            made_step,
            dense_info,
            stepsize_controller_result,
        ) = step_maybe_jit(
            tprev,
            tnext,
            y,
            solver_state,
            controller_state,
            solver,
            stepsize_controller,
            t1,
            args,
        )

        if vmap_any(made_step):
            if saveat.t is not None:
                tinterp, interp_cond = _pre_save_interp(
                    tnext_before, tinterp_index, saveat.t
                )
                while vmap_any(interp_cond):
                    ts.append(tinterp)
                    tinterp, yinterp, interp_cond, tinterp_index = _save_interp(
                        solver.interpolation_cls,
                        tprev_before,
                        tnext_before,
                        dense_info,
                        tinterp,
                        tinterp_index,
                        saveat.t,
                        interp_cond,
                    )
                    out_indices.append(jnp.where(interp_cond, len(ys), 0))
                    ys.append(yinterp)
            if saveat.steps:
                out_indices.append(len(ys))
                ts.append(tnext_before)
                ys.append(y)
            if saveat.dense:
                dense_ts.append(tprev_before)
                dense_infos.append(dense_info)
        if vmap_any(_neq(stepsize_controller_result, RESULTS.successful)):
            result = jnp.maximum(result, stepsize_controller_result)
            break

    if num_steps >= max_steps:
        result = jnp.where(_lt(tnext, t1), RESULTS.max_steps_reached, result)

    if throw and vmap_any(_neq(result, RESULTS.successful)):
        raise RuntimeError(f"diffeqsolve did not succeed. Error: {RESULTS[result]}")

    # saveat.steps will include the final timepoint anyway
    if saveat.t1 and not saveat.steps:
        out_indices.append(len(ys))
        ts.append(tprev)
        ys.append(y)

    if saveat.dense:
        dense_ts.append(t1)
        dense_ts = jnp.stack(dense_ts)
        if not len(dense_infos):
            assert vmap_all(t0 == t1)
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

    t0 = t0 * direction
    t1 = t1 * direction
    if len(ts):
        assert len(ys)
        out_len = 0
        if saveat.t0:
            out_len = out_len + 1
        if saveat.t is not None:
            out_len = out_len + len(saveat.t)
        if saveat.t1 and not saveat.steps:
            out_len = out_len + 1
        if saveat.steps:
            out_len = out_len + num_steps
        ts, ys = _compress_output(ts, ys, out_indices, direction, out_len, unravel_y)
    else:
        assert not len(ys)
        ts = None
        ys = None
    if not saveat.controller_state:
        controller_state = None
    if not saveat.solver_state:
        solver_state = None

    stats = {"num_steps": num_steps}

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
