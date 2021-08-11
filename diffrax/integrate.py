from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from .custom_types import Array, PyTree, Scalar, SquashTreeDef
from .global_interpolation import DenseInterpolation
from .misc import stack_pytrees, tree_squash, tree_unsquash, vmap_all, vmap_any
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
    y_treedef: SquashTreeDef,
    t1: Scalar,
    args: PyTree,
):

    (y_candidate, y_error, dense_info, solver_state_candidate) = solver.step(
        tprev, tnext, y, args, y_treedef, solver_state
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
        y_treedef,
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


# By default we exclude bools from being JIT'd,as they can be used to indicate
# flags for special behaviour.
def _filter_fn(elem):
    return eqx.is_array_like(elem) and not isinstance(elem, bool)


_jit_step = eqx.jitf(_step, filter_fn=_filter_fn)
# TODO: understand warnings being throw about donated argnums not being used.
#  eqx.jitf(_step, donate_argnums=(0, 1, 2, 3, 4), filter_fn=eqx.is_array_like)


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

    # TODO: support reverse integration
    if vmap_any(t0 > t1):
        raise ValueError("Must have t0 <= t1")
    if dt0 is not None and dt0 <= 0:
        raise ValueError("Must have dt0 > 0")
    if saveat.t:
        if vmap_any((saveat.t >= t1) | (saveat.t <= t0)):
            raise ValueError("saveat.t must lie strictly between t0 and t1.")
        tinterp_index = 0

    y, y_treedef = tree_squash(y0)

    tprev = t0
    if controller_state is None:
        (tnext, controller_state) = stepsize_controller.init(
            t0, y, dt0, args, y_treedef, solver.order, solver.func_for_init
        )
    else:
        assert dt0 is not None
        tnext = t0 + dt0
    tnext = jnp.minimum(tnext, t1)

    if solver_state is None:
        solver_state = solver.init(t0, tnext, y, args, y_treedef)

    ts = []
    ys = []
    controller_states = []
    solver_states = []
    dense_ts = []
    dense_infos = []
    if saveat.t0:
        ts.append(t0)
        ys.append(y0)
        if saveat.controller_state:
            controller_states.append(controller_state)
        if saveat.solver_state:
            solver_states.append(solver_state)
    else:
        del y0, dt0

    if jit:
        step_maybe_jit = _jit_step
    else:
        step_maybe_jit = _step

    num_steps = 0
    result = jnp.full_like(t1, RESULTS.successful)
    # We don't use lax.while_loop as it doesn't support reverse-mode autodiff
    while vmap_any(tprev < t1) and num_steps < max_steps:
        # We have to keep track of several different times -- tprev, tnext,
        # tprev_before, tnext_before, t1.
        #
        # tprev and tnext are the start and end points of the interval we're about to
        # step over. tprev_before and tnext_before are tprev and tnext from the
        # previous iteration. t1 is the terminal time.
        #
        # Note that t_after != tprev in general. If we have discontinuities in the
        # vector field then it may be the case that tprev = nextafter(t_after). (They
        # should never differ by more than this though.)
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
            y_treedef,
            t1,
            args,
        )

        if vmap_any(made_step):
            if saveat.t is not None:
                tinterp = saveat.t[tinterp_index]
                interp_cond = tinterp < tprev
                while vmap_any(interp_cond):
                    interpolator = solver.interpolation_cls(
                        t0=tprev_before, t1=tnext_before, **dense_info
                    )
                    # Note that interp_cond will only be True if we've made a step in
                    # that batch element. If the step is rejected then tprev == tprev
                    # before, and tinterp >= tprev_before, due to the previous
                    # iteration through this loop on the previous step.
                    tinterp = jnp.where(interp_cond, tinterp, tnext_before)
                    yinterp = jnp.where(interp_cond, interpolator.evaluate(tinterp), y)
                    ts.append(tinterp)
                    ys.append(yinterp)
                    if saveat.controller_state:
                        controller_states.append(None)
                    if saveat.solver_state:
                        solver_states.append(None)
                    tinterp_index = tinterp_index + jnp.where(interp_cond, 1, 0)
                    tinterp = saveat.t[tinterp_index]
                    interp_cond = tinterp < tprev
            if saveat.steps:
                ts.append(tnext_before)
                ys.append(tree_unsquash(y_treedef, y))
                if saveat.controller_state:
                    controller_states.append(controller_state)
                if saveat.solver_state:
                    solver_states.append(solver_state)
            if saveat.dense:
                dense_ts.append(tprev_before)
                dense_infos.append(dense_info)
        if vmap_any(stepsize_controller_result != RESULTS.successful):
            result = jnp.maximum(result, stepsize_controller_result)
            break

    if num_steps >= max_steps:
        result = jnp.where(tnext < t1, RESULTS.max_steps_reached, result)

    if throw and vmap_any(result != RESULTS.successful):
        raise RuntimeError(f"diffeqsolve did not succeed. Error: {RESULTS[result]}")

    if saveat.t1:
        ts.append(tprev)
        ys.append(tree_unsquash(y_treedef, y))
        if saveat.controller_state:
            controller_states.append(controller_state)
        if saveat.solver_state:
            solver_states.append(solver_state)

    if len(ts):
        ts = jnp.stack(ts)
    else:
        ts = None
    if len(ys):
        ys = stack_pytrees(ys)
    else:
        ys = None
    if not len(controller_states):
        controller_states = None
    if not len(solver_states):
        solver_states = None

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
            y_treedef=y_treedef,
        )
    else:
        interpolation = None
    return Solution(
        t0=t0,
        t1=t1,
        ts=ts,
        ys=ys,
        controller_states=controller_states,
        solver_states=solver_states,
        interpolation=interpolation,
        result=result,
    )
