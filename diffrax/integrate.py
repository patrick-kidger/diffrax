import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Optional

from .custom_types import Array, PyTree, Scalar, SquashTreeDef
from .interpolation import DenseInterpolation
from .misc import stack_pytrees, tree_squash, tree_unsquash, vmap_all, vmap_any, vmap_max
from .saveat import SaveAt
from .solution import RESULTS, Solution
from .solver import AbstractSolver, AbstractSolverState
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

    (y_candidate, y_error, dense_info,
     solver_state_candidate) = solver.step(tprev, tnext, y, args, y_treedef, solver_state)

    (keep_step, tprev_candidate, tnext_candidate, controller_state_candidate,
     stepsize_controller_result) = stepsize_controller.adapt_step_size(
         tprev, tnext, y, y_candidate, args, y_error, y_treedef, solver.order, controller_state
     )

    tnext_candidate = jnp.minimum(tnext_candidate, t1)
    not_done = tnext < t1
    keep_step = keep_step & not_done
    keep = lambda a, b: jnp.where(keep_step, a, b)
    y = keep(y_candidate, y)
    tprev = keep(tprev_candidate, tprev)
    tnext = keep(tnext_candidate, tnext)
    solver_state = jax.tree_map(keep, solver_state_candidate, solver_state)
    controller_state = jax.tree_map(keep, controller_state_candidate, controller_state)

    return tprev, tnext, y, solver_state, controller_state, keep_step, not_done, dense_info, stepsize_controller_result


_jit_step = eqx.jitf(_step, filter_fn=eqx.is_array_like)
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
    solver_state: Optional[AbstractSolverState] = None,
    controller_state: Optional[PyTree] = None,
    max_steps: Scalar = 2**31 - 1,
    jit: bool = True,
    throw: bool = True,
) -> Solution:

    y, y_treedef = tree_squash(y0)

    tprev = t0
    if controller_state is None:
        (tnext,
         controller_state) = stepsize_controller.init(t0, y, dt0, args, y_treedef, solver.order, solver.func_for_init)
    else:
        assert dt0 is not None
        tnext = t0 + dt0

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
        del t0, y0, dt0

    if jit:
        step_maybe_jit = _jit_step
    else:
        step_maybe_jit = _step

    not_done = tprev < t1
    num_steps = 0
    result = jnp.full_like(t0, RESULTS.successful)
    if saveat.t:
        if not vmap_all((saveat.t < t1) & (saveat.t > t0)):
            raise ValueError("saveat.t must lie between t0 and t1.")
        tinterp_index = 0
    # We don't use lax.while_loop as it doesn't support reverse-mode autodiff
    while vmap_any(not_done) and num_steps < max_steps:
        # We have to keep track of several different times -- tprev, tnext, tprev_before, tnext_before, t1.
        #
        # tprev and tnext are the start and end points of the interval we're about to step over.
        # tprev_before and tnext_before are tprev and tnext from the previous iteration.
        # t1 is the terminal time.
        #
        # Note that t_after != tprev in general. If we have discontinuities in the vector field then
        # it may be the case that tprev = nextafter(t_after). (They should never differe by more than
        # this though.)
        #
        # In particular this means that y technically corresponds to the value of the solution at tnext_before.
        num_steps = num_steps + 1
        tprev_before = tprev
        tnext_before = tnext
        (tprev, tnext, y, solver_state, controller_state, keep_step, not_done, dense_info,
         stepsize_controller_result) = step_maybe_jit(
             tprev, tnext, y, solver_state, controller_state, solver, stepsize_controller, y_treedef, t1, args,
         )

        if vmap_any(keep_step):
            if saveat.t is not None:
                tinterp = saveat.t[tinterp_index]
                interp_cond = tinterp < tprev
                while vmap_any(interp_cond):
                    interpolator = solver.interpolation_cls(t0=tprev_before, t1=tnext_before, **dense_info)
                    # Note that interp_cond will only be True if we've made a step in that batch element.
                    # If the step is rejected then tprev == tprev_before, and tinterp >= tprev_before, due
                    # to the previous iteration through this loop on the previous step.
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
        if vmap_any(stepsize_controller_result > 0):
            result = vmap_max(stepsize_controller_result).item()
            break

    if num_steps >= max_steps:
        result = result.at[not_done].set(RESULTS.max_steps_reached)

    if throw and vmap_any(result != RESULTS.successful):
        raise RuntimeError(f"diffeqsolve did not succeed. Error: {RESULTS[result]}")

    if saveat.t1:
        ts.append(tprev)
        ys.append(tree_unsquash(y_treedef, y))
        if saveat.controller_state:
            controller_states.append(controller_state)
        if saveat.solver_state:
            solver_states.append(solver_state)

    ts = jnp.stack(ts)
    ys = stack_pytrees(ys)
    if not saveat.controller_state:
        assert len(controller_states) == 0
        controller_states = None
    if not saveat.solver_state:
        assert len(solver_states) == 0
        solver_states = None

    if saveat.dense:
        dense_ts.append(t1)
        dense_ts = jnp.stack(dense_ts)
        dense_infos = stack_pytrees(dense_infos)
        interpolation = DenseInterpolation(interpolation_cls=solver.interpolation_cls, ts=dense_ts, infos=dense_infos)
    else:
        interpolation = None
    return Solution(
        ts=ts,
        ys=ys,
        controller_states=controller_states,
        solver_states=solver_states,
        interpolation=interpolation,
        result=result
    )
