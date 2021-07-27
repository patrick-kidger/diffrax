import jax
import jax.lax as lax
import jax.numpy as jnp
from typing import Optional, Type

from .autojit import autojit
from .custom_types import Array, PyTree, Scalar, SquashTreeDef
from .interpolation import AbstractInterpolation
from .misc import stack_pytrees
from .saveat import SaveAt
from .solution import RESULTS, Solution
from .solver import AbstractSolver, AbstractSolverState
from .step_size_controller import AbstractStepSizeController, ConstantStepSize
from .tree import tree_squash, tree_unsquash


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
    requested_state: frozenset,
):
    y_candidate, solver_state_candidate = solver.step(y_treedef, tprev, tnext, y, args, solver_state, requested_state)
    (keep_step, tprev, tnext, controller_state_candidate) = stepsize_controller.adapt_step_size(
        tprev, tnext, y, y_candidate, solver_state, solver_state_candidate, solver.order, controller_state
    )
    tprev = lax.stop_gradient(tprev)
    tnext = lax.stop_gradient(tnext)
    tnext = jnp.minimum(tnext, t1)
    not_done = tprev < t1
    keep_step = keep_step & not_done
    keep = lambda a, b: jnp.where(keep_step, a, b)
    y = keep(y_candidate, y)
    solver_state = jax.tree_map(keep, solver_state_candidate, solver_state)
    controller_state = jax.tree_map(keep, controller_state_candidate, controller_state)
    return tprev, tnext, y, solver_state, controller_state, keep_step, not_done


def diffeqint(
    solver: AbstractSolver,
    t0: Scalar,
    t1: Scalar,
    y0: PyTree,
    dt0: Optional[Scalar],
    args: Optional[PyTree] = None,
    *,
    stepsize_controller: AbstractStepSizeController = ConstantStepSize(),
    interpolation: Optional[Type[AbstractInterpolation]] = None,
    solver_state: Optional[AbstractSolverState] = None,
    controller_state: Optional[PyTree] = None,
    saveat: SaveAt = SaveAt(t1=True),
    jit: bool = True,
    max_steps: Scalar = 2**31 - 1
) -> Solution:

    if interpolation is None:
        interpolation = solver.recommended_interpolation

    requested_state = interpolation.requested_state | stepsize_controller.requested_state
    missing_state = requested_state - solver.available_state
    if missing_state:
        missing_state = set(missing_state)
        raise ValueError(
            f"This combination of interpolation={interpolation}, stepsize_controller={type(stepsize_controller)}, "
            f"solver={type(solver)} is not valid. This interpolation and stepsize_controller needs {missing_state} "
            "which this solver does not provide."
        )

    y, y_treedef = tree_squash(y0)

    tprev = t0
    if controller_state is None:
        (tnext,
         controller_state) = stepsize_controller.init(solver.func_for_init, y_treedef, t0, y, args, dt0, solver.order)
    else:
        assert dt0 is not None
        tnext = t0 + dt0

    if solver_state is None:
        solver_state = solver.init(y_treedef, t0, tnext, y, args, requested_state)

    ts = []
    ys = []
    controller_states = []
    solver_states = []
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
        # TODO: understand warnings being throw about donated argnums not being used.
        # step_maybe_jit = autojit(_step, donate_argnums=(0, 1, 2, 3, 4))
        step_maybe_jit = autojit(_step)
    else:
        step_maybe_jit = _step

    not_done = tprev < t1
    num_steps = 0
    # We don't use lax.while_loop as it doesn't support reverse-mode autodiff
    while jnp.any(not_done) and num_steps < max_steps:
        num_steps = num_steps + 1
        (tprev, tnext, y, solver_state, controller_state, keep_step, not_done) = step_maybe_jit(
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
            requested_state
        )
        if saveat.steps & jnp.any(keep_step):
            ts.append(tprev)
            ys.append(tree_unsquash(y_treedef, y))
            if saveat.controller_state:
                controller_states.append(controller_state)
            if saveat.solver_state:
                solver_states.append(solver_state)

    if num_steps >= max_steps:
        result = RESULTS.max_steps_reached
    else:
        result = RESULTS.successful

    # TODO: interpolate into ts and ys

    if saveat.t1:
        ts.append(tprev)
        ys.append(tree_unsquash(y_treedef, y))
        if saveat.controller_state:
            controller_states.append(controller_state)
        if saveat.solver_state:
            solver_states.append(solver_state)

    ts = jnp.stack(ts)
    ys = stack_pytrees(ys)
    if saveat.controller_state:
        controller_states = stack_pytrees(controller_states)
    else:
        controller_states = None
    if saveat.solver_state:
        solver_states = stack_pytrees(solver_states)
    else:
        solver_states = None

    interpolation = interpolation(ts=ts, ys=ys)
    return Solution(
        ts=ts,
        ys=ys,
        controller_states=controller_states,
        solver_states=solver_states,
        interpolation=interpolation,
        result=result
    )
