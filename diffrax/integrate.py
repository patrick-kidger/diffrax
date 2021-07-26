from typing import Optional, Type

import jax
import jax.numpy as jnp

from .custom_types import PyTree, Scalar
from .interpolation import AbstractInterpolation
from .misc import stack_pytrees
from .saveat import SaveAt
from .solution import Solution
from .solver import AbstractSolver
from .step_size_controller import AbstractStepSizeController, ConstantStepSize
from .tree import tree_squash, tree_unsquash


def diffeqint(
    solver: AbstractSolver,
    t0: Scalar,
    t1: Scalar,
    y0: PyTree,
    dt0: Optional[Scalar],
    args: Optional[PyTree] = None,
    stepsize_controller: AbstractStepSizeController = ConstantStepSize(),
    interpolation: Optional[Type[AbstractInterpolation]] = None,
    solver_state: Optional[PyTree] = None,
    controller_state: Optional[PyTree] = None,
    saveat: SaveAt = SaveAt(t1=True)
) -> Solution:

    if interpolation is None:
        interpolation = solver.recommended_interpolation

    tprev = t0
    if controller_state is None:
        tnext, controller_state = stepsize_controller.init(t0, dt0)
    else:
        assert dt0 is not None
        tnext = t0 + dt0

    y, y_treedef = tree_squash(y0)

    if solver_state is None:
        solver_state = solver.init(t0, y)

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

    # We don't use lax.while_loop as it doesn't support reverse-mode autodiff
    # variable step size solvers have a variable-size computation graph so they're
    # never going to be jit-able anyway.
    not_done = tprev < t1
    while jnp.any(not_done):
        y_candidate, solver_state_candidate = solver.step(y_treedef, tprev, tnext, y, args, solver_state)
        (keep_step, tprev, tnext, controller_state_candidate) = stepsize_controller.adapt_step_size(
            tprev, tnext, y, y_candidate, solver_state, solver_state_candidate, controller_state
        )
        tnext = jnp.minimum(tnext, t1)
        not_done = tprev < t1
        keep_step = keep_step & not_done
        keep = lambda a, b: jnp.where(keep_step, a, b)
        y = keep(y_candidate, y)
        solver_state = jax.tree_map(keep, solver_state_candidate, solver_state)
        controller_state = jax.tree_map(keep, controller_state_candidate, controller_state)
        if saveat.steps & jnp.any(keep_step):
            ts.append(tprev)
            ys.append(tree_unsquash(y_treedef, y))
            if saveat.controller_state:
                controller_states.append(controller_state)
            if saveat.solver_state:
                solver_states.append(solver_state)

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
        ts=ts, ys=ys, controller_states=controller_states, solver_states=solver_states, interpolation=interpolation
    )
