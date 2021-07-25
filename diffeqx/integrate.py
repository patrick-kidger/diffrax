import jax
import jax.numpy as jnp
from typing import Optional, Type

from .custom_types import PyTree, Scalar
from .interpolation import AbstractInterpolation
from .saveat import SaveAt
from .solution import Solution
from .solver import AbstractSolver
from .step_size_controller import AbstractStepSizeController, ConstantStepSize
from .tree import tree_squash


def diffeqint(solver: AbstractSolver, t0: Scalar, t1: Scalar, y0: PyTree, dt0: Optional[Scalar], stepsize_controller: AbstractStepSizeController = ConstantStepSize(), interpolation: Optional[Type[AbstractInterpolation]] = None, solver_state: Optional[PyTree] = None, controller_state: Optional[PyTree] = None, saveat: SaveAt = SaveAt(t1=True)) -> Solution:

    y = y0
    if not saveat.t0:
        y0 = None

    if interpolation is None:
        interpolation = solver.recommended_interpolation

    tprev = t0
    if controller_state is None:
        tnext, controller_state = stepsize_controller.init(t0, dt0)
    else:
        assert dt0 is not None
        tnext = t0 + dt0

    if solver_state is None:
        solver_state = solver.init(t0, y0)

    y, treedef = tree_squash(y)

    not_done = tprev < t1
    while not_done.any():
        y_candidate, solver_state_candidate = solver.step(treedef, tprev, tnext, y, solver_state)
        keep_step, tprev, tnext, controller_state_candidate = stepsize_controller.adapt_step_size(tprev, tnext, y, y_candidate, solver_state, solver_state_candidate, controller_state)
        tnext = jnp.minimum(tnext, t1)
        not_done = tprev < t1
        keep_step = keep_step & not_done
        keep = lambda a, b: jnp.where(keep_step, a, b)
        y = keep(y_candidate, y)
        solver_state = jax.tree_map(keep, solver_state_candidate, solver_state)
        controller_state = jax.tree_map(keep, controller_state_candidate, controller_state)
    # TODO: record into ts and ys
        
    if not saveat.controller_state:
        controller_state = None
    if not saveat.solver_state:
        solver_state = None
    interpolation = interpolation(ts=ts, ys=ys)
    return Solution(ts=ts, ys=ys, controller_state=controller_state, solver_state=solver_state, interpolation=interpolation)

