from typing import Callable, Tuple

from ..brownian import AbstractBrownianPath
from ..custom_types import Array, PyTree, Scalar, SquashTreeDef
from ..interpolation import LinearInterpolation
from ..term import AbstractTerm, ControlTerm, ODETerm
from .base import AbstractSolver, EmptySolverState


# This solver has two modes of operation. In both cases it computes the same thing,
# it just does them in a slightly different order.
# By default (when `"f1" not in requested_state`) it's just the Euler solver the way you'd usually write it down.
# When `"f1" in requested_state`, then after each step it evaluates the vector field at the end point, and stores
# the result. Then on the next step it uses the cached result, and evaluates at the end of the step it makes after
# that, and so on.
# This is useful when some other part of the ecosystem wants to use the end-of-interval evaluation for something.
# For example the Adam step size controller uses it to scale the learning rate.
class Euler(AbstractSolver):
    terms: tuple[AbstractTerm]

    order = 1
    recommended_interpolation = LinearInterpolation

    def init(
        self,
        y_treedef: SquashTreeDef,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
        requested_state: frozenset,
    ) -> EmptySolverState:
        return EmptySolverState(extras={})

    def step(
        self,
        y_treedef: SquashTreeDef,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
        solver_state: EmptySolverState,
        requested_state: frozenset,
    ) -> Tuple[Array["state"], EmptySolverState]:  # noqa: F821
        y1 = y0
        for term in self.terms:
            control_, control_treedef = term.contr_(t0, t1)
            y1 = y1 + term.vf_prod_(y_treedef, control_treedef, t0, y0, args, control_)
        return y1, solver_state

    def func(self, y_treedef: SquashTreeDef, t: Scalar, y_: Array["state"],  # noqa: F821
             args: PyTree) -> Array["state"]:  # noqa: F821
        vf = 0
        for term in self.terms:
            vf = vf + term.func(y_treedef, t, y_, args)
        return vf


def euler(vector_field: Callable[[Scalar, PyTree, PyTree], PyTree], **kwargs):
    return Euler(terms=(ODETerm(vector_field=vector_field),), **kwargs)


def euler_maruyama(
    drift: Callable[[Scalar, PyTree, PyTree], PyTree],
    diffusion: Callable[[Scalar, PyTree, PyTree], PyTree],
    bm: AbstractBrownianPath,
    **kwargs
):
    return Euler(terms=(ODETerm(vector_field=drift), ControlTerm(vector_field=diffusion, control=bm)), **kwargs)
