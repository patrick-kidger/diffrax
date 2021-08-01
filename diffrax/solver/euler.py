from typing import Callable, Tuple

from ..brownian import AbstractBrownianPath
from ..custom_types import Array, DenseInfo, PyTree, Scalar, SquashTreeDef
from ..local_interpolation import LocalLinearInterpolation
from ..term import AbstractTerm, ControlTerm, ODETerm
from .base import AbstractSolver


_SolverState = None


class Euler(AbstractSolver):
    terms: Tuple[AbstractTerm]

    order = 1
    interpolation_cls = LocalLinearInterpolation

    def step(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
        y_treedef: SquashTreeDef,
        solver_state: _SolverState,
    ) -> Tuple[Array["state"], None, DenseInfo, _SolverState]:  # noqa: F821
        y1 = y0
        for term in self.terms:
            control_, control_treedef = term.contr_(t0, t1)
            y1 = y1 + term.vf_prod_(y_treedef, control_treedef, t0, y0, args, control_)
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None

    def func_for_init(self, t: Scalar, y_: Array["state"], args: PyTree,  # noqa: F821
                      y_treedef: SquashTreeDef) -> Array["state"]:  # noqa: F821
        vf = 0
        for term in self.terms:
            vf = vf + term.func_for_init(t, y_, args, y_treedef)
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
