from typing import Callable, Tuple

from ..brownian import AbstractBrownianPath
from ..custom_types import Array, PyTree, Scalar, SquashTreeDef
from ..interpolation import LinearInterpolation
from ..term import AbstractTerm, ControlTerm, ODETerm
from ..tree import tree_dataclass
from .base import AbstractSolver

# Autojit is the secret sauce here.
# It means that we can pass in `self` to the jitted function, and as this is a tree_dataclass, it is treated as a
# PyTree and not recompiled against if another instance with the same structure is used later.


@tree_dataclass
class Euler(AbstractSolver):
    terms: tuple[AbstractTerm]

    order = 1
    recommended_interpolation = LinearInterpolation

    def step(
        self,
        y_treedef: SquashTreeDef,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
        solver_state: None,
        requested_state: frozenset,
    ) -> Tuple[Array["state"], None]:  # noqa: F821
        del requested_state
        y1 = y0
        for term in self.terms:
            control_, control_treedef = term.contr_(t0, t1)
            y1 = y1 + term.vf_prod_(y_treedef, control_treedef, t0, y0, args, control_)
        return y1, None

    def func_for_init(self, y_treedef: SquashTreeDef, t: Scalar, y_: Array["state"],  # noqa: F821
                      args: PyTree) -> Array["state"]:  # noqa: F821
        vf = 0
        for term in self.terms:
            vf = vf + term.func_for_init(y_treedef, t, y_, args)
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
