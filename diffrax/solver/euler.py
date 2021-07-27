from typing import Callable, Tuple, Type

from ..autojit import autojit
from ..custom_types import Array, PyTree, Scalar, SquashTreeDef
from ..interpolation import AbstractInterpolation, LinearInterpolation
from ..term import AbstractTerm, ODETerm
from ..tree import tree_dataclass
from .base import AbstractSolver

# Autojit is the secret sauce here.
# It means that we can pass in `self` to the jitted function, and as this is a tree_dataclass, it is treated as a
# PyTree and not recompiled against if another instance with the same structure is used later.


@tree_dataclass
class Euler(AbstractSolver):
    term: AbstractTerm
    recommended_interpolation: Type[AbstractInterpolation] = LinearInterpolation

    @autojit
    def step(
        self,
        y_treedef: SquashTreeDef,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
        solver_state: None
    ) -> Tuple[Array["state"], None]:  # noqa: F821
        control_, control_treedef = self.term.contr_(t0, t1)
        y1 = y0 + self.term.vf_prod_(y_treedef, control_treedef, t0, y0, args, control_)
        return y1, None


def euler(vector_field: Callable[[Scalar, PyTree, PyTree], PyTree]):
    return Euler(term=ODETerm(vector_field=vector_field))
