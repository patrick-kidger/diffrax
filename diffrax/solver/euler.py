import functools as ft
from typing import Callable, Tuple

from ..autojit import autojit
from ..custom_types import Array, PyTree, Scalar, SquashTreeDef
from ..interpolation import LinearInterpolation
from ..term import AbstractTerm, ODETerm
from .base import AbstractSolver


# Autojit is the secret sauce here.
# It means that we can pass in functions diff_control_, vector_field_prod_, and as long as these functions are pytrees
# (i.e. because they're tree_methods of a tree_dataclass), then we can avoid re-jitting these functions in many cases;
# it is common for these functions to be parameterised by some jnp.arrays and we do not re-jit when these change value,
# only when they change shape/dtype.
@ft.partial(autojit, inline=True)
def _euler_diff_step(
    diff_control_: Callable[[Scalar], Array["control"]],  # noqa: F821
    vf_prod_: Callable[[SquashTreeDef, Scalar, Array["state"], Array["control"]],  # noqa: F821
                       Array["state"]],  # noqa: F821
    y_treedef: SquashTreeDef,
    t0: Scalar,
    t1: Scalar,
    y0: Array["state"],  # noqa: F821
    args: PyTree,
) -> Array["state"]:  # noqa: F821

    control0_, control_treedef = diff_control_(t0)
    return y0 + vf_prod_(y_treedef, control_treedef, t0, y0, args, control0_ * (t1 - t0))


@ft.partial(autojit, inline=True)
def _euler_eval_step(
    eval_control_: Callable[[Scalar, Scalar], Array["control"]],  # noqa: F821
    vf_prod_: Callable[[SquashTreeDef, Scalar, Array["state"], Array["control"]],  # noqa: F821
                       Array["state"]],  # noqa: F821
    y_treedef: SquashTreeDef,
    t0: Scalar,
    t1: Scalar,
    y0: Array["state"],  # noqa: F821
    args: PyTree,
) -> Array["state"]:  # noqa: F821

    control_, control_treedef = eval_control_(t0, t1)
    return y0 + vf_prod_(y_treedef, control_treedef, t0, y0, args, control_)


class Euler(AbstractSolver):
    def __init__(self, *, term: AbstractTerm, diff_control: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.term = term
        self.diff_control = diff_control
        if diff_control:
            self.step = self.diff_step
        else:
            self.step = self.eval_step
        self.recommended_interpolation = LinearInterpolation

    # To avoid errors due to lacking an abstractmethod
    def step(
        self,
        y_treedef: SquashTreeDef,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
        solver_state: None
    ) -> Tuple[Array["state"], None]:  # noqa: F821
        pass

    def diff_step(
        self,
        y_treedef: SquashTreeDef,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
        solver_state: None
    ) -> Tuple[Array["state"], None]:  # noqa: F821
        return (_euler_diff_step(self.term.diff_control_, self.term.vf_prod_, y_treedef, t0, t1, y0, args), None)

    def eval_step(
        self,
        y_treedef: SquashTreeDef,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
        solver_state: None
    ) -> Tuple[Array["state"], None]:  # noqa: F821
        return (_euler_eval_step(self.term.eval_control_, self.term.vf_prod_, y_treedef, t0, t1, y0, args), None)


def euler(vector_field: Callable[[Scalar, PyTree, PyTree], PyTree]):
    return Euler(term=ODETerm(vector_field=vector_field))
