import functools as ft
from typing import Callable, Tuple

import jax

from ..brownian import AbstractBrownianPath
from ..custom_types import Array, PyTree, Scalar, SquashTreeDef
from ..interpolation import LinearInterpolation
from ..term import AbstractTerm, ControlTerm, ODETerm
from .base import AbstractSolver, SplittingMethod


@ft.partial(jax.jit, static_argnums=[0, 1])
def _euler_diff_step(
    diff_control_: Callable[[Scalar], Array["control":...]],  # noqa: F821
    vector_field_prod_: Callable[[SquashTreeDef, Scalar, Array["state":...], Array["control":...]],  # noqa: F821
                                 Array["state":...]],  # noqa: F821
    treedef: SquashTreeDef,
    t0: Scalar,
    t1: Scalar,
    y0: Array["state":...]  # noqa: F821
) -> Array["state":...]:  # noqa: F821
    control0 = diff_control_(t0)
    return y0 + vector_field_prod_(treedef, t0, y0, control0) * (t1 - t0)


@ft.partial(jax.jit, static_argnums=[0, 1])
def _euler_eval_step(
    eval_control_: Callable[[Scalar, Scalar], Array["control":...]],  # noqa: F821
    vector_field_prod_: Callable[[SquashTreeDef, Scalar, Array["state":...], Array["control":...]],  # noqa: F821
                                 Array["state":...]],  # noqa: F821
    treedef: SquashTreeDef,
    t0: Scalar,
    t1: Scalar,
    y0: Array["state":...]  # noqa: F821
) -> Array["state":...]:  # noqa: F821
    control = eval_control_(t0, t1)
    return y0 + vector_field_prod_(treedef, t0, y0, control)


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
    def step(self, t0, t1, y0):
        pass

    def init(self, t0: Scalar, y0: Array["state":...]) -> None:  # noqa: F821
        return None

    def diff_step(
        self, treedef: SquashTreeDef, t0: Scalar, t1: Scalar, y0: Array["state":...],  # noqa: F821
        solver_state: None
    ) -> Tuple[Array["state":...], None]:  # noqa: F821
        return (_euler_diff_step(self.term.diff_control_, self.term.vector_field_prod_, treedef, t0, t1, y0), None)

    def eval_step(
        self, treedef: SquashTreeDef, t0: Scalar, t1: Scalar, y0: Array["state":...],  # noqa: F821
        solver_state: None
    ) -> Tuple[Array["state":...], None]:  # noqa: F821
        return (_euler_eval_step(self.term.eval_control_, self.term.vector_field_prod_, treedef, t0, t1, y0), None)


def euler(vector_field: Callable[[Scalar, PyTree], PyTree]):
    return Euler(term=ODETerm(vector_field=vector_field))


def euler_maruyama(
    drift: Callable[[Scalar, PyTree], PyTree], diffusion: Callable[[Scalar, PyTree], PyTree], bm: AbstractBrownianPath
):
    drift = Euler(term=ODETerm(vector_field=drift))
    diffusion = Euler(term=ControlTerm(vector_field=diffusion, control=bm))
    return SplittingMethod(solvers=[[drift, diffusion]])
