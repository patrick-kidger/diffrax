from typing import Callable, Tuple

from ..brownian import AbstractBrownianPath
from ..custom_types import Array, DenseInfo, PyTree, Scalar
from ..local_interpolation import LocalLinearInterpolation
from ..solution import RESULTS
from ..term import AbstractTerm, ControlTerm, MultiTerm, ODETerm, WrapTerm
from .base import AbstractSolver


_SolverState = None


class Euler(AbstractSolver):
    """Euler's method.

    Explicit 1st order RK method. Does not support adaptive timestepping.
    """

    term: AbstractTerm

    interpolation_cls = LocalLinearInterpolation
    order = 1

    def _wrap(self, t0: Scalar, y0: PyTree, args: PyTree, direction: Scalar):
        kwargs = super()._wrap(t0, y0, args, direction)
        kwargs["term"] = WrapTerm(
            term=self.term, t=t0, y=y0, args=args, direction=direction
        )
        return kwargs

    def step(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
        solver_state: _SolverState,
        made_jump: Array[(), bool],
    ) -> Tuple[Array["state"], None, DenseInfo, _SolverState, RESULTS]:  # noqa: F821
        del solver_state, made_jump
        control = self.term.contr(t0, t1)
        y1 = y0 + self.term.vf_prod(t0, y0, args, control)
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func_for_init(
        self,
        t0: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
    ) -> Array["state"]:  # noqa: F821
        return self.term.func_for_init(t0, y0, args)


def euler(vector_field: Callable[[Scalar, PyTree, PyTree], PyTree], **kwargs):
    return Euler(term=ODETerm(vector_field=vector_field), **kwargs)


def euler_maruyama(
    drift: Callable[[Scalar, PyTree, PyTree], PyTree],
    diffusion: Callable[[Scalar, PyTree, PyTree], PyTree],
    bm: AbstractBrownianPath,
    **kwargs
):
    term = MultiTerm(
        terms=(
            ODETerm(vector_field=drift),
            ControlTerm(vector_field=diffusion, control=bm),
        )
    )
    return Euler(term=term, **kwargs)
