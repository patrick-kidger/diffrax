from typing import Callable, Optional, Tuple

from ..brownian import AbstractBrownianPath
from ..custom_types import Array, DenseInfo, PyTree, Scalar
from ..local_interpolation import LocalLinearInterpolation
from ..term import AbstractTerm, ControlTerm, MultiTerm, ODETerm, WrapTerm
from .base import AbstractSolver


_SolverState = Tuple[Array["state"], Array["state*control"]]  # noqa: F821


class ReversibleHeun(AbstractSolver):
    term: AbstractTerm

    interpolation_cls = LocalLinearInterpolation  # TODO use something better than this?
    order = 2

    def wrap(self, t0: Scalar, y0: PyTree, args: PyTree, direction: bool):
        return type(self)(
            term=WrapTerm(term=self.term, t=t0, y=y0, args=args, direction=direction)
        )

    def init(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
    ) -> _SolverState:
        vf0 = self.term.vf(t0, y0, args)
        return (y0, vf0)

    def step(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
        solver_state: _SolverState,
    ) -> Tuple[Array["state"], Array["state"], DenseInfo, _SolverState]:  # noqa: F821
        yhat0, vf0 = solver_state
        control = self.term.contr(t0, t1)
        yhat1 = 2 * y0 - yhat0 + self.term.prod(vf0, control)
        vf1 = self.term.vf(t1, yhat1, args)
        y1 = y0 + 0.5 * self.term.prod(vf0 + vf1, control)
        y1_error = 0.5 * self.term.prod(vf1 - vf0, control)

        dense_info = {"y0": y0, "y1": y1}
        solver_state = (yhat1, vf1)
        return y1, y1_error, dense_info, solver_state


def reversible_heun(
    vector_field: Callable[[Scalar, PyTree, PyTree], PyTree],
    diffusion: Optional[Callable[[Scalar, PyTree, PyTree], PyTree]] = None,
    bm: Optional[AbstractBrownianPath] = None,
    **kwargs,
):
    if diffusion is None:
        if bm is not None:
            raise ValueError
        return ReversibleHeun(term=ODETerm(vector_field=vector_field), **kwargs)
    else:
        if bm is None:
            raise ValueError
        term = MultiTerm(
            terms=(
                ODETerm(vector_field=vector_field),
                ControlTerm(vector_field=diffusion, control=bm),
            )
        )
        return ReversibleHeun(term=term, **kwargs)
