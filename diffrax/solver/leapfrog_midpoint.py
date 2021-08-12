from typing import Callable, Tuple

import jax.lax as lax

from ..custom_types import Array, DenseInfo, PyTree, Scalar
from ..local_interpolation import LocalLinearInterpolation
from ..term import AbstractTerm, ODETerm, WrapTerm
from .base import AbstractSolver


_SolverState = Tuple[Scalar, Array["state"], bool]  # noqa: F821


# TODO: support arbitrary linear multistep methods
class LeapfrogMidpoint(AbstractSolver):
    term: AbstractTerm

    interpolation_cls = LocalLinearInterpolation
    order = 2

    def wrap(self, t0: Scalar, y0: PyTree, args: PyTree, direction: Scalar):
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
        del t1, args
        return (t0, y0, True)

    def step(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
        solver_state: _SolverState,
    ) -> Tuple[Array["state"], None, DenseInfo, _SolverState]:  # noqa: F821
        tm1, ym1, firststep = solver_state
        y1 = lax.cond(
            firststep, self._firststep, self._laterstep, (tm1, t0, t1, ym1, y0, args)
        )
        dense_info = {"y0": y0, "y1": y1}
        solver_state = (t0, y0, False)
        return y1, None, dense_info, solver_state

    def _firststep(self, operand):
        # Euler method on the first step
        _, t0, t1, _, y0, args = operand
        control = self.term.contr(t0, t1)
        y1 = y0 + self.term.vf_prod(t0, y0, args, control)
        return y1

    def _laterstep(self, operand):
        tm1, t0, t1, ym1, y0, args = operand
        control = self.term.contr(tm1, t1)
        y1 = ym1 + self.term.vf_prod(t0, y0, args, control)
        return y1

    def func_for_init(
        self,
        t0: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
    ) -> Array["state"]:  # noqa: F821
        return self.term.func_for_init(t0, y0, args)


def leapfrog_midpoint(
    vector_field: Callable[[Scalar, PyTree, PyTree], PyTree],
    **kwargs,
):
    return LeapfrogMidpoint(term=ODETerm(vector_field=vector_field), **kwargs)
