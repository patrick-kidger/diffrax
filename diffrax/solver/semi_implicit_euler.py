from dataclasses import field
from typing import Callable, Tuple

import jax

from ..custom_types import Array, DenseInfo, PyTree, Scalar
from ..local_interpolation import LocalLinearInterpolation
from ..misc import ravel_pytree
from ..solution import RESULTS
from ..term import AbstractTerm, ODETerm, WrapTerm
from .base import AbstractSolver


_SolverState = None


_do_not_set_at_init = object()


# TODO: improve the efficiency of this a bit? It's doing quite a lot of ravelling +
# unravelling. (Probably shouldn't matter too much under JIT, at least.)
class SemiImplicitEuler(AbstractSolver):
    term1: AbstractTerm
    term2: AbstractTerm
    unravel_y: jax.tree_util.Partial = field(repr=False, default=_do_not_set_at_init)

    interpolation_cls = LocalLinearInterpolation
    order = 1

    def wrap(
        self, t0: Scalar, y0: Tuple[PyTree, PyTree], args: PyTree, direction: Scalar
    ):
        y0_1, y0_2 = y0
        _, unravel_y = ravel_pytree(y0)
        return type(self)(
            term1=WrapTerm(
                term=self.term1, t=t0, y=y0_2, args=args, direction=direction
            ),
            term2=WrapTerm(
                term=self.term2, t=t0, y=y0_1, args=args, direction=direction
            ),
            unravel_y=unravel_y,
        )

    def step(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
        solver_state: _SolverState,
        made_jump: Array[(), bool],
    ) -> Tuple[Array["state"], None, DenseInfo, _SolverState]:  # noqa: F821
        del made_jump

        control1 = self.term1.contr(t0, t1)
        control2 = self.term2.contr(t0, t1)

        y0_1, y0_2 = self.unravel_y(y0)
        y0_1, unravel_y1 = ravel_pytree(y0_1)
        y0_2, unravel_y2 = ravel_pytree(y0_2)

        y1_1 = y0_1 + self.term1.vf_prod(t0, y0_2, args, control1)
        y1_2 = y0_2 + self.term2.vf_prod(t0, y1_1, args, control2)

        y1_1 = unravel_y1(y1_1)
        y1_2 = unravel_y2(y1_2)
        y1, _ = ravel_pytree((y0_1, y0_2))

        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func_for_init(
        self,
        t0: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
    ) -> Array["state"]:  # noqa: F821

        y0_1, y0_2 = self.unravel_y(y0)
        y0_1, unravel_y1 = ravel_pytree(y0_1)
        y0_2, unravel_y2 = ravel_pytree(y0_2)

        f1 = self.term1.func_for_init(t0, y0_2, args)
        f2 = self.term2.func_for_init(t0, y0_1, args)

        f1 = unravel_y1(f1)
        f2 = unravel_y2(f2)
        f, _ = ravel_pytree((f1, f2))

        return f


def semi_implicit_euler(
    vector_field1: Callable[[Scalar, PyTree, PyTree], PyTree],
    vector_field2: Callable[[Scalar, PyTree, PyTree], PyTree],
    **kwargs
):
    return SemiImplicitEuler(
        term1=ODETerm(vector_field=vector_field1),
        term2=ODETerm(vector_field=vector_field2),
        **kwargs
    )
