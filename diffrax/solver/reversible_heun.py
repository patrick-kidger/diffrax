from typing import Callable, Optional, Tuple

import jax.lax as lax

from ..brownian import AbstractBrownianPath
from ..custom_types import Array, DenseInfo, PyTree, Scalar
from ..local_interpolation import LocalLinearInterpolation
from ..solution import RESULTS
from ..term import AbstractTerm, ControlTerm, MultiTerm, ODETerm, WrapTerm
from .base import AbstractSolver


_SolverState = Tuple[Array["state"], Array["state*control"]]  # noqa: F821


class ReversibleHeun(AbstractSolver):
    """Reversible Heun method.

    Algebraically reversible 2nd order method. Has an embedded 1st order method.

    @article{kidger2021efficient,
        author={Kidger, Patrick and Foster, James and Li, Xuechen and Lyons, Terry},
        title={Efficient and Accurate Gradients for Neural {SDE}s},
        year={2021},
        journal={Advances in Neural Information Processing Systems}
    }
    """

    term: AbstractTerm

    interpolation_cls = LocalLinearInterpolation  # TODO use something better than this?
    order = 2

    def _wrap(self, t0: Scalar, y0: PyTree, args: PyTree, direction: Scalar):
        kwargs = super()._wrap(t0, y0, args, direction)
        kwargs["term"] = WrapTerm(
            term=self.term, t=t0, y=y0, args=args, direction=direction
        )
        return kwargs

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
        made_jump: Array[(), bool],
    ) -> Tuple[Array["state"], Array["state"], DenseInfo, _SolverState]:  # noqa: F821

        yhat0, vf0 = solver_state

        vf0 = lax.cond(
            made_jump, lambda _: self.term.vf(t0, y0, args), lambda _: vf0, None
        )

        control = self.term.contr(t0, t1)
        yhat1 = 2 * y0 - yhat0 + self.term.prod(vf0, control)
        vf1 = self.term.vf(t1, yhat1, args)
        y1 = y0 + 0.5 * self.term.prod(vf0 + vf1, control)
        y1_error = 0.5 * self.term.prod(vf1 - vf0, control)

        dense_info = {"y0": y0, "y1": y1}
        solver_state = (yhat1, vf1)
        return y1, y1_error, dense_info, solver_state, RESULTS.successful

    def func_for_init(
        self,
        t0: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
    ) -> Array["state"]:  # noqa: F821
        return self.term.func_for_init(t0, y0, args)


def reversible_heun(
    vector_field: Callable[[Scalar, PyTree, PyTree], PyTree],
    diffusion: Optional[Callable[[Scalar, PyTree, PyTree], PyTree]] = None,
    bm: Optional[AbstractBrownianPath] = None,
    **kwargs,
) -> ReversibleHeun:
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
