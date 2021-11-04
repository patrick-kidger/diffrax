from typing import Callable, Tuple

import jax.numpy as jnp

from ..brownian import AbstractBrownianPath
from ..custom_types import Array, DenseInfo, PyTree, Scalar
from ..local_interpolation import LocalLinearInterpolation
from ..nonlinear_solver import AbstractNonlinearSolver, NewtonNonlinearSolver
from ..solution import RESULTS
from ..term import AbstractTerm, ControlTerm, MultiTerm, ODETerm, WrapTerm
from .base import AbstractSolver


_SolverState = None


def _implicit_relation(z1, nonlinear_solve_args):
    vf_prod, t1, y0, args, control = nonlinear_solve_args
    return vf_prod(t1, y0 + z1, args, control) - z1


class ImplicitEuler(AbstractSolver):
    r"""Implicit Euler method.

    A-B-L stable 1st order SDIRK method. Does not support adaptive timestepping.
    """

    term: AbstractTerm
    nonlinear_solver: AbstractNonlinearSolver = NewtonNonlinearSolver()

    interpolation_cls = LocalLinearInterpolation
    order = 1

    def _wrap(self, t0: Scalar, y0: PyTree, args: PyTree, direction: Scalar):
        kwargs = super()._wrap(t0, y0, args, direction)
        kwargs["term"] = WrapTerm(
            term=self.term, t=t0, y=y0, args=args, direction=direction
        )
        kwargs["nonlinear_solver"] = self.nonlinear_solver
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
        # TODO: offer explicit Euler as a predictor
        zero = jnp.zeros_like(y0)
        jac = self.nonlinear_solver.jac(
            _implicit_relation, zero, (self.term.vf_prod, t1, y0, args, control)
        )
        z1, result = self.nonlinear_solver(
            _implicit_relation, zero, (self.term.vf_prod, t1, y0, args, control), jac
        )
        y1 = y0 + z1
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, result

    def func_for_init(
        self,
        t0: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
    ) -> Array["state"]:  # noqa: F821
        return self.term.func_for_init(t0, y0, args)


def implicit_euler(vector_field: Callable[[Scalar, PyTree, PyTree], PyTree], **kwargs):
    return ImplicitEuler(term=ODETerm(vector_field=vector_field), **kwargs)


def implicit_euler_maruyama(
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
    return ImplicitEuler(term=term, **kwargs)
