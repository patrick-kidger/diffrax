from typing import Tuple

import jax

from ..custom_types import Bool, DenseInfo, PyTree, Scalar
from ..local_interpolation import LocalLinearInterpolation
from ..misc import ω
from ..solution import RESULTS
from ..term import AbstractTerm
from .base import AbstractImplicitSolver


_ErrorEstimate = None
_SolverState = None


def _implicit_relation(z1, nonlinear_solve_args):
    vf_prod, t1, y0, args, control = nonlinear_solve_args
    diff = (vf_prod(t1, (y0**ω + z1**ω).ω, args, control) ** ω - z1**ω).ω
    return diff


class ImplicitEuler(AbstractImplicitSolver):
    r"""Implicit Euler method.

    A-B-L stable 1st order SDIRK method. Does not support adaptive step sizing.
    """

    term_structure = jax.tree_structure(0)
    interpolation_cls = LocalLinearInterpolation

    def order(self, terms):
        return 1

    def step(
        self,
        terms: AbstractTerm,
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
        solver_state: _SolverState,
        made_jump: Bool,
    ) -> Tuple[PyTree, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        control = terms.contr(t0, t1)
        pred = terms.vf_prod(t0, y0, args, control)
        jac = self.nonlinear_solver.jac(
            _implicit_relation, pred, (terms.vf_prod, t1, y0, args, control)
        )
        nonlinear_sol = self.nonlinear_solver(
            _implicit_relation, pred, (terms.vf_prod, t1, y0, args, control), jac
        )
        z1 = nonlinear_sol.root
        y1 = (y0**ω + z1**ω).ω
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, nonlinear_sol.result

    def func(
        self,
        terms: AbstractTerm,
        t0: Scalar,
        y0: PyTree,
        args: PyTree,
    ) -> PyTree:
        return terms.vf(t0, y0, args)
