from typing import Tuple

import jax
import jax.flatten_util as fu

from ..custom_types import Bool, DenseInfo, PyTree, Scalar
from ..local_interpolation import LocalLinearInterpolation
from ..misc import ω
from ..nonlinear_solver import AbstractNonlinearSolver, NewtonNonlinearSolver
from ..solution import RESULTS
from ..term import AbstractTerm
from .base import AbstractSolver


_ErrorEstimate = None
_SolverState = None


def _implicit_relation(z1, nonlinear_solve_args):
    vf_prod, t1, y0, args, control = nonlinear_solve_args
    _, unravel = fu.ravel_pytree(y0)
    z1 = unravel(z1)
    diff = (vf_prod(t1, (y0 ** ω + z1 ** ω).ω, args, control) ** ω - z1 ** ω).ω
    diff, _ = fu.ravel_pytree(diff)
    return diff


class ImplicitEuler(AbstractSolver):
    r"""Implicit Euler method.

    A-B-L stable 1st order SDIRK method. Does not support adaptive timestepping.
    """

    nonlinear_solver: AbstractNonlinearSolver = NewtonNonlinearSolver()

    term_structure = jax.tree_structure(0)
    interpolation_cls = LocalLinearInterpolation
    order = 1

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
        pred, unravel = fu.ravel_pytree(pred)
        jac = self.nonlinear_solver.jac(
            _implicit_relation, pred, (terms.vf_prod, t1, y0, args, control)
        )
        z1, result = self.nonlinear_solver(
            _implicit_relation, pred, (terms.vf_prod, t1, y0, args, control), jac
        )
        z1 = unravel(z1)
        y1 = (y0 ** ω + z1 ** ω).ω
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, result

    def func_for_init(
        self,
        terms: AbstractTerm,
        t0: Scalar,
        y0: PyTree,
        args: PyTree,
    ) -> PyTree:
        return terms.func_for_init(t0, y0, args)
