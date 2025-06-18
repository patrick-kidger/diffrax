# Stuff for custom solver
from collections.abc import Callable
from typing import ClassVar

import diffrax
from diffrax._custom_types import VF, Args, BoolScalarLike, DenseInfo, RealScalarLike, Y
from diffrax._local_interpolation import LocalLinearInterpolation
from diffrax._solution import RESULTS
from diffrax._term import AbstractTerm, MultiTerm, ODETerm, LinearODETerm
from typing_extensions import TypeAlias

_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = None
import jax
import jax.numpy as jnp
import lineax as lx
from jax.scipy.linalg import expm


def make_operators(operator, control):
    match operator:
        case lx.DiagonalLinearOperator():
            # If the operator is diagonal things are fairly trivial
            # Technically we could do the part below through lineax too
            # but it just complicates things here as we return two different
            # operators (linear vs full)
            exp_Ah = jnp.exp(control * operator.diagonal)
            phi = jnp.expm1(control * operator.diagonal) / (operator.diagonal * control)
            return lx.DiagonalLinearOperator(exp_Ah), lx.DiagonalLinearOperator(phi)

        case _:
            # If the linear operator is not diagonal we need to calculate the matrix exponential
            # and solve the linear problem
            exp_Ah = expm(control * operator.as_matrix())
            A, b = operator * control, exp_Ah - jnp.eye(exp_Ah.shape[-1])
            phi = jax.vmap(lx.linear_solve, in_axes=(None, 1))(A, b).value
            return lx.MatrixLinearOperator(exp_Ah), lx.MatrixLinearOperator(phi)


class ExponentialEuler(diffrax.AbstractItoSolver):
    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    def order(self, terms):
        return 1.0

    def strong_order(self, terms):
        return 0.5

    def init(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        return None

    def step(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump

        # We split the terms into linear and the non-linear plus possible noise
        linear, non_linear = terms.terms[0].term, MultiTerm(*terms.terms[1:])
        exp_Ah, phi = make_operators(linear.operator, linear.contr(t0, t1))
        Gh_sdW = non_linear.vf_prod(t0, y0, args, non_linear.contr(t0, t1))
        y1 = exp_Ah.mv(y0) + phi.mv(Gh_sdW)
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)
