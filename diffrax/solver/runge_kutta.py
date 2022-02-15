import abc
from dataclasses import dataclass, field
from typing import Optional, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

from ..custom_types import Bool, DenseInfo, PyTree, Scalar
from ..misc import ω
from ..solution import RESULTS
from ..term import AbstractTerm
from .base import AbstractAdaptiveSolver, AbstractImplicitSolver, vector_tree_dot


# Entries must be np.arrays, and not jnp.arrays, so that we can index into them during
# trace time.
@dataclass(frozen=True)
class ButcherTableau:
    # Explicit RK methods
    c: np.ndarray
    b_sol: np.ndarray
    b_error: np.ndarray
    a_lower: Tuple[np.ndarray]

    # Implicit RK methods
    a_diagonal: Optional[np.ndarray] = None
    a_predictor: Optional[Tuple[np.ndarray]] = None

    # Determine the use of fast-paths
    ssal: bool = field(init=False)
    fsal: bool = field(init=False)

    def __post_init__(self):
        assert self.c.ndim == 1
        for a_i in self.a_lower:
            assert a_i.ndim == 1
        assert self.b_sol.ndim == 1
        assert self.b_error.ndim == 1
        assert self.c.shape[0] == len(self.a_lower)
        assert all(i + 1 == a_i.shape[0] for i, a_i in enumerate(self.a_lower))
        assert self.c.shape[0] + 1 == self.b_sol.shape[0]
        assert self.c.shape[0] + 1 == self.b_error.shape[0]
        for i, (a_i, c_i) in enumerate(zip(self.a_lower, self.c)):
            diagonal = 0 if self.a_diagonal is None else self.a_diagonal[i + 1]
            assert np.allclose(sum(a_i) + diagonal, c_i)
        assert np.allclose(sum(self.b_sol), 1.0)
        assert np.allclose(sum(self.b_error), 0.0)

        if self.a_diagonal is None:
            assert self.a_predictor is None
            implicit = False
        else:
            assert self.a_predictor is not None
            implicit = True
        if implicit:
            assert self.a_diagonal.ndim == 1
            assert self.c.shape[0] + 1 == self.a_diagonal.shape[0]
            assert len(self.a_lower) == len(self.a_predictor)
            for a_lower_i, a_predictor_i in zip(self.a_lower, self.a_predictor):
                assert a_lower_i.shape == a_predictor_i.shape
                assert np.allclose(sum(a_predictor_i), 1.0)

        lower_b_sol_equal = (self.b_sol[:-1] == self.a_lower[-1]).all()
        last_diagonal = 0 if self.a_diagonal is None else self.a_diagonal[-1]
        diagonal_b_sol_equal = self.b_sol[-1] == last_diagonal
        explicit_first_stage = self.a_diagonal is None or (self.a_diagonal[0] == 0)
        explicit_last_stage = self.a_diagonal is None or (self.a_diagonal[-1] == 0)
        # Solution y1 is the same as the last stage
        object.__setattr__(
            self,
            "ssal",
            lower_b_sol_equal and diagonal_b_sol_equal and explicit_last_stage,
        )
        # Vector field - control product k1 is the same across first/last stages.
        object.__setattr__(
            self,
            "fsal",
            lower_b_sol_equal and diagonal_b_sol_equal and explicit_first_stage,
        )


_SolverState = Optional[Tuple[Optional[PyTree], Scalar]]


# TODO: examine termination criterion for Newton iteration
# TODO: consider dividing by diagonal and control
# TODO: replace ki with ki=(zi + predictor), where this relation defines some zi, and
#       iterate to find zi, using zi=0 as the predictor. This should give better
#       numerical behaviour since the iteration is close to 0. (Although we have
#       multiplied by the increment of the control, i.e. dt, which is small...)
def _implicit_relation(ki, nonlinear_solve_args):
    # c.f:
    # https://github.com/SciML/DiffEqDevMaterials/blob/master/newton/output/main.pdf
    # (Bearing in mind that our ki is dt times smaller than theirs.)
    diagonal, vf_prod, ti, yi_partial, args, control = nonlinear_solve_args
    diff = (
        vf_prod(ti, (yi_partial**ω + diagonal * ki**ω).ω, args, control) ** ω
        - ki**ω
    ).ω
    return diff


class AbstractRungeKutta(AbstractAdaptiveSolver):

    term_structure = jax.tree_structure(0)

    @property
    @abc.abstractmethod
    def tableau(self) -> ButcherTableau:
        pass

    @abc.abstractmethod
    def _recompute_jac(self, i: int) -> bool:
        pass

    def init(
        self,
        terms: AbstractTerm,
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
    ) -> _SolverState:
        if self.tableau.fsal:
            control = terms.contr(t0, t1)
            k0 = terms.vf_prod(t0, y0, args, control)
            dt = t1 - t0
            return k0, dt
        else:
            return None

    def step(
        self,
        terms: AbstractTerm,
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
        solver_state: _SolverState,
        made_jump: Bool,
    ) -> Tuple[PyTree, PyTree, DenseInfo, _SolverState, RESULTS]:

        control = terms.contr(t0, t1)
        dt = t1 - t0

        if self.tableau.fsal:
            k0, prev_dt = solver_state
            k0 = lax.cond(
                made_jump,
                lambda _: terms.vf_prod(t0, y0, args, control),
                lambda _: (k0**ω * (dt / prev_dt)).ω,
                None,
            )
            jac = None
            result = RESULTS.successful
        else:
            if self.tableau.a_diagonal is None:
                t0_ = t0
            else:
                if self.tableau.diagonal[0] == 1:
                    # No floating point error
                    t0_ = t1
                else:
                    t0_ = t0 + self.tableau.diagonal[0] * dt
            k0, jac, result = self._eval_stage(
                terms, 0, t0_, y0, args, control, jac=None, k=None
            )

        # Note that our `k` is (for an ODE) `dt` times smaller than the usual
        # implementation (e.g. what you see in torchdiffeq or in the reference texts).
        # This is because of our vector-field-control approach.
        lentime = (len(self.tableau.c) + 1,)
        k = jax.tree_map(lambda y: jnp.empty(lentime + jnp.shape(y)), y0)
        k = (k**ω).at[0].set(k0**ω).ω

        for i, (a_i, c_i) in enumerate(zip(self.tableau.a_lower, self.tableau.c)):
            if c_i == 1:
                # No floating point error
                ti = t1
            else:
                ti = t0 + c_i * dt
            yi_partial = (y0**ω + vector_tree_dot(a_i, ω(k)[: i + 1].ω) ** ω).ω
            ki, jac, new_result = self._eval_stage(
                terms, i + 1, ti, yi_partial, args, control, jac, k
            )
            result = jnp.where(result == RESULTS.successful, new_result, result)
            # TODO: fast path to skip the rest of the stages if result is not successful
            k = ω(k).at[i + 1].set(ω(ki)).ω

        if self.tableau.ssal:
            y1 = yi_partial
        else:
            y1 = (y0**ω + vector_tree_dot(self.tableau.b_sol, k) ** ω).ω
        if self.tableau.fsal:
            k1 = (k**ω)[-1].ω
        else:
            k1 = None
        y_error = vector_tree_dot(self.tableau.b_error, k)
        y_error = jax.tree_map(
            lambda _y_error: jnp.where(result == RESULTS.successful, _y_error, jnp.inf),
            y_error,
        )
        dense_info = dict(y0=y0, y1=y1, k=k)
        if self.tableau.fsal:
            solver_state = (k1, dt)
        else:
            solver_state = None
        return y1, y_error, dense_info, solver_state, result

    def func_for_init(
        self,
        terms: AbstractTerm,
        t0: Scalar,
        y0: PyTree,
        args: PyTree,
    ) -> PyTree:
        return terms.func_for_init(t0, y0, args)

    def _eval_stage(self, terms, i, ti, yi_partial, args, control, jac, k):
        if self.tableau.a_diagonal is None:
            diagonal = 0
        else:
            diagonal = self.tableau.a_diagonal[i]
        if diagonal == 0:
            # Explicit stage
            ki = terms.vf_prod(ti, yi_partial, args, control)
            return ki, jac, RESULTS.successful
        else:
            # Implicit stage
            if i == 0:
                # Implicit first stage. Make an extra function evaluation to use as a
                # predictor for the solution to the first stage.
                ki_pred = terms.vf_prod(ti, yi_partial, args, control)
            else:
                ki_pred = vector_tree_dot(self.tableau.a_predictor[i - 1], ω(k)[:i].ω)
            if self._recompute_jac(i):
                jac = self.nonlinear_solver.jac(
                    _implicit_relation,
                    ki_pred,
                    (diagonal, terms.vf_prod, ti, yi_partial, args, control),
                )
            assert jac is not None
            nonlinear_sol = self.nonlinear_solver(
                _implicit_relation,
                ki_pred,
                (diagonal, terms.vf_prod, ti, yi_partial, args, control),
                jac,
            )
            ki = nonlinear_sol.root
            return ki, jac, nonlinear_sol.result


class AbstractERK(AbstractRungeKutta):
    def _recompute_jac(self, i: int) -> bool:
        assert False


class AbstractDIRK(AbstractRungeKutta, AbstractImplicitSolver):
    def _recompute_jac(self, i: int) -> bool:
        return True


class AbstractSDIRK(AbstractDIRK):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.tableau is not None:  # Abstract subclasses may not have a tableau.
            diagonal = cls.tableau.a_diagonal[0]
            assert (cls.tableau.a_diagonal == diagonal).all()

    def _recompute_jac(self, i: int) -> bool:
        return i == 0


class AbstractESDIRK(AbstractDIRK):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.tableau is not None:  # Abstract subclasses may not have a tableau.
            assert cls.tableau.a_diagonal[0] == 0
            diagonal = cls.tableau.a_diagonal[1]
            assert (cls.tableau.a_diagonal[1:] == diagonal).all()

    def _recompute_jac(self, i: int) -> bool:
        return i == 1
