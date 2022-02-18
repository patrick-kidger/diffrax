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
from ..term import AbstractExpensiveVFTerm, AbstractTerm
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


_SolverState = Optional[PyTree]


# TODO: examine termination criterion for Newton iteration
# TODO: replace fi with fi=(zi + predictor), where this relation defines some zi, and
#       iterate to find zi, using zi=0 as the predictor. This should give better
#       numerical behaviour since the iteration is close to 0. (Although we have
#       multiplied by the increment of the control, i.e. dt, which is small...)
def _implicit_relation_f(fi, nonlinear_solve_args):
    diagonal, vf, prod, ti, yi_partial, args, control = nonlinear_solve_args
    diff = (
        vf(ti, (yi_partial**ω + diagonal * prod(fi, control) ** ω).ω, args) ** ω
        - fi**ω
    ).ω
    return diff


# TODO: consider dividing by diagonal and control
def _implicit_relation_k(ki, nonlinear_solve_args):
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

    def _is_fsal(self, terms):
        is_expensive = lambda x: isinstance(x, AbstractExpensiveVFTerm)
        leaves = jax.tree_flatten(terms, is_leaf=is_expensive)[0]
        return self.tableau.fsal and not any(map(is_expensive, leaves))

    def init(
        self,
        terms: AbstractTerm,
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
    ) -> _SolverState:
        if self._is_fsal(terms):
            return terms.vf_prod(t0, y0, args)
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
        fsal = self._is_fsal(terms)
        implicit = self.tableau.a_diagonal is not None and any(
            self.tableau.a_diagonal != 0
        )

        if fsal:
            f0 = solver_state
            k0 = lax.cond(
                made_jump,
                lambda _: terms.vf_prod(t0, y0, args, control),
                lambda _: terms.prod(f0, control),
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
            # return_fi = fsal and ... (see below). But on this branch fsal is False,
            # so return_fi is False as well.
            _, k0, jac, result = self._eval_stage(
                terms,
                0,
                t0_,
                y0,
                args,
                control,
                jac=None,
                fs=None,
                ks=None,
                return_fi=False,
            )

        lentime = (len(self.tableau.c) + 1,)
        if fsal and implicit:
            fs = jax.tree_map(lambda f: jnp.empty(lentime + jnp.shape(f)), f0)
            fs = (fs**ω).at[0].set(f0**ω).ω
            ks = None
        else:
            fs = None
            ks = jax.tree_map(lambda y: jnp.empty(lentime + jnp.shape(y)), y0)
            ks = (ks**ω).at[0].set(k0**ω).ω

        for i, (a_i, c_i) in enumerate(zip(self.tableau.a_lower, self.tableau.c)):
            if c_i == 1:
                # No floating point error
                ti = t1
            else:
                ti = t0 + c_i * dt
            if fsal and implicit:
                increment = vector_tree_dot(a_i, ω(fs)[: i + 1].ω)
                increment = terms.prod(increment, control)
            else:
                increment = vector_tree_dot(a_i, ω(ks)[: i + 1].ω)
            yi_partial = (y0**ω + increment**ω).ω
            if implicit:
                return_fi = fsal
            else:
                return_fi = fsal and (i + 1 == len(self.tableau.a_lower))
            fi, ki, jac, new_result = self._eval_stage(
                terms, i + 1, ti, yi_partial, args, control, jac, fs, ks, return_fi
            )
            result = jnp.where(result == RESULTS.successful, new_result, result)
            if fsal and implicit:
                fs = ω(fs).at[i + 1].set(ω(fi)).ω
            else:
                ks = ω(ks).at[i + 1].set(ω(ki)).ω

        if self.tableau.ssal:
            y1 = yi_partial
        else:
            if fsal and implicit:
                increment = vector_tree_dot(self.tableau.b_sol, fs)
                increment = terms.prod(increment, control)
            else:
                increment = vector_tree_dot(self.tableau.b_sol, ks)
            y1 = (y0**ω + increment**ω).ω
        if fsal and implicit:
            y_error = vector_tree_dot(self.tableau.b_error, fs)
            y_error = terms.prod(y_error, control)
        else:
            y_error = vector_tree_dot(self.tableau.b_error, ks)
        y_error = jax.tree_map(
            lambda _y_error: jnp.where(result == RESULTS.successful, _y_error, jnp.inf),
            y_error,
        )
        if fsal and implicit:
            ks = jax.vmap(lambda f: terms.prod(f, control))(fs)
        dense_info = dict(y0=y0, y1=y1, k=ks)
        if fsal:
            solver_state = fi
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

    def _eval_stage(
        self, terms, i, ti, yi_partial, args, control, jac, fs, ks, return_fi
    ):
        if self.tableau.a_diagonal is None:
            diagonal = 0
        else:
            diagonal = self.tableau.a_diagonal[i]
        if diagonal == 0:
            # Explicit stage
            if return_fi:
                fi = terms.vf(ti, yi_partial, args)
                ki = terms.prod(fi, control)
            else:
                fi = None
                ki = terms.vf_prod(ti, yi_partial, args, control)
            return fi, ki, jac, RESULTS.successful
        else:
            # Implicit stage
            if return_fi:
                if i == 0:
                    # Implicit first stage. Make an extra function evaluation to use as
                    # a predictor for the solution to the first stage.
                    fi_pred = terms.vf(ti, yi_partial, args)
                else:
                    fi_pred = vector_tree_dot(
                        self.tableau.a_predictor[i - 1], ω(fs)[:i].ω
                    )
                if self._recompute_jac(i):
                    jac = self.nonlinear_solver.jac(
                        _implicit_relation_f,
                        fi_pred,
                        (diagonal, terms.vf, terms.prod, ti, yi_partial, args, control),
                    )
                assert jac is not None
                nonlinear_sol = self.nonlinear_solver(
                    _implicit_relation_f,
                    fi_pred,
                    (diagonal, terms.vf, terms.prod, ti, yi_partial, args, control),
                    jac,
                )
                fi = nonlinear_sol.root
                ki = terms.prod(fi, control)
                return fi, ki, jac, nonlinear_sol.result
            else:
                if i == 0:
                    # Implicit first stage. Make an extra function evaluation to use as
                    # a predictor for the solution to the first stage.
                    ki_pred = terms.vf_prod(ti, yi_partial, args, control)
                else:
                    ki_pred = vector_tree_dot(
                        self.tableau.a_predictor[i - 1], ω(ks)[:i].ω
                    )
                if self._recompute_jac(i):
                    jac = self.nonlinear_solver.jac(
                        _implicit_relation_k,
                        ki_pred,
                        (diagonal, terms.vf_prod, ti, yi_partial, args, control),
                    )
                assert jac is not None
                nonlinear_sol = self.nonlinear_solver(
                    _implicit_relation_k,
                    ki_pred,
                    (diagonal, terms.vf_prod, ti, yi_partial, args, control),
                    jac,
                )
                fi = None
                ki = nonlinear_sol.root
                return fi, ki, jac, nonlinear_sol.result


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
