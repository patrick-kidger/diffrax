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
        """Called on the i'th stage for all i. Used to determine when the Jacobian
        should be recomputed or not.
        """
        pass

    def func_for_init(
        self,
        terms: AbstractTerm,
        t0: Scalar,
        y0: PyTree,
        args: PyTree,
    ) -> PyTree:
        return terms.func_for_init(t0, y0, args)

    def init(
        self,
        terms: AbstractTerm,
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
    ) -> _SolverState:
        vf_expensive = terms.is_vf_expensive(t0, t1, y0, args)
        fsal = self.tableau.fsal and not vf_expensive
        if fsal:
            return terms.vf(t0, y0, args)
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

        #
        # Some Runge--Kutta methods have special structure that we can use to improve
        # efficiency.
        #
        # The famous one is FSAL; "first same as last". That is, the final evaluation
        # of the vector field on the previous step is the same as the first evaluation
        # on the subsequent step. We can reuse it and save an evaluation.
        # However note that this requires saving a vf evaluation, not a
        # vf-control-product. (This comes up when we have a different control on the
        # next step, e.g. as with adaptive step sizes, or with SDEs.)
        # As such we disable this is a vf is expensive and a vf-control-product is
        # cheap. (The canonical example is the optimise-then-discretise adjoint SDE.
        # For this SDE, the vf-control product is a vector-Jacobian product, which is
        # notably cheaper than evaluating a full Jacobian.)
        #
        # Next we have SSAL; "solution same as last". That is, the output of the step
        # has already been calculated during the internal stage calculations. We can
        # reuse those and save a dot product.
        #
        # Finally we have a choice whether to save and work with vector field
        # evaluations (fs), or to save and work with (vector field)-control products
        # (ks).
        # The former is needed for implicit FSAL solvers: they need to obtain the
        # final f1 for the FSAL property, which means they need to do the implicit
        # solve in vf-space rather than (vf-control-product)-space, which means they
        # need to use `fs` to predict the initial point for the root finding operation.
        # Meanwhile the latter is needed when solving optimise-then-discretise adjoint
        # SDEs, for which vector field evaluations are prohibitively expensive, and we
        # must necessarily work only with the (much cheap) vf-control-products. (In
        # this case this is the difference between computing a Jacobian and computing a
        # vector-Jacobian product.)
        # For other probles, we choose to use `ks`. This doesn't have a strong
        # rationale although it does have some minor efficiency points in its favour,
        # e.g. we need `ks` to perform dense interpolation if needed.
        #
        _vf_expensive = terms.is_vf_expensive(t0, t1, y0, args)
        _implicit_later_stages = self.tableau.a_diagonal is not None and any(
            self.tableau.a_diagonal[1:] != 0
        )
        fsal = self.tableau.fsal and not _vf_expensive
        ssal = self.tableau.ssal
        if _implicit_later_stages and fsal:
            use_fs = True
        elif _vf_expensive:
            use_fs = False
        else:  # Choice not as important here; we use ks for minor efficiency reasons.
            use_fs = False

        #
        # Initialise values. Evaluate the first stage if not FSAL.
        #

        control = terms.contr(t0, t1)
        dt = t1 - t0

        if fsal:
            if use_fs:
                f0 = solver_state
            else:
                f0 = solver_state
                k0 = lax.cond(
                    made_jump,
                    lambda _: terms.vf_prod(t0, y0, args, control),
                    lambda _: terms.prod(f0, control),
                    None,
                )
                del f0
            jac_f = None
            jac_k = None
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
            f0, k0, jac_f, jac_k, result = self._eval_stage(
                terms,
                0,
                t0_,
                y0,
                args,
                control,
                jac_f=None,
                jac_k=None,
                fs=None,
                ks=None,
                return_fi=use_fs,
                return_ki=not use_fs,
            )
            if use_fs:
                assert k0 is None
                del k0
            else:
                assert f0 is None
                del f0

        #
        # Initialise `fs` or `ks` as a place to store the stage evaluations.
        #

        lentime = (len(self.tableau.c) + 1,)
        if use_fs:
            fs = jax.tree_map(lambda f: jnp.empty(lentime + jnp.shape(f)), f0)
            fs = (fs**ω).at[0].set(f0**ω).ω
            ks = None
        else:
            fs = None
            ks = jax.tree_map(lambda k: jnp.empty(lentime + jnp.shape(k)), k0)
            ks = (ks**ω).at[0].set(k0**ω).ω

        #
        # Iterate through the stages
        #

        for i, (a_i, c_i) in enumerate(zip(self.tableau.a_lower, self.tableau.c)):
            if c_i == 1:
                # No floating point error
                ti = t1
            else:
                ti = t0 + c_i * dt
            if use_fs:
                increment = vector_tree_dot(a_i, ω(fs)[: i + 1].ω)
                increment = terms.prod(increment, control)
            else:
                increment = vector_tree_dot(a_i, ω(ks)[: i + 1].ω)
            yi_partial = (y0**ω + increment**ω).ω
            last_iteration = i == len(self.tableau.a_lower) - 1
            return_fi = use_fs or (fsal and last_iteration)
            return_ki = not use_fs
            fi, ki, jac_f, jac_k, new_result = self._eval_stage(
                terms,
                i + 1,
                ti,
                yi_partial,
                args,
                control,
                jac_f,
                jac_k,
                fs,
                ks,
                return_fi,
                return_ki,
            )
            if not return_fi:
                assert fi is None
                del fi
            if use_fs:
                assert ki is None
                del ki
            result = jnp.where(result == RESULTS.successful, new_result, result)
            if use_fs:
                fs = ω(fs).at[i + 1].set(ω(fi)).ω
            else:
                ks = ω(ks).at[i + 1].set(ω(ki)).ω

        #
        # Compute step output
        #

        if ssal:
            y1 = yi_partial
        else:
            if use_fs:
                increment = vector_tree_dot(self.tableau.b_sol, fs)
                increment = terms.prod(increment, control)
            else:
                increment = vector_tree_dot(self.tableau.b_sol, ks)
            y1 = (y0**ω + increment**ω).ω

        #
        # Compute error estimate
        #

        if use_fs:
            y_error = vector_tree_dot(self.tableau.b_error, fs)
            y_error = terms.prod(y_error, control)
        else:
            y_error = vector_tree_dot(self.tableau.b_error, ks)
        y_error = jax.tree_map(
            lambda _y_error: jnp.where(result == RESULTS.successful, _y_error, jnp.inf),
            y_error,
        )  # i.e. an implicit step failed to converge

        #
        # Compute dense info
        #

        if use_fs:
            if fs is None:
                # Edge case for diffeqsolve(y0=None)
                ks = None
            else:
                ks = jax.vmap(lambda f: terms.prod(f, control))(fs)
        dense_info = dict(y0=y0, y1=y1, k=ks)

        #
        # Compute next solver state
        #

        if fsal:
            solver_state = fi
        else:
            solver_state = None

        return y1, y_error, dense_info, solver_state, result

    def _eval_stage(
        self,
        terms,
        i,
        ti,
        yi_partial,
        args,
        control,
        jac_f,
        jac_k,
        fs,
        ks,
        return_fi,
        return_ki,
    ):
        assert return_fi or return_ki
        if self.tableau.a_diagonal is None:
            diagonal = 0
        else:
            diagonal = self.tableau.a_diagonal[i]
        if diagonal == 0:
            # Explicit stage
            if return_fi:
                fi = terms.vf(ti, yi_partial, args)
                if return_ki:
                    ki = terms.prod(fi, control)
                else:
                    ki = None
            else:
                fi = None
                if return_ki:
                    ki = terms.vf_prod(ti, yi_partial, args, control)
                else:
                    assert False
            return fi, ki, jac_f, jac_k, RESULTS.successful
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
                    jac_f = self.nonlinear_solver.jac(
                        _implicit_relation_f,
                        fi_pred,
                        (diagonal, terms.vf, terms.prod, ti, yi_partial, args, control),
                    )
                assert jac_f is not None
                nonlinear_sol = self.nonlinear_solver(
                    _implicit_relation_f,
                    fi_pred,
                    (diagonal, terms.vf, terms.prod, ti, yi_partial, args, control),
                    jac_f,
                )
                fi = nonlinear_sol.root
                if return_ki:
                    ki = terms.prod(fi, control)
                else:
                    ki = None
                return fi, ki, jac_f, jac_k, nonlinear_sol.result
            else:
                if return_ki:
                    if i == 0:
                        # Implicit first stage. Make an extra function evaluation to
                        # use as a predictor for the solution to the first stage.
                        ki_pred = terms.vf_prod(ti, yi_partial, args, control)
                    else:
                        ki_pred = vector_tree_dot(
                            self.tableau.a_predictor[i - 1], ω(ks)[:i].ω
                        )
                    if self._recompute_jac(i):
                        jac_k = self.nonlinear_solver.jac(
                            _implicit_relation_k,
                            ki_pred,
                            (diagonal, terms.vf_prod, ti, yi_partial, args, control),
                        )
                    assert jac_k is not None
                    nonlinear_sol = self.nonlinear_solver(
                        _implicit_relation_k,
                        ki_pred,
                        (diagonal, terms.vf_prod, ti, yi_partial, args, control),
                        jac_k,
                    )
                    fi = None
                    ki = nonlinear_sol.root
                    return fi, ki, jac_f, jac_k, nonlinear_sol.result
                else:
                    assert False


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
