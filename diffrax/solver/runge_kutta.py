import abc
from dataclasses import dataclass, field
from typing import Optional, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

from ..custom_types import Array, DenseInfo, PyTree, Scalar
from ..nonlinear_solver import AbstractNonlinearSolver, NewtonNonlinearSolver
from ..solution import RESULTS
from ..term import AbstractTerm, WrapTerm
from .base import AbstractSolver


@dataclass(frozen=True)
class ButcherTableau:
    alpha: np.ndarray
    beta: Tuple[np.ndarray]
    c_sol: np.ndarray
    c_error: np.ndarray
    diagonal: Optional[np.ndarray] = None
    ssal: bool = field(init=False)
    fsal: bool = field(init=False)

    def __post_init__(self):
        assert self.alpha.ndim == 1
        for beta_i in self.beta:
            assert beta_i.ndim == 1
        assert self.c_sol.ndim == 1
        assert self.c_error.ndim == 1
        assert self.alpha.shape[0] == len(self.beta)
        assert all(i + 1 == beta_i.shape[0] for i, beta_i in enumerate(self.beta))
        assert self.alpha.shape[0] + 1 == self.c_sol.shape[0]
        assert self.alpha.shape[0] + 1 == self.c_error.shape[0]
        if self.diagonal is not None:
            assert self.diagonal.ndim == 1
            assert self.alpha.shape[0] + 1 == self.diagonal.shape[0]

        lower_c_sol_equal = (self.c_sol[:-1] == self.beta[-1]).all()
        last_diagonal = 0 if self.diagonal is None else self.diagonal[-1]
        diagonal_c_sol_equal = self.c_sol[-1] == last_diagonal
        explicit_first_stage = self.diagonal is None or (self.diagonal[0] == 0)
        explicit_last_stage = self.diagonal is None or (self.diagonal[-1] == 0)
        # Solution y1 is the same as the last stage
        object.__setattr__(
            self,
            "ssal",
            lower_c_sol_equal and diagonal_c_sol_equal and explicit_last_stage,
        )
        # Vector field - control product k1 is the same across first/last stages.
        object.__setattr__(
            self,
            "fsal",
            lower_c_sol_equal and diagonal_c_sol_equal and explicit_first_stage,
        )


_SolverState = Tuple[Optional[Array["state"]], Scalar]  # noqa: F821


# TODO: examine termination criterion for Newton iteration
# TODO: consider dividing by diagonal and control
def _implicit_relation(ki, diagonal, vf_prod, ti, yi_partial, args, control):
    # c.f:
    # https://github.com/SciML/DiffEqDevMaterials/blob/master/newton/output/main.pdf
    return vf_prod(ti, yi_partial + diagonal * ki, args, control) - ki


class AbstractRungeKutta(AbstractSolver):
    term: AbstractTerm

    @property
    @abc.abstractmethod
    def tableau(self) -> ButcherTableau:
        pass

    @abc.abstractmethod
    def _init_jac(self, t0, y0, args, control):
        pass

    @abc.abstractmethod
    def _stage_jac(self, i, ti, yi_partial, args, control, jac):
        pass

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
    ) -> _SolverState:  # noqa: F821
        control = self.term.contr(t0, t1)
        if self.tableau.fsal:
            k0 = self.term.vf_prod(t0, y0, args, control)
        else:
            k0 = None
        dt = t1 - t0
        return (k0, dt)

    def step(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
        solver_state: _SolverState,
        made_jump: Array[(), bool],
    ) -> Tuple[Array["state"], Array["state"], DenseInfo, _SolverState]:  # noqa: F821

        control = self.term.contr(t0, t1)
        dt = t1 - t0
        k0, prev_dt = solver_state
        jac = self._init_jac(t0, y0, args, control)

        if self.tableau.fsal:
            k0 = lax.cond(
                made_jump,
                lambda _: self.term.vf_prod(t0, y0, args, control),
                lambda _: k0 * (dt.astype(k0.dtype) / prev_dt),
                None,
            )
            result = RESULTS.successful
        else:
            k0, result = self._eval_stage(0, t0, y0, args, control, jac)

        # Note that our `k` is (for an ODE) `dt` times smaller than the usual
        # implementation (e.g. what you see in torchdiffeq or in the reference texts).
        # This is because of our vector-field-control approach.
        k = jnp.empty(
            (len(self.tableau.alpha) + 1,) + y0.shape
        )  # y0.shape is actually single-dimensional
        k = k.at[0].set(k0)

        for i, (alpha_i, beta_i) in enumerate(
            zip(self.tableau.alpha, self.tableau.beta)
        ):
            if alpha_i == 1:
                # No floating point error
                ti = t1
            else:
                ti = t0 + alpha_i * dt
            yi_partial = y0 + beta_i @ k[: i + 1]
            jac = self._stage_jac(i + 1, ti, yi_partial, args, control, jac)
            ki, new_result = self._eval_stage(i + 1, ti, yi_partial, args, control, jac)
            result = jnp.where(result == RESULTS.successful, new_result, result)
            # TODO: fast path to skip the rest of the stages if result is not successful
            k = k.at[i + 1].set(ki)

        if self.tableau.ssal:
            y1 = yi_partial
        else:
            y1 = y0 + self.tableau.c_sol @ k
        if self.tableau.fsal:
            k1 = k[-1]
        else:
            k1 = None
        y_error = jnp.where(
            result == RESULTS.successful, self.tableau.c_error @ k, jnp.inf
        )
        dense_info = {"y0": y0, "y1": y1, "k": k}
        return y1, y_error, dense_info, (k1, dt), result

    def func_for_init(
        self,
        t0: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
    ) -> Array["state"]:  # noqa: F821
        return self.term.func_for_init(t0, y0, args)

    def _eval_stage(self, i, ti, yi_partial, args, control, jac):
        if self.tableau.diagonal is None:
            diagonal = 0
        else:
            diagonal = self.tableau.diagonal[i]
        if diagonal == 0:
            # Explicit stage
            ki = self.term.vf_prod(ti, yi_partial, args, control)
            return ki, RESULTS.successful
        else:
            # Implicit stage
            # TODO: use a better predictor than zero
            zero = jax.tree_map(jnp.zeros_like, yi_partial)
            ki, result = self.nonlinear_solver(
                _implicit_relation,
                zero,
                (diagonal, self.term.vf_prod, ti, yi_partial, args, control),
                jac,
            )
            return ki, result


class AbstractERK(AbstractRungeKutta):
    def _init_jac(self, t0, y0, args, control):
        pass

    def _stage_jac(self, i, ti, yi_partial, args, control, jac):
        pass


class AbstractDIRK(AbstractRungeKutta):
    nonlinear_solver: AbstractNonlinearSolver = NewtonNonlinearSolver()

    def _wrap(self, t0: Scalar, y0: PyTree, args: PyTree, direction: Scalar):
        kwargs = super()._wrap(t0, y0, args, direction)
        kwargs["nonlinear_solver"] = self.nonlinear_solver
        return kwargs

    def _init_jac(self, t0, y0, args, control):
        pass

    def _stage_jac(self, i, ti, yi_partial, args, control, jac):
        del jac
        diagonal = self.tableau.diagonal[i]
        zero = jax.tree_map(jnp.zeros_like, yi_partial)
        return self.nonlinear_solver.jac(
            _implicit_relation,
            zero,
            (diagonal, self.term.vf_prod, ti, yi_partial, args, control),
        )


class AbstractSDIRK(AbstractDIRK):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.tableau is not None:  # Abstract subclasses may not have a tableau.
            diagonal = cls.tableau.diagonal[0]
            assert (cls.tableau.diagonal == diagonal).all()

    def _init_jac(self, t0, y0, args, control):
        diagonal = self.tableau.diagonal[0]
        zero = jax.tree_map(jnp.zeros_like, y0)
        return self.nonlinear_solver.jac(
            _implicit_relation,
            zero,
            (diagonal, self.term.vf_prod, t0, y0, args, control),
        )

    def _stage_jac(self, i, ti, yi_partial, args, control, jac):
        return jac


class AbstractESDIRK(AbstractDIRK):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.tableau is not None:  # Abstract subclasses may not have a tableau.
            assert cls.tableau.diagonal[0] == 0
            diagonal = cls.tableau.diagonal[1]
            assert (cls.tableau.diagonal[1:] == diagonal).all()

    def _init_jac(self, t0, y0, args, control):
        diagonal = self.tableau.diagonal[1]
        zero = jax.tree_map(jnp.zeros_like, y0)
        return self.nonlinear_solver.jac(
            _implicit_relation,
            zero,
            (diagonal, self.term.vf_prod, t0, y0, args, control),
        )

    def _stage_jac(self, i, ti, yi_partial, args, control, jac):
        return jac
