import abc
from dataclasses import dataclass, field
from typing import Optional, Tuple

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

        c_sol_equal = (self.c_sol[:-1] == self.beta[-1]).all()
        last_diagonal = 0 if self.diagonal is None else self.diagonal[-1]
        object.__setattr__(
            self, "fsal", c_sol_equal and self.c_sol[-1] == last_diagonal
        )


_SolverState = Tuple[Array["state"], Scalar]  # noqa: F821


class AbstractRungeKutta(AbstractSolver):
    term: AbstractTerm

    @property
    @abc.abstractmethod
    def tableau(self) -> ButcherTableau:
        pass

    @abc.abstractmethod
    def _init_jac(self, i, yi_prev, ti, yi_partial, args, control):
        pass

    @abc.abstractmethod
    def _stage_jac(self, i, yi_prev, ti, yi_partial, args, control, jac):
        pass

    @abc.abstractmethod
    def _implicit_solve(self, i, yi_prev, jac, t0, yi_partial, args, control):
        return yi_partial

    def wrap(self, t0: Scalar, y0: PyTree, args: PyTree, direction: Scalar):
        return type(self)(
            term=WrapTerm(term=self.term, t=t0, y=y0, args=args, direction=direction)
        )

    def init(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
    ) -> _SolverState:  # noqa: F821
        control = self.term.contr(t0, t1)
        f0 = self.term.vf_prod(t0, y0, args, control)
        dt = t1 - t0
        return (f0, dt)

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
        f0, prev_dt = solver_state
        f0 = lax.cond(
            made_jump,
            lambda _: self.term.vf_prod(t0, y0, args, control),
            lambda _: f0 * (dt.astype(f0.dtype) / prev_dt),
            None,
        )

        # Note that our `k` is (for an ODE) `dt` times smaller than the usual
        # implementation (e.g. what you see in torchdiffeq or in the reference texts).
        # This is because of our vector-field-control approach.
        k = jnp.empty(
            (len(self.tableau.alpha) + 1,) + y0.shape
        )  # y0.shape is actually single-dimensional
        k = k.at[0].set(f0)

        jac = self._init_jac(0, y0, t0, y0, args, control)
        yi_prev = y0

        result = RESULTS.successful
        for i, (alpha_i, beta_i) in enumerate(
            zip(self.tableau.alpha, self.tableau.beta)
        ):
            if alpha_i == 1:
                # No floating point error
                ti = t1
            else:
                ti = t0 + alpha_i * dt
            yi_partial = y0 + beta_i @ k[: i + 1]
            jac = self._stage_jac(i + 1, yi_prev, ti, yi_partial, args, control, jac)
            # TODO: use a better predictor than just the previous stage value
            yi, new_result = self._implicit_solve(
                i + 1, yi_prev, ti, yi_partial, args, control, jac
            )
            result = jnp.where(result == RESULTS.successful, new_result, result)
            # TODO: fast path to skip the rest of the stages if result is not successful
            fi = self.term.vf_prod(ti, yi, args, control)
            k = k.at[i + 1].set(fi)
            yi_prev = yi

        if self.tableau.fsal:
            y1 = yi
            f1 = fi
        else:
            y1 = y0 + self.tableau.c_sol @ k
            f1 = self.term.vf_prod(t1, y1, args, control)
        y_error = jnp.where(
            result == RESULTS.successful, self.tableau.c_error @ k, jnp.inf
        )
        dense_info = {"y0": y0, "y1": y1, "k": k}
        return y1, y_error, dense_info, (f1, dt), result

    def func_for_init(
        self,
        t0: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
    ) -> Array["state"]:  # noqa: F821
        return self.term.func_for_init(t0, y0, args)


class AbstractERK(AbstractRungeKutta):
    def _init_jac(self, i, yi_prev, ti, yi_partial, args, control):
        pass

    def _stage_jac(self, i, yi_prev, ti, yi_partial, args, control, jac):
        pass

    def _implicit_solve(self, i, yi_prev, ti, yi_partial, args, control, jac):
        return yi_partial, RESULTS.successful


def _criterion(zi, diagonal, vf_prod, ti, yi_partial, args, control):
    return diagonal * vf_prod(ti, yi_partial + zi, args, control) - zi


class AbstractDIRK(AbstractRungeKutta):
    nonlinear_solver: AbstractNonlinearSolver = NewtonNonlinearSolver()

    def _init_jac(self, i, yi_prev, ti, yi_partial, args, control):
        pass

    def _stage_jac(self, i, yi_prev, ti, yi_partial, args, control, jac):
        del jac
        diagonal = self.tableau.diagonal[i]
        return self.nonlinear_solver.jac(
            _criterion,
            yi_prev - yi_partial,
            (diagonal, self.term.vf_prod, ti, yi_partial, args, control),
        )

    def _implicit_solve(self, i, yi_prev, ti, yi_partial, args, control, jac):
        diagonal = self.tableau.diagonal[i]
        if diagonal == 0:  # ESDIRK methods
            return yi_partial
        zi, result = self.nonlinear_solver(
            _criterion,
            yi_prev - yi_partial,
            (diagonal, self.term.vf_prod, ti, yi_partial, args, control),
            jac,
        )
        yi = yi_partial + zi
        return yi, result


class AbstractSDIRK(AbstractDIRK):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.tableau is not None:  # Abstract subclasses may not have a tableau.
            diagonal = cls.tableau.diagonal[0]
            assert (cls.tableau.diagonal == diagonal).all()

    def _init_jac(self, i, yi_prev, ti, yi_partial, args, control):
        diagonal = self.tableau.diagonal[i]
        return self.nonlinear_solver.jac(
            _criterion,
            yi_prev - yi_partial,
            (diagonal, self.term.vf_prod, ti, yi_partial, args, control),
        )

    def _stage_jac(self, i, yi_prev, ti, yi_partial, args, control, jac):
        return jac


class AbstractESDIRK(AbstractDIRK):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.tableau is not None:  # Abstract subclasses may not have a tableau.
            assert cls.tableau.diagonal[0] == 0
            diagonal = cls.tableau.diagonal[1]
            assert (cls.tableau.diagonal[1:] == diagonal).all()

    def _init_jac(self, i, yi_prev, ti, yi_partial, args, control):
        diagonal = self.tableau.diagonal[i]
        return self.nonlinear_solver.jac(
            _criterion,
            yi_prev - yi_partial,
            (diagonal, self.term.vf_prod, ti, yi_partial, args, control),
        )

    def _stage_jac(self, i, yi_prev, ti, yi_partial, args, control, jac):
        return jac
