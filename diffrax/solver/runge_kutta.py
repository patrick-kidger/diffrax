import abc
from dataclasses import dataclass
from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as np

from ..custom_types import Array, DenseInfo, PyTree, Scalar
from ..misc import frozenndarray
from ..term import AbstractTerm, WrapTerm
from .base import AbstractSolver


# The entries are frozen so that they can be hashed, which is needed because the whole
# ButcherTableau will be treated as a static_argnum and therefore used as a dictionary
# key.
@dataclass(frozen=True)
class ButcherTableau:
    alpha: frozenndarray
    beta: Tuple[frozenndarray]
    c_sol: frozenndarray
    c_error: frozenndarray

    def __post_init__(self):
        alpha = np.asarray(self.alpha)
        beta = tuple(np.asarray(beta_i) for beta_i in self.beta)
        c_sol = np.asarray(self.c_sol)
        c_error = np.asarray(self.c_error)
        assert alpha.ndim == 1
        for beta_i in beta:
            assert beta_i.ndim == 1
        assert c_sol.ndim == 1
        assert c_error.ndim == 1
        assert alpha.shape[0] == len(beta_i)
        assert all(i + 1 == beta_i.shape[0] for i, beta_i in enumerate(beta))
        assert alpha.shape[0] + 1 == c_sol.shape[0]
        assert alpha.shape[0] + 1 == c_error.shape[0]


_SolverState = Dict[str, Array]


class RungeKutta(AbstractSolver):
    term: AbstractTerm

    @property
    @abc.abstractmethod
    def tableau(self) -> ButcherTableau:
        pass

    def wrap(self, t0: Scalar, y0: PyTree, args: PyTree):
        return type(self)(term=WrapTerm(term=self.term, t=t0, y=y0, args=args))

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
        return dict(f0=f0, dt=dt)

    def step(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
        solver_state: _SolverState,
    ) -> Tuple[Array["state"], Array["state"], DenseInfo, _SolverState]:  # noqa: F821
        # Convert from frozenarray to array
        # Operations (+,*,@ etc.) aren't defined for frozenarray
        alpha = np.asarray(self.tableau.alpha)
        beta = [np.asarray(beta_i) for beta_i in self.tableau.beta]
        c_sol = np.asarray(self.tableau.c_sol)
        c_error = np.asarray(self.tableau.c_error)

        control = self.term.contr(t0, t1)
        f0 = solver_state["f0"]
        prev_dt = solver_state["dt"]
        dt = t1 - t0

        # Note that our `k` is (for an ODE) `dt` times smaller than the usual
        # implementation (e.g. what you see in torchdiffeq or in the reference texts).
        # This is because of our vector-field-control approach.
        k = jnp.empty(
            (len(alpha) + 1,) + y0.shape
        )  # y0.shape is actually single-dimensional
        k = k.at[0].set(f0 * (dt / prev_dt))

        # lax.fori_loop is not reverse differentiable
        # Since we're JITing I'm not sure it'd necessarily be faster anyway.
        for i, (alpha_i, beta_i) in enumerate(zip(alpha, beta)):
            if alpha_i == 1:
                # No floating point error
                ti = t1
            else:
                ti = t0 + alpha_i * dt
            yi = y0 + beta_i @ k[: i + 1]
            fi = self.term.vf_prod(ti, yi, args, control)
            k = k.at[i + 1].set(fi)

        if not (c_sol[-1] == 0 and (c_sol[:-1] == beta[-1]).all()):
            yi = y0 + c_sol @ k

        y1 = yi
        f1 = k[-1]
        y_error = c_error @ k
        dense_info = {"y0": y0, "y1": y1, "k": k}
        return y1, y_error, dense_info, dict(f0=f1, dt=dt)

    def func_for_init(
        self,
        t0: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
    ) -> Array["state"]:  # noqa: F821
        return self.term.func_for_init(t0, y0, args)
