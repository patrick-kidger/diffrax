from dataclasses import dataclass, field
import jax.numpy as jnp
import numpy as np
from typing import Tuple

from ..custom_types import Array, PyTree, Scalar, SquashTreeDef
from ..interpolation import FourthOrderPolynomialInterpolation
from ..misc import frozenndarray
from ..term import AbstractTerm
from ..tree import tree_dataclass
from .base import AbstractSolver, AbstractSolverState


# Not a tree_dataclass as we want to compile against the values of alpha, beta etc.
@dataclass(frozen=True)
class ButcherTableau:
    alpha: frozenndarray
    beta: tuple[frozenndarray]
    c_sol: frozenndarray
    c_error: frozenndarray
    order: int = field(init=False)

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
        assert all(alpha.shape[0] == beta_i.shape[0] for beta_i in beta)
        assert alpha.shape[0] + 1 == c_sol.shape[0]
        assert alpha.shape[0] + 1 == c_error.shape[0]
        object.__setattr__(self, "order", len(alpha) + 1)


@tree_dataclass
class RungeKuttaSolverState(AbstractSolverState):
    f0: Array["state"]  # noqa: F821
    dt: Scalar


@tree_dataclass
class RungeKutta(AbstractSolver):
    terms: list[AbstractTerm]
    tableau: ButcherTableau

    @property
    def order(self):
        return self.tableau.order

    @property
    def available_state(self) -> frozenset:
        return frozenset({"y_error", "k"}) | super().available_state

    recommended_interpolation = FourthOrderPolynomialInterpolation

    def init(
        self,
        y_treedef: SquashTreeDef,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
        requested_state: frozenset
    ):  # noqa: F821
        f0 = 0
        for term in self.terms:
            control_, control_treedef = term.contr_(t0, t1)
            f0 = f0 + term.vf_prod_(y_treedef, control_treedef, t0, y0, args, control_)
        dt = t1 - t0
        extras = {}
        if "y_error" in requested_state:
            extras["y_error"] = jnp.zeros(y0.shape)
        if "k" in requested_state:
            extras["k"] = jnp.zeros(y0.shape + (self.tableau.order,))
        return RungeKuttaSolverState(f0=f0, dt=dt, extras=extras)

    def step(
        self,
        y_treedef: SquashTreeDef,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
        solver_state: RungeKuttaSolverState,
        requested_state: frozenset,
    ) -> Tuple[Array["state"], RungeKuttaSolverState]:  # noqa: F821
        # Convert from frozenarray to array
        # Operations (+,*,@ etc.) aren't defined for frozenarray
        alpha = np.asarray(self.tableau.alpha)
        beta = [np.asarray(beta_i) for beta_i in self.tableau.beta]
        c_sol = np.asarray(self.tableau.c_sol)
        c_error = np.asarray(self.tableau.c_error)

        controls_ = []
        control_treedefs = []
        f0 = solver_state.f0
        prev_dt = solver_state.dt
        dt = t1 - t0
        for term in self.terms:
            control_, control_treedef = term.contr_(t0, t1)
            controls_.append(control_)
            control_treedefs.append(control_treedef)

        k = jnp.zeros(y0.shape + (self.tableau.order,))
        k = k.at[..., 0].add(f0 * (dt / prev_dt))

        # lax.fori_loop is not reverse differentiable
        # Since we're JITing I'm not sure it'd necessarily be faster anyway.
        for i, (alpha_i, beta_i) in enumerate(zip(alpha, beta)):
            if alpha_i == 1:
                # No floating point error
                ti = t1
            else:
                ti = t0 + alpha_i * dt
            yi = y0 + k[..., :i + 1] @ beta_i
            for term, control_, control_treedef in zip(self.terms, controls_, control_treedefs):
                fi = term.vf_prod_(y_treedef, control_treedef, ti, yi, args, control_)
                k = k.at[..., i + 1].add(fi)

        if not (c_sol[-1] == 0 and (c_sol[:-1] == beta[-1]).all()):
            yi = y0 + k @ c_sol

        y1 = yi
        f1 = k[..., -1]
        extras = {}
        if "y_error" in requested_state:
            extras["y_error"] = k @ c_error
        if "k" in requested_state:
            extras["k"] = k
        solver_state = RungeKuttaSolverState(f0=f1, dt=dt, extras=extras)
        return y1, solver_state

    def func_for_init(self, y_treedef: SquashTreeDef, t: Scalar, y_: Array["state"],  # noqa: F821
                      args: PyTree) -> Array["state"]:  # noqa: F821
        vf = 0
        for term in self.terms:
            vf = vf + term.func_for_init(y_treedef, t, y_, args)
        return vf
