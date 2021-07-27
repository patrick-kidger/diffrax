from dataclasses import dataclass
import jax.numpy as jnp
import numpy as np
from typing import Any, Tuple

from ..autojit import autojit
from ..custom_types import Array, PyTree, Scalar, SquashTreeDef
from ..misc import frozenndarray
from ..term import AbstractTerm
from ..tree import tree_dataclass
from .base import AbstractSolver


# Not a tree_dataclass as we want to compile against the values of alpha, beta etc.
@dataclass(frozen=True)
class ButcherTableau:
    alpha: frozenndarray
    beta: tuple[frozenndarray]
    c_sol: frozenndarray
    c_error: frozenndarray


@tree_dataclass
class RungeKutta(AbstractSolver):
    terms: list[AbstractTerm]
    tableau: ButcherTableau

    def init(self, y_treedef: SquashTreeDef, t0: Scalar, t1: Scalar, y0: Array["state"], args: PyTree):  # noqa: F821
        f0s = []
        for term in self.terms:
            control_, control_treedef = term.contr_(t0, t1)
            f0 = term.vf_prod_(y_treedef, control_treedef, t0, y0, args, control_)
            f0s.append(f0)
        return [jnp.zeros(y0.shape) for _ in range(len(self.terms))], f0s

    @autojit
    def step(
        self,
        y_treedef: SquashTreeDef,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
        solver_state,
    ) -> Tuple[Array["state"], Any]:  # noqa: F821
        # Convert from frozenarray to array
        # Operations (+,*,@ etc.) aren't defined for frozenarray
        alpha = np.asarray(self.tableau.alpha)
        beta = [np.asarray(beta_i) for beta_i in self.tableau.beta]
        c_sol = np.asarray(self.tableau.c_sol)
        c_error = np.asarray(self.tableau.c_error)

        controls_ = []
        control_treedefs = []
        ks = []
        _, f0s = solver_state
        for term, f0 in zip(self.terms, f0s):
            control_, control_treedef = term.contr_(t0, t1)
            k = jnp.empty(y0.shape + (len(alpha) + 1,))
            k = k.at[..., 0].set(f0)
            controls_.append(control_)
            control_treedefs.append(control_treedef)
            ks.append(k)
        dt = t1 - t0
        # lax.fori_loop is not reverse differentiable
        # Since we're JITing I'm not sure it'd necessarily be faster anyway.
        for tableau_index, (alpha_i, beta_i) in enumerate(zip(alpha, beta)):
            if alpha_i == 1:
                # No floating point error
                ti = t1
            else:
                ti = t0 + alpha_i * dt
            yi = y0
            for k in ks:
                yi = yi + k[..., :tableau_index + 1] @ (beta_i * dt)
            for term_index, (term, control_, control_treedef, k) in enumerate(zip(self.terms,
                                                                                  controls_,
                                                                                  control_treedefs,
                                                                                  ks)):
                fi = term.vf_prod_(y_treedef, control_treedef, ti, yi, args, control_)
                k = k.at[..., tableau_index + 1].set(fi)
                ks[term_index] = k

        if not (c_sol[-1] == 0 and (c_sol[:-1] == beta[-1]).all()):
            yi = y0
            for k in ks:
                yi = yi + k @ (c_sol * dt)

        y1 = yi
        f1s = [k[..., -1] for k in ks]
        y1_error = [k @ (c_error * dt) for k in ks]
        solver_state = (y1_error, f1s)
        return y1, solver_state
