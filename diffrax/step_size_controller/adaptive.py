import jax.lax as lax
import jax.numpy as jnp
from typing import Callable, Optional, Tuple

from ..custom_types import Array, PyTree, Scalar, SquashTreeDef
from ..misc import tree_squash, tree_unsquash
from ..solution import RESULTS
from .base import AbstractStepSizeController


def _rms_norm(x: PyTree) -> Scalar:
    x, _ = tree_squash(x)
    return jnp.sqrt(jnp.mean(x**2))


# Empirical initial step selection algorithm from:
# E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential Equations I: Nonstiff Problems", Sec. II.4,
# 2nd edition.
def _select_initial_step(
    t0: Scalar,
    y0: Array["state"],  # noqa: F821
    args: PyTree,
    y_treedef: SquashTreeDef,
    solver_order: int,
    func_for_init: Callable[[SquashTreeDef, Scalar, Array["state"], PyTree], Array["state"]],  # noqa: F821
    rtol: Scalar,
    atol: Scalar,
    norm: Callable[[Array], Scalar]
):
    f0 = func_for_init(y_treedef, t0, y0, args)
    scale = atol + jnp.abs(y0) * rtol
    d0 = norm(y0 / scale)
    d1 = norm(f0 / scale)

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * (d0 / d1)

    t1 = t0 + h0
    y1 = y0 + h0 * f0
    f1 = func_for_init(y_treedef, t1, y1, args)
    d2 = norm((f1 - f0) / scale) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = jnp.maximum(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 * jnp.maximum(d1, d2))**(1 / solver_order)

    return jnp.minimum(100 * h0, h1)


def _scale_error_estimate(
    y_error: Array["state"],  # noqa: F821
    y0: Array["state"],  # noqa: F821
    y1_candidate: Array["state"],  # noqa: F821
    y_treedef: SquashTreeDef,
    rtol: Scalar,
    atol: Scalar,
    norm: Callable[[Array], Scalar]
) -> Scalar:
    scale = y_error / (atol + jnp.maximum(y0, y1_candidate) * rtol)
    scale = tree_unsquash(y_treedef, scale)
    return norm(scale)


# https://diffeq.sciml.ai/stable/extras/timestepping/
# are good notes on different step size control algorithms.
class IController(AbstractStepSizeController):
    # Default tolerances taken from scipy.integrate.solve_ivp
    rtol: Scalar = 1e-3
    atol: Scalar = 1e-6
    safety: Scalar = 0.9
    ifactor: Scalar = 10.0
    dfactor: Scalar = 0.2
    norm: Callable = _rms_norm
    dtmin: Optional[Scalar] = None
    dtmax: Optional[Scalar] = None
    force_dtimin: bool = True

    requested_state = frozenset({"y_error"})

    def init(
        self,
        t0: Scalar,
        y0: Array["state"],  # noqa: F821
        dt0: Optional[Scalar],
        args: PyTree,
        y_treedef: SquashTreeDef,
        solver_order: int,
        func_for_init: Callable[[SquashTreeDef, Scalar, Array["state"], PyTree], Array["state"]],  # noqa: F821
    ) -> Tuple[Scalar, None]:
        if dt0 is None:
            dt0 = _select_initial_step(
                t0, y0, args, y_treedef, solver_order, func_for_init, self.rtol, self.atol, self.norm
            )
        return t0 + dt0, None

    def adapt_step_size(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        y1_candidate: Array["state"],  # noqa: F821
        args: PyTree,
        y_error: Optional[Array["state"]],  # noqa: F821
        y_treedef: SquashTreeDef,
        solver_order: int,
        controller_state: None
    ) -> Tuple[bool, Scalar, Scalar, None, int]:
        del args, controller_state
        if y_error is None:
            raise ValueError("Cannot use adaptive step sizes with a solver that does not provide error estimates.")
        prev_dt = t1 - t0

        scaled_error = _scale_error_estimate(y_error, y0, y1_candidate, y_treedef, self.rtol, self.atol, self.norm)
        keep_step = scaled_error < 1
        factor = lax.cond(
            scaled_error == 0, lambda _: self.ifactor, self._scale_factor, (solver_order, keep_step, scaled_error)
        )
        dt = prev_dt * factor
        results = jnp.full_like(t0, RESULTS.successful)
        if self.dtmin is not None:
            if not self.force_dtmin:
                result = results.at[dt < self.dtmin].set(RESULTS.dt_min_reached)
            dt = jnp.maximum(dt, self.dtmin)

        if self.dtmax is not None:
            dt = jnp.minimum(dt, self.dtmax)

        return keep_step, t1, t1 + dt, None, result

    def _scale_factor(self, operand):
        order, keep_step, scaled_error = operand
        dfactor = jnp.where(keep_step, 1, self.dfactor)
        exponent = 1 / order
        return jnp.clip(self.safety / scaled_error**exponent, a_min=dfactor, a_max=self.ifactor)
