from dataclasses import field
from typing import Callable, Optional, Tuple

import jax.lax as lax
import jax.numpy as jnp

from ..custom_types import Array, PyTree, Scalar
from ..misc import ravel_pytree
from ..solution import RESULTS
from .base import AbstractStepSizeController


def _rms_norm(x: PyTree) -> Scalar:
    x, _ = ravel_pytree(x)
    if x.size == 0:
        return 0
    return jnp.sqrt(jnp.mean(x ** 2))


# Empirical initial step selection algorithm from:
# E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential Equations I:
# Nonstiff Problems", Sec. II.4, 2nd edition.
def _select_initial_step(
    t0: Scalar,
    y0: Array["state"],  # noqa: F821
    args: PyTree,
    solver_order: int,
    func_for_init: Callable[
        [Scalar, Array["state"], PyTree], Array["state"]  # noqa: F821
    ],
    unravel_y: callable,
    rtol: Scalar,
    atol: Scalar,
    norm: Callable[[Array], Scalar],
):
    f0 = func_for_init(t0, y0, args)
    scale = atol + jnp.abs(y0) * rtol
    d0 = norm(unravel_y(y0 / scale))
    d1 = norm(unravel_y(f0 / scale))

    h0 = jnp.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * (d0 / d1))

    t1 = t0 + h0
    y1 = y0 + h0 * f0
    f1 = func_for_init(t1, y1, args)
    d2 = norm(unravel_y((f1 - f0) / scale)) / h0

    h1 = jnp.where(
        (d1 <= 1e-15) | (d2 <= 1e-15),
        jnp.maximum(1e-6, h0 * 1e-3),
        (0.01 * jnp.maximum(d1, d2)) ** (1 / solver_order),
    )

    return jnp.minimum(100 * h0, h1)


def _scale_error_estimate(
    y_error: Array["state"],  # noqa: F821
    y0: Array["state"],  # noqa: F821
    y1_candidate: Array["state"],  # noqa: F821
    unravel_y: callable,
    rtol: Scalar,
    atol: Scalar,
    norm: Callable[[Array], Scalar],
) -> Scalar:
    scale = y_error / (atol + jnp.maximum(y0, y1_candidate) * rtol)
    scale = unravel_y(scale)
    return norm(scale)


DO_NOT_SET = object()  # Is set during wrap instead


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
    force_dtmin: bool = True
    unravel_y: callable = field(repr=False, default=DO_NOT_SET)
    direction: Scalar = field(repr=False, default=DO_NOT_SET)

    def wrap(self, unravel_y: callable, direction: Scalar):
        return type(self)(
            rtol=self.rtol,
            atol=self.atol,
            safety=self.safety,
            ifactor=self.ifactor,
            dfactor=self.dfactor,
            norm=self.norm,
            dtmin=self.dtmin,
            dtmax=self.dtmax,
            force_dtmin=self.force_dtmin,
            unravel_y=unravel_y,
            direction=direction,
        )

    def init(
        self,
        t0: Scalar,
        y0: Array["state"],  # noqa: F821
        dt0: Optional[Scalar],
        args: PyTree,
        solver_order: int,
        func_for_init: Callable[
            [Scalar, Array["state"], PyTree],  # noqa: F821
            Array["state"],  # noqa: F821
        ],
    ) -> Tuple[Scalar, None]:
        if dt0 is None:
            dt0 = _select_initial_step(
                t0,
                y0,
                args,
                solver_order,
                func_for_init,
                self.unravel_y,
                self.rtol,
                self.atol,
                self.norm,
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
        solver_order: int,
        controller_state: None,
    ) -> Tuple[bool, Scalar, Scalar, None, int]:
        del args, controller_state
        if y_error is None:
            raise ValueError(
                "Cannot use adaptive step sizes with a solver that does not provide "
                "error estimates."
            )
        prev_dt = t1 - t0

        scaled_error = _scale_error_estimate(
            y_error, y0, y1_candidate, self.unravel_y, self.rtol, self.atol, self.norm
        )
        keep_step = scaled_error < 1
        factor = lax.cond(
            scaled_error == 0,
            lambda _: self.ifactor,
            self._scale_factor,
            (solver_order, keep_step, scaled_error),
        )
        dt = prev_dt * factor
        result = jnp.full_like(t0, RESULTS.successful)
        if self.dtmin is not None:
            if not self.force_dtmin:
                result = result.at[dt < self.dtmin].set(RESULTS.dt_min_reached)
            dt = jnp.maximum(dt, self.dtmin)

        if self.dtmax is not None:
            dt = jnp.minimum(dt, self.dtmax)

        next_t0 = jnp.where(keep_step, t1, t0)

        return keep_step, next_t0, next_t0 + dt, None, result

    def _scale_factor(self, operand):
        order, keep_step, scaled_error = operand
        dfactor = jnp.where(keep_step, 1, self.dfactor)
        exponent = 1 / order
        return jnp.clip(
            self.safety / scaled_error ** exponent, a_min=dfactor, a_max=self.ifactor
        )
