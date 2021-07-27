import abc
import jax.lax as lax
import jax.numpy as jnp
from typing import Callable, Optional, Tuple, TypeVar

from .custom_types import Array, PyTree, Scalar, SquashTreeDef
from .tree import tree_dataclass


T = TypeVar('T', bound=PyTree)
T2 = TypeVar('T2', bound=PyTree)


@tree_dataclass
class AbstractStepSizeController(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def init(
        self,
        func: Callable[[SquashTreeDef, Scalar, Array["state"], PyTree], Array["state"]],  # noqa: F821
        y_treedef: SquashTreeDef,
        t0: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
        dt0: Optional[Scalar],
        solver_order: int
    ) -> Tuple[Scalar, T]:
        pass

    @abc.abstractmethod
    def adapt_step_size(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state":...],  # noqa: F821
        y1_candidate: Array["state":...],  # noqa: F821
        solver_state0: T2,
        solver_state1_candidate: T2,
        solver_order: int,
        controller_state: T
    ) -> Tuple[bool, Scalar, Scalar, T]:
        pass


@tree_dataclass
class ConstantStepSize(AbstractStepSizeController):
    def init(
        self,
        func: Callable[[SquashTreeDef, Scalar, Array["state"], PyTree], Array["state"]],  # noqa: F821
        y_treedef: SquashTreeDef,
        t0: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
        dt0: Optional[Scalar],
        solver_order: int
    ) -> Tuple[Scalar, Scalar]:
        del func, y_treedef, y0, args, solver_order
        assert dt0 is not None
        return t0 + dt0, dt0

    def adapt_step_size(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state":...],  # noqa: F821
        y1_candidate: Array["state":...],  # noqa: F821
        solver_state0: T2,
        solver_state1_candidate: T2,
        solver_order: int,
        controller_state: Scalar
    ) -> Tuple[bool, Scalar, Scalar, Scalar]:
        del t0, y0, solver_state0, solver_state1_candidate, solver_order
        return True, t1, t1 + controller_state, controller_state


def _rms_norm(x: Array) -> Scalar:
    return jnp.sqrt(jnp.mean(x**2))


# Empirical initial step selection algorithm from:
# E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential Equations I: Nonstiff Problems", Sec. II.4,
# 2nd edition.
def _select_initial_step(
    func_for_init: Callable[[SquashTreeDef, Scalar, Array["state"], PyTree], Array["state"]],  # noqa: F821
    y_treedef: SquashTreeDef,
    t0: Scalar,
    y0: Array["state"],  # noqa: F821
    args: PyTree,
    solver_order: int,
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
    rtol: Scalar,
    atol: Scalar,
    norm: Callable[[Array], Scalar]
) -> Scalar:
    return norm(y_error / (atol + jnp.maximum(y0, y1_candidate) * rtol))


# https://diffeq.sciml.ai/stable/extras/timestepping/
# are good notes on different step size control algorithms.
@tree_dataclass
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

    def init(
        self,
        func_for_init: Callable[[SquashTreeDef, Scalar, Array["state"], PyTree], Array["state"]],  # noqa: F821
        y_treedef: SquashTreeDef,
        t0: Scalar,
        y0: Array["state"],  # noqa: F821
        dt0: Optional[Scalar],
        args: PyTree,
        solver_order: int
    ) -> Tuple[Scalar, None]:
        if dt0 is None:
            dt0 = _select_initial_step(
                func_for_init, y_treedef, t0, y0, args, solver_order, self.rtol, self.atol, self.norm
            )
        return dt0, None

    def adapt_step_size(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state":...],  # noqa: F821
        y1_candidate: Array["state":...],  # noqa: F821
        solver_state0: T2,
        solver_state1_candidate: T2,
        solver_order: int,
        controller_state: None
    ) -> Tuple[bool, Scalar, Scalar, None]:
        del solver_state0, controller_state
        prev_dt = t1 - t0
        y_error = solver_state1_candidate.y_error

        scaled_error = _scale_error_estimate(y_error, y0, y1_candidate, self.rtol, self.atol, self.norm)
        keep_step = scaled_error < 1
        factor = lax.cond(
            scaled_error == 0, lambda _: self.ifactor, self._scale_factor, (solver_order, keep_step, scaled_error)
        )
        dt = prev_dt * factor
        if self.dtmin is not None:
            dt = jnp.maximum(dt, self.dtmin)
        if self.dtmax is not None:
            dt = jnp.minimum(dt, self.dtmax)

        return keep_step, t1, t1 + dt, None

    def _scale_factor(self, operand):
        order, keep_step, scaled_error = operand
        dfactor = jnp.where(keep_step, 1, self.dfactor)
        exponent = 1 / order
        return jnp.minimum(self.ifactor, jnp.maximum(self.safety / scaled_error * exponent, dfactor))
