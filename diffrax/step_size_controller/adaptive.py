import typing
from typing import Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp

from ..custom_types import Array, Bool, PyTree, Scalar
from ..misc import nextafter, nextbefore, rms_norm, ω
from ..solution import RESULTS
from ..solver import AbstractImplicitSolver, AbstractSolver
from ..term import AbstractTerm
from .base import AbstractStepSizeController


# Empirical initial step selection algorithm from:
# E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential Equations I:
# Nonstiff Problems", Sec. II.4, 2nd edition.
def _select_initial_step(
    terms: PyTree[AbstractTerm],
    t0: Scalar,
    y0: PyTree,
    args: PyTree,
    solver: AbstractSolver,
    rtol: Scalar,
    atol: Scalar,
    norm: Callable[[Array], Scalar],
) -> Scalar:
    f0 = solver.func_for_init(terms, t0, y0, args)
    scale = (atol + ω(y0).call(jnp.abs) * rtol).ω
    d0 = norm((y0 ** ω / scale ** ω).ω)
    d1 = norm((f0 ** ω / scale ** ω).ω)

    _cond = (d0 < 1e-5) | (d1 < 1e-5)
    _d1 = jnp.where(_cond, 1, d1)
    h0 = jnp.where(_cond, 1e-6, 0.01 * (d0 / _d1))

    t1 = t0 + h0
    y1 = (y0 ** ω + h0 * f0 ** ω).ω
    f1 = solver.func_for_init(terms, t1, y1, args)
    d2 = norm(((f1 ** ω - f0 ** ω) / scale ** ω).ω) / h0

    max_d = jnp.maximum(d1, d2)
    h1 = jnp.where(
        max_d <= 1e-15,
        jnp.maximum(1e-6, h0 * 1e-3),
        (0.01 / max_d) ** (1 / (solver.order + 1)),
    )

    return jnp.minimum(100 * h0, h1)


def _scale_error_estimate(
    y_error: PyTree,
    y0: PyTree,
    y1_candidate: PyTree,
    rtol: Scalar,
    atol: Scalar,
    norm: Callable[[Array], Scalar],
) -> Scalar:
    def _calc(_y0, _y1_candidate, _y_error):
        return _y_error / (
            atol + jnp.maximum(jnp.abs(_y0), jnp.abs(_y1_candidate)) * rtol
        )

    scale = jax.tree_map(_calc, y0, y1_candidate, y_error)
    return norm(scale)


_IControllerState = Tuple[Bool, Bool]


_gendocs = getattr(typing, "GENERATING_DOCUMENTATION", False)


class _gendocs_norm:
    def __repr__(self):
        return str(rms_norm)


class AbstractAdaptiveStepSizeController(AbstractStepSizeController):
    # Default tolerances taken from scipy.integrate.solve_ivp
    rtol: Scalar = 1e-3
    atol: Scalar = 1e-6

    def wrap_solver(self, solver: AbstractSolver) -> AbstractSolver:
        # Poor man's multiple dispatch
        if isinstance(solver, AbstractImplicitSolver):
            if solver.nonlinear_solver.rtol is None:
                solver = eqx.tree_at(
                    lambda s: s.nonlinear_solver.rtol,
                    solver,
                    self.rtol,
                    is_leaf=lambda x: x is None,
                )
            if solver.nonlinear_solver.atol is None:
                solver = eqx.tree_at(
                    lambda s: s.nonlinear_solver.atol,
                    solver,
                    self.atol,
                    is_leaf=lambda x: x is None,
                )
        return solver


# https://diffeq.sciml.ai/stable/extras/timestepping/
# are good notes on different step size control algorithms.
class IController(AbstractAdaptiveStepSizeController):
    """Adapts the step size to produce a solution accurate to a given tolerance.
    The tolerance is calculated as `atol + rtol * y` for the evolving solution `y`.

    Steps are adapted using an I-controller.
    """

    dtmin: Optional[Scalar] = None
    dtmax: Optional[Scalar] = None
    force_dtmin: bool = True
    step_ts: Optional[Array["steps"]] = None  # noqa: F821
    jump_ts: Optional[Array["jumps"]] = None  # noqa: F821
    dfactor: Scalar = 0.2
    ifactor: Scalar = 10.0
    # The documentation treats callables as methods and displays `norm` twice: as both
    # an attribute and a method.
    norm: Callable[[PyTree], Scalar] = _gendocs_norm() if _gendocs else rms_norm
    safety: Scalar = 0.9

    def wrap(self, direction: Scalar):
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
            step_ts=None if self.step_ts is None else self.step_ts * direction,
            jump_ts=None if self.jump_ts is None else self.jump_ts * direction,
        )

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        dt0: Optional[Scalar],
        args: PyTree,
        solver: AbstractSolver,
    ) -> Tuple[Scalar, _IControllerState]:
        del t1
        if dt0 is None:
            dt0 = _select_initial_step(
                terms,
                t0,
                y0,
                args,
                solver,
                self.rtol,
                self.atol,
                self.norm,
            )
            # So this stop_gradient is a choice I'm not 100% convinced by.
            #
            # (Note that we also do something similar lower down, by stopping the
            # gradient through the multiplicative factor updating the step size, and
            # the following discussion is in reference to them both, collectively.)
            #
            # - This dramatically speeds up gradient computations. e.g. at time of
            #   writing, the neural ODE example goes from 0.3 seconds/iteration down to
            #   0.1 seconds/iteration.
            # - On some problems this actually improves training behaviour. e.g. at
            #   time of writing, the neural CDE example fails to train if these
            #   stop_gradients are removed.
            # - I've never observed this hurting training behaviour.
            # - Other libraries (notably torchdiffeq) do this by default without
            #   remark. The idea is that "morally speaking" the time discretisation
            #   shouldn't really matter, it's just some minor implementation detail of
            #   the ODE solve. (e.g. part of the folklore of neural ODEs is that "you
            #   don't need to backpropagate through rejected steps".)
            #
            # However:
            # - This feels morally wrong from the point of view of differentiable
            #   programming.
            # - That "you don't need to backpropagate through rejected steps" feels a
            #   bit questionable. They _are_ part of the computational graph and do
            #   have a subtle effect on the choice of step size, and the choice of step
            #   step size does have a not-so-subtle effect on the solution computed.
            # - This does mean that certain esoteric optimisation criteria, like
            #   optimising wrt parameters of the adaptive step size controller itself,
            #   might fail?
            # - It's entirely opaque why these stop_gradients should either improve the
            #   speed of backpropagation, or why they should improve training behavior.
            #
            # I would welcome your thoughts, dear reader, if you have any insight!
            dt0 = lax.stop_gradient(dt0)
        if self.dtmax is not None:
            dt0 = jnp.minimum(dt0, self.dtmax)
        if self.dtmin is None:
            at_dtmin = jnp.array(False)
        else:
            at_dtmin = dt0 <= self.dtmin
            dt0 = jnp.maximum(dt0, self.dtmin)

        t1 = self._clip_step_ts(t0, t0 + dt0)
        t1, jump_next_step = self._clip_jump_ts(t0, t1)

        return t1, (jump_next_step, at_dtmin)

    def adapt_step_size(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        y1_candidate: PyTree,
        args: PyTree,
        y_error: Optional[PyTree],
        solver_order: int,
        controller_state: _IControllerState,
    ) -> Tuple[Bool, Scalar, Scalar, Bool, _IControllerState, RESULTS]:
        del args
        if y_error is None and y0 is not None:
            # y0 is not None check is included to handle the edge case that the state
            # is just a trivial `None` PyTree. In this case `y_error` has the same
            # PyTree structure and thus overlaps with our special usage of `None` to
            # indicate a lack of error estimate.
            raise RuntimeError(
                "Cannot use adaptive step sizes with a solver that does not provide "
                "error estimates."
            )
        prev_dt = t1 - t0
        made_jump, at_dtmin = controller_state

        #
        # Figure out how things went on the last step: error, and whether to
        # accept/reject it.
        #

        scaled_error = _scale_error_estimate(
            y_error, y0, y1_candidate, self.rtol, self.atol, self.norm
        )
        keep_step = scaled_error < 1
        if self.dtmin is not None:
            keep_step = keep_step | at_dtmin

        #
        # Adjust next step size
        #

        # Double-where trick to avoid NaN gradients.
        # See JAX issues #5039 and #1052.
        #
        # (Although we've actually since added a stop_gradient afterwards, this is kept
        # for completeness, e.g. just in case we ever remove the stop_gradient.)
        cond = scaled_error == 0
        _scaled_error = jnp.where(cond, 1.0, scaled_error)
        factor = lax.cond(
            cond,
            lambda _: self.ifactor,
            self._scale_factor,
            (solver_order, keep_step, _scaled_error),
        )
        factor = lax.stop_gradient(factor)  # See note in init above.
        dt = prev_dt * factor

        #
        # Clip next step size based on dtmin/dtmax
        #

        result = jnp.full_like(t0, RESULTS.successful, dtype=int)
        if self.dtmax is not None:
            dt = jnp.minimum(dt, self.dtmax)
        if self.dtmin is None:
            at_dtmin = jnp.array(False)
        else:
            if not self.force_dtmin:
                result = jnp.where(dt < self.dtmin, RESULTS.dt_min_reached, result)
            at_dtmin = dt <= self.dtmin
            dt = jnp.maximum(dt, self.dtmin)

        #
        # Clip next step size based on step_ts/jump_ts
        #

        if jnp.issubdtype(t1.dtype, jnp.inexact):
            _t1 = jnp.where(made_jump, nextafter(t1), t1)
        else:
            _t1 = t1
        next_t0 = jnp.where(keep_step, _t1, t0)
        next_t1 = self._clip_step_ts(next_t0, next_t0 + dt)
        next_t1, next_made_jump = self._clip_jump_ts(next_t0, next_t1)

        controller_state = (next_made_jump, at_dtmin)
        return keep_step, next_t0, next_t1, made_jump, controller_state, result

    def _scale_factor(self, operand):
        order, keep_step, scaled_error = operand
        dfactor = jnp.where(keep_step, 1, self.dfactor)
        exponent = 1 / (order + 1)  # +1 to convert from global error to local error
        return jnp.clip(
            self.safety / scaled_error ** exponent, a_min=dfactor, a_max=self.ifactor
        )

    def _clip_step_ts(self, t0: Scalar, t1: Scalar) -> Scalar:
        if self.step_ts is None:
            return t1

        # TODO: it should be possible to switch this O(nlogn) for just O(n) by keeping
        # track of where we were last, and using that as a hint for the next search.
        t0_index = jnp.searchsorted(self.step_ts, t0)
        t1_index = jnp.searchsorted(self.step_ts, t1)
        # This minimum may or may not actually be necessary. The left branch is taken
        # iff t0_index < t1_index <= len(self.step_ts), so all valid t0_index s must
        # already satisfy the minimum.
        # However, that branch is actually executed unconditionally and then where'd,
        # so we clamp it just to be sure we're not hitting undefined behaviour.
        t1 = jnp.where(
            t0_index < t1_index,
            self.step_ts[jnp.minimum(t0_index, len(self.step_ts) - 1)],
            t1,
        )
        return t1

    def _clip_jump_ts(self, t0: Scalar, t1: Scalar) -> Tuple[Scalar, Array[(), bool]]:
        if self.jump_ts is None:
            return t1, jnp.full_like(t1, fill_value=False, dtype=bool)
        if self.jump_ts is not None and not jnp.issubdtype(
            self.jump_ts.dtype, jnp.inexact
        ):
            raise ValueError(
                f"jump_ts must be floating point, not {self.jump_ts.dtype}"
            )
        if not jnp.issubdtype(t1.dtype, jnp.inexact):
            raise ValueError(
                "t0, t1, dt0 must be floating point when specifying jump_t. Got "
                f"{t1.dtype}."
            )
        t0_index = jnp.searchsorted(self.step_ts, t0)
        t1_index = jnp.searchsorted(self.step_ts, t1)
        cond = t0_index < t1_index
        t1 = jnp.where(
            cond,
            nextbefore(self.jump_ts[jnp.minimum(t0_index, len(self.step_ts) - 1)]),
            t1,
        )
        next_made_jump = jnp.where(cond, True, False)
        return t1, next_made_jump


IController.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance.
- `atol`: Absolute tolerance.
- `dtmin`: Minimum step size. The step size is either clipped to this value, or an
    error raised if the step size decreases below this, depending on `force_dtmin`.
- `dtmax`: Maximum step size; the step size is clipped to this value.
- `force_dtmin`: How to handle the step size hitting the minimum. If `True` then the
    step size is clipped to `dtmin`. If `False` then the step fails and the integration
    errors. (Which will in turn either sets an error flag, or raises an exception,
    depending on the `throw` value for `diffeqsolve(..., throw=...).)
- `step_ts`: Denotes *extra* times that must be stepped to. This can be used to help
    deal with a vector field that has a known derivative discontinuity, by stepping
    to exactly the derivative discontinuity.
- `jump_ts`: Denotes times at which the vector field has a known discontinuity. This
    can be used to step exactly to the discontinuity. (And any other appropriate action
    taken, like FSAL algorithms re-evaluating the vector field.)
- `dfactor`: Minimum amount a step size can be decreased relative to the previous step.
- `ifactor`: Maximum amount a step size can be increased relative to the previous step.
- `norm`: A function `PyTree -> Scalar` used in the error control. Precisely, step
    sizes are chosen so that `norm(error / (atol + rtol * y))` is approximately
    one.
- `safety`: Multiplicative safety factor.
"""
