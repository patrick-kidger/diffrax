import typing
from collections.abc import Callable
from typing import cast, Optional, TYPE_CHECKING, TypeVar

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax.internal as lxi
import optimistix as optx
from jaxtyping import Real


if TYPE_CHECKING:
    from typing import ClassVar as AbstractVar
else:
    from equinox import AbstractVar
from equinox.internal import ω
from jaxtyping import Array, PyTree
from lineax.internal import complex_to_real_dtype

from .._custom_types import (
    Args,
    BoolScalarLike,
    IntScalarLike,
    RealScalarLike,
    VF,
    Y,
)
from .._misc import static_select, upcast_or_raise
from .._solution import RESULTS
from .._term import AbstractTerm, ODETerm
from .base import AbstractStepSizeController


ω = cast(Callable, ω)


def _select_initial_step(
    terms: PyTree[AbstractTerm],
    t0: RealScalarLike,
    y0: Y,
    args: Args,
    func: Callable[
        [PyTree[AbstractTerm], RealScalarLike, Y, Args],
        VF,
    ],
    error_order: RealScalarLike,
    rtol: RealScalarLike,
    atol: RealScalarLike,
    norm: Callable[[PyTree], RealScalarLike],
) -> RealScalarLike:
    # TODO: someone needs to figure out an initial step size algorithm for SDEs.
    if not isinstance(terms, ODETerm):
        return 0.01

    def fn(carry):
        t, y, _h0, _d1, _f, _ = carry
        f = func(terms, t, y, args)
        return t, y, _h0, _d1, _f, f

    def intermediate(carry):
        _, _, _, _, _, f0 = carry
        d0 = norm((y0**ω / scale**ω).ω)
        d1 = norm((f0**ω / scale**ω).ω)
        _cond = (d0 < 1e-5) | (d1 < 1e-5)
        _d1 = jnp.where(_cond, 1, d1)
        h0 = jnp.where(_cond, 1e-6, 0.01 * (d0 / _d1))
        t1 = t0 + h0
        y1 = (y0**ω + h0 * f0**ω).ω
        return t1, y1, h0, d1, f0, f0

    scale = (atol + ω(y0).call(jnp.abs) * rtol).ω
    dummy_h = t0
    dummy_d = eqxi.eval_empty(norm, y0)
    dummy_f = eqxi.eval_empty(lambda: func(terms, t0, y0, args))
    _, _, h0, d1, f0, f1 = eqxi.scan_trick(
        fn, [intermediate], (t0, y0, dummy_h, dummy_d, dummy_f, dummy_f)
    )
    d2 = norm(((f1**ω - f0**ω) / scale**ω).ω) / h0
    max_d = jnp.maximum(d1, d2)
    h1 = jnp.where(
        max_d <= 1e-15,
        jnp.maximum(1e-6, h0 * 1e-3),
        (0.01 / max_d) ** (1 / error_order),
    )
    return jnp.minimum(100 * h0, h1)


_ControllerState = TypeVar("_ControllerState")
_Dt0 = TypeVar("_Dt0", None, RealScalarLike, Optional[RealScalarLike])


class AbstractAdaptiveStepSizeController(
    AbstractStepSizeController[_ControllerState, _Dt0]
):
    """Indicates an adaptive step size controller.

    Accepts tolerances `rtol` and `atol`. When used in conjunction with an implicit
    solver ([`diffrax.AbstractImplicitSolver`][]), then these tolerances will
    automatically be used as the tolerances for the nonlinear solver passed to the
    implicit solver, if they are not specified manually.
    """

    rtol: AbstractVar[RealScalarLike]
    atol: AbstractVar[RealScalarLike]
    norm: AbstractVar[Callable[[PyTree], RealScalarLike]]

    def __check_init__(self):
        if self.rtol is None or self.atol is None:
            raise ValueError(
                "The default values for `rtol` and `atol` were removed in Diffrax "
                "version 0.1.0. (As the choice of tolerance is nearly always "
                "something that you, as an end user, should make an explicit choice "
                "about.)\n"
                "If you want to match the previous defaults then specify "
                "`rtol=1e-3`, `atol=1e-6`. For example:\n"
                "```\n"
                "diffrax.PIDController(rtol=1e-3, atol=1e-6)\n"
                "```\n"
            )


_PidState = tuple[
    BoolScalarLike, BoolScalarLike, RealScalarLike, RealScalarLike, RealScalarLike
]


def _none_or_array(x):
    if x is None:
        return None
    else:
        return jnp.asarray(x)


if TYPE_CHECKING:
    rms_norm = optx.rms_norm
else:
    # We can't use `optx.rms_norm` itself as a default attribute value. This is because
    # it is a callable, and then the doc stack thinks that it is a method.
    if getattr(typing, "GENERATING_DOCUMENTATION", False):

        class _RmsNorm:
            def __repr__(self):
                return "<function rms_norm>"

        old_rms_norm = optx.rms_norm
        rms_norm = _RmsNorm()
    else:
        rms_norm = optx.rms_norm


# https://diffeq.sciml.ai/stable/extras/timestepping/
# are good introductory notes on different step size control algorithms.
# TODO: we don't currently offer a limiter, or a variant accept/reject scheme, as given
#       in Soderlind and Wang 2006.
class PIDController(
    AbstractAdaptiveStepSizeController[_PidState, Optional[RealScalarLike]]
):
    r"""Adapts the step size to produce a solution accurate to a given tolerance.
    The tolerance is calculated as `atol + rtol * y` for the evolving solution `y`.

    Steps are adapted using a PID controller.

    ??? tip "Choosing tolerances"

        The choice of `rtol` and `atol` are used to determine how accurately you would
        like the numerical approximation to your equation.

        Typically this is something you already know; or alternatively something for
        which you try a few different values of `rtol` and `atol` until you are getting
        good enough solutions.

        If you're not sure, then a good default for easy ("non-stiff") problems is
        often something like `rtol=1e-3`, `atol=1e-6`. When more accurate solutions
        are required then something like `rtol=1e-7`, `atol=1e-9` are typical (along
        with using `float64` instead of `float32`).

        (Note that technically speaking, the meaning of `rtol` and `atol` is entirely
        dependent on the choice of `solver`. In practice however, most solvers tend to
        provide similar behaviour for similar values of `rtol`, `atol`. As such it is
        common to refer to solving an equation to specific tolerances, without
        necessarily stating which solver was used.)

        ??? Example

            The choice of `rtol` and `atol` can have a significant impact on the
            accuracy of even simple systems.
            Consider a simple pendulum with a small angle kick:
            ```python
            import diffrax as dfx

            def dynamics(t, y, args):
                dtheta = y["omega"]
                domega = - jnp.sin(y["theta"])
                return dict(theta=dtheta, omega=domega)

            y0 = dict(theta=0.1, omega=0)
            term = dfx.ODETerm(dynamics)
            sol = dfx.diffeqsolve(
                term, solver, t0=0, t1=1000, dt0=0.1, y0,
                saveat=dfx.SaveAts(ts=jnp.linspace(0, 1000, 10000),
                max_steps=2**20,
                stepsize_controller=...
            )
            ```
            to compare the effect of different tolerances:
            ```python
            PID_controller_incorrect = diffrax.PIDController(rtol=1e-3, atol=1e-6)
            PID_controller_correct = diffrax.PIDController(rtol=1e-7, atol=1e-9)
            Constant_controller = diffrax.ConstantStepSize()
            ```
            The phase portraits of the pendulum from the different tolerances clearly
            illustrate the impact of the choice of `rtol` and `atol` on the accuracy of
            the solution.
            ![Phase portrait of pendulum](../imgs/pendulum_adaptive_steps.png)

    ??? tip "Choosing PID coefficients"

        This controller can be reduced to any special case (e.g. just a PI controller,
        or just an I controller) by setting `pcoeff`, `icoeff` or `dcoeff` to zero
        as appropriate.

        For smoothly-varying (i.e. easy to solve) problems then an I controller, or a
        PI controller with `icoeff=1`, will often be most efficient.
        ```python
        PIDController(pcoeff=0,   icoeff=1, dcoeff=0)  # default coefficients
        PIDController(pcoeff=0.4, icoeff=1, dcoeff=0)
        ```

        For moderate difficulty problems that may have an error estimate that does
        not vary smoothly, then a less sensitive controller will often do well. (This
        includes many mildly stiff problems.) Several different coefficients are
        suggested in the literature, e.g.
        ```python
        PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0)
        PIDController(pcoeff=0.3, icoeff=0.3, dcoeff=0)
        PIDController(pcoeff=0.2, icoeff=0.4, dcoeff=0)
        ```

        For SDEs (an extreme example of a problem type that does not have smooth
        behaviour) then an insensitive PI controller is recommended. For example:
        ```python
        PIDController(pcoeff=0.1, icoeff=0.3, dcoeff=0)
        ```

        The best choice is largely empirical, and problem/solver dependent. For most
        moderately difficult ODE problems it is recommended to try tuning these
        coefficients subject to `pcoeff>=0.2`, `icoeff>=0.3`, `pcoeff + icoeff <= 0.7`.
        You can check the number of steps made via:
        ```python
        sol = diffeqsolve(...)
        print(sol.stats["num_steps"])
        ```


    ??? cite "References"

        Both the initial step size selection algorithm for ODEs, and the use of
        an I controller for ODEs, are from Section II.4 of:

        ```bibtex
        @book{hairer2008solving-i,
          address={Berlin},
          author={Hairer, E. and N{\o}rsett, S.P. and Wanner, G.},
          edition={Second Revised Edition},
          publisher={Springer},
          title={{S}olving {O}rdinary {D}ifferential {E}quations {I} {N}onstiff
                 {P}roblems},
          year={2008}
        }
        ```

        The use of a PI controller for ODEs are from Section IV.2 of:

        ```bibtex
        @book{hairer2002solving-ii,
          address={Berlin},
          author={Hairer, E. and Wanner, G.},
          edition={Second Revised Edition},
          publisher={Springer},
          title={{S}olving {O}rdinary {D}ifferential {E}quations {II} {S}tiff and
                 {D}ifferential-{A}lgebraic {P}roblems},
          year={2002}
        }
        ```

        and Sections 1--3 of:

        ```bibtex
        @article{soderlind2002automatic,
            title={Automatic control and adaptive time-stepping},
            author={Gustaf S{\"o}derlind},
            year={2002},
            journal={Numerical Algorithms},
            volume={31},
            pages={281--310}
        }
        ```

        The use of PID controllers are from:

        ```bibtex
        @article{soderlind2003digital,
            title={{D}igital {F}ilters in {A}daptive {T}ime-{S}tepping,
            author={Gustaf S{\"o}derlind},
            year={2003},
            journal={ACM Transactions on Mathematical Software},
            volume={20},
            number={1},
            pages={1--26}
        }
        ```

        The use of PI and PID controllers for SDEs are from:

        ```bibtex
        @article{burrage2004adaptive,
          title={Adaptive stepsize based on control theory for stochastic
                 differential equations},
          journal={Journal of Computational and Applied Mathematics},
          volume={170},
          number={2},
          pages={317--336},
          year={2004},
          doi={https://doi.org/10.1016/j.cam.2004.01.027},
          author={P.M. Burrage and R. Herdiana and K. Burrage},
        }

        @article{ilie2015adaptive,
          author={Ilie, Silvana and Jackson, Kenneth R. and Enright, Wayne H.},
          title={{A}daptive {T}ime-{S}tepping for the {S}trong {N}umerical {S}olution
                 of {S}tochastic {D}ifferential {E}quations},
          year={2015},
          publisher={Springer-Verlag},
          address={Berlin, Heidelberg},
          volume={68},
          number={4},
          doi={https://doi.org/10.1007/s11075-014-9872-6},
          journal={Numer. Algorithms},
          pages={791–-812},
        }
        ```
    """

    rtol: RealScalarLike
    atol: RealScalarLike
    pcoeff: RealScalarLike = 0
    icoeff: RealScalarLike = 1
    dcoeff: RealScalarLike = 0
    dtmin: Optional[RealScalarLike] = None
    dtmax: Optional[RealScalarLike] = None
    force_dtmin: bool = True
    step_ts: Optional[Real[Array, " steps"]] = eqx.field(
        default=None, converter=_none_or_array
    )
    jump_ts: Optional[Real[Array, " jumps"]] = eqx.field(
        default=None, converter=_none_or_array
    )
    factormin: RealScalarLike = 0.2
    factormax: RealScalarLike = 10.0
    norm: Callable[[PyTree], RealScalarLike] = rms_norm
    safety: RealScalarLike = 0.9
    error_order: Optional[RealScalarLike] = None

    def __check_init__(self):
        if self.jump_ts is not None and not jnp.issubdtype(
            self.jump_ts.dtype, jnp.inexact
        ):
            raise ValueError(
                f"jump_ts must be floating point, not {self.jump_ts.dtype}"
            )

    def wrap(self, direction: IntScalarLike):
        step_ts = None if self.step_ts is None else self.step_ts * direction
        jump_ts = None if self.jump_ts is None else self.jump_ts * direction
        return eqx.tree_at(
            lambda s: (s.step_ts, s.jump_ts),
            self,
            (step_ts, jump_ts),
            is_leaf=lambda x: x is None,
        )

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        dt0: Optional[RealScalarLike],
        args: Args,
        func: Callable[[PyTree[AbstractTerm], RealScalarLike, Y, Args], VF],
        error_order: Optional[RealScalarLike],
    ) -> tuple[RealScalarLike, _PidState]:
        del t1
        if dt0 is None:
            error_order = self._get_error_order(error_order)
            dt0 = _select_initial_step(
                terms,
                t0,
                y0,
                args,
                func,
                error_order,
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

        y_leaves = jtu.tree_leaves(y0)
        if len(y_leaves) == 0:
            y_dtype = lxi.default_floating_dtype()
        else:
            y_dtype = jnp.result_type(*y_leaves)
        return t1, (
            jump_next_step,
            at_dtmin,
            dt0,
            jnp.array(1.0, dtype=complex_to_real_dtype(y_dtype)),
            jnp.array(1.0, dtype=complex_to_real_dtype(y_dtype)),
        )

    def adapt_step_size(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        y1_candidate: Y,
        args: Args,
        y_error: Optional[Y],
        error_order: RealScalarLike,
        controller_state: _PidState,
    ) -> tuple[
        BoolScalarLike,
        RealScalarLike,
        RealScalarLike,
        BoolScalarLike,
        _PidState,
        RESULTS,
    ]:
        # Note that different implementations, and different papers, do slightly
        # different things here. It's generally not clear which of these choices are
        # best. (If you know anything about which of these choices is best then please
        # let me know!)
        #
        # Some will compute
        # `scaled_error = norm(y_error / (atol + y * rtol))`.       (1)
        # Some will compute
        # `scaled_error = norm(y_error) / (atol + norm(y) * rtol)`  (2)
        # We do (1). torchdiffeq and torchsde do (1). Soderlind's papers and
        # OrdinaryDiffEq.jl do (2).
        # We choose to do (1) by considering what if `y` were to contain different
        # components at very different scales. The errors in the small components may
        # be drowned out by the errors in the big components if we were using (2).
        #
        # Some will put the multiplication by `safety` outside the `coeff/error_order`
        # exponent. (1) Some will put it inside. (2)
        # We do (1). torchdiffeq and OrdinaryDiffEq.jl does (1). torchsde and
        # Soderlind's papers do (2).
        # We choose to do (1) arbitrarily.
        #
        # Some will perform PI or PID control via
        # h_{n+1} = (ε_n / r_n)^β_1 * (ε_n / r_{n-1})^β_2 * (ε_n / r_{n-2})^β_3 * h_n            (1) # noqa: E501
        # Some will perform
        # h_{n+1} = (ε_n / r_n)^β_1 * (ε_{n-1} / r_{n-1})^β_2 * (ε_{n-2} / r_{n-2})^β_3 * h_n    (2) # noqa: E501
        # Some will perform
        # h_{n+1} = δ_{n,n}^β_1 * δ_{n,n-1}^β_2 * δ_{n,n-2}^β_3 * h_n                            (3) # noqa: E501
        # Some could perform
        # h_{n+1} = δ_{n,n}^β_1 * δ_{n-1,n-1}^β_2 * δ_{n-2,n-2}^β_3 * h_n                        (4) # noqa: E501
        # where
        # h_n is the nth step size
        # ε_n     = atol + norm(y) * rtol with y on the nth step
        # r_n     = norm(y_error) with y_error on the nth step
        # δ_{n,m} = norm(y_error / (atol + norm(y) * rtol))^(-1) with y_error on the nth
        # step and y on the mth step
        # β_1     = pcoeff + icoeff + dcoeff
        # β_2     = -(pcoeff + 2 * dcoeff)
        # β_3     = dcoeff
        # We do (4). torchsde tries to do (3). (But looks like it has a bug in that the
        # numerator and denominator for the P-control have been swapped, I think?)
        # Soderlind's papers do (1). OrdinaryDiffEq.jl does (2).
        # We choose to do (4) by rejecting the others. We reject (1) and (2) for the
        # same reason as computing `scaled_error`, above. (`atol` scaling.) We reject
        # (3) because (whilst it is more similar to Soderlind's work with (1)), it is
        # more inefficient than (4) to implement, as it requires storing the y-shaped
        # (atol + norm(y) * rtol) between steps rather than just the scalar δ_{n,n}
        # between steps.

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
        (
            made_jump,
            at_dtmin,
            prev_dt,
            prev_inv_scaled_error,
            prev_prev_inv_scaled_error,
        ) = controller_state
        error_order = self._get_error_order(error_order)
        # t1 - t0 is the step we actually took, so that's usually what we mean by the
        # "previous dt".
        # However if we made a jump then this t1 was clipped relatively to what it
        # could have been, so for guessing the next step size it's probably better to
        # use the size the step would have been, had there been no jump.
        # There are cases in which something besides the step size controller modifies
        # the step locations t0, t1; most notably the main integration routine clipping
        # steps when we're right at the end of the interval.
        prev_dt = jnp.where(made_jump, prev_dt, t1 - t0)

        #
        # Figure out how things went on the last step: error, and whether to
        # accept/reject it.
        #

        def _scale(_y0, _y1_candidate, _y_error):
            # In case the solver steps into a region for which the vector field isn't
            # defined.
            _nan = jnp.isnan(_y1_candidate).any()
            _y1_candidate = jnp.where(_nan, _y0, _y1_candidate)
            _y = jnp.maximum(jnp.abs(_y0), jnp.abs(_y1_candidate))
            with jax.numpy_dtype_promotion("standard"):
                return _y_error / (self.atol + _y * self.rtol)

        scaled_error = self.norm(jtu.tree_map(_scale, y0, y1_candidate, y_error))
        keep_step = scaled_error < 1
        if self.dtmin is not None:
            keep_step = keep_step | at_dtmin
        # Make sure it's not a Python scalar and thus getting a ZeroDivisionError.
        inv_scaled_error = 1 / jnp.asarray(scaled_error)
        inv_scaled_error = lax.stop_gradient(
            inv_scaled_error
        )  # See note in init above.
        # Note: if you ever remove this lax.stop_gradient, then you'll need to do a lot
        # of work to get safe gradients through these operations.
        # When `inv_scaled_error` has a (non-symbolic) zero cotangent, and `y_error`
        # is either zero or inf, then we get a `0 * inf = nan` on the backward pass.

        #
        # Adjust next step size
        #

        _zero_coeff = lambda c: isinstance(c, (int, float)) and c == 0
        coeff1 = (self.icoeff + self.pcoeff + self.dcoeff) / error_order
        coeff2 = -cast(RealScalarLike, self.pcoeff + 2 * self.dcoeff) / error_order
        coeff3 = self.dcoeff / error_order
        factor1 = 1 if _zero_coeff(coeff1) else inv_scaled_error**coeff1
        factor2 = 1 if _zero_coeff(coeff2) else prev_inv_scaled_error**coeff2
        factor3 = 1 if _zero_coeff(coeff3) else prev_prev_inv_scaled_error**coeff3
        factormin = jnp.where(keep_step, 1, self.factormin)
        factor = jnp.clip(
            self.safety * factor1 * factor2 * factor3,
            a_min=factormin,
            a_max=self.factormax,
        )
        # Once again, see above. In case we have gradients on {i,p,d}coeff.
        # (Probably quite common for them to have zero tangents if passed across
        # a grad API boundary as part of a larger model.)
        factor = lax.stop_gradient(factor)
        factor = eqxi.nondifferentiable(factor)
        dt = prev_dt * factor.astype(prev_dt)

        # E.g. we failed an implicit step, so y_error=inf, so inv_scaled_error=0,
        # so factor=factormin, and we shrunk our step.
        # If we're using a PI or PID controller we shouldn't then force shrinking on
        # the next or next two steps as well!
        pred = (inv_scaled_error == 0) | jnp.isinf(inv_scaled_error)
        inv_scaled_error = jnp.where(pred, 1, inv_scaled_error)

        #
        # Clip next step size based on dtmin/dtmax
        #

        result = RESULTS.successful
        if self.dtmax is not None:
            dt = jnp.minimum(dt, self.dtmax)
        if self.dtmin is None:
            at_dtmin = jnp.array(False)
        else:
            if not self.force_dtmin:
                result = RESULTS.where(dt < self.dtmin, RESULTS.dt_min_reached, result)
            at_dtmin = dt <= self.dtmin
            dt = jnp.maximum(dt, self.dtmin)

        #
        # Clip next step size based on step_ts/jump_ts
        #

        if jnp.issubdtype(jnp.result_type(t1), jnp.inexact):
            # Two nextafters. If made_jump then t1 = prevbefore(jump location)
            # so now _t1 = nextafter(jump location)
            # This is important because we don't know whether or not the jump is as a
            # result of a left- or right-discontinuity, so we have to skip the jump
            # location altogether.
            _t1 = static_select(made_jump, eqxi.nextafter(eqxi.nextafter(t1)), t1)
        else:
            _t1 = t1
        next_t0 = jnp.where(keep_step, _t1, t0)
        next_t1 = self._clip_step_ts(next_t0, next_t0 + dt)
        next_t1, next_made_jump = self._clip_jump_ts(next_t0, next_t1)

        inv_scaled_error = jnp.where(keep_step, inv_scaled_error, prev_inv_scaled_error)
        prev_inv_scaled_error = jnp.where(
            keep_step, prev_inv_scaled_error, prev_prev_inv_scaled_error
        )
        controller_state = (
            next_made_jump,
            at_dtmin,
            dt,
            inv_scaled_error,
            prev_inv_scaled_error,
        )
        return keep_step, next_t0, next_t1, made_jump, controller_state, result

    def _get_error_order(self, error_order: Optional[RealScalarLike]) -> RealScalarLike:
        # Attribute takes priority, if the user knows the correct error order better
        # than our guess.
        error_order = error_order if self.error_order is None else self.error_order
        if error_order is None:
            raise ValueError(
                "The order of convergence for the solver has not been specified; pass "
                "`PIDController(..., error_order=...)` manually instead. If solving "
                "an ODE then this should be equal to the (global) order plus one. If "
                "solving an SDE then should be equal to the (global) order plus 0.5."
            )
        return error_order

    def _clip_step_ts(self, t0: RealScalarLike, t1: RealScalarLike) -> RealScalarLike:
        if self.step_ts is None:
            return t1

        step_ts0 = upcast_or_raise(
            self.step_ts,
            t0,
            "`PIDController.step_ts`",
            "time (the result type of `t0`, `t1`, `dt0`, `SaveAt(ts=...)` etc.)",
        )
        step_ts1 = upcast_or_raise(
            self.step_ts,
            t1,
            "`PIDController.step_ts`",
            "time (the result type of `t0`, `t1`, `dt0`, `SaveAt(ts=...)` etc.)",
        )
        # TODO: it should be possible to switch this O(nlogn) for just O(n) by keeping
        # track of where we were last, and using that as a hint for the next search.
        t0_index = jnp.searchsorted(step_ts0, t0, side="right")
        t1_index = jnp.searchsorted(step_ts1, t1, side="right")
        # This minimum may or may not actually be necessary. The left branch is taken
        # iff t0_index < t1_index <= len(self.step_ts), so all valid t0_index s must
        # already satisfy the minimum.
        # However, that branch is actually executed unconditionally and then where'd,
        # so we clamp it just to be sure we're not hitting undefined behaviour.
        t1 = jnp.where(
            t0_index < t1_index,
            step_ts1[jnp.minimum(t0_index, len(self.step_ts) - 1)],
            t1,
        )
        return t1

    def _clip_jump_ts(
        self, t0: RealScalarLike, t1: RealScalarLike
    ) -> tuple[RealScalarLike, BoolScalarLike]:
        if self.jump_ts is None:
            return t1, False
        assert jnp.issubdtype(self.jump_ts.dtype, jnp.inexact)
        if not jnp.issubdtype(jnp.result_type(t0), jnp.inexact):
            raise ValueError(
                "`t0`, `t1`, `dt0` must be floating point when specifying `jump_ts`. "
                f"Got {jnp.result_type(t0)}."
            )
        if not jnp.issubdtype(jnp.result_type(t1), jnp.inexact):
            raise ValueError(
                "`t0`, `t1`, `dt0` must be floating point when specifying `jump_ts`. "
                f"Got {jnp.result_type(t1)}."
            )
        jump_ts0 = upcast_or_raise(
            self.jump_ts,
            t0,
            "`PIDController.jump_ts`",
            "time (the result type of `t0`, `t1`, `dt0`, `SaveAt(ts=...)` etc.)",
        )
        jump_ts1 = upcast_or_raise(
            self.jump_ts,
            t1,
            "`PIDController.jump_ts`",
            "time (the result type of `t0`, `t1`, `dt0`, `SaveAt(ts=...)` etc.)",
        )
        t0_index = jnp.searchsorted(jump_ts0, t0, side="right")
        t1_index = jnp.searchsorted(jump_ts1, t1, side="right")
        next_made_jump = t0_index < t1_index
        t1 = jnp.where(
            next_made_jump,
            eqxi.prevbefore(jump_ts1[jnp.minimum(t0_index, len(self.jump_ts) - 1)]),
            t1,
        )
        return t1, next_made_jump


PIDController.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance.
- `atol`: Absolute tolerance.
- `pcoeff`: The coefficient of the proportional part of the step size control.
- `icoeff`: The coefficient of the integral part of the step size control.
- `dcoeff`: The coefficient of the derivative part of the step size control.
- `dtmin`: Minimum step size. The step size is either clipped to this value, or an
    error raised if the step size decreases below this, depending on `force_dtmin`.
- `dtmax`: Maximum step size; the step size is clipped to this value.
- `force_dtmin`: How to handle the step size hitting the minimum. If `True` then the
    step size is clipped to `dtmin`. If `False` then the differential equation solve
    halts with an error.
- `step_ts`: Denotes extra times that must be stepped to.
- `jump_ts`: Denotes extra times that must be stepped to, and at which the vector field
    has a known discontinuity. (This is used to force FSAL solvers so re-evaluate the
    vector field.)
- `factormin`: Minimum amount a step size can be decreased relative to the previous
    step.
- `factormax`: Maximum amount a step size can be increased relative to the previous
    step.
- `norm`: A function `PyTree -> Scalar` used in the error control. Precisely, step
    sizes are chosen so that `norm(error / (atol + rtol * y))` is approximately
    one.
- `safety`: Multiplicative safety factor.
- `error_order`: Optional. The order of the error estimate for the solver. Can be used
    to override the error order determined automatically, if extra structure is known
    about this particular problem. (Typically when solving SDEs with known structure.)
"""
