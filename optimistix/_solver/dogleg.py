from collections.abc import Callable
from typing import Any, cast, Generic, Union

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, PyTree, Scalar

from .._custom_types import Aux, Out, Y
from .._misc import (
    max_norm,
    sum_squares,
    tree_dot,
    tree_full_like,
    tree_where,
    two_norm,
)
from .._root_find import AbstractRootFinder, root_find
from .._search import AbstractDescent, FunctionInfo
from .._solution import RESULTS
from .bisection import Bisection
from .gauss_newton import AbstractGaussNewton, newton_step
from .trust_region import ClassicalTrustRegion


class _DoglegDescentState(eqx.Module, Generic[Y], strict=True):
    newton: Y
    cauchy: Y
    newton_norm: Scalar
    cauchy_norm: Scalar
    result: RESULTS


class DoglegDescent(
    AbstractDescent[
        Y,
        Union[FunctionInfo.EvalGradHessian, FunctionInfo.ResidualJac],
        _DoglegDescentState,
    ],
    strict=True,
):
    """The Dogleg descent step, which switches between the Cauchy and the Newton
    descent directions.
    """

    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)
    root_finder: AbstractRootFinder[Scalar, Scalar, None, Any] = Bisection(
        rtol=1e-3, atol=1e-3
    )
    trust_region_norm: Callable[[PyTree], Scalar] = two_norm

    def init(
        self,
        y: Y,
        f_info_struct: Union[FunctionInfo.EvalGradHessian, FunctionInfo.ResidualJac],
    ) -> _DoglegDescentState:
        # Dummy values; unused
        del f_info_struct
        return _DoglegDescentState(
            newton=y,
            cauchy=y,
            newton_norm=jnp.array(0.0),
            cauchy_norm=jnp.array(0.0),
            result=RESULTS.successful,
        )

    def query(
        self,
        y: Y,
        f_info: Union[FunctionInfo.EvalGradHessian, FunctionInfo.ResidualJac],
        state: _DoglegDescentState,
    ) -> _DoglegDescentState:
        del y, state
        # Compute `denom = grad^T Hess grad.`
        if isinstance(f_info, FunctionInfo.EvalGradHessian):
            grad = f_info.grad
            denom = tree_dot(f_info.grad, f_info.hessian.mv(grad))
        elif isinstance(f_info, FunctionInfo.ResidualJac):
            # Use Gauss--Newton approximation `Hess ~ J^T J`
            grad = f_info.compute_grad()
            denom = sum_squares(f_info.jac.mv(grad))
        else:
            raise ValueError(
                "`DoglegDescent` can only be used with least-squares solvers, or "
                "quasi-Newton minimisers which make approximations to the Hessian "
                "(like `optx.BFGS(use_inverse=False)`)"
            )
        denom_nonzero = denom > jnp.finfo(denom.dtype).eps
        safe_denom = jnp.where(denom_nonzero, denom, 1)
        # Compute `grad^T grad / (grad^T Hess grad)`
        scaling = jnp.where(denom_nonzero, sum_squares(grad) / safe_denom, 0.0)
        scaling = cast(Array, scaling)

        # Downhill towards the bottom of the quadratic basin.
        newton_sol, result = newton_step(f_info, self.linear_solver)
        newton = (-(newton_sol**ω)).ω
        newton_norm = self.trust_region_norm(newton_sol)

        # Downhill steepest descent.
        cauchy = (-scaling * grad**ω).ω
        cauchy_norm = self.trust_region_norm(cauchy)

        return _DoglegDescentState(
            newton=newton,
            cauchy=cauchy,
            newton_norm=newton_norm,
            cauchy_norm=cauchy_norm,
            result=result,
        )

    def step(self, step_size: Scalar, state: _DoglegDescentState) -> tuple[Y, RESULTS]:
        # For trust-region methods like `DoglegDescent`, the trust-region size directly
        # controls how large a step we take. This is actually somewhat annoying,
        # as a trust region algorithm has no understanding of the scale of a
        # problem unless initialised with a complex initialisation procedure.
        #
        # A simple, heuristic way around this is to scale the trust region `step_size`
        # so that a `step_size` of `1` corresponds to the full length of the Newton
        # step (anything greater than `1` will still accept the Newton step.)
        scaled_step_size = state.newton_norm * step_size

        def accept_scaled_cauchy(cauchy, newton):
            """Scale and return the Cauchy step."""
            norm_nonzero = state.cauchy_norm > jnp.finfo(state.cauchy_norm.dtype).eps
            safe_norm = jnp.where(norm_nonzero, state.cauchy_norm, 1)
            # Return zeros in degenerate case instead of inf because if `cauchy_norm` is
            # near zero, then so is the gradient and `delta` must be tiny to accept
            # this.
            normalised_cauchy = tree_where(
                norm_nonzero,
                ((cauchy**ω / safe_norm) * scaled_step_size).ω,
                tree_full_like(cauchy, 0),
            )
            return normalised_cauchy, RESULTS.successful

        def interpolate_cauchy_and_newton(cauchy, newton):
            """Find the point interpolating the Cauchy and Newton steps which
            intersects the trust region radius.
            """

            def interpolate(t):
                return (cauchy**ω + (t - 1) * (newton**ω - cauchy**ω)).ω

            # The vast majority of the time we expect users to use `two_norm`,
            # ie. the classic, elliptical trust region radius. In this case, we
            # compute the value of `t` to hit the trust region radius using by solving
            # a quadratic equation `a * t**2 + b * t + c = 0`
            # See section 4.1 of Nocedal & Wright "Numerical Optimization" for details.
            #
            # If they pass a norm other than `two_norm`, ie. they use a more exotic
            # trust region shape, we use a root find to approximately
            # find the value which hits the trust region radius.
            if self.trust_region_norm is two_norm:
                a = sum_squares((newton**ω - cauchy**ω).ω)
                inner_prod = tree_dot(cauchy, (newton**ω - cauchy**ω).ω)
                b = 2 * (inner_prod - a)
                c = state.cauchy_norm**2 - 2 * inner_prod + a - scaled_step_size**2
                quadratic_1 = jnp.clip(
                    0.5 * (-b + jnp.sqrt(b**2 - 4 * a * c)) / a, min=1, max=2
                )
                quadratic_2 = jnp.clip(
                    ((2 * c) / (-b - jnp.sqrt(b**2 - 4 * a * c))), min=1, max=2
                )
                # The quadratic formula is not numerically stable, and it is best to
                # use slightly different formulas when `b >=` and `b < 0`.
                # See https://github.com/fortran-lang/stdlib/issues/674 for a number of
                # references.
                t_interp = jnp.where(b >= 0, quadratic_1, quadratic_2)
                result = RESULTS.successful
            else:
                root_find_options = {"lower": jnp.array(1.0), "upper": jnp.array(2.0)}
                root_sol = root_find(
                    lambda t, _: (
                        self.trust_region_norm(interpolate(t)) - scaled_step_size
                    ),
                    self.root_finder,
                    y0=jnp.array(1.5),
                    options=root_find_options,
                    throw=False,
                )
                t_interp = root_sol.value
                result = root_sol.result
            return interpolate(t_interp), result

        def accept_newton(cauchy, newton):
            """Return the Newton step."""
            return newton, RESULTS.successful

        index = jnp.where(
            state.cauchy_norm > scaled_step_size, 0, jnp.where(step_size < 1, 1, 2)
        )
        y_diff, root_result = lax.switch(
            index,
            [accept_scaled_cauchy, interpolate_cauchy_and_newton, accept_newton],
            state.cauchy,
            state.newton,
        )
        result = RESULTS.where(
            state.result == RESULTS.successful, root_result, state.result
        )
        return y_diff, result


DoglegDescent.__init__.__doc__ = """**Arguments:**

- `linear_solver`: The linear solver used to compute the Newton step.
- `root_finder`: The root finder used to find the point where the trust-region
    intersects the dogleg path. This is ignored if
    `trust_region_norm=optimistix.two_norm`, for which there is an analytic formula
    instead.
- `trust_region_norm`: The norm used to determine the trust-region shape.
"""


class Dogleg(AbstractGaussNewton[Y, Out, Aux], strict=True):
    """Dogleg algorithm. Used for nonlinear least squares problems.

    Given a quadratic bowl that locally approximates the function to be minimised, then
    there are two different ways we might try to move downhill: in the steepest descent
    direction (as in gradient descent; this is also sometimes called the Cauchy
    direction), and in the direction of the minima of the quadratic bowl (as in Newton's
    method; correspondingly this is called the Newton direction).

    The distinguishing feature of this algorithm is the "dog leg" shape of its descent
    path, in which it begins by moving in the steepest descent direction, and then
    switches to moving in the Newton direction.

    Supports the following `options`:

    - `jac`: whether to use forward- or reverse-mode autodifferentiation to compute the
        Jacobian. Can be either `"fwd"` or `"bwd"`. Defaults to `"fwd"`, which is
        usually more efficient. Changing this can be useful when the target function has
        a `jax.custom_vjp`, and so does not support forward-mode autodifferentiation.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    descent: DoglegDescent[Y]
    search: ClassicalTrustRegion[Y]
    verbose: frozenset[str]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None),
        verbose: frozenset[str] = frozenset(),
    ):
        # We don't expose root_finder to the default API for Dogleg because
        # we assume the `trust_region_norm` norm is `two_norm`, which has
        # an analytic formula for the intersection with the dogleg path.
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = DoglegDescent(linear_solver=linear_solver)
        self.search = ClassicalTrustRegion()
        self.verbose = verbose


Dogleg.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `linear_solver`: The linear solver used to compute the Newton part of the dogleg step.
- `verbose`: Whether to print out extra information about how the solve is proceeding.
    Should be a frozenset of strings, specifying what information to print out. Valid
    entries are `step`, `loss`, `accepted`, `step_size`, `y`. For example
    `verbose=frozenset({"loss", "step_size"})`.
"""
