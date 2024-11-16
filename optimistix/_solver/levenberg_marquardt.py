from collections.abc import Callable
from typing import cast, Generic, Union

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, Float, PyTree, Scalar, ScalarLike

from .._custom_types import Aux, Out, Y
from .._misc import max_norm, tree_full_like, two_norm
from .._root_find import AbstractRootFinder, root_find
from .._search import AbstractDescent, FunctionInfo
from .._solution import RESULTS
from .gauss_newton import AbstractGaussNewton, newton_step
from .newton_chord import Newton
from .trust_region import ClassicalTrustRegion


class _Damped(eqx.Module, strict=True):
    operator: lx.AbstractLinearOperator
    damping: Float[Array, ""]

    def __call__(self, y: PyTree[Array]):
        residual = self.operator.mv(y)
        with jax.numpy_dtype_promotion("standard"):
            damped = jtu.tree_map(lambda yi: jnp.sqrt(self.damping) * yi, y)
        return residual, damped


def damped_newton_step(
    step_size: Scalar,
    f_info: Union[FunctionInfo.EvalGradHessian, FunctionInfo.ResidualJac],
    linear_solver: lx.AbstractLinearSolver,
) -> tuple[PyTree[Array], RESULTS]:
    """Compute a damped Newton step.

    For a minimisation problem, this means solving `(Hess + λI)^{-1} grad`.

    In the (nonlinear) least-squares case, for which the minimisation objective
    is given by `0.5*residual^2`, then we know that `grad=J^T residual`, and we make
    the Gauss--Newton approximation `Hess ~ J^T J`. This reduces the above to
    solving the (linear) least-squares problem
    ```
    [    J   ] [diff]  =  [residual]
    [sqrt(λ)I]         =  [   0    ]
    ```
    This can be seen by observing that the normal equations for the this linear
    least-squares problem is the original linear problem we wanted to solve.
    """

    pred = step_size > jnp.finfo(step_size.dtype).eps
    safe_step_size = jnp.where(pred, step_size, 1)
    lm_param = jnp.where(pred, 1 / safe_step_size, jnp.finfo(step_size.dtype).max)
    lm_param = cast(Array, lm_param)
    if isinstance(f_info, FunctionInfo.EvalGradHessian):
        operator = f_info.hessian + lm_param * lx.IdentityLinearOperator(
            f_info.hessian.in_structure()
        )
        vector = f_info.grad
        if lx.is_positive_semidefinite(f_info.hessian):
            operator = lx.TaggedLinearOperator(operator, lx.positive_semidefinite_tag)
    elif isinstance(f_info, FunctionInfo.ResidualJac):
        y_structure = f_info.jac.in_structure()
        operator = lx.FunctionLinearOperator(_Damped(f_info.jac, lm_param), y_structure)
        vector = (f_info.residual, tree_full_like(y_structure, 0))
    else:
        raise ValueError(
            "Damped newton descent cannot be used with a solver that does not "
            "provide (approximate) Hessian information."
        )
    linear_sol = lx.linear_solve(operator, vector, linear_solver, throw=False)
    return linear_sol.value, RESULTS.promote(linear_sol.result)


class _DampedNewtonDescentState(eqx.Module, strict=True):
    f_info: Union[FunctionInfo.EvalGradHessian, FunctionInfo.ResidualJac]


class DampedNewtonDescent(
    AbstractDescent[
        Y,
        Union[FunctionInfo.EvalGradHessian, FunctionInfo.ResidualJac],
        _DampedNewtonDescentState,
    ],
    strict=True,
):
    """The damped Newton (Levenberg--Marquardt) descent.

    That is: gradient descent is about moving in the direction of `-grad`.
    (Quasi-)Newton descent is about moving in the direction of `-Hess^{-1} grad`. Damped
    Newton interpolates between these two regimes, by moving in the direction of
    `-(Hess + λI)^{-1} grad`.

    The value λ is often referred to as a the "Levenberg--Marquardt" parameter, and in
    version is handled directly, as λ = 1/step_size. Larger step sizes correspond to
    Newton directions; smaller step sizes correspond to gradient directions. (And
    additionally also reduces the step size, hence the term "damping".) This is because
    a line search expects the step to be smaller as the step size decreases.
    """

    # Will probably resolve to either Cholesky (for minimisation problems) or
    # QR (for least-squares problems).
    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)

    def init(
        self,
        y: Y,
        f_info_struct: Union[FunctionInfo.EvalGradHessian, FunctionInfo.ResidualJac],
    ) -> _DampedNewtonDescentState:
        del y
        f_info_init = tree_full_like(f_info_struct, 0, allow_static=True)
        return _DampedNewtonDescentState(f_info_init)

    def query(
        self,
        y: Y,
        f_info: Union[FunctionInfo.EvalGradHessian, FunctionInfo.ResidualJac],
        state: _DampedNewtonDescentState,
    ) -> _DampedNewtonDescentState:
        del y, state
        return _DampedNewtonDescentState(f_info)

    def step(
        self, step_size: Scalar, state: _DampedNewtonDescentState
    ) -> tuple[Y, RESULTS]:
        sol_value, result = damped_newton_step(
            step_size, state.f_info, self.linear_solver
        )
        y_diff = (-(sol_value**ω)).ω
        return y_diff, result


DampedNewtonDescent.__init__.__doc__ = """**Arguments:**

- `linear_solver`: The linear solver used to compute the Newton step.
"""


class _IndirectDampedNewtonDescentState(eqx.Module, Generic[Y], strict=True):
    f_info: Union[FunctionInfo.EvalGradHessian, FunctionInfo.ResidualJac]
    newton: Y
    newton_norm: Scalar
    result: RESULTS


class IndirectDampedNewtonDescent(
    AbstractDescent[
        Y,
        Union[FunctionInfo.EvalGradHessian, FunctionInfo.ResidualJac],
        _IndirectDampedNewtonDescentState,
    ],
    strict=True,
):
    """The indirect damped Newton (Levenberg--Marquardt) trust-region descent.

    If the above line just looks like technical word soup, here's what's going on:

    Gradient descent is about moving in the direction of `-grad`. (Quasi-)Newton descent
    is about moving in the direction of `-Hess^{-1} grad`. Damped Newton interpolates
    between these two regimes, by moving in the direction of
    `-(Hess + λI)^{-1} grad`.

    This can be derived as the dual problem of a trust region method, see Conn, Gould,
    Toint: "Trust-Region Methods" section 7.3. λ is interpreted as a Lagrange
    multiplier. This involves solving a one-dimensional root-finding problem for λ at
    each descent.
    """

    lambda_0: ScalarLike = 1.0
    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=False)
    # Default tol for `root_finder` because the tol doesn't have to be very strict.
    root_finder: AbstractRootFinder = Newton(rtol=1e-2, atol=1e-2)
    trust_region_norm: Callable[[PyTree], Scalar] = two_norm

    def init(
        self,
        y: Y,
        f_info_struct: Union[FunctionInfo.EvalGradHessian, FunctionInfo.ResidualJac],
    ) -> _IndirectDampedNewtonDescentState:
        return _IndirectDampedNewtonDescentState(
            f_info=tree_full_like(f_info_struct, 0, allow_static=True),
            newton=y,
            newton_norm=jnp.array(0.0),
            result=RESULTS.successful,
        )

    def query(
        self,
        y: Y,
        f_info: Union[FunctionInfo.EvalGradHessian, FunctionInfo.ResidualJac],
        state: _IndirectDampedNewtonDescentState,
    ) -> _IndirectDampedNewtonDescentState:
        del y
        newton, result = newton_step(f_info, self.linear_solver)
        newton_norm = self.trust_region_norm(newton)
        return _IndirectDampedNewtonDescentState(
            f_info=f_info, newton=newton, newton_norm=newton_norm, result=result
        )

    def step(
        self, step_size: Scalar, state: _IndirectDampedNewtonDescentState
    ) -> tuple[Y, RESULTS]:
        # For trust-region methods like `IndirectDampedNewtonDescent`, the trust-region
        # size directly controls how large a step we take. This is actually somewhat
        # annoying, as a trust region algorithm has no understanding of the scale of a
        # problem unless initialised with a complex initialisation procedure.
        #
        # A simple, heuristic way around this is to scale the trust region `step_size`
        # so that a `step_size` of `1` corresponds to the full length of the Newton
        # step (anything greater than `1` will still accept the Newton step.)
        scaled_step_size = state.newton_norm * step_size

        def comparison_fn(lambda_i: Scalar, _):
            step, _ = damped_newton_step(1 / lambda_i, state.f_info, self.linear_solver)
            return self.trust_region_norm(step) - scaled_step_size

        def reject_newton():
            lambda_out = root_find(
                fn=comparison_fn,
                has_aux=False,
                solver=self.root_finder,
                y0=jnp.asarray(self.lambda_0),
                options=dict(lower=1e-5),
                max_steps=32,
                throw=False,
            ).value
            y_diff, result = damped_newton_step(
                1 / lambda_out, state.f_info, self.linear_solver
            )
            return y_diff, result

        def accept_newton():
            return state.newton, state.result

        # Only do a root-find if we have a small step size, and our Newton step was
        # successful.
        do_root_solve = (state.result == RESULTS.successful) & (step_size < 1)
        neg_y_diff, new_result = lax.cond(do_root_solve, reject_newton, accept_newton)
        return (-(neg_y_diff**ω)).ω, new_result


IndirectDampedNewtonDescent.__init__.__doc__ = """**Arguments:**    

- `lambda_0`: The initial value of the Levenberg--Marquardt parameter used in the root-
    find to hit the trust-region radius. If `IndirectDampedNewtonDescent` is failing,
    this value may need to be increased.
- `linear_solver`: The linear solver used to compute the Newton step.
- `root_finder`: The root finder used to find the Levenberg--Marquardt parameter which
    hits the trust-region radius.
- `trust_region_norm`: The norm used to determine the trust-region shape.
"""


class LevenbergMarquardt(AbstractGaussNewton[Y, Out, Aux], strict=True):
    """The Levenberg--Marquardt method.

    This is a classical solver for nonlinear least squares, which works by regularising
    [`optimistix.GaussNewton`][] with a damping factor. This serves to (a) interpolate
    between Gauss--Newton and steepest descent, and (b) limit step size to a local
    region around the current point.

    This is a good algorithm for many least squares problems.

    Supports the following `options`:

    - `jac`: whether to use forward- or reverse-mode autodifferentiation to compute the
        Jacobian. Can be either `"fwd"` or `"bwd"`. Defaults to `"fwd"`, which is
        usually more efficient. Changing this can be useful when the target function has
        a `jax.custom_vjp`, and so does not support forward-mode autodifferentiation.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    descent: DampedNewtonDescent[Y]
    search: ClassicalTrustRegion[Y]
    verbose: frozenset[str]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        linear_solver: lx.AbstractLinearSolver = lx.QR(),
        verbose: frozenset[str] = frozenset(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = DampedNewtonDescent(linear_solver=linear_solver)
        self.search = ClassicalTrustRegion()
        self.verbose = verbose


LevenbergMarquardt.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `linear_solver`: The linear solver to use to solve the damped Newton step. Defaults to
    `lineax.QR`.
- `verbose`: Whether to print out extra information about how the solve is proceeding.
    Should be a frozenset of strings, specifying what information to print out. Valid
    entries are `step`, `loss`, `accepted`, `step_size`, `y`. For example
    `verbose=frozenset({"loss", "step_size"})`.
"""


class IndirectLevenbergMarquardt(AbstractGaussNewton[Y, Out, Aux], strict=True):
    """The Levenberg--Marquardt method as a true trust-region method.

    This is a variant of [`optimistix.LevenbergMarquardt`][]. The other algorithm works
    by updating the damping factor directly -- this version instead updates a trust
    region, and then fits the damping factor to the size of the trust region.

    Generally speaking [`optimistix.LevenbergMarquardt`][] is preferred, as it performs
    nearly the same algorithm, without the computational overhead of an extra (scalar)
    nonlinear solve.

    Supports the following `options`:

    - `jac`: whether to use forward- or reverse-mode autodifferentiation to compute the
        Jacobian. Can be either `"fwd"` or `"bwd"`. Defaults to `"fwd"`, which is
        usually more efficient. Changing this can be useful when the target function has
        a `jax.custom_vjp`, and so does not support forward-mode autodifferentiation.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    descent: IndirectDampedNewtonDescent[Y]
    search: ClassicalTrustRegion[Y]
    verbose: frozenset[str]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        lambda_0: ScalarLike = 1.0,
        linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=False),
        root_finder: AbstractRootFinder = Newton(rtol=0.01, atol=0.01),
        verbose: frozenset[str] = frozenset(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = IndirectDampedNewtonDescent(
            lambda_0=lambda_0,
            linear_solver=linear_solver,
            root_finder=root_finder,
        )
        self.search = ClassicalTrustRegion()
        self.verbose = verbose


IndirectLevenbergMarquardt.__init__.__doc__ = """**Arguments:**
    
- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `lambda_0`: The initial value of the Levenberg--Marquardt parameter used in the root-
    find to hit the trust-region radius. If `IndirectLevenbergMarquardt` is failing,
    this value may need to be increased.
- `linear_solver`: The linear solver used to compute the Newton step.
- `root_finder`: The root finder used to find the Levenberg--Marquardt parameter which
    hits the trust-region radius.
- `verbose`: Whether to print out extra information about how the solve is proceeding.
    Should be a frozenset of strings, specifying what information to print out. Valid
    entries are `step`, `loss`, `accepted`, `step_size`, `y`. For example
    `verbose=frozenset({"loss", "step_size"})`.
"""
