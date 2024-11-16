from collections.abc import Callable
from typing import Any, Generic, Literal, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from equinox import AbstractVar
from equinox.internal import ω
from jaxtyping import Array, Bool, Int, PyTree, Scalar

from .._custom_types import Args, Aux, DescentState, Fn, Out, SearchState, Y
from .._least_squares import AbstractLeastSquaresSolver
from .._misc import (
    cauchy_termination,
    filter_cond,
    max_norm,
    sum_squares,
    tree_full_like,
    verbose_print,
)
from .._search import (
    AbstractDescent,
    AbstractSearch,
    FunctionInfo,
)
from .._solution import RESULTS
from .learning_rate import LearningRate


def newton_step(
    f_info: Union[
        FunctionInfo.EvalGradHessian,
        FunctionInfo.EvalGradHessianInv,
        FunctionInfo.ResidualJac,
    ],
    linear_solver: lx.AbstractLinearSolver,
) -> tuple[PyTree[Array], RESULTS]:
    """Compute a Newton step.

    For a minimisation problem, this means computing `Hess^{-1} grad`.

    For a least-squares problem, we convert to a minimisation problem via
    `0.5*residuals^2`, which then implies `Hess^{-1} ~ J^T J` (Gauss--Newton
    approximation) and `grad = J^T residuals`.

    Thus `Hess^{-1} grad ~ (J^T J)^{-1} J^T residuals`.   [Equation A]

    Now if `J` is well-posed then this equals `J^{-1} residuals`, which is exactly what
    we compute here.

    And if `J` is ill-posed then [Equation A] is just the normal equations, which should
    almost never be treated directly! (Squares the condition number blahblahblah.) The
    solution of the normal equations matches the pseudoinverse solution `J^{dagger}`
    residuals, which is what we get using an ill-posed-capable linear solver (typically
    QR). So we solve the same linear system as in the well-posed case, we just need to
    set a different linear solver. (Which happens with
    `linear_solver=lx.AutoLinearSolver(well_posed=None)`, which is the recommended
    value.)
    """
    if isinstance(f_info, FunctionInfo.EvalGradHessianInv):
        newton = f_info.hessian_inv.mv(f_info.grad)
        result = RESULTS.successful
    else:
        if isinstance(f_info, FunctionInfo.EvalGradHessian):
            operator = f_info.hessian
            vector = f_info.grad
        elif isinstance(f_info, FunctionInfo.ResidualJac):
            operator = f_info.jac
            vector = f_info.residual
        else:
            raise ValueError(
                "Cannot use a Newton descent with a solver that only evaluates the "
                "gradient, or only the function itself."
            )
        out = lx.linear_solve(operator, vector, linear_solver)
        newton = out.value
        result = RESULTS.promote(out.result)
    return newton, result


class _NewtonDescentState(eqx.Module, Generic[Y], strict=True):
    newton: Y
    result: RESULTS


class NewtonDescent(
    AbstractDescent[
        Y,
        Union[
            FunctionInfo.EvalGradHessian,
            FunctionInfo.EvalGradHessianInv,
            FunctionInfo.ResidualJac,
        ],
        _NewtonDescentState,
    ],
    strict=True,
):
    """Newton descent direction.

    Given a quadratic bowl `x -> x^T Hess x` -- a local quadratic approximation
    to the target function -- this corresponds to moving in the direction of the bottom
    of the bowl. (Which is *not* the same as steepest descent.)

    This is done by solving a linear system of the form `Hess^{-1} grad`.
    """

    norm: Optional[Callable[[PyTree], Scalar]] = None
    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)

    def init(self, y: Y, f_info_struct: FunctionInfo) -> _NewtonDescentState:
        del f_info_struct
        # Dummy values of the right shape; unused.
        return _NewtonDescentState(y, RESULTS.successful)

    def query(
        self,
        y: Y,
        f_info: Union[
            FunctionInfo.EvalGradHessian,
            FunctionInfo.EvalGradHessianInv,
            FunctionInfo.ResidualJac,
        ],
        state: _NewtonDescentState,
    ) -> _NewtonDescentState:
        del state
        newton, result = newton_step(f_info, self.linear_solver)
        if self.norm is not None:
            newton = (newton**ω / self.norm(newton)).ω
        return _NewtonDescentState(newton, result)

    def step(self, step_size: Scalar, state: _NewtonDescentState) -> tuple[Y, RESULTS]:
        return (-step_size * state.newton**ω).ω, state.result


NewtonDescent.__init__.__doc__ = """**Arguments:**

- `norm`: If passed, then normalise the gradient using this norm. (The returned step
    will have length `step_size` with respect to this norm.) Optimistix includes three
    built-in norms: [`optimistix.max_norm`][], [`optimistix.rms_norm`][], and
    [`optimistix.two_norm`][].
- `linear_solver`: The linear solver used to compute the Newton step.
"""


class _GaussNewtonState(
    eqx.Module, Generic[Y, Out, Aux, SearchState, DescentState], strict=True
):
    # Updated every search step
    first_step: Bool[Array, ""]
    y_eval: Y
    search_state: SearchState
    # Updated after each descent step
    f_info: FunctionInfo.ResidualJac
    aux: Aux
    descent_state: DescentState
    # Used for termination
    terminate: Bool[Array, ""]
    result: RESULTS
    # Use for verbose logging
    num_steps: Int[Array, ""]
    num_accepted_steps: Int[Array, ""]
    num_steps_since_acceptance: Int[Array, ""]


def _make_f_info(
    fn: Callable[[Y, Args], tuple[Any, Aux]],
    y: Y,
    args: Args,
    tags: frozenset,
    jac: Literal["fwd", "bwd"],
) -> tuple[FunctionInfo.ResidualJac, Aux]:
    if jac == "fwd":
        f_eval, lin_fn, aux_eval = jax.linearize(
            lambda _y: fn(_y, args), y, has_aux=True
        )
        jac_eval = lx.FunctionLinearOperator(lin_fn, jax.eval_shape(lambda: y), tags)
    elif jac == "bwd":
        # Materialise the Jacobian in this case.
        def _for_jacrev(_y):
            f_eval, aux_eval = fn(_y, args)
            return f_eval, (f_eval, aux_eval)

        jac_pytree, (f_eval, aux_eval) = jax.jacrev(_for_jacrev, has_aux=True)(y)
        output_structure = jax.eval_shape(lambda: f_eval)
        jac_eval = lx.PyTreeLinearOperator(jac_pytree, output_structure, tags)
    else:
        raise ValueError("Only `jac='fwd'` or `jac='bwd'` are valid.")
    return FunctionInfo.ResidualJac(f_eval, jac_eval), aux_eval


class AbstractGaussNewton(
    AbstractLeastSquaresSolver[Y, Out, Aux, _GaussNewtonState], strict=True
):
    """Abstract base class for all Gauss-Newton type methods.

    This includes methods such as [`optimistix.GaussNewton`][],
    [`optimistix.LevenbergMarquardt`][], and [`optimistix.Dogleg`][].

    Subclasses must provide the following attributes, with the following types:

    - `rtol`: `float`
    - `atol`: `float`
    - `norm`: `Callable[[PyTree], Scalar]`
    - `descent`: `AbstractDescent`
    - `search`: `AbstractSearch`
    - `verbose`: `frozenset[str]`

    Supports the following `options`:

    - `jac`: whether to use forward- or reverse-mode autodifferentiation to compute the
        Jacobian. Can be either `"fwd"` or `"bwd"`. Defaults to `"fwd"`, which is
        usually more efficient. Changing this can be useful when the target function has
        a `jax.custom_vjp`, and so does not support forward-mode autodifferentiation.
    """

    rtol: AbstractVar[float]
    atol: AbstractVar[float]
    norm: AbstractVar[Callable[[PyTree], Scalar]]
    descent: AbstractVar[AbstractDescent[Y, FunctionInfo.ResidualJac, Any]]
    search: AbstractVar[
        AbstractSearch[Y, FunctionInfo.ResidualJac, FunctionInfo.ResidualJac, Any]
    ]
    verbose: AbstractVar[frozenset[str]]

    def init(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _GaussNewtonState:
        jac = options.get("jac", "fwd")
        f_info_struct, _ = eqx.filter_eval_shape(_make_f_info, fn, y, args, tags, jac)
        f_info = tree_full_like(f_info_struct, 0, allow_static=True)
        return _GaussNewtonState(
            first_step=jnp.array(True),
            y_eval=y,
            search_state=self.search.init(y, f_info_struct),
            f_info=f_info,
            aux=tree_full_like(aux_struct, 0),
            descent_state=self.descent.init(y, f_info_struct),
            terminate=jnp.array(False),
            result=RESULTS.successful,
            num_steps=jnp.array(0),
            num_accepted_steps=jnp.array(0),
            num_steps_since_acceptance=jnp.array(0),
        )

    def step(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _GaussNewtonState,
        tags: frozenset[object],
    ) -> tuple[Y, _GaussNewtonState, Aux]:
        jac = options.get("jac", "fwd")
        f_eval_info, aux_eval = _make_f_info(fn, state.y_eval, args, tags, jac)
        # We have a jaxpr in `f_info.jac`, which are compared by identity. Here we
        # arrange to use the same one so that downstream equality checks (e.g. in the
        # `filter_cond` below)
        dynamic = eqx.filter(f_eval_info.jac, eqx.is_array)
        static = eqx.filter(state.f_info.jac, eqx.is_array, inverse=True)
        jac = eqx.combine(dynamic, static)
        f_eval_info = eqx.tree_at(lambda f: f.jac, f_eval_info, jac)

        step_size, accept, search_result, search_state = self.search.step(
            state.first_step,
            y,
            state.y_eval,
            state.f_info,
            f_eval_info,
            state.search_state,
        )
        num_steps = state.num_steps + 1
        num_accepted_steps = state.num_accepted_steps + jnp.where(accept, 1, 0)
        num_steps_since_acceptance = jnp.where(
            accept, 0, state.num_steps_since_acceptance + 1
        )

        def accepted(descent_state):
            descent_state = self.descent.query(state.y_eval, f_eval_info, descent_state)
            y_diff = (state.y_eval**ω - y**ω).ω
            f_diff = (f_eval_info.residual**ω - state.f_info.residual**ω).ω
            terminate = cauchy_termination(
                self.rtol,
                self.atol,
                self.norm,
                state.y_eval,
                y_diff,
                f_eval_info.residual,
                f_diff,
            )
            return state.y_eval, f_eval_info, aux_eval, descent_state, terminate

        def rejected(descent_state):
            return y, state.f_info, state.aux, descent_state, jnp.array(False)

        y, f_info, aux, descent_state, terminate = filter_cond(
            accept, accepted, rejected, state.descent_state
        )

        if len(self.verbose) > 0:
            verbose_step = "step" in self.verbose
            verbose_loss = "loss" in self.verbose
            verbose_accepted = "accepted" in self.verbose
            verbose_step_size = "step_size" in self.verbose
            verbose_y = "y" in self.verbose
            loss_eval = 0.5 * sum_squares(f_eval_info.residual)
            loss = 0.5 * sum_squares(state.f_info.residual)
            verbose_print(
                (verbose_step, "Step", state.num_steps),
                (
                    verbose_step and verbose_accepted,
                    "Accepted steps",
                    state.num_accepted_steps,
                ),
                (
                    verbose_step and verbose_accepted,
                    "Steps since acceptance",
                    state.num_steps_since_acceptance,
                ),
                (verbose_loss, "Loss on this step", loss_eval),
                (
                    verbose_loss and verbose_accepted,
                    "Loss on the last accepted step",
                    loss,
                ),
                (verbose_step_size, "Step size", step_size),
                (verbose_y, "y", state.y_eval),
                (verbose_y and verbose_accepted, "y on the last accepted step", y),
            )

        y_descent, descent_result = self.descent.step(step_size, descent_state)
        y_eval = (y**ω + y_descent**ω).ω
        result = RESULTS.where(
            search_result == RESULTS.successful, descent_result, search_result
        )

        state = _GaussNewtonState(
            first_step=jnp.array(False),
            y_eval=y_eval,
            search_state=search_state,
            f_info=f_info,
            aux=aux,
            descent_state=descent_state,
            terminate=terminate,
            result=result,
            num_steps=num_steps,
            num_accepted_steps=num_accepted_steps,
            num_steps_since_acceptance=num_steps_since_acceptance,
        )
        return y, state, aux

    def terminate(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _GaussNewtonState,
        tags: frozenset[object],
    ):
        return state.terminate, state.result

    def postprocess(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _GaussNewtonState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}


class GaussNewton(AbstractGaussNewton[Y, Out, Aux], strict=True):
    """Gauss-Newton algorithm, for solving nonlinear least-squares problems.

    Note that regularised approaches like [`optimistix.LevenbergMarquardt`][] are
    usually preferred instead.

    Supports the following `options`:

    - `jac`: whether to use forward- or reverse-mode autodifferentiation to compute the
        Jacobian. Can be either `"fwd"` or `"bwd"`. Defaults to `"fwd"`, which is
        usually more efficient. Changing this can be useful when the target function has
        a `jax.custom_vjp`, and so does not support forward-mode autodifferentiation.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    descent: NewtonDescent[Y]
    search: LearningRate[Y]
    verbose: frozenset[str]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None),
        verbose: frozenset[str] = frozenset(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = NewtonDescent(linear_solver=linear_solver)
        self.search = LearningRate(1.0)
        self.verbose = verbose


GaussNewton.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `linear_solver`: The linear solver used to compute the Newton step.
- `verbose`: Whether to print out extra information about how the solve is proceeding.
    Should be a frozenset of strings, specifying what information to print out. Valid
    entries are `step`, `loss`, `accepted`, `step_size`, `y`. For example
    `verbose=frozenset({"loss", "step_size"})`.
"""
