from collections.abc import Callable
from typing import Any, Generic, Optional, Union
from typing_extensions import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import AbstractVar
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import Aux, DescentState, Fn, Out, SearchState, Y
from .._minimise import AbstractMinimiser
from .._misc import (
    cauchy_termination,
    filter_cond,
    lin_to_grad,
    max_norm,
    tree_full_like,
)
from .._search import (
    AbstractDescent,
    AbstractSearch,
    FunctionInfo,
)
from .._solution import RESULTS
from .learning_rate import LearningRate


class _SteepestDescentState(eqx.Module, Generic[Y], strict=True):
    grad: Y


_FnInfo: TypeAlias = Union[
    FunctionInfo.EvalGrad,
    FunctionInfo.EvalGradHessian,
    FunctionInfo.EvalGradHessianInv,
    FunctionInfo.ResidualJac,
]


class SteepestDescent(AbstractDescent[Y, _FnInfo, _SteepestDescentState], strict=True):
    """The descent direction given by locally following the gradient."""

    norm: Optional[Callable[[PyTree], Scalar]] = None

    def init(self, y: Y, f_info_struct: _FnInfo) -> _SteepestDescentState:
        del f_info_struct
        # Dummy; unused
        return _SteepestDescentState(y)

    def query(
        self, y: Y, f_info: _FnInfo, state: _SteepestDescentState
    ) -> _SteepestDescentState:
        if isinstance(
            f_info,
            (
                FunctionInfo.EvalGrad,
                FunctionInfo.EvalGradHessian,
                FunctionInfo.EvalGradHessianInv,
            ),
        ):
            grad = f_info.grad
        elif isinstance(f_info, FunctionInfo.ResidualJac):
            grad = f_info.compute_grad()
        else:
            raise ValueError(
                "Cannot use `SteepestDescent` with this solver. This is because "
                "`SteepestDescent` requires gradients of the target function, but "
                "this solver does not evaluate such gradients."
            )
        if self.norm is not None:
            grad = (grad**ω / self.norm(grad)).ω
        return _SteepestDescentState(grad)

    def step(
        self, step_size: Scalar, state: _SteepestDescentState
    ) -> tuple[Y, RESULTS]:
        return (-step_size * state.grad**ω).ω, RESULTS.successful


SteepestDescent.__init__.__doc__ = """**Arguments:**

- `norm`: If passed, then normalise the gradient using this norm. (The returned step
    will have length `step_size` with respect to this norm.) Optimistix includes three
    built-in norms: [`optimistix.max_norm`][], [`optimistix.rms_norm`][], and
    [`optimistix.two_norm`][].
"""


class _GradientDescentState(
    eqx.Module, Generic[Y, Out, Aux, SearchState, DescentState], strict=True
):
    # Updated every search step
    first_step: Bool[Array, ""]
    y_eval: Y
    search_state: SearchState
    # Updated after each descent step
    f_info: FunctionInfo.EvalGrad
    aux: Aux
    descent_state: DescentState
    # Used for termination
    terminate: Bool[Array, ""]
    result: RESULTS


class AbstractGradientDescent(
    AbstractMinimiser[Y, Aux, _GradientDescentState], strict=True
):
    """The gradient descent method for unconstrained minimisation.

    At every step, this algorithm performs a line search along the steepest descent
    direction. You should subclass this to provide it with a particular choice of line
    search. (E.g. [`optimistix.GradientDescent`][] uses a simple learning rate step.)

    Subclasses must provide the following abstract attributes, with the following types:

    - `rtol: float`
    - `atol: float`
    - `norm: Callable[[PyTree], Scalar]`
    - `descent: AbstractDescent`
    - `search: AbstractSearch`
    """

    rtol: AbstractVar[float]
    atol: AbstractVar[float]
    norm: AbstractVar[Callable[[PyTree], Scalar]]
    descent: AbstractVar[AbstractDescent[Y, FunctionInfo.EvalGrad, Any]]
    search: AbstractVar[
        AbstractSearch[Y, FunctionInfo.EvalGrad, FunctionInfo.Eval, Any]
    ]

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: jax.ShapeDtypeStruct,
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _GradientDescentState:
        f_info = FunctionInfo.EvalGrad(jnp.zeros(f_struct.shape, f_struct.dtype), y)
        f_info_struct = jax.eval_shape(lambda: f_info)
        return _GradientDescentState(
            first_step=jnp.array(True),
            y_eval=y,
            search_state=self.search.init(y, f_info_struct),
            f_info=f_info,
            aux=tree_full_like(aux_struct, 0),
            descent_state=self.descent.init(y, f_info_struct),
            terminate=jnp.array(False),
            result=RESULTS.successful,
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _GradientDescentState,
        tags: frozenset[object],
    ) -> tuple[Y, _GradientDescentState, Aux]:
        f_eval, lin_fn, aux_eval = jax.linearize(
            lambda _y: fn(_y, args), state.y_eval, has_aux=True
        )
        step_size, accept, search_result, search_state = self.search.step(
            state.first_step,
            y,
            state.y_eval,
            state.f_info,
            FunctionInfo.Eval(f_eval),
            state.search_state,
        )

        def accepted(descent_state):
            (grad,) = lin_to_grad(lin_fn, y)
            f_eval_info = FunctionInfo.EvalGrad(f_eval, grad)
            descent_state = self.descent.query(state.y_eval, f_eval_info, descent_state)
            y_diff = (state.y_eval**ω - y**ω).ω
            f_diff = (f_eval**ω - state.f_info.f**ω).ω
            terminate = cauchy_termination(
                self.rtol, self.atol, self.norm, state.y_eval, y_diff, f_eval, f_diff
            )
            terminate = jnp.where(
                state.first_step, jnp.array(False), terminate
            )  # Skip termination on first step
            return state.y_eval, f_eval_info, aux_eval, descent_state, terminate

        def rejected(descent_state):
            return y, state.f_info, state.aux, descent_state, jnp.array(False)

        y, f_info, aux, descent_state, terminate = filter_cond(
            accept, accepted, rejected, state.descent_state
        )

        y_descent, descent_result = self.descent.step(step_size, descent_state)
        y_eval = (y**ω + y_descent**ω).ω
        result = RESULTS.where(
            search_result == RESULTS.successful, descent_result, search_result
        )

        state = _GradientDescentState(
            first_step=jnp.array(False),
            y_eval=y_eval,
            search_state=search_state,
            f_info=f_info,
            aux=aux,
            descent_state=descent_state,
            terminate=terminate,
            result=result,
        )
        return y, state, aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _GradientDescentState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return state.terminate, state.result

    def postprocess(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _GradientDescentState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}


class GradientDescent(AbstractGradientDescent[Y, Aux], strict=True):
    """Classic gradient descent with a learning rate `learning_rate`."""

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    descent: SteepestDescent[Y]
    search: LearningRate[Y]

    def __init__(
        self,
        learning_rate: float,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
    ):
        """**Arguments:**

        - `learning_rate`: Specifies a constant learning rate to use at each step.
        - `rtol`: Relative tolerance for terminating the solve.
        - `atol`: Absolute tolerance for terminating the solve.
        - `norm`: The norm used to determine the difference between two iterates in the
            convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
            includes three built-in norms: [`optimistix.max_norm`][],
            [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
        """
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = SteepestDescent()
        self.search = LearningRate(learning_rate)
