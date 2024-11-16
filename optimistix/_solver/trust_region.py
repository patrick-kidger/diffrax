import abc
from typing import TypeVar, Union
from typing_extensions import TypeAlias

import equinox as eqx
import jax.numpy as jnp
from equinox import AbstractVar
from equinox.internal import ω
from jaxtyping import Array, Bool, Scalar, ScalarLike

from .._custom_types import Y
from .._misc import (
    sum_squares,
    tree_dot,
)
from .._search import AbstractSearch, FunctionInfo
from .._solution import RESULTS


class _TrustRegionState(eqx.Module, strict=True):
    step_size: Scalar


_FnInfo = TypeVar("_FnInfo", bound=FunctionInfo)
_FnEvalInfo: TypeAlias = FunctionInfo


class _AbstractTrustRegion(
    AbstractSearch[Y, _FnInfo, _FnEvalInfo, _TrustRegionState], strict=True
):
    """The abstract base class of the trust-region update algorithm.

    Trust region line searches compute the ratio
    `true_reduction/predicted_reduction`, where `true_reduction` is the decrease in `fn`
    between `y` and `new_y`, and `predicted_reduction` is how much we expected the
    function to decrease using an approximation to `fn`.

    The trust-region ratio determines whether to accept or reject a step and the
    next choice of step-size. Specifically:

    - reject the step and decrease stepsize if the ratio is smaller than a
        cutoff `low_cutoff`
    - accept the step and increase the step-size if the ratio is greater than
        another cutoff `high_cutoff` with `low_cutoff < high_cutoff`.
    - else, accept the step and make no change to the step-size.
    """

    high_cutoff: AbstractVar[ScalarLike]
    low_cutoff: AbstractVar[ScalarLike]
    high_constant: AbstractVar[ScalarLike]
    low_constant: AbstractVar[ScalarLike]

    def __post_init__(self):
        # You would not expect `self.low_cutoff` or `self.high_cutoff` to
        # be below zero, but this is technically not incorrect so we don't
        # require it.
        self.low_cutoff, self.high_cutoff = eqx.error_if(  # pyright: ignore
            (self.low_cutoff, self.high_cutoff),
            self.low_cutoff > self.high_cutoff,  # pyright: ignore
            "`low_cutoff` must be below `high_cutoff` in `ClassicalTrustRegion`",
        )
        self.low_constant = eqx.error_if(  # pyright: ignore
            self.low_constant,
            self.low_constant < 0,  # pyright: ignore
            "`low_constant` must be greater than `0` in `ClassicalTrustRegion`",
        )
        self.high_constant = eqx.error_if(  # pyright: ignore
            self.high_constant,
            self.high_constant < 0,  # pyright: ignore
            "`high_constant` must be greater than `0` in `ClassicalTrustRegion`",
        )

    @abc.abstractmethod
    def predict_reduction(self, y_diff: Y, f_info: _FnInfo) -> Scalar:
        ...

    def init(self, y: Y, f_info_struct: _FnInfo) -> _TrustRegionState:
        del f_info_struct
        return _TrustRegionState(step_size=jnp.array(1.0))

    def step(
        self,
        first_step: Bool[Array, ""],
        y: Y,
        y_eval: Y,
        f_info: _FnInfo,
        f_eval_info: _FnEvalInfo,
        state: _TrustRegionState,
    ) -> tuple[Scalar, Bool[Array, ""], RESULTS, _TrustRegionState]:
        y_diff = (y_eval**ω - y**ω).ω
        predicted_reduction = self.predict_reduction(y_diff, f_info)
        # We never actually compute the ratio
        # `true_reduction/predicted_reduction`. Instead, we rewrite the conditions as
        # `true_reduction < const * predicted_reduction` instead, where the inequality
        # flips because we assume `predicted_reduction` is negative.
        # This avoids an expensive division.
        f_min = f_info.as_min()
        f_min_eval = f_eval_info.as_min()
        f_min_diff = f_min_eval - f_min  # This number is probably negative
        accept = f_min_diff <= self.low_cutoff * predicted_reduction
        good = f_min_diff < self.high_cutoff * predicted_reduction
        good = good & (predicted_reduction < 0)

        good = good & jnp.invert(first_step)
        accept = accept | first_step
        mul = jnp.where(good, self.high_constant, 1)
        mul = jnp.where(accept, mul, self.low_constant)
        new_step_size = mul * state.step_size
        new_state = _TrustRegionState(step_size=new_step_size)
        return new_step_size, accept, RESULTS.successful, new_state


class ClassicalTrustRegion(
    _AbstractTrustRegion[
        Y, Union[FunctionInfo.EvalGradHessian, FunctionInfo.ResidualJac]
    ],
    strict=True,
):
    """The classic trust-region update algorithm which uses a quadratic approximation of
    the objective function to predict reduction.

    Building a quadratic approximation requires an approximation to the Hessian of the
    overall minimisation function. This means that trust region is suitable for use with
    least-squares algorithms (which make the Gauss--Newton approximation
    Hessian~Jac^T J) and for quasi-Newton minimisation algorithms like
    [`optimistix.BFGS`][]. (An error will be raised if you use this with an incompatible
    solver.)
    """

    # This choice of default parameters comes from Gould et al.
    # "Sensitivity of trust region algorithms to their parameters."
    high_cutoff: ScalarLike = 0.99
    low_cutoff: ScalarLike = 0.01
    high_constant: ScalarLike = 3.5
    low_constant: ScalarLike = 0.25

    def predict_reduction(
        self,
        y_diff: Y,
        f_info: Union[FunctionInfo.EvalGradHessian, FunctionInfo.ResidualJac],
    ) -> Scalar:
        """Compute the expected decrease in loss from taking the step `y_diff`.

        The true reduction is
        ```
        fn(y0 + y_diff) - fn(y0)
        ```
        so if `B` is the approximation to the Hessian coming from the quasi-Newton
        method at `y`, and `g` is the gradient of `fn` at `y`, then the predicted
        reduction is
        ```
        g^T y_diff + 1/2 y_diff^T B y_diff
        ```

        **Arguments**:

        - `y_diff`: the proposed step by the descent method.
        - `deriv_info`: the derivative information (on the gradient and Hessian)
            provided by the outer loop.

        **Returns**:

        The expected decrease in loss from moving from `y0` to `y0 + y_diff`.
        """

        if isinstance(f_info, FunctionInfo.EvalGradHessian):
            # Minimisation algorithm. Directly compute the quadratic approximation.
            return tree_dot(
                y_diff,
                (f_info.grad**ω + 0.5 * f_info.hessian.mv(y_diff) ** ω).ω,
            )
        elif isinstance(f_info, FunctionInfo.ResidualJac):
            # Least-squares algorithm. So instead of considering fn (which returns the
            # residuals), instead consider `0.5*fn(y)^2`, and then apply the logic as
            # for minimisation.
            # We get that `g = J^T f0` and `B = J^T J + dJ/dx^T J`.
            # (Here, `f0 = fn(y0)` are the residuals, and `J = dfn/dy(y0)` is the
            # Jacobian of the residuals wrt y.)
            # Then neglect the second term in B (the usual Gauss--Newton approximation)
            # and complete the square.
            # We find that the predicted reduction is
            # `0.5 * ((J y_diff + f0)^T (J y_diff + f0) - f0^T f0)`
            # and this is what is written below.
            #
            # The reason we go through this hassle is because this now involves only
            # a single Jacobian-vector product, rather than the three we would have to
            # make by naively substituting `B = J^T J `and `g = J^T f0` into the general
            # algorithm used for minimisation.
            rtr = sum_squares(f_info.residual)
            jacobian_term = sum_squares(
                (f_info.jac.mv(y_diff) ** ω + f_info.residual**ω).ω
            )
            return 0.5 * (jacobian_term - rtr)
        else:
            raise ValueError(
                "Cannot use `ClassicalTrustRegion` with this solver. This is because "
                "`ClassicalTrustRegion` requires (an approximation to) the Hessian of "
                "the target function, but this solver does not make any estimate of "
                "that information."
            )


# When using a gradient-based method, `LinearTrustRegion` is actually a variant of
# `BacktrackingArmijo`. The linear predicted reduction is the same as the Armijo
# condition. The difference is that unlike standard backtracking,
# `LinearTrustRegion` chooses its next step size based on how well it did in the
# previous iteration.
class LinearTrustRegion(
    _AbstractTrustRegion[
        Y,
        Union[
            FunctionInfo.EvalGrad,
            FunctionInfo.EvalGradHessian,
            FunctionInfo.EvalGradHessianInv,
            FunctionInfo.ResidualJac,
        ],
    ],
    strict=True,
):
    """The trust-region update algorithm which uses a linear approximation of
    the objective function to predict reduction.

    Generally speaking you should prefer [`optimistix.ClassicalTrustRegion`][], unless
    you happen to be using a solver (e.g. a non-quasi-Newton minimiser) with which that
    is incompatible.
    """

    # This choice of default parameters comes from Gould et al.
    # "Sensitivity of trust region algorithms to their parameters."
    high_cutoff: ScalarLike = 0.99
    low_cutoff: ScalarLike = 0.01
    high_constant: ScalarLike = 3.5
    low_constant: ScalarLike = 0.25

    def predict_reduction(
        self,
        y_diff: Y,
        f_info: Union[
            FunctionInfo.EvalGrad,
            FunctionInfo.EvalGradHessian,
            FunctionInfo.EvalGradHessianInv,
            FunctionInfo.ResidualJac,
        ],
    ) -> Scalar:
        """Compute the expected decrease in loss from taking the step `y_diff`.

        The true reduction is
        ```
        fn(y0 + y_diff) - fn(y0)
        ```
        so if `g` is the gradient of `fn` at `y`, then the predicted reduction is
        ```
        g^T y_diff
        ```

        **Arguments**:

        - `y_diff`: the proposed step by the descent method.
        - `deriv_info`: the derivative information (on the gradient and Hessian)
            provided by the outer loop.

        **Returns**:

        The expected decrease in loss from moving from `y0` to `y0 + y_diff`.
        """

        if isinstance(
            f_info,
            (
                FunctionInfo.EvalGrad,
                FunctionInfo.EvalGradHessian,
                FunctionInfo.EvalGradHessianInv,
                FunctionInfo.ResidualJac,
            ),
        ):
            return f_info.compute_grad_dot(y_diff)
        else:
            raise ValueError(
                "Cannot use `LinearTrustRegion` with this solver. This is because "
                "`LinearTrustRegion` requires gradients of the target function, but "
                "this solver does not evaluate such gradients."
            )


_init_doc = """In the following, `ratio` refers to the ratio
`true_reduction/predicted_reduction`.

**Arguments**:

- `high_cutoff`: the cutoff such that `ratio > high_cutoff` will accept the step
and increase the step-size on the next iteration.
- `low_cutoff`: the cutoff such that `ratio < low_cutoff` will reject the step
and decrease the step-size on the next iteration.
- `high_constant`: when `ratio > high_cutoff`, multiply the previous step-size by
high_constant`.
- `low_constant`: when `ratio < low_cutoff`, multiply the previous step-size by
low_constant`.
"""

LinearTrustRegion.__init__.__doc__ = _init_doc
ClassicalTrustRegion.__init__.__doc__ = _init_doc
