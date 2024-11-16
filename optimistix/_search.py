"""We use notions of "searches" and "descents" to generalise line searches, trust
regions, and learning rates.

For example, the classical gradient descent algorithm is given by setting the search
to be `optimistix.LearningRate`, and setting the descent to be
`optimistix.SteepestDescent`.

As another example, Levenberg--Marquardt is given by setting the search to be
`optimistix.ClassicalTrustRegion`, and setting the descent to be
`optimistix.DampedNewtonDescent`.

As a third example, BFGS is given by setting the search to be
`optimistix.BacktrackingArmijo`, and setting the descent to be
`optimistix.NewtonDescent`.

This gives us a flexible way to mix-and-match ideas across different optimisers.

Right now, these are used exclusively for minimisers and least-squares optimisers. (As
the latter are a special case of the former, with `to_minimise = 0.5 * residuals^2`)

For an in-depth discussion of how these pieces fit together, see the documentation at
`docs/api/searches/introduction.md`.
"""

import abc
from typing import ClassVar, Generic, Type, TypeVar

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from jaxtyping import Array, Bool, Scalar

from ._custom_types import (
    DescentState,
    Out,
    SearchState,
    Y,
)
from ._misc import sum_squares, tree_dot
from ._solution import RESULTS


class FunctionInfo(eqx.Module, strict=eqx.StrictConfig(allow_abstract_name=True)):
    """Different solvers (BFGS, Levenberg--Marquardt, ...) evaluate different
    quantities of the objective function. Some may compute gradient information,
    some may provide approximate Hessian information, etc.

    This enumeration-ish object captures the different variants.

    Available variants are
    `optimistix.FunctionInfo.{Eval, EvalGrad, EvalGradHessian, EvalGradHessianInv, Residual, ResidualJac}`.
    """  # noqa: E501

    Eval: ClassVar[Type["Eval"]]
    EvalGrad: ClassVar[Type["EvalGrad"]]
    EvalGradHessian: ClassVar[Type["EvalGradHessian"]]
    EvalGradHessianInv: ClassVar[Type["EvalGradHessianInv"]]
    Residual: ClassVar[Type["Residual"]]
    ResidualJac: ClassVar[Type["ResidualJac"]]

    @abc.abstractmethod
    def as_min(self) -> Scalar:
        """For a minimisation problem, returns f(y). For a least-squares problem,
        returns 0.5*residuals^2 -- i.e. its loss as a minimisation problem.
        """


# NOT PUBLIC, despite lacking an underscore. This is so pyright gets the name right.
class Eval(FunctionInfo, strict=True):
    """Has a `.f` attribute describing `fn(y)`. Used when no gradient information is
    available.
    """

    f: Scalar

    def as_min(self):
        return self.f


# NOT PUBLIC, despite lacking an underscore. This is so pyright gets the name right.
class EvalGrad(FunctionInfo, Generic[Y], strict=True):
    """Has a `.f` attribute as with [`optimistix.FunctionInfo.Eval`][]. Also has a
    `.grad` attribute describing `d(fn)/dy`. Used with first-order solvers for
    minimisation problems. (E.g. gradient descent; nonlinear CG.)
    """

    f: Scalar
    grad: Y

    def as_min(self):
        return self.f

    def compute_grad_dot(self, y: Y):
        return tree_dot(self.grad, y)


# NOT PUBLIC, despite lacking an underscore. This is so pyright gets the name right.
class EvalGradHessian(FunctionInfo, Generic[Y], strict=True):
    """Has `.f` and `.grad` attributes as with [`optimistix.FunctionInfo.EvalGrad`][].
    Also has a `.hessian` attribute describing (an approximation to) the Hessian of
    `fn` at `y`. Used with quasi-Newton minimisation algorithms, like BFGS.
    """

    f: Scalar
    grad: Y
    hessian: lx.AbstractLinearOperator

    def as_min(self):
        return self.f

    def compute_grad_dot(self, y: Y):
        return tree_dot(self.grad, y)


# NOT PUBLIC, despite lacking an underscore. This is so pyright gets the name right.
class EvalGradHessianInv(FunctionInfo, Generic[Y], strict=True):
    """As [`optimistix.FunctionInfo.EvalGradHessian`][], but records the (approximate)
    inverse-Hessian instead. Has `.f` and `.grad` and `.hessian_inv` attributes.
    """

    f: Scalar
    grad: Y
    hessian_inv: lx.AbstractLinearOperator

    def as_min(self):
        return self.f

    def compute_grad_dot(self, y: Y):
        return tree_dot(self.grad, y)


# NOT PUBLIC, despite lacking an underscore. This is so pyright gets the name right.
class Residual(FunctionInfo, Generic[Out], strict=True):
    """Has a `.residual` attribute describing `fn(y)`. Used with least squares problems,
    for which `fn` returns residuals.
    """

    residual: Out

    def as_min(self):
        return 0.5 * sum_squares(self.residual)


# NOT PUBLIC, despite lacking an underscore. This is so pyright gets the name right.
class ResidualJac(FunctionInfo, Generic[Y, Out], strict=True):
    """Records the Jacobian `d(fn)/dy` as a linear operator. Used for least squares
    problems, for which `fn` returns residuals. Has `.residual` and `.jac` attributes,
    where `residual = fn(y)`, `jac = d(fn)/dy`.
    """

    residual: Out
    jac: lx.AbstractLinearOperator

    def as_min(self):
        return 0.5 * sum_squares(self.residual)

    def compute_grad(self):
        conj_residual = jtu.tree_map(jnp.conj, self.residual)
        return self.jac.transpose().mv(conj_residual)

    def compute_grad_dot(self, y: Y):
        # If `self.jac` is a `lx.JacobianLinearOperator` (or a
        # `lx.FunctionLinearOperator` wrapping the result of `jax.linearize`), then
        # `min = 0.5 * residual^2`, so `grad = jac^T residual`, i.e. the gradient of
        # this. So that what we want to compute is `residual^T jac y`. Doing the
        # reduction in this order means we hit forward-mode rather than reverse-mode
        # autodiff.
        #
        # For the complex case: in this case then actually
        # `min = 0.5 * residual residual^bar`
        # which implies
        # `grad = jac^T residual^bar`
        # and thus that we want
        # `grad^T^bar y = residual^T jac^bar y = (jac y^bar)^T^bar residual`.
        # Notes:
        # (a) the `grad` derivation is not super obvious. Note that
        # `grad(z -> 0.5 z z^bar)` is `z^bar` in JAX (yes, twice the Wirtinger
        # derivative!) It uses a non-Wirtinger derivative for nonholomorphic functions:
        # https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#complex-numbers-and-differentiation
        # (b) our convention is that the first term of a dot product gets the conjugate:
        # https://github.com/patrick-kidger/diffrax/pull/454#issuecomment-2210296643
        return tree_dot(self.jac.mv(jtu.tree_map(jnp.conj, y)), self.residual)


Eval.__qualname__ = "FunctionInfo.Eval"
EvalGrad.__qualname__ = "FunctionInfo.EvalGrad"
EvalGradHessian.__qualname__ = "FunctionInfo.EvalGradHessian"
EvalGradHessianInv.__qualname__ = "FunctionInfo.EvalGradHessianInv"
Residual.__qualname__ = "FunctionInfo.Residual"
ResidualJac.__qualname__ = "FunctionInfo.ResidualJac"
FunctionInfo.Eval = Eval
FunctionInfo.EvalGrad = EvalGrad
FunctionInfo.EvalGradHessian = EvalGradHessian
FunctionInfo.EvalGradHessianInv = EvalGradHessianInv
FunctionInfo.Residual = Residual
FunctionInfo.ResidualJac = ResidualJac


_FnInfo = TypeVar("_FnInfo", contravariant=True, bound=FunctionInfo)
_FnEvalInfo = TypeVar("_FnEvalInfo", contravariant=True, bound=FunctionInfo)


class AbstractDescent(eqx.Module, Generic[Y, _FnInfo, DescentState], strict=True):
    """The abstract base class for descents. A descent consumes a scalar (e.g. a step
    size), and returns the `diff` to take at point `y`, so that `y + diff` is the next
    iterate in a nonlinear optimisation problem.

    See [this documentation](./introduction.md) for more information.
    """

    @abc.abstractmethod
    def init(self, y: Y, f_info_struct: _FnInfo) -> DescentState:
        """Is called just once, at the very start of the entire optimisation problem.
        This is used to set up the evolving state for the descent.

        **Arguments:**

        - `y`: The initial guess for the optimisation problem.
        - `f_info`: An [`optimistix.FunctionInfo`][] describing information about `f`
            evaluated at `y`, and potentially the gradient of `f` at `y`, etc.

        **Returns:**

        The initial state, prior to doing any descents or searches.
        """

    @abc.abstractmethod
    def query(self, y: Y, f_info: _FnInfo, state: DescentState) -> DescentState:
        """Is called whenever the search decides to halt and accept its current iterate,
        and then query the descent for a new descent direction.

        This method can be used to precompute information that does not depend on the
        step size.

        For example, consider running a minimisation algorithm that performs multiple
        line searches. (E.g. nonlinear CG.) A single line search may reject many steps
        until it finds a location that it is happy with, and then accept that one. At
        that point, we must compute the direction for the next line search. That
        should be done in this method.

        **Arguments:**

        - `y`: the value of the current (just accepted) iterate.
        - `f_info`: An [`optimistix.FunctionInfo`][] describing information about `f`
            evaluated at `y`, the gradient of `f` at `y`, etc.
        - `state`: the evolving state of the repeated descents.

        **Returns:**

        The updated descent state.
        """

    @abc.abstractmethod
    def step(self, step_size: Scalar, state: DescentState) -> tuple[Y, RESULTS]:
        """Computes the descent of size `step_size`.

        **Arguments:**

        - `step_size`: a non-negative scalar describing the step size to take.
        - `state`: the evolving state of the repeated descents. This will typically
            convey most of the information about the problem, having been computed in
            [`optimistix.AbstractDescent.query`][].

        **Returns:**

        A 2-tuple of `(y_diff, result)`, describing the delta to apply in
        y-space, and any error if computing the update was not successful.
        """


class AbstractSearch(
    eqx.Module, Generic[Y, _FnInfo, _FnEvalInfo, SearchState], strict=True
):
    """The abstract base class for all searches. (Which are our generalisation of
    line searches, trust regions, and learning rates.)

    See [this documentation](./introduction.md) for more information.
    """

    @abc.abstractmethod
    def init(self, y: Y, f_info_struct: _FnInfo) -> SearchState:
        """Is called just once, at the very start of the entire optimisation problem.
        This is used to set up the evolving state for the search.

        **Arguments:**

        - `y`: The initial guess for the optimisation problem.
        - `f_info`: An [`optimistix.FunctionInfo`][] describing information about the
            shape/dtype of `f` evaluated at `y`, and potentially the gradient of `f` at
            `y`, etc.

        **Returns:**

        The initial state, prior to doing any descents or searches.
        """

    @abc.abstractmethod
    def step(
        self,
        first_step: Bool[Array, ""],
        y: Y,
        y_eval: Y,
        f_info: _FnInfo,
        f_eval_info: _FnEvalInfo,
        state: SearchState,
    ) -> tuple[Scalar, Bool[Array, ""], RESULTS, SearchState]:
        """Performs a step within a search. For example, one step within a line search.

        **Arguments:**

        - `first_step`: True on the first step, then False afterwards. On the first
            step, then `y` and `f_info` will be meaningless dummy data, as no steps have
            been accepted yet. The search will normally need to special-case this case.
            On the first step, it is assumed that `accept=True` will be returned.
        - `y`: the value of the most recently "accepted" iterate, i.e. the last point at
            which the descent was `query`'d.
        - `y_eval`: the value of the most recent iterate within the search.
        - `y_diff`: equal to `y_eval - y`. Provided as a convenience, to save some
            computation, as this is commonly available from the calling solver.
        - `f_info`: An [`optimistix.FunctionInfo`][] describing information about `f`
            evaluated at `y`, the gradient of `f` at `y`, etc.
        - `f_eval_info`: An [`optimistix.FunctionInfo`][] describing information about
            `f` evaluated at `y`, the gradient of `f` at `y`, etc.
        - `state`: the evolving state of the repeated searches.

        **Returns:**

        A 4-tuple of `(step_size, accept, result, state)`, where:

        - `step_size`: the size of the next step to make. (Relative to `y`, not
            `y_eval`.) Will be passed to the descent.
        - `accept`: whether this step should be "accepted", in which case the descent
            will be queried for a new direction. Must be `True` if `first_step=True`.
        - `result`: an [`optimistix.RESULTS`][] object, for declaring any errors.
        - `state`: the updated state for the next search.
        """
