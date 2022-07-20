import abc
from typing import Callable, Optional, Tuple, TypeVar

import equinox as eqx

from ..custom_types import Bool, PyTree, Scalar
from ..solution import RESULTS
from ..solver import AbstractSolver
from ..term import AbstractTerm


_ControllerState = TypeVar("ControllerState", bound=PyTree)


class AbstractStepSizeController(eqx.Module):
    """Abstract base class for all step size controllers."""

    @abc.abstractmethod
    def wrap(self, direction: Scalar) -> "AbstractStepSizeController":
        """Remakes this step size controller, adding additional information.

        Most step size controllers can't be used without first calling `wrap` to give
        them the extra information they need.

        **Arguments:**

        - `direction`: Either 1 or -1, indicating whether the integration is going to
            be performed forwards-in-time or backwards-in-time respectively.

        **Returns:**

        A copy of the the step size controller, updated to reflect the additional
        information.
        """

    def wrap_solver(self, solver: AbstractSolver) -> AbstractSolver:
        """Remakes the solver, adding additional information.

        Some step size controllers need to modify the solver slightly. For example,
        adaptive step size controllers can automatically set the tolerances used in
        implicit solvers.

        **Arguments:**

        - `solver`: The solver to modify.

        **Returns:**

        The modified solver.
        """
        return solver

    @abc.abstractmethod
    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        dt0: Optional[Scalar],
        args: PyTree,
        func: Callable[[Scalar, PyTree, PyTree], PyTree],
        error_order: Optional[Scalar],
    ) -> Tuple[Scalar, _ControllerState]:
        r"""Determines the size of the first step, and initialise any hidden state for
        the step size controller.

        **Arguments:** As `diffeqsolve`.

        - `func`: The value of `solver.func`.
        - `error_order`: The order of the error estimate. If solving an ODE this will
            typically be `solver.order()`. If solving an SDE this will typically be
            `solver.strong_order() + 0.5`.

        **Returns:**

        A 2-tuple of:

        - The endpoint $\tau$ for the initial first step: the first step will be made
            over the interval $[t_0, \tau]$. If `dt0` is specified (not `None`) then
            this is typically `t0 + dt0`. (Although in principle the step size
            controller doesn't have to respect this if it doesn't want to.)
        - The initial hidden state for the step size controller, which is used the
            first time `adapt_step_size` is called.
        """

    @abc.abstractmethod
    def adapt_step_size(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        y1_candidate: PyTree,
        args: PyTree,
        y_error: Optional[PyTree],
        error_order: Scalar,
        controller_state: _ControllerState,
    ) -> Tuple[Bool, Scalar, Scalar, Bool, _ControllerState, RESULTS]:
        """Determines whether to accept or reject the current step, and determines the
        step size to use on the next step.

        **Arguments:**

        - `t0`: The start of the interval that the current step was just made over.
        - `t1`: The end of the interval that the current step was just made over.
        - `y0`: The value of the solution at `t0`.
        - `y1_candidate`: The value of the solution at `t1`, as estimated by the main
            solver. Only a "candidate" as it is now up to the step size controller to
            accept or reject it.
        - `args`: Any extra arguments passed to the vector field; as
            [`diffrax.diffeqsolve`][].
        - `y_error`: An estimate of the local truncation error, as calculated by the
            main solver.
        - `error_order`: The order of `y_error`. For an ODE this is typically equal to
            `solver.order()`; for an SDE this is typically equal to
            `solver.strong_order() + 0.5`.
        - `controller_state`: Any evolving state for the step size controller itself,
            at `t0`.

        **Returns:**

        A tuple of several objects:

        - A boolean indicating whether the step was accepted/rejected.
        - The time at which the next step is to be started. (Typically equal to the
            argument `t1`, but not always -- if there was a vector field discontinuity
            there then it may be `nextafter(t1)` instead.)
        - The time at which the next step is to finish.
        - A boolean indicating whether a discontinuity in the vector field has just
            been passed. (Which for example some solvers use to recalculate their
            hidden state; in particular the FSAL property of some Runge--Kutta
            methods.)
        - The value of the step size controller state at `t1`.
        - An integer (corresponding to `diffrax.RESULTS`) indicating whether the step
            happened successfully, or if it failed for some reason. (e.g. hitting a
            minimum allowed step size in the solver.)
        """
