import typing
from dataclasses import field
from typing import Any, Dict, Optional

from .custom_types import Array, PyTree, Scalar
from .global_interpolation import DenseInterpolation
from .misc import ContainerMeta
from .path import AbstractPath


class RESULTS(metaclass=ContainerMeta):
    successful = ""
    max_steps_reached = "The maximum number of solver steps was reached."
    dt_min_reached = "The minimum step size was reached."
    nan_time = "NaN time encountered during timestepping."
    implicit_divergence = "Implicit method diverged."
    implicit_nonconvergence = (
        "Implicit method did not converge within the required number of iterations."
    )


if getattr(typing, "GENERATING_DOCUMENTATION", False):
    RESULTS = int  # noqa: F811


class Solution(AbstractPath):
    """The solution to a differential equation.

    **Attributes:**

    - `t0`: The start of the interval that the differential equation was solved over.
    - `t1`: The end of the interval that the differential equation was solved over.
    - `ts`: Some ordered collection of times. Might be `None` if no values were saved.
        (i.e. just `diffeqsolve(..., saveat=SaveAt(dense=True))` is used.)
    - `ys`: The value of the solution at each of the times in `ts`. Might `None` if no
        values were saved.
    - `solver_state`: If saved, is the final internal state of the numerical solver.
    - `controller_state`: If saved, is the final internal state for the step size
        controller.
    - `stats`: Statistics for the solve (number of steps etc.).
    - `result`: Integer specifying the success or cause of failure of the solve. A
        value of `0` corresponds to a successful solve. Any other value is a failure;
        a human-readable message can be obtained via [`diffrax.Solution.message`][] or
        via `diffrax.RESULTS[result]`.
    """

    t0: Scalar = field(init=True)
    t1: Scalar = field(init=True)  # override init=False in AbstractPath
    ts: Optional[Array["times"]]  # noqa: F821
    ys: Optional[PyTree["times", ...]]  # noqa: F821
    solver_state: Optional[PyTree]
    controller_state: Optional[PyTree]
    interpolation: Optional[DenseInterpolation]
    stats: Dict[str, Any]
    result: RESULTS

    @property
    def message(self):
        """Human-readable version of `result`."""
        return RESULTS[self.result]

    def evaluate(
        self, t0: Scalar, t1: Optional[Scalar] = None, left: bool = True
    ) -> PyTree:
        """If dense output was saved, then evaluate the solution at any point in the
        region of integration `self.t0` to `self.t1`.

        **Arguments:**

        - `t0`: The point to evaluate the solution at.
        - `t1`: If passed, then the increment from `t0` to `t1` is returned.
            (`=evaluate(t1) - evaluate(t0)`)
        - `left`: When evaluating at a jump in the solution, whether to return the
            left-limit or the right-limit at that point.
        """
        if self.interpolation is None:
            raise ValueError(
                "Dense solution has not been saved; pass saveat.dense=True."
            )
        return self.interpolation.evaluate(t0, t1, left)

    def derivative(self, t: Scalar, left: bool = True) -> PyTree:
        r"""If dense output was saved, then calculate an approximation to the
        derivative of the solution at any point in the region of integration `self.t0`
        to `self.t1`.

        That is, letting $y$ denote the solution over the interval `[t0, t1]`, then
        this calculates an approximation to $\frac{\mathrm{d}y}{\mathrm{d}t}$.

        (This is *not* backpropagating through the differential equation -- that
        typically corresponds to compute something like
        $\frac{\mathrm{d}y(t_1)}{\mathrm{d}y(t_0)}$.)

        !!! note
            For an ODE satisfying

            $\frac{\mathrm{d}y}{\mathrm{d}t} = f(t, y(t))$

            then this value is approximately equal to $f(t, y(t))$.

        !!! warning
            The value calculated here is not necessarily very close to the
            derivative of the true solution.

            Differential equation solvers typically produce dense output as a spline.
            The value returned here is the derivative of that spline.

            This spline will be close to the true solution of the differential
            equation, but this does not mean that the derivative will be.

            Thus, precisely, this `derivative` method returns the *derivative of the
            numerical solution*, and *not* an approximation to the derivative of the
            true solution.

            If solving an ODE and wanting the derivative to the true solution, then
            evaluating the vector field on `self.evaluate(t)` will typically be much
            more accurate.

        **Arguments:**

        - `t`: The point to calculate the derivative of the solution at.
        - `left`: When evaluating at a jump in the solution, whether to return the
            left-limit or the right-limit at that point.
        """
        if self.interpolation is None:
            raise ValueError(
                "Dense solution has not been saved; pass saveat.dense=True."
            )
        return self.interpolation.derivative(t, left)
