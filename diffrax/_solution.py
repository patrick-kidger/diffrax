from typing import Any, Optional

import jax
import optimistix as optx
from jaxtyping import Array, Bool, PyTree, Real, Shaped

from ._custom_types import BoolScalarLike, RealScalarLike
from ._global_interpolation import DenseInterpolation
from ._path import AbstractPath


class RESULTS(optx.RESULTS):  # pyright: ignore
    successful = ""
    discrete_terminating_event_occurred = (
        "Terminating differential equation solve because a discrete terminating event "
        "occurred."
    )
    max_steps_reached = (
        "The maximum number of solver steps was reached. Try increasing `max_steps`."
    )
    dt_min_reached = (
        "The minimum step size was reached in the differential equation solver."
    )


def is_okay(result: RESULTS) -> Bool[Array, ""]:
    with jax.ensure_compile_time_eval():  # for the `|` between two `Bool[Array, ""]`.
        return is_successful(result) | is_event(result)


def is_successful(result: RESULTS) -> Bool[Array, ""]:
    return result == RESULTS.successful


# TODO: In the future we may support other event types, in which case this function
# should be updated.
def is_event(result: RESULTS) -> Bool[Array, ""]:
    return result == RESULTS.discrete_terminating_event_occurred


def update_result(old_result: RESULTS, new_result: RESULTS) -> RESULTS:
    """
    Returns:

        old | success event_o error_o
    new     |
    --------+-------------------------
    success | success event_o error_o
    event_n | event_n event_o error_o
    error_n | error_n error_n error_o
    """
    out_result = RESULTS.where(is_okay(old_result), new_result, old_result)
    with jax.ensure_compile_time_eval():
        pred = is_okay(new_result) & is_event(old_result)
    return RESULTS.where(pred, old_result, out_result)


class Solution(AbstractPath):
    """The solution to a differential equation.

    **Attributes:**

    - `t0`: The start of the interval that the differential equation was solved over.
    - `t1`: The end of the interval that the differential equation was solved over.
    - `ts`: Some ordered collection of times. Might be `None` if no values were saved.
        (i.e. just `diffeqsolve(..., saveat=SaveAt(dense=True))` is used.)
    - `ys`: The value of the solution at each of the times in `ts`. Might `None` if no
        values were saved.
    - `stats`: Statistics for the solve (number of steps etc.).
    - `result`: Integer specifying the success or cause of failure of the solve. A
        value of `0` corresponds to a successful solve. Any other value is a failure.
        A human-readable message can be obtained by looking up messages via
        `diffrax.RESULTS[<integer>]`.
    - `solver_state`: If saved, the final internal state of the numerical solver.
    - `controller_state`: If saved, the final internal state for the step size
        controller.
    - `made_jump`: If saved, the final internal state for the jump tracker.

    !!! note

        If `diffeqsolve(..., saveat=SaveAt(steps=True))` is set, then the `ts` and `ys`
        in the solution object will be padded with `NaN`s, out to the value of
        `max_steps` passed to [`diffrax.diffeqsolve`][].

        This is because JAX demands that shapes be known statically ahead-of-time. As
        we do not know how many steps we will take until the solve is performed, we
        must allocate enough space for the maximum possible number of steps.
    """

    t0: RealScalarLike
    t1: RealScalarLike
    # SaveAt(subs=...) means that the pytree structures of `ts` and `ys` will lead with
    # the structure of `subs`.
    # SaveAt(fn=...) means that `ys` will then follow with arbitrary sub-dependent
    # PyTree structures.
    ts: Optional[PyTree[Real[Array, " ?times"], " S"]]
    ys: Optional[PyTree[Shaped[Array, "?times ?*shape"], "S ..."]]
    interpolation: Optional[DenseInterpolation]
    stats: dict[str, Any]
    result: RESULTS
    solver_state: Optional[PyTree]
    controller_state: Optional[PyTree]
    made_jump: Optional[BoolScalarLike]

    def evaluate(
        self, t0: RealScalarLike, t1: Optional[RealScalarLike] = None, left: bool = True
    ) -> PyTree[Shaped[Array, "?*shape"], " Y"]:
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
                "Dense solution has not been saved; pass SaveAt(dense=True)."
            )
        return self.interpolation.evaluate(t0, t1, left)

    def derivative(
        self, t: RealScalarLike, left: bool = True
    ) -> PyTree[Shaped[Array, "?*shape"], " Y"]:
        r"""If dense output was saved, then calculate an **approximation** to the
        derivative of the solution at any point in the region of integration `self.t0`
        to `self.t1`.

        That is, letting $y$ denote the solution over the interval `[t0, t1]`, then
        this calculates an approximation to $\frac{\mathrm{d}y}{\mathrm{d}t}$.

        (This is *not* backpropagating through the differential equation -- that
        typically corresponds to e.g. $\frac{\mathrm{d}y(t_1)}{\mathrm{d}y(t_0)}$.)

        !!! example

            For an ODE satisfying

            $\frac{\mathrm{d}y}{\mathrm{d}t} = f(t, y(t))$

            then this value is approximately equal to $f(t, y(t))$.

        !!! warning

            This value is generally not very accurate. Differential equation solvers
            are usually designed to produce splines whose value is close to the true
            solution; not to produce splines whose derivative is close to the
            derivative of the true solution.

            If you need accurate derivatives for the solution of an ODE, it is usually
            best to calculate `vector_field(t, sol.evaluate(t), args)`. That is, to
            pay the extra computational cost of another vector field evaluation, in
            order to get a more accurate value.

            Put precisely: this `derivative` method returns the *derivative of the
            numerical solution*, and *not* an approximation to the derivative of the
            true solution.

        **Arguments:**

        - `t`: The point to calculate the derivative of the solution at.
        - `left`: When evaluating at a jump in the solution, whether to return the
            left-limit or the right-limit at that point.
        """
        if self.interpolation is None:
            raise ValueError(
                "Dense solution has not been saved; pass SaveAt(dense=True)."
            )
        return self.interpolation.derivative(t, left)
