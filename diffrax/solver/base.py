import abc
from typing import Optional, Tuple, Type, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp

from ..custom_types import Bool, DenseInfo, PyTree, PyTreeDef, Scalar
from ..local_interpolation import AbstractLocalInterpolation
from ..nonlinear_solver import AbstractNonlinearSolver, NewtonNonlinearSolver
from ..solution import RESULTS
from ..term import AbstractTerm


_SolverState = TypeVar("SolverState", bound=Optional[PyTree])


def vector_tree_dot(a, b):
    return jax.tree_map(lambda bi: jnp.tensordot(a, bi, axes=1), b)


class AbstractSolver(eqx.Module):
    """Abstract base class for all differential equation solvers."""

    @property
    @abc.abstractmethod
    def term_structure(self) -> PyTreeDef:
        """What PyTree structure `terms` should have when used with this solver."""

    @property
    @abc.abstractmethod
    def interpolation_cls(self) -> Type[AbstractLocalInterpolation]:
        """How to interpolate the solution in between steps."""

    @property
    @abc.abstractmethod
    def order(self) -> int:
        """Order of the solver."""

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
    ) -> _SolverState:
        """Initialises any hidden state for the solver.

        **Arguments** as [`diffrax.diffeqsolve`][].

        **Returns:**

        The initial solver state, which should be used the first time `step` is called.
        """
        return None

    @abc.abstractmethod
    def step(
        self,
        terms: PyTree[AbstractTerm],
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
        solver_state: _SolverState,
        made_jump: Bool,
    ) -> Tuple[PyTree, Optional[PyTree], DenseInfo, _SolverState, RESULTS]:
        """Make a single step of the solver.

        Each step is made over the specified interval $[t_0, t_1]$.

        **Arguments:**

        - `terms`: The PyTree of terms representing the vector fields and controls.
        - `t0`: The start of the interval that the step is made over.
        - `t1`: The end of the interval that the step is made over.
        - `y0`: The current value of the solution at `t0`.
        - `args`: Any extra arguments passed to the vector field.
        - `solver_state`: Any evolving state for the solver itself, at `t0`.
        - `made_jump`: Whether there was a discontinuity in the vector field at `t0`.
            Some solvers (notably FSAL Runge--Kutta solvers) usually assume that there
            are no jumps and for efficiency re-use information between steps; this
            indicates that a jump has just occurred and this assumption is not true.

        **Returns:**

        A tuple of several objects:

        - The value of the solution at `t1`.
        - A local error estimate made during the step. (Used by adaptive step size
            controllers to change the step size.) May be `None` if no estimate was
            made.
        - Some dictionary of information that is passed to the solver's interpolation
            routine to calculate dense output. (Used with `SaveAt(ts=...)` or
            `SaveAt(dense=...)`.)
        - The value of the solver state at `t1`.
        - An integer (corresponding to `diffrax.RESULTS`) indicating whether the step
            happened successfully, or if (unusually) it failed for some reason.
        """

    def func_for_init(
        self, terms: PyTree[AbstractTerm], t0: Scalar, y0: PyTree, args: PyTree
    ) -> PyTree:
        """Provides vector field evaluations to select the initial step size.

        This is used to make a point evaluation. This is unlike
        [`diffrax.AbstractSolver.step`][], which operates over an interval.

        In general differential equation solvers are interval-based. There is precisely
        one place where point evaluations are needed: selecting the initial step size
        automatically in an ODE solve. And that is what this function is for.

        **Arguments:** As [`diffrax.diffeqsolve`][]

        **Returns:**

        The evaluation of the vector field at `t0`.
        """

        raise ValueError(
            "An initial step size cannot be selected automatically. The most common "
            "scenario for this error to occur is when trying to use adaptive step "
            "size solvers with SDEs. Please specify an initial `dt0` instead."
        )


class AbstractImplicitSolver(AbstractSolver):
    nonlinear_solver: AbstractNonlinearSolver = NewtonNonlinearSolver()
