import abc
from typing import Optional, Tuple, Type, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp

from ..custom_types import Bool, DenseInfo, PyTree, PyTreeDef, Scalar
from ..local_interpolation import AbstractLocalInterpolation
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

        **Arguments** are as `diffeqsolve`, with the exception that `y0` must be a
        flattened one-dimensional JAX array. (Obtained via
        `jax.flatten_util.ravel_pytree` if `y0` was originally a PyTree.)

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

        A step is made over some interval $[t_0, t_1]$.

        **Arguments:**

        - `t0`: The start of the interval that the step is made over.
        - `t1`: The end of the interval that the step is amde over.
        - `y0`: The current value of the solution at `t0`.
        - `args`: Any extra arguments passed to the vector field.
        - `solver_state`: Any evolving state for the solver itself, at `t0`.
        - `made_jump`: Whether there was a discontinuity in the vector field at `t0`.
            Some solvers (notably FSAL Runge--Kutta solvers) usually assume that there
            are no jumps and for efficiency re-use information between steps, which is
            kept in their `solver_state`.

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
            happened successfully, or if it failed for some reason. (e.g. no solution
            could be found to the nonlinear problem in an implicit solver.)
        """

    def func_for_init(
        self, terms: PyTree[AbstractTerm], t0: Scalar, y0: PyTree, args: PyTree
    ) -> PyTree:
        """Provides vector field evaluations to select the initial step size.

        Note that `step` operates over an interval, rather than acting at
        a point $t$. (This is important for handling SDEs and CDEs.)

        However, ODEs have one trick up their sleeve that require point evaluations:
        selecting the initial step size automatically.

        This function is used for that one purpose.

        **Arguments:**

        - `t0`: The initial point of the overall region of integration.
        - `y0`: The (ravelled) initial value for the ODE at `t0`.
        - `args`: Any extra arguments to pass to the vector field.

        **Returns:**

        The (ravelled) evaluation of the vector field at `t0`.
        """

        raise ValueError(
            "An initial step size cannot be selected automatically. The most common "
            "scenario for this error to occur is when trying to use adaptive step "
            "size solvers with SDEs. Please specify an initial `dt0` instead."
        )
