import abc
from collections.abc import Callable
from typing import (
    Any,
    ClassVar,
    Generic,
    get_args,
    get_origin,
    Optional,
    Type,
    TYPE_CHECKING,
    TypeVar,
)

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import optimistix as optx


if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar, ClassVar as AbstractVar
else:
    from equinox import AbstractClassVar, AbstractVar
from equinox.internal import ω
from jaxtyping import PyTree

from .._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
from .._heuristics import is_sde
from .._local_interpolation import AbstractLocalInterpolation
from .._solution import RESULTS, update_result
from .._term import AbstractTerm, MultiTerm


_SolverState = TypeVar("_SolverState")


def vector_tree_dot(a, b):
    return jtu.tree_map(
        lambda bi: jnp.tensordot(a, bi, axes=1, precision=lax.Precision.HIGHEST),
        b,
    )


class _MetaAbstractSolver(type(eqx.Module), type):
    def __instancecheck__(cls, obj):
        if super(_MetaAbstractSolver, AbstractWrappedSolver).__instancecheck__(obj):  # pyright: ignore
            # Either one will suffice.
            return super().__instancecheck__(obj) or super().__instancecheck__(
                obj.solver
            )
        else:
            return super().__instancecheck__(obj)


# Sneak the metaclass past pyright, as otherwise it disables the dataclass-ness of
# `eqx.Module`.
_set_metaclass = dict(metaclass=_MetaAbstractSolver)


def _term_compatible_contr_kwargs(term_structure):
    origin = get_origin(term_structure)
    if origin is MultiTerm:
        [terms] = get_args(term_structure)
        return tuple(_term_compatible_contr_kwargs(term) for term in get_args(terms))
    if origin is not None:
        term_structure = origin
    if isinstance(term_structure, type) and issubclass(term_structure, AbstractTerm):
        return {}
    return jtu.tree_map(_term_compatible_contr_kwargs, term_structure)


class AbstractSolver(eqx.Module, Generic[_SolverState], **_set_metaclass):
    """Abstract base class for all differential equation solvers.

    Subclasses should have a class-level attribute `terms`, specifying the PyTree
    structure of `terms` in `diffeqsolve(terms, ...)`.
    """

    # What PyTree structure `terms` should have when used with this solver.
    term_structure: AbstractClassVar[PyTree[Type[AbstractTerm]]]
    # How to interpolate the solution in between steps.
    interpolation_cls: AbstractClassVar[Callable[..., AbstractLocalInterpolation]]

    # Any keyword arguments needed in `_term_compatible` in `_integrate.py`.
    term_compatible_contr_kwargs: ClassVar[PyTree[dict[str, Any]]] = property(
        lambda self: _term_compatible_contr_kwargs(self.term_structure)
    )

    def order(self, terms: PyTree[AbstractTerm]) -> Optional[int]:
        """Order of the solver for solving ODEs."""
        return None

    def strong_order(self, terms: PyTree[AbstractTerm]) -> Optional[RealScalarLike]:
        """Strong order of the solver for solving SDEs."""
        return None

    def error_order(self, terms: PyTree[AbstractTerm]) -> Optional[RealScalarLike]:
        """Order of the error estimate used for adaptive stepping.

        The default (slightly heuristic) implementation is as follows.

        The error estimate is assumed to come from the difference of two methods. If
        these two methods have orders `p` and `q` then the local order of the error
        estimate is `min(p, q) + 1` for an ODE and `min(p, q) + 0.5` for an SDE.

        - In the SDE case then we assume `p == q == solver.strong_order()`.
        - In the ODE case then we assume `p == q + 1 == solver.order()`.
        - We assume that non-SDE/ODE cases do not arise.

        This is imperfect as these assumptions may not be true. In addition in the SDE
        case, then solvers will sometimes exhibit higher orders of convergence for
        specific noise types (see issue #47).
        """
        if is_sde(terms):
            order = self.strong_order(terms)
            if order is not None:
                order = order + 0.5
            return order
        else:
            return self.order(terms)

    @abc.abstractmethod
    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        """Initialises any hidden state for the solver.

        **Arguments** as [`diffrax.diffeqsolve`][].

        **Returns:**

        The initial solver state, which should be used the first time `step` is called.
        """

    @abc.abstractmethod
    def step(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Optional[Y], DenseInfo, _SolverState, RESULTS]:
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

    @abc.abstractmethod
    def func(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        """Evaluate the vector field at a point. (This is unlike
        [`diffrax.AbstractSolver.step`][], which operates over an interval.)

        For most operations differential equation solvers are interval-based, so this
        opertion should be used sparingly. This operation is needed for things like
        selecting an initial step size.

        **Arguments:** As [`diffrax.diffeqsolve`][]

        **Returns:**

        The evaluation of the vector field at `t0`, `y0`.
        """


class AbstractImplicitSolver(AbstractSolver[_SolverState]):
    """Indicates that this is an implicit differential equation solver, and as such
    that it should take a root finder as an argument.
    """

    root_finder: AbstractVar[optx.AbstractRootFinder]
    root_find_max_steps: AbstractVar[int]


class AbstractItoSolver(AbstractSolver[_SolverState]):
    """Indicates that when used as an SDE solver that this solver will converge to the
    Itô solution.
    """


class AbstractStratonovichSolver(AbstractSolver[_SolverState]):
    """Indicates that when used as an SDE solver that this solver will converge to the
    Stratonovich solution.
    """


class AbstractAdaptiveSolver(AbstractSolver[_SolverState]):
    """Indicates that this solver provides error estimates, and that as such it may be
    used with an adaptive step size controller.
    """


class AbstractWrappedSolver(AbstractSolver[_SolverState]):
    """Wraps another solver "transparently", in the sense that all `isinstance` checks
    will be forwarded on to the wrapped solver, e.g. when testing whether the solver is
    implicit/adaptive/SDE-compatible/etc.

    Inherit from this class if that is desired behaviour. (Do not inherit from this
    class if that is not desired behaviour.)
    """

    solver: AbstractVar[AbstractSolver]


class HalfSolver(
    AbstractAdaptiveSolver[_SolverState], AbstractWrappedSolver[_SolverState]
):
    """Wraps another solver, trading cost in order to provide error estimates. (That
    is, it means the solver can be used with an adaptive step size controller,
    regardless of whether the underlying solver supports adaptive step sizing.)

    For every step of the wrapped solver, it does this by also making two half-steps,
    and comparing the results between the full step and the two half steps. Hence the
    name "HalfSolver".

    As such each step costs 3 times the computational cost of the wrapped solver,
    whilst producing results that are roughly twice as accurate, in addition to
    producing error estimates.

    !!! tip

        Many solvers already provide error estimates, making `HalfSolver` primarily
        useful when using a solver that doesn't provide error estimates -- e.g.
        [`diffrax.Euler`][]. Such solvers are most common when solving SDEs.
    """

    solver: AbstractSolver[_SolverState]

    @property
    def term_structure(self):
        return self.solver.term_structure

    @property
    def interpolation_cls(self):  # pyright: ignore
        return self.solver.interpolation_cls

    @property
    def term_compatible_contr_kwargs(self):
        return self.solver.term_compatible_contr_kwargs

    def order(self, terms: PyTree[AbstractTerm]) -> Optional[int]:
        return self.solver.order(terms)

    def strong_order(self, terms: PyTree[AbstractTerm]) -> Optional[RealScalarLike]:
        return self.solver.strong_order(terms)

    def error_order(self, terms: PyTree[AbstractTerm]) -> Optional[RealScalarLike]:
        if is_sde(terms):
            order = self.strong_order(terms)
            if order is not None:
                order = order + 0.5
        else:
            order = self.order(terms)
            if order is not None:
                order = order + 1
        return order

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        return self.solver.init(terms, t0, t1, y0, args)

    def step(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Optional[Y], DenseInfo, _SolverState, RESULTS]:
        original_solver_state = solver_state
        thalf = t0 + 0.5 * (t1 - t0)

        yhalf, _, _, solver_state, result1 = self.solver.step(
            terms, t0, thalf, y0, args, solver_state, made_jump
        )
        y1, _, _, solver_state, result2 = self.solver.step(
            terms, thalf, t1, yhalf, args, solver_state, made_jump=False
        )

        # TODO: use dense_info from the pair of half-steps instead
        y1_alt, _, dense_info, _, result3 = self.solver.step(
            terms, t0, t1, y0, args, original_solver_state, made_jump
        )

        y_error = (y1**ω - y1_alt**ω).call(jnp.abs).ω
        result = update_result(result1, update_result(result2, result3))

        return y1, y_error, dense_info, solver_state, result

    def func(
        self, terms: PyTree[AbstractTerm], t0: RealScalarLike, y0: Y, args: Args
    ) -> VF:
        return self.solver.func(terms, t0, y0, args)


HalfSolver.__init__.__doc__ = """**Arguments:**

- `solver`: The solver to wrap.
"""
