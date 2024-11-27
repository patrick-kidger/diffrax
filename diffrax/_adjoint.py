import abc
import functools as ft
import warnings
from collections.abc import Callable, Iterable
from typing import Any, cast, Optional, TypeAlias, TypeVar, Union

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
import optimistix.internal as optxi
from equinox.internal import ω
from jaxtyping import PyTree

from ._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
from ._heuristics import is_sde, is_unsafe_sde
from ._saveat import save_y, SaveAt, SubSaveAt
from ._solution import RESULTS, update_result
from ._solver import (
    AbstractAdaptiveSolver,
    AbstractItoSolver,
    AbstractRungeKutta,
    AbstractSolver,
    AbstractStratonovichSolver,
    AbstractWrappedSolver,
    LeapfrogMidpoint,
    ReversibleHeun,
    SemiImplicitEuler,
)
from ._term import AbstractTerm, AdjointTerm


ω = cast(Callable, ω)


def _is_none(x):
    return x is None


def _is_subsaveat(x: Any) -> bool:
    return isinstance(x, SubSaveAt)


def _nondiff_solver_controller_state(
    adjoint, init_state, passed_solver_state, passed_controller_state
):
    if passed_solver_state:
        name = (
            f"When using `adjoint={adjoint.__class__.__name__}()`, then `solver_state`"
        )
        solver_fn = ft.partial(
            eqxi.nondifferentiable,
            name=name,
        )
    else:
        solver_fn = lax.stop_gradient
    if passed_controller_state:
        name = (
            f"When using `adjoint={adjoint.__class__.__name__}()`, then "
            "`controller_state`"
        )
        controller_fn = ft.partial(
            eqxi.nondifferentiable,
            name=name,
        )
    else:
        controller_fn = lax.stop_gradient
    init_state = eqx.tree_at(
        lambda s: s.solver_state,
        init_state,
        replace_fn=solver_fn,
        is_leaf=_is_none,
    )
    init_state = eqx.tree_at(
        lambda s: s.controller_state,
        init_state,
        replace_fn=controller_fn,
        is_leaf=_is_none,
    )
    return init_state


def _only_transpose_ys(final_state):
    from ._integrate import SaveState

    is_save_state = lambda x: isinstance(x, SaveState)

    def get_ys(_final_state):
        return [
            s.ys
            for s in jtu.tree_leaves(_final_state.save_state, is_leaf=is_save_state)
        ]

    ys = get_ys(final_state)

    named_nondiff_entries = (
        "y",
        "tprev",
        "tnext",
        "solver_state",
        "controller_state",
        "dense_ts",
        "dense_infos",
    )
    named_nondiff_values = tuple(
        eqxi.nondifferentiable_backward(getattr(final_state, k), name=k, symbolic=False)
        for k in named_nondiff_entries
    )

    final_state = eqxi.nondifferentiable_backward(final_state, symbolic=False)

    get_named_nondiff_entries = lambda s: tuple(
        getattr(s, k) for k in named_nondiff_entries
    )
    final_state = eqx.tree_at(
        get_named_nondiff_entries, final_state, named_nondiff_values, is_leaf=_is_none
    )

    final_state = eqx.tree_at(get_ys, final_state, ys)
    return final_state


class AbstractAdjoint(eqx.Module):
    """Abstract base class for all adjoint methods."""

    @abc.abstractmethod
    def loop(
        self,
        *,
        args,
        terms,
        solver,
        stepsize_controller,
        event,
        saveat,
        t0,
        t1,
        dt0,
        max_steps,
        throw,
        init_state,
        passed_solver_state,
        passed_controller_state,
        progress_meter,
    ) -> Any:
        """Runs the main solve loop. Subclasses can override this to provide custom
        backpropagation behaviour; see for example the implementation of
        [`diffrax.BacksolveAdjoint`][].
        """

    # Eurgh, delayed imports to handle circular dependencies.
    #
    # `integrate.py` defines the forward pass. `adjoint.py` defines the backward pass.
    # These pretty much necessarily depend on each other:
    #
    # - diffeqsolve needs to know about AbstractAdjoint, since it's one its arguments.
    # - BacksolveAdjoint needs to know about how to integrate a differential equation,
    #   since that's what it does.
    #
    # As such we get a circular dependency. We resolve it by lazily importing from
    # `integrate.py`. For convenience we make them available as properties here so all
    # adjoint methods can access these.
    @property
    def _loop(self):
        from ._integrate import loop

        return loop

    @property
    def _diffeqsolve(self):
        from ._integrate import diffeqsolve

        return diffeqsolve


_inner_loop = jax.named_call(eqxi.while_loop, name="inner-loop")
_outer_loop = jax.named_call(eqxi.while_loop, name="outer-loop")


def _uncallable(*args, **kwargs):
    assert False


class RecursiveCheckpointAdjoint(AbstractAdjoint):
    """Backpropagate through [`diffrax.diffeqsolve`][] by differentiating the numerical
    solution directly. This is sometimes known as "discretise-then-optimise", or
    described as "backpropagation through the solver".

    Uses a binomial checkpointing scheme to keep memory usage low.

    For most problems this is the preferred technique for backpropagating through a
    differential equation.

    !!! info

        Note that this cannot be forward-mode autodifferentiated. (E.g. using
        `jax.jvp`.) Try using [`diffrax.DirectAdjoint`][] if that is something you need.

    ??? cite "References"

        Selecting which steps at which to save checkpoints (and when this is done, which
        old checkpoint to evict) is important for minimising the amount of recomputation
        performed.

        The implementation here performs "online checkpointing", as the number of steps
        is not known in advance. This was developed in:

        ```bibtex
        @article{stumm2010new,
            author = {Stumm, Philipp and Walther, Andrea},
            title = {New Algorithms for Optimal Online Checkpointing},
            journal = {SIAM Journal on Scientific Computing},
            volume = {32},
            number = {2},
            pages = {836--854},
            year = {2010},
            doi = {10.1137/080742439},
        }

        @article{wang2009minimal,
            author = {Wang, Qiqi and Moin, Parviz and Iaccarino, Gianluca},
            title = {Minimal Repetition Dynamic Checkpointing Algorithm for Unsteady
                     Adjoint Calculation},
            journal = {SIAM Journal on Scientific Computing},
            volume = {31},
            number = {4},
            pages = {2549--2567},
            year = {2009},
            doi = {10.1137/080727890},
        }
        ```

        For reference, the classical "offline checkpointing" (also known as "treeverse",
        "recursive binary checkpointing", "revolve" etc.) was developed in:

        ```bibtex
        @article{griewank1992achieving,
            author = {Griewank, Andreas},
            title = {Achieving logarithmic growth of temporal and spatial complexity in
                     reverse automatic differentiation},
            journal = {Optimization Methods and Software},
            volume = {1},
            number = {1},
            pages = {35--54},
            year  = {1992},
            publisher = {Taylor & Francis},
            doi = {10.1080/10556789208805505},
        }

        @article{griewank2000revolve,
            author = {Griewank, Andreas and Walther, Andrea},
            title = {Algorithm 799: Revolve: An Implementation of Checkpointing for the
                     Reverse or Adjoint Mode of Computational Differentiation},
            year = {2000},
            publisher = {Association for Computing Machinery},
            volume = {26},
            number = {1},
            doi = {10.1145/347837.347846},
            journal = {ACM Trans. Math. Softw.},
            pages = {19--45},
        }
        ```
    """

    checkpoints: Optional[int] = None

    def loop(
        self,
        *,
        terms,
        saveat,
        init_state,
        max_steps,
        throw,
        passed_solver_state,
        passed_controller_state,
        **kwargs,
    ):
        del throw, passed_solver_state, passed_controller_state
        if is_unsafe_sde(terms):
            raise ValueError(
                "`adjoint=RecursiveCheckpointAdjoint()` does not support "
                "`UnsafeBrownianPath`. Consider using `adjoint=DirectAdjoint()` "
                "instead."
            )
        if self.checkpoints is None and max_steps is None:
            inner_while_loop = ft.partial(_inner_loop, kind="lax")
            outer_while_loop = ft.partial(_outer_loop, kind="lax")
            msg = (
                "Cannot reverse-mode autodifferentiate when using "
                "`diffeqsolve(..., max_steps=None, adjoint=RecursiveCheckpointAdjoint(checkpoints=None))`. "  # noqa: E501
                "This is because JAX needs to know how much memory to allocate for "
                "saving the forward pass. You should either put a bound on the maximum "
                "number of steps, or explicitly specify how many checkpoints to use."
            )
        else:
            inner_while_loop = ft.partial(_inner_loop, kind="checkpointed")
            outer_while_loop = ft.partial(
                _outer_loop, kind="checkpointed", checkpoints=self.checkpoints
            )
            msg = None
        final_state = self._loop(
            terms=terms,
            saveat=saveat,
            init_state=init_state,
            max_steps=max_steps,
            inner_while_loop=inner_while_loop,
            outer_while_loop=outer_while_loop,
            **kwargs,
        )
        if msg is not None:
            final_state = eqxi.nondifferentiable_backward(
                final_state, msg=msg, symbolic=True
            )
        return final_state


RecursiveCheckpointAdjoint.__init__.__doc__ = """
**Arguments:**

- `checkpoints`: the number of checkpoints to save. The amount of memory used by the
    differential equation solve will be roughly equal to the number of checkpoints
    multiplied by the size of `y0`. You can speed up backpropagation by allocating more
    checkpoints. (So it makes sense to set as many checkpoints as you have memory for.)
    This value can also be set to `None` (the default), in which case it will be set to
    `log(max_steps)`, for which a theoretical result is available guaranteeing that
    backpropagation will take `O(n log n)` time in the number of steps `n <= max_steps`.

You must pass either `diffeqsolve(..., max_steps=...)` or
`RecursiveCheckpointAdjoint(checkpoints=...)` to be able to backpropagate; otherwise
the computation will not be autodifferentiable.
"""


class DirectAdjoint(AbstractAdjoint):
    """A variant of [`diffrax.RecursiveCheckpointAdjoint`][]. The differences are that
    `DirectAdjoint`:

    - Is less time+memory efficient at reverse-mode autodifferentiation (specifically,
      these will increase every time `max_steps` increases passes a power of 16);
    - Cannot be reverse-mode autodifferentated if `max_steps is None`;
    - Supports forward-mode autodifferentiation.

    So unless you need forward-mode autodifferentiation then
    [`diffrax.RecursiveCheckpointAdjoint`][] should be preferred.
    """

    def loop(
        self,
        *,
        solver,
        max_steps,
        terms,
        throw,
        passed_solver_state,
        passed_controller_state,
        **kwargs,
    ):
        del throw, passed_solver_state, passed_controller_state
        # TODO: remove the `is_unsafe_sde` guard.
        # We need JAX to release bloops, so that we can deprecate `kind="bounded"`.
        if is_unsafe_sde(terms):
            kind = "lax"
            msg = (
                "Cannot reverse-mode autodifferentiate when using "
                "`UnsafeBrownianPath`."
            )
        elif max_steps is None:
            kind = "lax"
            msg = (
                "Cannot reverse-mode autodifferentiate when using "
                "`diffeqsolve(..., max_steps=None, adjoint=DirectAdjoint())`. "
                "This is because JAX needs to know how much memory to allocate for "
                "saving the forward pass. You should either put a bound on the maximum "
                "number of steps, or switch to "
                "`adjoint=RecursiveCheckpointAdjoint(checkpoints=...)`, with an "
                "explicitly specified number of checkpoints."
            )
        else:
            kind = "bounded"
            msg = None
        # Support forward-mode autodiff.
        # TODO: remove this hack once we can JVP through custom_vjps.
        if isinstance(solver, AbstractRungeKutta) and solver.scan_kind is None:
            solver = eqx.tree_at(
                lambda s: s.scan_kind, solver, "bounded", is_leaf=_is_none
            )
        inner_while_loop = ft.partial(_inner_loop, kind=kind)
        outer_while_loop = ft.partial(_outer_loop, kind=kind)
        final_state = self._loop(
            **kwargs,
            solver=solver,
            max_steps=max_steps,
            terms=terms,
            inner_while_loop=inner_while_loop,
            outer_while_loop=outer_while_loop,
        )
        if msg is not None:
            final_state = eqxi.nondifferentiable_backward(
                final_state, msg=msg, symbolic=True
            )
        return final_state


def _vf(ys, residual, inputs):
    state_no_y, _ = residual
    t = state_no_y.tprev

    def _unpack(_y):
        (_y1,) = _y
        return _y1

    y = jtu.tree_map(_unpack, ys)
    args, terms, _, _, solver, _, _ = inputs
    return solver.func(terms, t, y, args)


def _solve(inputs):
    args, terms, self, kwargs, solver, saveat, init_state = inputs
    final_state, aux_stats = self._loop(
        **kwargs,
        args=args,
        terms=terms,
        solver=solver,
        saveat=saveat,
        init_state=init_state,
        inner_while_loop=ft.partial(_inner_loop, kind="lax"),
        outer_while_loop=ft.partial(_outer_loop, kind="lax"),
    )
    # Note that we use .ys not .y here. The former is what is actually returned
    # by diffeqsolve, so it is the thing we want to attach the tangent to.
    #
    # Note that `final_state.save_state` has type PyTree[SaveState]. To access `.ys`
    # we are assuming that this PyTree has trivial structure. This is the case because
    # of the guard in `ImplicitAdjoint` that `saveat` be `SaveAt(t1=True)`.
    return final_state.save_state.ys, (
        eqx.tree_at(lambda s: s.save_state.ys, final_state, None),
        aux_stats,
    )


# Unwrap jaxtyping decorator during tests, so that these are global functions.
# This is needed to ensure `optx.implicit_jvp` is happy.
if _vf.__globals__["__name__"].startswith("jaxtyping"):
    _vf = _vf.__wrapped__  # pyright: ignore[reportFunctionMemberAccess]
if _solve.__globals__["__name__"].startswith("jaxtyping"):
    _solve = _solve.__wrapped__  # pyright: ignore[reportFunctionMemberAccess]


def _frozenset(x: Union[object, Iterable[object]]) -> frozenset[object]:
    try:
        iter_x = iter(x)  # pyright: ignore
    except TypeError:
        return frozenset([x])
    else:
        return frozenset(iter_x)


class ImplicitAdjoint(AbstractAdjoint):
    r"""Backpropagate via the [implicit function theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem#Statement_of_the_theorem).

    This is used when solving towards a steady state, typically using
    [`diffrax.Event`][] where the condition function is obtained by calling
    [`diffrax.steady_state_event`][]. In this case, the output of the solver is $y(θ)$
    for which $f(t, y(θ), θ) = 0$. (Where $θ$ corresponds to all parameters found
    through `terms` and `args`, but not `y0`.) Then we can skip backpropagating through
    the solver and instead directly compute
    $\frac{\mathrm{d}y}{\mathrm{d}θ} = - (\frac{\mathrm{d}f}{\mathrm{d}y})^{-1}\frac{\mathrm{d}f}{\mathrm{d}θ}$
    via the implicit function theorem.

    Observe that this involves solving a linear system with matrix given by the Jacobian
    `df/dy`.
    """  # noqa: E501

    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)
    tags: frozenset[object] = eqx.field(
        default_factory=frozenset, converter=_frozenset, static=True
    )

    def loop(
        self,
        *,
        args,
        terms,
        solver,
        saveat,
        throw,
        init_state,
        passed_solver_state,
        passed_controller_state,
        **kwargs,
    ):
        del throw

        # `is` check because this may return a Tracer from SaveAt(ts=<array>)
        if eqx.tree_equal(saveat, SaveAt(t1=True)) is not True:
            raise ValueError(
                "Can only use `adjoint=ImplicitAdjoint()` with "
                "`saveat=SaveAt(t1=True)`."
            )
        init_state = _nondiff_solver_controller_state(
            self, init_state, passed_solver_state, passed_controller_state
        )
        inputs = (args, terms, self, kwargs, solver, saveat, init_state)
        ys, residual = optxi.implicit_jvp(
            _solve, _vf, inputs, self.tags, self.linear_solver
        )

        final_state_no_ys, aux_stats = residual
        # Note that `final_state.save_state` has type PyTree[SaveState]. To access `.ys`
        # we are assuming that this PyTree has trivial structure. This is the case
        # because of the guard that `saveat` be `SaveAt(t1=True)`.
        final_state = eqx.tree_at(
            lambda s: s.save_state.ys, final_state_no_ys, ys, is_leaf=_is_none
        )
        final_state = _only_transpose_ys(final_state)
        return final_state, aux_stats


ImplicitAdjoint.__init__.__doc__ = """**Arguments:**

- `linear_solver`: A [Lineax](https://github.com/google/lineax) solver for solving the
    linear system.
- `tags`: Any Lineax [tags](https://docs.kidger.site/lineax/api/tags/) describing the
    Jacobian matrix `df/dy`.
"""


# Compute derivatives with respect to the first argument:
# - y, corresponding to the initial state;
# - args, corresponding to explicit parameters;
# - terms, corresponding to implicit parameters as part of the vector field.
@eqx.filter_custom_vjp
def _loop_backsolve(y__args__terms, *, self, throw, init_state, **kwargs):
    del throw
    y, args, terms = y__args__terms
    init_state = eqx.tree_at(lambda s: s.y, init_state, y)
    del y
    return self._loop(
        args=args,
        terms=terms,
        init_state=init_state,
        inner_while_loop=ft.partial(_inner_loop, kind="lax"),
        outer_while_loop=ft.partial(_outer_loop, kind="lax"),
        **kwargs,
    )


@_loop_backsolve.def_fwd
def _loop_backsolve_fwd(perturbed, y__args__terms, **kwargs):
    del perturbed
    final_state, aux_stats = _loop_backsolve(y__args__terms, **kwargs)
    # Note that `final_state.save_state` has type `PyTree[SaveState]`; here we are
    # relying on the guard in `BacksolveAdjoint` that it have trivial structure.
    ts = final_state.save_state.ts
    ys = final_state.save_state.ys
    return (final_state, aux_stats), (ts, ys)


def _materialise_none(y, grad_y):
    if grad_y is None and eqx.is_inexact_array(y):
        return jnp.zeros_like(y)
    else:
        return grad_y


@_loop_backsolve.def_bwd
def _loop_backsolve_bwd(
    residuals,
    grad_final_state__aux_stats,
    perturbed,
    y__args__terms,
    *,
    self,
    solver,
    stepsize_controller,
    event,
    saveat,
    t0,
    t1,
    dt0,
    max_steps,
    throw,
    init_state,
    progress_meter,
):
    assert event is None

    #
    # Unpack our various arguments. Delete a lot of things just to make sure we're not
    # using them later.
    #

    del perturbed, init_state, t1, progress_meter
    ts, ys = residuals
    del residuals
    grad_final_state, _ = grad_final_state__aux_stats
    # Note that `grad_final_state.save_state` has type `PyTree[SaveState]`; here we are
    # relying on the guard in `BacksolveAdjoint` that it have trivial structure.
    grad_ys = grad_final_state.save_state.ys
    # We take the simple way out and don't try to handle symbolic zeros.
    grad_ys = jtu.tree_map(_materialise_none, ys, grad_ys)
    del grad_final_state, grad_final_state__aux_stats
    y, args, terms = y__args__terms
    del y__args__terms
    diff_args = eqx.filter(args, eqx.is_inexact_array)
    diff_terms = eqx.filter(terms, eqx.is_inexact_array)
    zeros_like_y = jtu.tree_map(jnp.zeros_like, y)
    zeros_like_diff_args = jtu.tree_map(jnp.zeros_like, diff_args)
    zeros_like_diff_terms = jtu.tree_map(jnp.zeros_like, diff_terms)
    del diff_args, diff_terms
    # TODO: have this look inside MultiTerms? Need to think about the math. i.e.:
    # is_leaf=lambda x: isinstance(x, AbstractTerm) and not isinstance(x, MultiTerm)
    adjoint_terms = jtu.tree_map(
        AdjointTerm, terms, is_leaf=lambda x: isinstance(x, AbstractTerm)
    )
    diffeqsolve = self._diffeqsolve
    kwargs = dict(
        args=args,
        adjoint=self,
        solver=solver,
        stepsize_controller=stepsize_controller,
        terms=adjoint_terms,
        dt0=None if dt0 is None else -dt0,
        max_steps=max_steps,
        throw=throw,
    )
    kwargs.update(self.kwargs)
    del self, solver, stepsize_controller, adjoint_terms, dt0, max_steps, throw
    del y, args, terms
    # Note that `saveat.subs` has type `PyTree[SubSaveAt]`. Here we use the assumption
    # (checked in `BacksolveAdjoint`) that it has trivial pytree structure.
    saveat_t0 = saveat.subs.t0
    del saveat

    #
    # Now run a scan backwards in time, diffeqsolve'ing between each pair of adjacent
    # timestamps.
    #

    def _scan_fun(_state, _vals, first=False):
        _t1, _t0, _y0, _grad_y0 = _vals
        _a0, _solver_state, _controller_state = _state
        _a_y0, _a_diff_args0, _a_diff_term0 = _a0
        _a_y0 = (_a_y0**ω + _grad_y0**ω).ω
        _aug0 = (_y0, _a_y0, _a_diff_args0, _a_diff_term0)

        _sol = diffeqsolve(
            t0=_t0,
            t1=_t1,
            y0=_aug0,
            solver_state=_solver_state,
            controller_state=_controller_state,
            made_jump=not first,  # Adding _grad_y0, above, is a jump.
            saveat=SaveAt(t1=True, solver_state=True, controller_state=True),
            **kwargs,
        )

        def __get(__aug):
            assert __aug.shape[0] == 1
            return __aug[0]

        _aug1 = ω(_sol.ys).call(__get).ω
        _, _a_y1, _a_diff_args1, _a_diff_term1 = _aug1
        _a1 = (_a_y1, _a_diff_args1, _a_diff_term1)
        _solver_state = _sol.solver_state
        _controller_state = _sol.controller_state

        return (_a1, _solver_state, _controller_state), None

    state = ((zeros_like_y, zeros_like_diff_args, zeros_like_diff_terms), None, None)
    del zeros_like_y, zeros_like_diff_args, zeros_like_diff_terms

    # We always start backpropagating from `ts[-1]`.
    # We always finish backpropagating at `t0`.
    #
    # We may or may not have included `t0` in `ts`. (Depending on the value of
    # SaveaAt(t0=...) on the forward pass.)
    #
    # For some of these options, we run _scan_fun once outside the loop to get access
    # to solver_state etc. of the correct PyTree structure.
    if saveat_t0:
        if len(ts) > 2:
            val0 = (ts[-2], ts[-1], ω(ys)[-1].ω, ω(grad_ys)[-1].ω)
            state, _ = _scan_fun(state, val0, first=True)
            vals = (
                ts[:-2],
                ts[1:-1],
                ω(ys)[1:-1].ω,
                ω(grad_ys)[1:-1].ω,
            )
            state, _ = lax.scan(_scan_fun, state, vals, reverse=True)

        elif len(ts) == 1:
            # nothing to do, diffeqsolve is the identity when merely SaveAt(t0=True).
            pass

        else:
            assert len(ts) == 2
            val = (ts[0], ts[1], ω(ys)[1].ω, ω(grad_ys)[1].ω)
            state, _ = _scan_fun(state, val, first=True)

        aug1, _, _ = state
        a_y1, a_diff_args1, a_diff_terms1 = aug1
        a_y1 = (ω(a_y1) + ω(grad_ys)[0]).ω

    else:
        if len(ts) > 1:
            # TODO: fold this `_scan_fun` into the `lax.scan`. This will reduce compile
            # time.
            val0 = (ts[-2], ts[-1], ω(ys)[-1].ω, ω(grad_ys)[-1].ω)
            state, _ = _scan_fun(state, val0, first=True)
            vals = (
                jnp.concatenate([t0[None], ts[:-2]]),
                ts[:-1],
                ω(ys)[:-1].ω,
                ω(grad_ys)[:-1].ω,
            )
            state, _ = lax.scan(_scan_fun, state, vals, reverse=True)

        else:
            assert len(ts) == 1
            val = (t0, ts[0], ω(ys)[0].ω, ω(grad_ys)[0].ω)
            state, _ = _scan_fun(state, val, first=True)

        aug1, _, _ = state
        a_y1, a_diff_args1, a_diff_terms1 = aug1

    return a_y1, a_diff_args1, a_diff_terms1


class BacksolveAdjoint(AbstractAdjoint):
    """Backpropagate through [`diffrax.diffeqsolve`][] by solving the continuous
    adjoint equations backwards-in-time. This is also sometimes known as
    "optimise-then-discretise", the "continuous adjoint method" or simply the "adjoint
    method".

    This method implies very low memory usage, but the
    computed gradients will only be approximate. As such other methods are generally
    preferred unless exceeding memory is a concern.

    This will compute gradients with respect to the `terms`, `y0` and `args` arguments
    passed to [`diffrax.diffeqsolve`][]. If you attempt to compute gradients with
    respect to anything else (for example `t0`, or arguments passed via closure), then
    a `CustomVJPException` will be raised. See also
    [this FAQ](../../further_details/faq/#im-getting-a-customvjpexception)
    entry.

    !!! note

        This was popularised by [this paper](https://arxiv.org/abs/1806.07366). For
        this reason it is sometimes erroneously believed to be a better method for
        backpropagation than the other choices available.

    !!! warning

        Using this method prevents computing forward-mode autoderivatives of
        [`diffrax.diffeqsolve`][]. (That is to say, `jax.jvp` will not work.)
    """  # noqa: E501

    kwargs: dict[str, Any]

    def __init__(self, **kwargs):
        """
        **Arguments:**

        - `**kwargs`: The arguments for the [`diffrax.diffeqsolve`][] operations that
            are called on the backward pass. For example use
            ```python
            BacksolveAdjoint(solver=Dopri5())
            ```
            to specify a particular solver to use on the backward pass.
        """
        valid_keys = {
            "dt0",
            "solver",
            "stepsize_controller",
            "adjoint",
            "max_steps",
            "throw",
        }
        given_keys = set(kwargs.keys())
        diff_keys = given_keys - valid_keys
        if len(diff_keys) > 0:
            raise ValueError(
                "The following keyword argments are not valid for `BacksolveAdjoint`: "
                f"{diff_keys}"
            )
        self.kwargs = kwargs

    def loop(
        self,
        *,
        args,
        terms,
        solver,
        saveat,
        init_state,
        passed_solver_state,
        passed_controller_state,
        event,
        **kwargs,
    ):
        if jtu.tree_structure(saveat.subs, is_leaf=_is_subsaveat) != jtu.tree_structure(
            0
        ):
            raise NotImplementedError(
                "Cannot use `adjoint=BacksolveAdjoint()` with `SaveAt(subs=...)`."
            )
        if saveat.dense or saveat.subs.steps:
            raise NotImplementedError(
                "Cannot use `adjoint=BacksolveAdjoint()` with "
                "`saveat=SaveAt(steps=True)` or saveat=SaveAt(dense=True)`."
            )
        if saveat.subs.fn is not save_y:
            raise NotImplementedError(
                "Cannot use `adjoint=BacksolveAdjoint()` with `saveat=SaveAt(fn=...)`."
            )
        if is_unsafe_sde(terms):
            raise ValueError(
                "`adjoint=BacksolveAdjoint()` does not support `UnsafeBrownianPath`. "
                "Consider using `adjoint=DirectAdjoint()` instead."
            )
        if is_sde(terms):
            if isinstance(solver, AbstractItoSolver):
                raise NotImplementedError(
                    f"`{solver.__class__.__name__}` converges to the Itô solution. "
                    "However `BacksolveAdjoint` currently only supports Stratonovich "
                    "SDEs."
                )
            elif not isinstance(solver, AbstractStratonovichSolver):
                warnings.warn(
                    f"{solver.__class__.__name__} is not marked as converging to "
                    "either the Itô or the Stratonovich solution. Note that "
                    "`BacksolveAdjoint` will only produce the correct solution for "
                    "Stratonovich SDEs."
                )
        if jtu.tree_structure(solver.term_structure) != jtu.tree_structure(0):
            raise NotImplementedError(
                "`diffrax.BacksolveAdjoint` is only compatible with solvers that take "
                "a single term."
            )
        if event is not None:
            raise NotImplementedError(
                "`diffrax.BacksolveAdjoint` is not compatible with events."
            )

        y = init_state.y
        init_state = eqx.tree_at(lambda s: s.y, init_state, object())
        init_state = _nondiff_solver_controller_state(
            self, init_state, passed_solver_state, passed_controller_state
        )

        final_state, aux_stats = _loop_backsolve(
            (y, args, terms),
            self=self,
            saveat=saveat,
            init_state=init_state,
            solver=solver,
            event=event,
            **kwargs,
        )
        final_state = _only_transpose_ys(final_state)
        return final_state, aux_stats


# Reversible Adjoint custom vjp computes gradients w.r.t.
# - y, corresponding to the initial state;
# - args, corresponding to explicit parameters;
# - terms, corresponding to implicit parameters as part of the vector field.


@eqx.filter_custom_vjp
def _loop_reversible(y__args__terms, *, self, throw, init_state, **kwargs):
    del throw
    y, args, terms = y__args__terms
    init_state = eqx.tree_at(lambda s: s.y, init_state, y)
    del y
    return self._loop(
        args=args,
        terms=terms,
        init_state=init_state,
        inner_while_loop=ft.partial(_inner_loop, kind="lax"),
        outer_while_loop=ft.partial(_outer_loop, kind="lax"),
        **kwargs,
    )


@_loop_reversible.def_fwd
def _loop_reversible_fwd(perturbed, y__args__terms, **kwargs):
    del perturbed
    final_state, aux_stats = _loop_reversible(y__args__terms, **kwargs)
    ts = final_state.reversible_ts
    ts_final_index = final_state.reversible_save_index
    ys = final_state.save_state.ys
    solver_state1 = final_state.solver_state
    return (final_state, aux_stats), (ts, ts_final_index, ys, solver_state1)


@_loop_reversible.def_bwd
def _loop_reversible_bwd(
    residuals,
    grad_final_state__aux_stats,
    perturbed,
    y__args__terms,
    *,
    self,
    saveat,
    init_state,
    solver,
    event,
    **kwargs,
):
    assert event is None

    del perturbed, self, init_state, kwargs
    ts, ts_final_index, ys, solver_state1 = residuals
    original_solver_state, z1 = solver_state1
    del residuals, solver_state1

    grad_final_state, _ = grad_final_state__aux_stats
    # If true we must be using SaveAt(t1=True).
    t1_only = saveat.subs.t1
    if t1_only:
        y1 = (ω(ys)[-1]).ω
        grad_ys = (ω(grad_final_state.save_state.ys)[-1]).ω
        grad_ys = jtu.tree_map(_materialise_none, y1, grad_ys)
        grad_y0_zeros = jtu.tree_map(jnp.zeros_like, grad_ys)

    # Otherwise we must be using SaveAt(..., steps=True) due to the guard in
    # ReversibleAdjoint. If y0 is not saved (t0=False) then we prepend grad_y0 (zeros).
    else:
        if saveat.subs.t0:
            y1 = (ω(ys)[ts_final_index]).ω
            grad_ys = grad_final_state.save_state.ys
        else:
            y1 = (ω(ys)[ts_final_index - 1]).ω
            grad_ys = grad_final_state.save_state.ys
            grad_y0 = jtu.tree_map(lambda x: jnp.zeros_like(x[0]), grad_ys)
            grad_ys = jtu.tree_map(
                lambda x, y: jnp.concatenate([x[None], y]), grad_y0, grad_ys
            )

        grad_ys = jtu.tree_map(_materialise_none, ys, grad_ys)

    del grad_final_state, grad_final_state__aux_stats

    y, args, terms = y__args__terms
    del y__args__terms

    diff_args = eqx.filter(args, eqx.is_inexact_array)
    diff_terms = eqx.filter(terms, eqx.is_inexact_array)
    diff_z1 = eqx.filter(z1, eqx.is_inexact_array)
    grad_args = jtu.tree_map(jnp.zeros_like, diff_args)
    grad_terms = jtu.tree_map(jnp.zeros_like, diff_terms)
    grad_z1 = jtu.tree_map(jnp.zeros_like, diff_z1)
    del diff_args, diff_terms, diff_z1

    def grad_step(state):
        def solver_step(t0, t1, original_solver_state, y0, args, terms):
            step, _, _, original_solver_state, _ = solver.solver.step(
                terms, t0, t1, y0, args, original_solver_state, False
            )
            return step, original_solver_state

        ts_index, y1, solver_state, grad_ys, grad_z1, grad_args, grad_terms = state
        original_solver_state, z1 = solver_state

        t1 = ts[ts_index]
        t0 = ts[ts_index - 1]

        if t1_only:
            grad_y1 = grad_ys
            grad_y0 = grad_y0_zeros  # pyright: ignore

        else:
            grad_y1 = (ω(grad_ys)[ts_index]).ω
            grad_y0 = (ω(grad_ys)[ts_index - 1]).ω

        solver_step_fn = ft.partial(solver_step, t1, t0, original_solver_state)
        step_y1, vjp_fun_y1, original_solver_state = eqx.filter_vjp(
            solver_step_fn, y1, args, terms, has_aux=True
        )
        z0 = (ω(z1) - ω(y1) + ω(step_y1)).ω

        solver_step_fn = ft.partial(solver_step, t0, t1, original_solver_state)
        step_z0, vjp_fun_z0, _ = eqx.filter_vjp(
            solver_step_fn, z0, args, terms, has_aux=True
        )

        y0 = ((1 / solver.l) * (ω(y1) - ω(step_z0)) + ω(z0)).ω

        grad_step_y1 = vjp_fun_y1(grad_z1)
        grad_y1 = (ω(grad_y1) + ω(grad_z1) - ω(grad_step_y1[0])).ω

        grad_step_z0 = vjp_fun_z0(grad_y1)
        grad_y0 = (solver.l * ω(grad_y1) + ω(grad_y0)).ω
        grad_z0 = (ω(grad_z1) - solver.l * ω(grad_y1) + ω(grad_step_z0[0])).ω

        grad_terms = (ω(grad_terms) - ω(grad_step_y1[2]) + ω(grad_step_z0[2])).ω
        grad_args = (ω(grad_args) - ω(grad_step_y1[1]) + ω(grad_step_z0[1])).ω

        if t1_only:
            grad_ys = grad_y0
        else:
            grad_ys = (ω(grad_ys).at[ts_index].set(ω(grad_y1))).ω
            grad_ys = (ω(grad_ys).at[ts_index - 1].set(ω(grad_y0))).ω

        ts_index = ts_index - 1

        return (
            ts_index,
            y0,
            (original_solver_state, z0),
            grad_ys,
            grad_z0,
            grad_args,
            grad_terms,
        )

    def cond_fun(state):
        ts_index = state[0]
        return ts_index > 0

    state = (
        ts_final_index,
        y1,
        (original_solver_state, z1),
        grad_ys,
        grad_z1,
        grad_args,
        grad_terms,
    )

    state = eqxi.while_loop(cond_fun, grad_step, state, kind="lax")
    _, _, _, grad_ys, grad_z0, grad_args, grad_terms = state
    if t1_only:
        grad_y0 = grad_ys
    else:
        grad_y0 = (ω(grad_ys)[0]).ω

    return (ω(grad_y0) + ω(grad_z0)).ω, grad_args, grad_terms


class ReversibleAdjoint(AbstractAdjoint):
    """
    Backpropagate through [`diffrax.diffeqsolve`][] using the reversible solver
    method.

    This method automatically wraps the passed solver to create an algebraically
    reversible version of that solver. In doing so, gradient calculation is exact
    (up to floating point errors) and backpropagation becomes a linear in time $O(n)$
    and constant in memory $O(1)$ algorithm in the number of steps $n$.

    The reversible adjoint can be used when solving ODEs/CDEs/SDEs and is
    compatible with any [`diffrax.AbstractSolver`][]. Adaptive step sizes are also
    supported.

    !!! note

        This adjoint can be less numerically stable than
        [`diffrax.RecursiveCheckpointAdjoint`][] and [`diffrax.DirectAdjoint`][].
        Stability can be largely improved by using [double (64bit) precision](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision)
        and [smaller/adaptive step sizes](https://docs.kidger.site/diffrax/api/stepsize_controller/).

    ??? cite "References"

        This algorithm was developed in:

        ```bibtex
        @article{mccallum2024efficient,
            title={Efficient, Accurate and Stable Gradients for Neural ODEs},
            author={McCallum, Sam and Foster, James},
            journal={arXiv preprint arXiv:2410.11648},
            year={2024}
        }
        ```

        And built on previous work by:

        ```bibtex
        @article{kidger2021efficient,
            title={Efficient and accurate gradients for neural sdes},
            author={Kidger, Patrick and Foster, James and Li, Xuechen Chen and Lyons,
                    Terry},
            journal={Advances in Neural Information Processing Systems},
            volume={34},
            pages={18747--18761},
            year={2021}
        }
        ```

        ```bibtex
        @article{zhuang2021mali,
            title={Mali: A memory efficient and reverse accurate integrator for neural
                    odes},
            author={Zhuang, Juntang and Dvornek, Nicha C and Tatikonda, Sekhar and
            Duncan, James S},
            journal={arXiv preprint arXiv:2102.04668},
            year={2021}
        }
        ```
    """

    l: float = 0.999

    def loop(
        self,
        *,
        args,
        terms,
        solver,
        saveat,
        init_state,
        passed_solver_state,
        passed_controller_state,
        event,
        **kwargs,
    ):
        # `is` check because this may return a Tracer from SaveAt(ts=<array>)
        if (
            eqx.tree_equal(saveat, SaveAt(t1=True)) is not True
            and eqx.tree_equal(saveat, SaveAt(steps=True)) is not True
            and eqx.tree_equal(saveat, SaveAt(t0=True, steps=True)) is not True
        ):
            raise ValueError(
                "Can only use `diffrax.ReversibleAdjoint` with "
                "`saveat=SaveAt(t1=True)` or `saveat=SaveAt(steps=True)`."
            )

        if event is not None:
            raise NotImplementedError(
                "`diffrax.ReversibleAdjoint` is not compatible with events."
            )

        if isinstance(solver, (SemiImplicitEuler, ReversibleHeun, LeapfrogMidpoint)):
            raise ValueError(
                "`diffrax.ReversibleAdjoint` is not compatible with solvers that are "
                f"intrinsically algebraically reversible, such as {solver}."
            )

        solver = _Reversible(solver, self.l)
        tprev = init_state.tprev
        tnext = init_state.tnext
        y = init_state.y

        init_state = eqx.tree_at(
            lambda s: s.solver_state,
            init_state,
            solver.init(terms, tprev, tnext, y, args),
            is_leaf=_is_none,
        )

        init_state = eqx.tree_at(lambda s: s.y, init_state, object())
        init_state = _nondiff_solver_controller_state(
            self, init_state, passed_solver_state, passed_controller_state
        )

        final_state, aux_stats = _loop_reversible(
            (y, args, terms),
            self=self,
            saveat=saveat,
            init_state=init_state,
            solver=solver,
            event=event,
            **kwargs,
        )
        final_state = _only_transpose_ys(final_state)
        return final_state, aux_stats


ReversibleAdjoint.__init__.__doc__ = r"""
**Arguments:**

- `l` - coupling parameter, defaults to `l=0.999`.

The reversible solver introduces the coupled state $\{y_n, z_n\}_{n\geq 0}$ and the 
coupling parameter $l\in (0, 1)$ mixes the states via $ly_n + (1-l)z_n$. This parameter 
effects the stability of the reversible solver; decreasing it's value leads to greater 
forward stability and increasing it's value leads to greater backward stability.

In most cases the default value is sufficient. However, if you find yourself needing 
greater control over stability it can be passed as an argument.
"""

_BaseSolverState = TypeVar("_BaseSolverState")
_SolverState: TypeAlias = tuple[_BaseSolverState, Y]


def _add_maybe_none(x, y):
    if x is None:
        return None
    else:
        return (ω(x) + ω(y)).ω


class _Reversible(
    AbstractAdaptiveSolver[_SolverState], AbstractWrappedSolver[_SolverState]
):
    """
    Reversible solver method.

    Allows any solver ([`diffrax.AbstractSolver`][]) to be made algebraically
    reversible. This is a private API, exclusively for [`diffrax.ReversibleAdjoint`][].
    """

    solver: AbstractSolver
    l: float = 0.999

    @property
    def term_structure(self):
        return self.solver.term_structure

    @property
    def interpolation_cls(self):  # pyright: ignore
        return self.solver.interpolation_cls

    @property
    def term_compatible_contr_kwargs(self):
        return self.solver.term_compatible_contr_kwargs

    @property
    def root_finder(self):
        return self.solver.root_finder  # pyright: ignore

    @property
    def root_find_max_steps(self):
        return self.solver.root_find_max_steps  # pyright: ignore

    def order(self, terms: PyTree[AbstractTerm]) -> Optional[int]:
        return self.solver.order(terms)

    def strong_order(self, terms: PyTree[AbstractTerm]) -> Optional[RealScalarLike]:
        return self.solver.strong_order(terms)

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        if isinstance(self.solver, AbstractRungeKutta):
            object.__setattr__(self.solver.tableau, "fsal", False)
            object.__setattr__(self.solver.tableau, "ssal", False)
        original_solver_init = self.solver.init(terms, t0, t1, y0, args)
        return (original_solver_init, y0)

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
        original_solver_state, z0 = solver_state

        step_z0, z_error, dense_info, original_solver_state, result1 = self.solver.step(
            terms, t0, t1, z0, args, original_solver_state, made_jump
        )
        y1 = (self.l * (ω(y0) - ω(z0)) + ω(step_z0)).ω

        step_y1, y_error, _, _, result2 = self.solver.step(
            terms, t1, t0, y1, args, original_solver_state, made_jump
        )
        z1 = (ω(y1) + ω(z0) - ω(step_y1)).ω

        solver_state = (original_solver_state, z1)
        result = update_result(result1, result2)

        return y1, _add_maybe_none(z_error, y_error), dense_info, solver_state, result

    def func(
        self, terms: PyTree[AbstractTerm], t0: RealScalarLike, y0: Y, args: Args
    ) -> VF:
        return self.solver.func(terms, t0, y0, args)
