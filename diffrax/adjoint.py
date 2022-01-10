import abc
import functools as ft
from typing import Any, Dict

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp

from .misc import nondifferentiable_output, ω
from .saveat import SaveAt
from .term import AdjointTerm


class AbstractAdjoint(eqx.Module):
    """Abstract base class for all adjoint methods."""

    @abc.abstractmethod
    def loop(
        self,
        solver,
        stepsize_controller,
        saveat,
        dt0,
        t1,
        max_steps,
        throw,
        terms,
        args,
        init_state,
    ):
        """Runs the main solve loop. Subclasses can override this to provide custom
        backpropagation behaviour; see for example the implementation of
        [`diffrax.BacksolveAdjoint`][].
        """

    # Eurgh, delayed imports to handle circular dependencies.
    #
    # `integrate.py` defines the forward pass. `adjoint.py` defines the backward pass.
    # These pretty much necessarily depend up on each other:
    # - diffeqsolve needs to know about AbstractAdjoint, since it's one its arguments.
    # - BacksolveAdjoint needs to know about how to integrate a differential equation,
    #   since that's what it does.
    # As such we get a circular dependency. We resolve it by lazily importing from
    # `integrate.py`. For convenience we make them available as properties here so all
    # adjoint methods can access these.
    @property
    def _loop_fn(self):
        from .integrate import loop

        return loop

    @property
    def _diffeqsolve(self):
        from .integrate import diffeqsolve

        return diffeqsolve


class RecursiveCheckpointAdjoint(AbstractAdjoint):
    """Backpropagate through [`diffrax.diffeqsolve`][] by differentiating the numerical
    solution directly. This is sometimes known as "discretise-then-optimise", or
    described as "backpropagation through the solver".

    For most problems this is the preferred technique for backpropagating through a
    differential equation.

    A binomial checkpointing scheme is used so that memory usage is low.
    """

    def loop(
        self,
        solver,
        stepsize_controller,
        saveat,
        _dt0,
        t1,
        max_steps,
        _throw,
        terms,
        args,
        init_state,
    ):
        return self._loop_fn(
            solver,
            stepsize_controller,
            saveat,
            t1,
            max_steps,
            terms,
            args,
            init_state,
            is_bounded=True,
        )


class NoAdjoint(AbstractAdjoint):
    """Disable backpropagation through [`diffrax.diffeqsolve`][].

    Forward-mode autodifferentiation (`jax.jvp`) will continue to work as normal.

    If you do not need to differentiate the results of [`diffrax.diffeqsolve`][] then
    this may sometimes improve the speed at which the differential equation is solved.
    """

    def loop(
        self,
        solver,
        stepsize_controller,
        saveat,
        _dt0,
        t1,
        max_steps,
        _throw,
        terms,
        args,
        init_state,
    ):
        return self._loop_fn(
            solver,
            stepsize_controller,
            saveat,
            t1,
            max_steps,
            terms,
            args,
            init_state,
            is_bounded=False,
        )


@ft.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8))
def _loop_backsolve(
    self,
    solver,
    stepsize_controller,
    saveat,
    _dt0,
    t1,
    max_steps,
    _throw,
    terms,
    args,
    init_state,
):
    return self._loop_fn(
        solver,
        stepsize_controller,
        saveat,
        t1,
        max_steps,
        terms,
        args,
        init_state,
        is_bounded=False,
    )


def _loop_backsolve_fwd(
    self,
    solver,
    stepsize_controller,
    saveat,
    _dt0,
    t1,
    max_steps,
    _throw,
    terms,
    args,
    init_state,
):
    final_state = self._loop_fn(
        solver,
        stepsize_controller,
        saveat,
        t1,
        max_steps,
        terms,
        args,
        init_state,
        is_bounded=False,
    )
    t0 = init_state.tprev
    ts = final_state.ts
    ys = final_state.ys
    context = (t0, terms, args, ts, ys)
    return final_state, context


# TODO: implement this as a single diffeqsolve with events, once events are supported.
def _loop_backsolve_bwd(
    self,
    solver,
    stepsize_controller,
    saveat,
    dt0,
    _t1,
    max_steps,
    throw,
    context,
    grad_final_state,
):
    t0, terms, args, ts, ys = context
    terms = jax.tree_map(AdjointTerm, terms)
    kwargs = dict(
        terms=terms,
        dt0=dt0,
        solver=solver,
        args=args,
        stepsize_controller=stepsize_controller,
        adjoint=self,
        max_steps=max_steps,
        throw=throw,
    )
    kwargs.update(self.kwargs)
    grad_ys = grad_final_state.ys
    had_t0 = saveat.t0
    diffeqsolve = self._diffeqsolve
    del (
        self,
        solver,
        stepsize_controller,
        saveat,
        dt0,
        _t1,
        max_steps,
        throw,
        context,
        terms,
        grad_final_state,
    )

    saveat = SaveAt(t1=True, solver_state=True, controller_state=True)

    def _scan_fun(_state, _vals, first=False):
        _t1, _t0, _y0, _grad_y0 = _vals
        _y_aug0, _solver_state, _controller_state = _state
        _a_y0, _a_args0 = _y_aug0
        _a_y0 = (_a_y0 ** ω + _grad_y0 ** ω).ω
        _y_aug0 = (_y0, _a_y0, _a_args0)
        _sol = diffeqsolve(
            t0=_t0,
            t1=_t1,
            y0=_y_aug0,
            solver_state=_solver_state,
            controller_state=_controller_state,
            made_jump=not first,
            saveat=saveat,
            **kwargs,
        )

        def __get(__y):
            assert __y.shape[0] == 1
            return __y[0]

        _y1 = ω(_sol.ys).call(__get).ω
        _, _a_y1, _a_args1 = _y1
        _y1 = (_a_y1, _a_args1)
        _solver_state = _sol.solver_state
        _controller_state = _sol.controller_state

        return (_y1, _solver_state, _controller_state), None

    a_y0 = ω(grad_ys)[-1].ω
    a_args0 = ω(args).call(jnp.zeros_like).ω
    y_aug0 = (a_y0, a_args0)
    scan_init = (y_aug0, None, None)
    # Run once outside the loop to get access to solver_state etc. of the correct
    # structure.
    val0 = (ts[-2], ts[-1], ω(ys)[-1].ω, ω(grad_ys)[-1].ω)
    if had_t0:
        vals = (ts[:-2], ts[1:-1], ω(ys)[1:-1].ω, ω(grad_ys)[1:-1].ω)
    else:
        vals = (
            jnp.concatenate([t0[None], ts[:-2]]),
            ts[:-1],
            ω(ys)[:-1].ω,
            ω(grad_ys)[:-1].ω,
        )
    scan_init = _scan_fun(scan_init, val0, first=True)
    scan_out, _ = lax.scan(_scan_fun, scan_init, vals, reverse=True)
    y_augm1, _, _, _ = scan_out
    a_ym1, a_argsm1 = y_augm1
    if had_t0:
        a_ym1 = (ω(a_ym1) + ω(grad_ys)[0]).ω

    # TODO: returns


_loop_backsolve.defvjp(_loop_backsolve_fwd, _loop_backsolve_bwd)


class BacksolveAdjoint(AbstractAdjoint):
    """Backpropagate through [`diffrax.diffeqsolve`][] by solving the continuous
    adjoint equations backwards-in-time. This is also sometimes known as
    "optimise-then-discretise", the "continuous adjoint method" or simply the "adjoint
    method".

    This method implies very low memory usage, but is usually relatively slow, and the
    computed gradients will only be approximate. As such other methods are generally
    preferred unless exceeding memory is a concern.

    !!! note

        This was popularised by [this paper](https://arxiv.org/abs/1806.07366). For
        this reason it is sometimes erroneously believed to be a better method for
        backpropagation than the other choices available.

    !!! warning

        Using this method prevents computing forward-mode autoderivatives of
        [`diffrax.diffeqsolve`][]. (That is to say, `jax.jvp` will not work.)
    """

    kwargs: Dict[str, Any]

    def __init__(self, **kwargs):
        """
        **Arguments:**

        - `**kwargs`: The arguments for the [`diffrax.diffeqsolve`][] operations that
            are called on the backward pass. For example use
            ```python
            BacksolveAdjoint(solver=Dopri5())
            ```
            to specify a particular solver to use on the backward pass.
            ```
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
        if len(diff_keys):
            raise ValueError(
                f"The following keys are not valid for `BacksolveAdjoint`: {diff_keys}"
            )
        self.kwargs = kwargs

    def loop(
        self,
        solver,
        stepsize_controller,
        saveat,
        dt0,
        t1,
        max_steps,
        throw,
        terms,
        args,
        init_state,
    ):
        if saveat.steps or saveat.dense:
            raise NotImplementedError(
                "Cannot use `adjoint=BacksolveAdjoint()` with "
                "`saveat=Steps(steps=True)` or `saveat=Steps(dense=True)`."
            )
        final_state = _loop_backsolve(
            self,
            solver,
            stepsize_controller,
            saveat,
            dt0,
            t1,
            max_steps,
            throw,
            terms,
            args,
            init_state,
        )

        # We only allow backpropagation through `ys`; in particular not through
        # `solver_state` etc.
        sentinel = object()
        final_state_no_ys = eqx.tree_at(lambda s: s.ys, final_state, sentinel)
        leaves, treedef = jax.tree_flatten(final_state_no_ys)
        leaves = [
            final_state.ys if x is sentinel else nondifferentiable_output(x)
            for x in leaves
        ]
        final_state = jax.tree_unflatten(treedef, leaves)

        return final_state
