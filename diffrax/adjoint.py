import abc
from typing import Any, Dict

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp

from .misc import implicit_jvp, nondifferentiable_output, ω
from .saveat import SaveAt
from .term import AbstractTerm, AdjointTerm


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
        discrete_terminating_event,
        saveat,
        t0,
        t1,
        dt0,
        max_steps,
        throw,
        init_state,
    ):
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

    def loop(self, *, throw, **kwargs):
        del throw
        return self._loop_fn(**kwargs, is_bounded=True)


class NoAdjoint(AbstractAdjoint):
    """Disable backpropagation through [`diffrax.diffeqsolve`][].

    Forward-mode autodifferentiation (`jax.jvp`) will continue to work as normal.

    If you do not need to differentiate the results of [`diffrax.diffeqsolve`][] then
    this may sometimes improve the speed at which the differential equation is solved.
    """

    def loop(self, *, throw, **kwargs):
        del throw
        final_state, aux_stats = self._loop_fn(**kwargs, is_bounded=False)
        final_state = jax.tree_map(nondifferentiable_output, final_state)
        return final_state, aux_stats


def _vf(ys, residual, args__terms, closure):
    state_no_y, _ = residual
    t = state_no_y.tprev
    (y,) = ys  # unpack length-1 dimension
    args, terms = args__terms
    _, _, solver, _, _ = closure
    return solver.func(terms, t, y, args)


def _solve(args__terms, closure):
    args, terms = args__terms
    self, kwargs, solver, saveat, init_state = closure
    final_state, aux_stats = self._loop_fn(
        **kwargs,
        args=args,
        terms=terms,
        solver=solver,
        saveat=saveat,
        init_state=init_state,
        is_bounded=False,
    )
    # Note that we use .ys not .y here. The former is what is actually returned
    # by diffeqsolve, so it is the thing we want to attach the tangent to.
    return final_state.ys, (
        eqx.tree_at(lambda s: s.ys, final_state, None),
        aux_stats,
    )


class ImplicitAdjoint(AbstractAdjoint):
    r"""Backpropagate via the [implicit function theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem#Statement_of_the_theorem).

    This is used when solving towards a steady state, typically using
    [`diffrax.SteadyStateEvent`][]. In this case, the output of the solver is $y(θ)$
    for which $f(t, y(θ), θ) = 0$. (Where $θ$ corresponds to all parameters found
    through `terms` and `args`, but not `y0`.) Then we can skip backpropagating through
    the solver and instead directly compute
    $\frac{\mathrm{d}y}{\mathrm{d}θ} = - (\frac{\mathrm{d}f}{\mathrm{d}y})^{-1}\frac{\mathrm{d}f}{\mathrm{d}θ}$
    via the implicit function theorem.
    """  # noqa: E501

    def loop(self, *, args, terms, solver, saveat, throw, init_state, **kwargs):
        del throw

        # `is` check because this may return a Tracer from SaveAt(ts=<array>)
        if eqx.tree_equal(saveat, SaveAt(t1=True)) is not True:
            raise ValueError(
                "Can only use `adjoint=ImplicitAdjoint()` with `SaveAt(t1=True)`."
            )

        init_state = eqx.tree_at(
            lambda s: (s.y, s.solver_state, s.controller_state),
            init_state,
            replace_fn=lax.stop_gradient,
        )
        closure = (self, kwargs, solver, saveat, init_state)
        ys, residual = implicit_jvp(_solve, _vf, (args, terms), closure)

        final_state_no_ys, aux_stats = residual
        return (
            eqx.tree_at(
                lambda s: s.ys, final_state_no_ys, ys, is_leaf=lambda x: x is None
            ),
            aux_stats,
        )


# Compute derivatives with respect to the first argument:
# - y, corresponding to the initial state;
# - args, corresponding to explicit parameters;
# - terms, corresponding to implicit parameters as part of the vector field.
@eqx.filter_custom_vjp
def _loop_backsolve(y__args__terms, *, self, throw, init_state, **kwargs):
    del throw
    y, args, terms = y__args__terms
    init_state = eqx.tree_at(
        lambda s: jax.tree_leaves(s.y), init_state, jax.tree_leaves(y)
    )
    del y
    return self._loop_fn(
        args=args, terms=terms, init_state=init_state, **kwargs, is_bounded=False
    )


def _loop_backsolve_fwd(y__args__terms, **kwargs):
    final_state, aux_stats = _loop_backsolve(y__args__terms, **kwargs)
    ts = final_state.ts
    ys = final_state.ys
    return (final_state, aux_stats), (ts, ys)


def _loop_backsolve_bwd(
    residuals,
    grad_final_state__aux_stats,
    y__args__terms,
    *,
    self,
    solver,
    stepsize_controller,
    discrete_terminating_event,
    saveat,
    t0,
    t1,
    dt0,
    max_steps,
    throw,
    init_state,
):

    #
    # Unpack our various arguments. Delete a lot of things just to make sure we're not
    # using them later.
    #

    del init_state, t1
    ts, ys = residuals
    del residuals
    grad_final_state, _ = grad_final_state__aux_stats
    grad_ys = grad_final_state.ys
    del grad_final_state, grad_final_state__aux_stats
    y, args, terms = y__args__terms
    del y__args__terms
    diff_args = eqx.filter(args, eqx.is_inexact_array)
    diff_terms = eqx.filter(terms, eqx.is_inexact_array)
    zeros_like_y = jax.tree_map(jnp.zeros_like, y)
    zeros_like_diff_args = jax.tree_map(jnp.zeros_like, diff_args)
    zeros_like_diff_terms = jax.tree_map(jnp.zeros_like, diff_terms)
    del diff_args, diff_terms
    adjoint_terms = jax.tree_map(
        AdjointTerm, terms, is_leaf=lambda x: isinstance(x, AbstractTerm)
    )
    diffeqsolve = self._diffeqsolve
    kwargs = dict(
        args=args,
        adjoint=self,
        solver=solver,
        stepsize_controller=stepsize_controller,
        discrete_terminating_event=discrete_terminating_event,
        terms=adjoint_terms,
        dt0=None if dt0 is None else -dt0,
        max_steps=max_steps,
        throw=throw,
    )
    kwargs.update(self.kwargs)
    del self, solver, stepsize_controller, adjoint_terms, dt0, max_steps, throw
    del y, args, terms
    saveat_t0 = saveat.t0
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


_loop_backsolve.defvjp(_loop_backsolve_fwd, _loop_backsolve_bwd)


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

    def loop(self, *, args, terms, saveat, init_state, **kwargs):
        if saveat.steps or saveat.dense:
            raise NotImplementedError(
                "Cannot use `adjoint=BacksolveAdjoint()` with "
                "`saveat=Steps(steps=True)` or `saveat=Steps(dense=True)`."
            )

        y = init_state.y
        sentinel = object()
        init_state = eqx.tree_at(
            lambda s: jax.tree_leaves(s.y), init_state, replace_fn=lambda _: sentinel
        )

        final_state, aux_stats = _loop_backsolve(
            (y, args, terms), self=self, saveat=saveat, init_state=init_state, **kwargs
        )

        # We only allow backpropagation through `ys`; in particular not through
        # `solver_state` etc.
        ys = final_state.ys
        final_state = jax.tree_map(nondifferentiable_output, final_state)
        final_state = eqx.tree_at(
            lambda s: jax.tree_leaves(s.ys), final_state, jax.tree_leaves(ys)
        )

        return final_state, aux_stats
