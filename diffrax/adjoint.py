import abc
import functools as ft
from typing import Any, Dict

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp

from .misc import bounded_while_loop
from .solution import RESULTS


def _at_max_steps(cond_fun, final_state):
    result = jnp.where(
        cond_fun(final_state), RESULTS.max_steps_reached, final_state.result
    )
    return eqx.tree_at(lambda s: s.result, final_state, result)


def _loop_bounded(make_cond_body_funs, max_steps, terms, args, init_state):
    cond_fun, body_fun = make_cond_body_funs(terms, args)
    final_state = bounded_while_loop(cond_fun, body_fun, init_state, max_steps)
    return _at_max_steps(cond_fun, final_state)


def _loop_while(make_cond_body_funs, max_steps, terms, args, init_state):
    cond_fun, body_fun = make_cond_body_funs(terms, args)

    def _cond_fun(state):
        return cond_fun(state) & (state.step < max_steps)

    def _body_fun(state):
        return _body_fun(state, lambda x: x)

    final_state = lax.while_loop(_cond_fun, _body_fun, init_state)
    return _at_max_steps(cond_fun, final_state)


class AbstractAdjoint(eqx.Module):
    """Abstract base class for all adjoint methods."""

    @abc.abstractmethod
    def loop(self, make_cond_body_funs, max_steps, terms, args, init_state):
        """Runs the main solve loop. Subclasses can override this to provide custom
        backpropagation behaviour; see for example the implementation of
        [`diffrax.BacksolveAdjoint`][].
        """


class RecursiveCheckpointAdjoint(AbstractAdjoint):
    """Backpropagate through [`diffrax.diffeqsolve`][] by differentiating the numerical
    solution directly. This is sometimes known as "discretise-then-optimise", or
    described as "backpropagation through the solver".

    For most problems this is the preferred technique for backpropagating through a
    differential equation.

    A binomial checkpointing scheme is used so that memory usage is low.
    """

    loop = staticmethod(_loop_bounded)


@ft.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _loop_backsolve(self, make_cond_body_funs, max_steps, terms, args, init_state):
    return _loop_while(make_cond_body_funs, max_steps, terms, args, init_state)


def _loop_backsolve_adjoint_fwd(
    self, make_cond_body_funs, max_steps, terms, args, init_state
):
    context = None
    return _loop_while(make_cond_body_funs, max_steps, terms, args, init_state), context


def _loop_backsolve_adjoint_bwd(
    self, make_cond_body_funs, max_steps, context, grad_final_state
):
    ...  # TODO


_loop_backsolve.defvjp(_loop_backsolve_adjoint_fwd, _loop_backsolve_adjoint_bwd)


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
        self.kwargs = kwargs

    # Not just loop = _loop_backsolve, as the latter doesn't have __get__ and so won't
    # implicitly bind self.
    def loop(self, make_cond_body_funs, max_steps, terms, args, init_state):
        return _loop_backsolve(
            self, make_cond_body_funs, max_steps, terms, args, init_state
        )


class NoAdjoint(AbstractAdjoint):
    """Disable backpropagation through [`diffrax.diffeqsolve`][].

    Forward-mode autodifferentiation (`jax.jvp`) will continue to work as normal.

    If you do not need to differentiate the results of [`diffrax.diffeqsolve`][] then
    this may sometimes improve the speed at which the differential equation is solved.
    """

    loop = staticmethod(_loop_while)
