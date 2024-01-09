from typing import cast

import diffrax
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array


def test_discrete_terminate1():
    term = diffrax.ODETerm(lambda t, y, args: y)
    solver = diffrax.Tsit5()
    t0 = 0
    t1 = jnp.inf
    dt0 = 1
    y0 = 1.0

    def event_fn(state, **kwargs):
        assert isinstance(state.y, jax.Array)
        return state.tprev > 10

    event = diffrax.DiscreteTerminatingEvent(event_fn)
    sol = diffrax.diffeqsolve(
        term, solver, t0, t1, dt0, y0, discrete_terminating_event=event
    )
    assert jnp.all(cast(Array, sol.ys) > 10)


def test_discrete_terminate2():
    term = diffrax.ODETerm(lambda t, y, args: y)
    solver = diffrax.Tsit5()
    t0 = 0
    t1 = jnp.inf
    dt0 = 1
    y0 = 1.0

    def event_fn(state, **kwargs):
        assert isinstance(state.y, jax.Array)
        return state.tprev > 10

    event = diffrax.DiscreteTerminatingEvent(event_fn)
    sol = diffrax.diffeqsolve(
        term, solver, t0, t1, dt0, y0, discrete_terminating_event=event
    )
    assert jnp.all(cast(Array, sol.ts) > 10)


def test_event_backsolve():
    term = diffrax.ODETerm(lambda t, y, args: y)
    solver = diffrax.Tsit5()
    t0 = 0
    t1 = jnp.inf
    dt0 = 1
    y0 = 1.0

    def event_fn(state, **kwargs):
        assert isinstance(state.y, jax.Array)
        return state.tprev > 10

    event = diffrax.DiscreteTerminatingEvent(event_fn)

    @jax.jit
    @jax.grad
    def run(y0):
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            y0,
            discrete_terminating_event=event,
            adjoint=diffrax.BacksolveAdjoint(),
        )
        return jnp.sum(cast(Array, sol.ys))

    # And in particular not some other error.
    with pytest.raises(NotImplementedError):
        run(y0)


# diffrax.SteadyStateEvent tested as part of test_adjoint.py::test_implicit
