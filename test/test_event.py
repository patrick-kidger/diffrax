from typing import cast

import diffrax
import jax
import jax.numpy as jnp
import optimistix as optx
import pytest
from jaxtyping import Array


@pytest.mark.skip(reason="Old event implementation")
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

    event = diffrax.DiscreteTerminatingEvent(event_fn)  # pyright: ignore
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        discrete_terminating_event=event,  # pyright: ignore
    )
    assert jnp.all(cast(Array, sol.ys) > 10)


@pytest.mark.skip(reason="Old event implementation")
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

    event = diffrax.DiscreteTerminatingEvent(event_fn)  # pyright: ignore
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        discrete_terminating_event=event,  # pyright: ignore
    )
    assert jnp.all(cast(Array, sol.ts) > 10)


@pytest.mark.skip(reason="Old event implementation")
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

    event = diffrax.DiscreteTerminatingEvent(event_fn)  # pyright: ignore

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
            discrete_terminating_event=event,  # pyright: ignore
            adjoint=diffrax.BacksolveAdjoint(),
        )
        return jnp.sum(cast(Array, sol.ys))

    # And in particular not some other error.
    with pytest.raises(NotImplementedError):
        run(y0)


# diffrax.SteadyStateEvent tested as part of test_adjoint.py::test_implicit


def test_continuous_terminate1():
    term = diffrax.ODETerm(lambda t, y, args: y)
    solver = diffrax.Tsit5()
    t0 = 0
    t1 = jnp.inf
    dt0 = 1
    y0 = 1.0

    def cond_fn(state, **kwargs):
        assert isinstance(state.y, jax.Array)
        return state.tprev > 10

    event = diffrax.Event(cond_fn=cond_fn)
    sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, event=event)
    assert jnp.all(cast(Array, sol.ys) > 10)


def test_continuous_terminate2():
    term = diffrax.ODETerm(lambda t, y, args: y)
    solver = diffrax.Tsit5()
    t0 = 0
    t1 = jnp.inf
    dt0 = 1
    y0 = 1.0

    def cond_fn(state, **kwargs):
        assert isinstance(state.y, jax.Array)
        return state.tprev - 10.0

    event = diffrax.Event(cond_fn=cond_fn)
    sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, event=event)
    assert jnp.all(cast(Array, sol.ts) >= 10)


def test_continuous_event_time():
    term = diffrax.ODETerm(lambda t, y, args: y)
    solver = diffrax.Tsit5()
    t0 = 0
    t1 = jnp.inf
    dt0 = 1.0
    y0 = 1.0

    def cond_fn(state, y, **kwargs):
        assert isinstance(state.y, jax.Array)
        return y - jnp.exp(1.0)

    root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
    event = diffrax.Event(cond_fn, root_finder)
    sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, event=event)
    assert jnp.all(jnp.isclose(cast(Array, sol.ts), 1.0, 1e-4))


def test_continuous_event_value():
    term = diffrax.ODETerm(lambda t, y, args: 1.0)
    solver = diffrax.Tsit5()
    t0 = 0
    t1 = jnp.inf
    dt0 = 1.0
    y0 = -10.0

    def cond_fn(state, y, **kwargs):
        assert isinstance(state.y, jax.Array)
        return y

    root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
    event = diffrax.Event(cond_fn, root_finder)
    sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, event=event)
    assert jnp.all(jnp.isclose(cast(Array, sol.ys), 0.0, 1e-5))


def test_continuous_no_event():
    term = diffrax.ODETerm(lambda t, y, args: 1.0)
    solver = diffrax.Tsit5()
    t0 = 0
    t1 = 5.0
    dt0 = 1.0
    y0 = -10.0

    def cond_fn(state, y, **kwargs):
        assert isinstance(state.y, jax.Array)
        return y

    root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
    event = diffrax.Event(cond_fn, root_finder)
    sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, event=event)
    assert not cast(Array, sol.event_mask)
    assert jnp.all(jnp.isclose(cast(Array, sol.ts), 5.0, 1e-5))


def test_continuous_two_events():
    term = diffrax.ODETerm(lambda t, y, args: 1.0)
    solver = diffrax.Tsit5()
    t0 = 0
    t1 = jnp.inf
    dt0 = 1.0
    y0 = -10.0

    def cond_fn_1(state, y, **kwargs):
        assert isinstance(state.y, jax.Array)
        return y

    def cond_fn_2(state, y, **kwargs):
        assert isinstance(state.y, jax.Array)
        return y + 5.0

    root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
    event = diffrax.Event([cond_fn_1, cond_fn_2], root_finder)
    sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, event=event)
    assert cast(Array, sol.event_mask)[1]
    assert not cast(Array, sol.event_mask)[0]
    assert jnp.all(jnp.isclose(cast(Array, sol.ts), 5.0, 1e-5))


def test_continuous_event_time_grad():
    def vector_field(t, y, args):
        x, v = y
        d_out = v, -8.0
        return jnp.array(d_out)

    def cond_fn(state, y, **kwargs):
        x, v = y
        return x

    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Tsit5()
    root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
    event = diffrax.Event(cond_fn, root_finder)
    t0 = 0
    t1 = jnp.inf
    dt0 = 0.01

    @jax.jit
    @jax.grad
    def first_bounce_time(x0):
        y0 = jnp.array([x0, 0.0])
        sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, event=event)
        assert sol.ts is not None
        return sol.ts[-1]

    def first_bounce_time_grad(x0):
        y0 = jnp.array([x0, 0.0])
        sol0 = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, event=event)
        assert sol0.ts is not None
        tevent = jax.lax.stop_gradient(sol0.ts[-1])

        def phi(_x):
            _y0 = jnp.array([_x, 0.0])
            _sol = diffrax.diffeqsolve(term, solver, t0, tevent, dt0, _y0)
            assert _sol.ys is not None
            return _sol.ys[-1, :]

        def event_fn(_y):
            return cond_fn(None, _y)

        _, num = jax.vjp(phi, x0)
        (num,) = num(jax.grad(event_fn)(y0))
        _, dem = jax.jvp(event_fn, (y0,), (vector_field(tevent, phi(x0), None),))
        return -num / dem

    x0_test = jnp.array([1.0, 3.0, 10.0])
    x0_autograd = jax.vmap(first_bounce_time)(x0_test)
    x0_truegrad = jax.vmap(first_bounce_time_grad)(x0_test)
    assert jnp.all(jnp.isclose(x0_autograd, x0_truegrad, 1e-5))
