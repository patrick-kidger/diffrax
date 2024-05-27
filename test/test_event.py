import functools as ft
from typing import cast

import diffrax
import jax
import jax.numpy as jnp
import optimistix as optx
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
        del kwargs
        assert isinstance(state.y, jax.Array)
        return state.tprev > 10

    event = diffrax.DiscreteTerminatingEvent(event_fn)
    with pytest.warns(DeprecationWarning, match="discrete_terminating_event"):
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            y0,
            discrete_terminating_event=event,
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
        del kwargs
        assert isinstance(state.y, jax.Array)
        return state.tprev > 10

    event = diffrax.DiscreteTerminatingEvent(event_fn)
    with pytest.warns(DeprecationWarning, match="discrete_terminating_event"):
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            y0,
            discrete_terminating_event=event,
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
        del kwargs
        assert isinstance(state.y, jax.Array)
        return state.tprev > 10

    event = diffrax.DiscreteTerminatingEvent(event_fn)

    @jax.jit
    @jax.grad
    def run(y0):
        with pytest.warns(DeprecationWarning, match="discrete_terminating_event"):
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


def test_steady_state_event():
    term = diffrax.ODETerm(lambda t, y, args: -1.0 * y)
    controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
    solver = diffrax.Tsit5()
    t0 = 0
    t1 = jnp.inf
    dt0 = 1
    y0 = 1.0
    cond_fn = diffrax.steady_state_event()
    event = diffrax.Event(cond_fn)
    sol = diffrax.diffeqsolve(
        term, solver, t0, t1, dt0, y0, stepsize_controller=controller, event=event
    )

    assert cast(Array, sol.event_mask)
    assert jnp.all(jnp.isclose(cast(Array, sol.ys), 0.0, atol=1e-5))


def test_no_step_event():
    term = diffrax.ODETerm(lambda t, y, args: jnp.array([1, 1]))
    solver = diffrax.Tsit5()
    t0 = 0
    t1 = 10
    dt0 = 1
    y0 = jnp.array([1, -1e-1])

    def cond_fn(t, y, args, **kwargs):
        del t, args, kwargs
        assert isinstance(y, Array)
        _, x = y
        return x < 0

    event = diffrax.Event(cond_fn)
    sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, event=event)

    assert sol.stats["num_steps"] == 0
    assert jnp.all(jnp.isclose(cast(Array, sol.ys)[-1], y0, 1e-7))
    assert jnp.all(jnp.isclose(cast(Array, sol.ts), t0, 1e-7))


def test_continuous_terminate1():
    term = diffrax.ODETerm(lambda t, y, args: y)
    solver = diffrax.Tsit5()
    t0 = 0
    t1 = jnp.inf
    dt0 = 1
    y0 = 1.0

    def cond_fn(t, y, args, **kwargs):
        del y, args, kwargs
        return t > 10

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

    def cond_fn(t, y, args, **kwargs):
        del args, kwargs
        assert isinstance(y, jax.Array)
        return t - 10.0

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

    def cond_fn(t, y, args, **kwargs):
        del t, args, kwargs
        assert isinstance(y, jax.Array)
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

    def cond_fn(t, y, args, **kwargs):
        del t, args, kwargs
        assert isinstance(y, jax.Array)
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

    def cond_fn(t, y, args, **kwargs):
        del t, args, kwargs
        assert isinstance(y, jax.Array)
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

    def cond_fn_1(t, y, args, **kwargs):
        del y, args, kwargs
        assert isinstance(t, jax.Array)
        return t - 10

    def cond_fn_2(t, y, args, **kwargs):
        del t, args, kwargs
        assert isinstance(y, jax.Array)
        return y + 5.0

    root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
    event = diffrax.Event([cond_fn_1, cond_fn_2], root_finder)
    sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, event=event)
    assert cast(Array, sol.event_mask)[1]
    assert not cast(Array, sol.event_mask)[0]
    assert jnp.all(jnp.isclose(cast(Array, sol.ts), 5.0, 1e-5))


def test_continuous_event_time_grad():
    def vector_field(t, y, args):
        del t, args
        _, v = y
        d_out = v, -8.0
        return jnp.array(d_out)

    def cond_fn(t, y, args, **kwargs):
        del t, args, kwargs
        x, _ = y
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
            return cond_fn(t=None, y=_y, args=None)

        _, num = jax.vjp(phi, x0)
        (num,) = num(jax.grad(event_fn)(y0))
        _, dem = jax.jvp(event_fn, (y0,), (vector_field(tevent, phi(x0), None),))
        return -num / dem

    x0_test = jnp.array([1.0, 3.0, 10.0])
    x0_autograd = jax.vmap(first_bounce_time)(x0_test)
    x0_truegrad = jax.vmap(first_bounce_time_grad)(x0_test)
    assert jnp.all(jnp.isclose(x0_autograd, x0_truegrad, 1e-5))


def test_adaptive_stepping_event():
    term = diffrax.ODETerm(lambda t, y, args: -y)
    controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
    solver = diffrax.Tsit5()
    t0 = 0
    t1 = jnp.inf
    dt0 = 1

    def cond_fn(t, y, args, **kwargs):
        del t, args, kwargs
        assert isinstance(y, Array)
        return y - 1

    root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
    event = diffrax.Event(cond_fn, root_finder)

    @jax.jit
    @jax.value_and_grad
    def run(y):
        sol = diffrax.diffeqsolve(
            term, solver, t0, t1, dt0, y, stepsize_controller=controller, event=event
        )
        return cast(Array, sol.ys)[-1]

    y0s = [10.0, 100.0, 1000.0]
    for y0 in y0s:
        val, grad = run(y0)
        assert jnp.all(jnp.isclose(val - 1, 0.0, atol=1e-5))
        assert not jnp.isnan(grad).any()


@pytest.mark.parametrize(
    "stepsize_controller",
    (diffrax.ConstantStepSize(), diffrax.PIDController(rtol=1e-3, atol=1e-6)),
)
def test_event_vmap_y0(stepsize_controller):
    term = diffrax.ODETerm(lambda t, y, args: -y)
    solver = diffrax.Tsit5()
    t0 = 0
    t1 = jnp.inf
    dt0 = 1

    def cond_fn(t, y, args, **kwargs):
        del t, args, kwargs
        assert isinstance(y, Array)
        return y - 1

    root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
    event = diffrax.Event(cond_fn, root_finder)

    @jax.vmap
    @jax.value_and_grad
    def run(y):
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            y,
            stepsize_controller=stepsize_controller,
            event=event,
        )
        return cast(Array, sol.ys)[-1]

    y0s = jnp.arange(10.0) + 2.0
    vals, grads = run(y0s)
    for val, grad in zip(vals, grads):
        assert jnp.all(jnp.isclose(val - 1, 0.0, atol=1e-5))
        assert not jnp.isnan(grad).any()


@pytest.mark.parametrize(
    "stepsize_controller",
    (diffrax.ConstantStepSize(), diffrax.PIDController(rtol=1e-3, atol=1e-6)),
)
def test_event_vmap_t0(stepsize_controller):
    term = diffrax.ODETerm(lambda t, y, args: -y)
    solver = diffrax.Tsit5()
    t1 = jnp.inf
    dt0 = 1
    y0 = 10

    def cond_fn(t, y, args, **kwargs):
        del t, args, kwargs
        assert isinstance(y, Array)
        return y - 1

    root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
    event = diffrax.Event(cond_fn, root_finder)

    @jax.vmap
    @jax.value_and_grad
    def run(t):
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t,
            t1,
            dt0,
            y0,
            stepsize_controller=stepsize_controller,
            event=event,
        )
        return cast(Array, sol.ys)[-1]

    t0s = jnp.arange(10.0) / 10
    vals, grads = run(t0s)
    for val, grad in zip(vals, grads):
        assert jnp.all(jnp.isclose(val - 1, 0.0, atol=1e-5))
        assert not jnp.isnan(grad).any()


@pytest.mark.parametrize(
    "stepsize_controller",
    (diffrax.ConstantStepSize(), diffrax.PIDController(rtol=1e-3, atol=1e-6)),
)
def test_event_vmap_event_def(stepsize_controller):
    term = diffrax.ODETerm(lambda t, y, args: -y)
    solver = diffrax.Tsit5()
    t0 = 0
    t1 = jnp.inf
    dt0 = 1
    y0 = 10

    def cond_fn(thr, t, y, args, **kwargs):
        del t, args, kwargs
        assert isinstance(y, Array)
        return y - thr

    root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)

    @jax.vmap
    @jax.value_and_grad
    def run(thr):
        _cond_fn = ft.partial(cond_fn, thr)
        event = diffrax.Event(_cond_fn, root_finder)
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            y0,
            stepsize_controller=stepsize_controller,
            event=event,
        )
        return cast(Array, sol.ys)[-1]

    thrs = (jnp.arange(10.0) + 1) / 5
    vals, grads = run(thrs)
    for thr, val, grad in zip(thrs, vals, grads):
        assert jnp.all(jnp.isclose(val - thr, 0.0, atol=1e-5))
        assert not jnp.isnan(grad).any()


@pytest.mark.parametrize(
    "stepsize_controller",
    (diffrax.ConstantStepSize(), diffrax.PIDController(rtol=1e-3, atol=1e-6)),
)
def test_event_vmap_cond_fn(stepsize_controller):
    term = diffrax.ODETerm(lambda t, y, args: -y)
    solver = diffrax.Tsit5()
    t0 = 0
    t1 = jnp.inf
    dt0 = 1

    def cond_fn(t, y, args, **kwargs):
        del t, args, kwargs

        @jax.vmap
        def _cond_fn(y):
            return y - 1

        return jnp.max(_cond_fn(y))

    root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
    event = diffrax.Event(cond_fn, root_finder)

    @jax.value_and_grad
    def run(y):
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            y,
            stepsize_controller=stepsize_controller,
            event=event,
        )
        return jnp.max(cast(Array, sol.ys)[-1])

    y0s = jnp.arange(10.0) + 2.0
    val, grad = run(y0s)
    assert jnp.all(jnp.isclose(val - 1, 0.0, atol=1e-5))
    assert not jnp.isnan(grad).any()


def test_event_scalar_error():
    term = diffrax.ODETerm(lambda t, y, args: y)
    solver = diffrax.Tsit5()
    t0 = 0
    t1 = jnp.inf
    dt0 = 1
    y0 = 1.0

    def cond_fn_1(t, y, args, **kwargs):
        del t, args, kwargs
        assert isinstance(y, jax.Array)
        return (y,)

    def cond_fn_2(t, y, args, **kwargs):
        del t, args, kwargs
        return jnp.array([y, 1.0])

    cond_fns = [cond_fn_1, cond_fn_2]
    for cond_fn in cond_fns:
        event = diffrax.Event(cond_fn=cond_fn)
        with pytest.raises(ValueError):
            diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, event=event)


def test_event_dtype_error():
    term = diffrax.ODETerm(lambda t, y, args: y)
    solver = diffrax.Tsit5()
    t0 = 0
    t1 = jnp.inf
    dt0 = 1
    y0 = 1.0

    def cond_fn_1(t, y, args, **kwargs):
        del t, args, kwargs
        assert isinstance(y, jax.Array)
        return jnp.array(1, dtype=int)

    def cond_fn_2(t, y, args, **kwargs):
        del t, args, kwargs
        return jnp.array(1, dtype=complex)

    cond_fns = [cond_fn_1, cond_fn_2]
    for cond_fn in cond_fns:
        event = diffrax.Event(cond_fn=cond_fn)
        with pytest.raises(AssertionError):
            diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, event=event)
