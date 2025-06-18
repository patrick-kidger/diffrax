import functools as ft
from typing import cast

import diffrax
import equinox as eqx
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
    with pytest.warns(match="discrete_terminating_event"):
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
    with pytest.warns(match="discrete_terminating_event"):
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
        with pytest.warns(match="discrete_terminating_event"):
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
        with pytest.raises(ValueError):
            diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, event=event)


@pytest.mark.parametrize("steps", (1, 2, 3, 4, 5))
def test_event_save_all_steps(steps):
    term = diffrax.ODETerm(lambda t, y, args: (1.0, 1.0))
    solver = diffrax.Tsit5()
    t0 = 0
    t1 = 10
    dt0 = 1
    thr = steps - 0.5
    y0 = (0.0, -thr)
    ts = jnp.array([0.5, 3.5, 5.5])

    def cond_fn(t, y, args, **kwargs):
        del t, args, kwargs
        x, _ = y
        return x - thr

    root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
    event = diffrax.Event(cond_fn, root_finder)

    def run(saveat):
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            y0,
            event=event,
            saveat=saveat,
        )
        return cast(Array, sol.ts), cast(tuple, sol.ys)

    saveats = [
        diffrax.SaveAt(steps=True),
        diffrax.SaveAt(steps=True, t1=True),
        diffrax.SaveAt(steps=True, t1=True, t0=True),
        diffrax.SaveAt(steps=True, fn=lambda t, y, args: (y[0], y[1] + thr)),
    ]
    num_steps = [steps, steps, steps + 1, steps]
    yevents = [(thr, 0), (thr, 0), (thr, 0), (thr, thr)]

    for saveat, n, yevent in zip(saveats, num_steps, yevents, strict=True):
        ts, ys = run(saveat)
        xs, zs = ys
        xevent, zevent = yevent
        assert jnp.sum(jnp.isfinite(ts)) == n
        assert jnp.sum(jnp.isfinite(xs)) == n
        assert jnp.sum(jnp.isfinite(zs)) == n
        assert jnp.all(jnp.isclose(ts[n - 1], thr, atol=1e-5))
        assert jnp.all(jnp.isclose(xs[n - 1], xevent, atol=1e-5))
        assert jnp.all(jnp.isclose(zs[n - 1], zevent, atol=1e-5))


@pytest.mark.parametrize("steps", (1, 2, 3, 4, 5))
def test_event_save_ts(steps):
    term = diffrax.ODETerm(lambda t, y, args: (1.0, 1.0))
    solver = diffrax.Tsit5()
    t0 = 0
    t1 = 10
    dt0 = 1
    thr = steps - 0.5
    y0 = (0.0, -thr)
    ts = jnp.array([0.5, 3.5, 5.5])

    def cond_fn(t, y, args, **kwargs):
        del t, args, kwargs
        x, _ = y
        return x - thr

    root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
    event = diffrax.Event(cond_fn, root_finder)

    def run(saveat):
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            y0,
            event=event,
            saveat=saveat,
        )
        return cast(Array, sol.ts), cast(tuple, sol.ys)

    saveats = [
        diffrax.SaveAt(ts=ts),
        diffrax.SaveAt(ts=ts, t1=True),
        diffrax.SaveAt(ts=ts, t0=True),
        diffrax.SaveAt(ts=ts, steps=True),
        diffrax.SaveAt(ts=ts, fn=lambda t, y, args: (y[0], y[1] + thr)),
    ]
    save_finals = [False, True, False, True, False]
    yevents = [(thr, 0), (thr, 0), (thr, 0), (thr, 0), (thr, thr)]
    for saveat, save_final, yevent in zip(saveats, save_finals, yevents):
        ts, ys = run(saveat)
        xs, zs = ys
        xevent, zevent = yevent
        if save_final:
            assert jnp.all(jnp.isclose(ts[jnp.isfinite(ts)][-1], thr, atol=1e-5))
            assert jnp.all(jnp.isclose(xs[jnp.isfinite(xs)][-1], xevent, atol=1e-5))
            assert jnp.all(jnp.isclose(zs[jnp.isfinite(zs)][-1], zevent, atol=1e-5))
        else:
            assert jnp.all(ts[jnp.isfinite(ts)] <= thr)


@pytest.mark.parametrize("steps", (1, 2, 3, 4, 5))
def test_event_save_subsaveat(steps):
    term = diffrax.ODETerm(lambda t, y, args: jnp.array([1.0, 1.0]))
    solver = diffrax.Tsit5()
    t0 = 0.0
    t1 = 10.0
    dt0 = 1.0
    thr = steps - 0.5
    y0 = jnp.array([0.0, -thr])
    ts = jnp.arange(t0, t1, 3.0)
    ts_event = jnp.sum(ts <= thr)
    last_t = jnp.array(ts[ts_event - 1])

    def cond_fn(t, y, args, **kwargs):
        del t, args, kwargs
        x, _ = y
        return x - thr

    root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
    event = diffrax.Event(cond_fn, root_finder)

    class Saved(eqx.Module):
        y: Array

    def save_fn(t, y, args):
        del t, args
        ynorm = jnp.einsum("i,i->", y, y)
        return Saved(jnp.array([ynorm]))

    last_save = save_fn(None, y0 + last_t, None).y
    subsaveat_a = diffrax.SubSaveAt(ts=ts, fn=save_fn)
    subsaveat_b = diffrax.SubSaveAt(steps=True)
    saveat = diffrax.SaveAt(subs=[subsaveat_a, subsaveat_b])
    sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, event=event, saveat=saveat)
    ts_1, ts_2 = cast(list, sol.ts)
    ys_1, ys_2 = cast(list, sol.ys)
    assert jnp.sum(jnp.isfinite(ts_1)) == ts_event
    assert jnp.sum(jnp.isfinite(ts_2)) == steps
    assert jnp.all(jnp.isclose(ys_2[steps - 1], jnp.array([thr, 0]), atol=1e-5))
    assert jnp.all(jnp.isclose(ys_1.y[ts_event - 1], last_save, atol=1e-5))


def test_event_trig_dir():
    term = diffrax.ODETerm(lambda t, y, args: jnp.array([1.0, 1.0]))
    solver = diffrax.Tsit5()
    t0 = 0.0
    t1 = 10.0
    dt0 = 1.0
    y0 = jnp.array([0, 1])

    def up_cond(t, y, args, **kwargs):
        del t, args, kwargs
        y0, _ = y
        return y0 - 5.0

    def down_cond(t, y, args, **kwargs):
        del t, args, kwargs
        _, y1 = y
        return -(y1 - 5.0)

    root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)

    def run(event):
        sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, event=event)
        assert sol.ts is not None
        assert sol.ys is not None
        [t_final] = sol.ts
        [y_final] = sol.ys
        return t_final, y_final

    event = diffrax.Event((up_cond, down_cond), root_finder, True)
    t_final, y_final = run(event)
    assert jnp.allclose(t_final, 5.0)
    assert jnp.allclose(y_final, jnp.array([5.0, 6.0]))

    event = diffrax.Event((up_cond, down_cond), root_finder, False)
    t_final, y_final = run(event)
    assert jnp.allclose(t_final, 4.0)
    assert jnp.allclose(y_final, jnp.array([4.0, 5.0]))

    event = diffrax.Event((up_cond, down_cond), root_finder, (True, True))
    t_final, y_final = run(event)
    assert jnp.allclose(t_final, 5.0)
    assert jnp.allclose(y_final, jnp.array([5.0, 6.0]))

    event = diffrax.Event((up_cond, down_cond), root_finder, (True, False))
    t_final, y_final = run(event)
    assert jnp.allclose(t_final, 4.0)
    assert jnp.allclose(y_final, jnp.array([4.0, 5.0]))

    event = diffrax.Event((up_cond, down_cond), root_finder, (False, True))
    t_final, y_final = run(event)
    assert jnp.allclose(t_final, 10.0)
    assert jnp.allclose(y_final, jnp.array([10.0, 11.0]))

    event = diffrax.Event((up_cond, down_cond), root_finder, (False, None))
    t_final, y_final = run(event)
    assert jnp.allclose(t_final, 4.0)
    assert jnp.allclose(y_final, jnp.array([4.0, 5.0]))


def test_event_trig_dir_pytree_structure():
    f = lambda x: x
    root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
    diffrax.Event([f, f, [f, f]], root_finder, [True, None, [False, True]])
    with pytest.raises(ValueError):
        diffrax.Event([f, f, [f, f, f]], root_finder, [True, None, [False, True]])


def test_event_with_pytree_valued_condition_function():
    term = diffrax.ODETerm(lambda t, y, args: jnp.array([1.0, 1.0]))
    solver = diffrax.Tsit5()
    t0 = 0.0
    t1 = 10.0
    dt0 = 1.0
    y0 = jnp.array([0, 1])

    class CondFn(eqx.Module):
        crossing: tuple[tuple[float]]
        downcrossing: bool

        def __call__(self, t, y, args, **kwargs):
            del t, args, kwargs
            y0, _ = y
            [[crossing]] = self.crossing
            out = y0 - crossing
            if self.downcrossing:
                out = -out
            return out

    def another_cond_fn(t, y, args, **kwargs):
        del t, y, args, kwargs
        return 5.0

    root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)

    def run(event):
        sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, event=event)
        assert sol.ts is not None
        assert sol.ys is not None
        [t_final] = sol.ts
        [y_final] = sol.ys
        return t_final, y_final

    event = diffrax.Event(
        (CondFn(((3.0,),), False), another_cond_fn), root_finder, (True, None)
    )
    t_final, y_final = run(event)
    assert jnp.allclose(t_final, 3.0)
    assert jnp.allclose(y_final, jnp.array([3.0, 4.0]))

    event = diffrax.Event(
        (CondFn(((3.0,),), False), another_cond_fn), root_finder, (None, False)
    )
    t_final, y_final = run(event)
    assert jnp.allclose(t_final, 3.0)
    assert jnp.allclose(y_final, jnp.array([3.0, 4.0]))

    event = diffrax.Event(
        (CondFn(((3.0,),), False), another_cond_fn), root_finder, (False, False)
    )
    t_final, y_final = run(event)
    assert jnp.allclose(t_final, 10.0)
    assert jnp.allclose(y_final, jnp.array([10.0, 11.0]))
