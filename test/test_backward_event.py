from typing import cast

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
import pytest


@pytest.mark.parametrize("backward", (False, True))
@pytest.mark.parametrize("root_find", (False, True))
@pytest.mark.parametrize("boolean", (False, True))
def test_event_uses_physical_time(backward, root_find, boolean):
    t0, t1, dt0 = (2.0, 0.0, -0.25) if backward else (0.0, 2.0, 0.25)
    term = diffrax.ODETerm(lambda t, y, args: 1.0)
    root_finder = optx.Newton(rtol=1e-9, atol=1e-9) if root_find else None

    def cond_fn(t, y, args, **kwargs):
        if boolean:
            return t < 1.0 if backward else t > 1.0
        else:
            return t - 1.0

    sol = diffrax.diffeqsolve(
        term,
        diffrax.Tsit5(),
        t0,
        t1,
        dt0,
        t0,
        event=diffrax.Event(cond_fn, root_finder),
    )
    assert sol.result == diffrax.RESULTS.event_occurred
    assert sol.stats["num_steps"] > 0
    expected_time = 1.0 if not boolean else (0.75 if backward else 1.25)
    assert jnp.allclose(cast(jax.Array, sol.ts), expected_time, atol=1e-9)
    assert jnp.allclose(cast(jax.Array, sol.ys), expected_time, atol=1e-9)


@pytest.mark.parametrize("backward", (False, True))
@pytest.mark.parametrize("root_find", (False, True))
def test_event_callback_kwargs_match_physical_problem(backward, root_find):
    t0, t1 = (2.0, 0.0) if backward else (0.0, 2.0)
    dt0 = None
    term = diffrax.ODETerm(lambda t, y, args: t)
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.StepTo(ts=jnp.linspace(t0, t1, 9))
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 5), t1=True)

    def cond_fn(t, y, args, **kwargs):
        # In particular, solver.func must not apply a second time transformation
        # to an already physical callback time.
        correct_kwargs = (
            (kwargs["t0"] == t0)
            & (kwargs["t1"] == t1)
            & (kwargs["dt0"] is None)
            & jnp.all(kwargs["saveat"].subs.ts == saveat.subs.ts)
            & jnp.all(kwargs["stepsize_controller"].ts == stepsize_controller.ts)
            & (solver.func(kwargs["terms"], t, y, args) == t)
        )
        if root_find:
            return jnp.where(correct_kwargs, t - 0.9, 1.0)
        else:
            crossed = t <= 1.0 if backward else t >= 1.0
            return correct_kwargs & crossed

    root_finder = optx.Newton(rtol=1e-9, atol=1e-9) if root_find else None

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        0.5 * t0**2,
        event=diffrax.Event(cond_fn, root_finder),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )
    assert sol.result == diffrax.RESULTS.event_occurred
    expected_time = 0.9 if root_find else 1.0
    ts, ys = cast(jax.Array, sol.ts), cast(jax.Array, sol.ys)
    finite_ts = ts[jnp.isfinite(ts)]
    finite_ys = ys[jnp.isfinite(ts)]
    assert jnp.isclose(finite_ts[-1], expected_time)
    assert jnp.isclose(finite_ys[-1], 0.5 * expected_time**2)


def test_backward_event_time_gradient():
    def run(threshold):
        term = diffrax.ODETerm(lambda t, y, args: t)
        event = diffrax.Event(
            lambda t, y, args, **kwargs: t - args,
            root_finder=optx.Newton(rtol=1e-9, atol=1e-9),
        )
        sol = diffrax.diffeqsolve(
            term, diffrax.Tsit5(), 2.0, 0.0, -0.3, 2.0, threshold, event=event
        )
        return cast(jax.Array, sol.ts)[0], cast(jax.Array, sol.ys)[0]

    time, value = run(0.9)
    assert jnp.isclose(time, 0.9)
    assert jnp.isclose(value, 0.5 * 0.9**2)
    grad_time, grad_value = jax.jacrev(run)(0.9)
    assert jnp.isclose(grad_time, 1.0)
    assert jnp.isclose(grad_value, 0.9)


def test_event_mixed_integration_directions_vmap():
    def run(t0, t1):
        term = diffrax.ODETerm(lambda t, y, args: 1.0)
        event = diffrax.Event(
            lambda t, y, args, **kwargs: t - 0.9,
            root_finder=optx.Newton(rtol=1e-9, atol=1e-9),
        )
        sol = diffrax.diffeqsolve(
            term, diffrax.Tsit5(), t0, t1, (t1 - t0) / 8, t0, event=event
        )
        return cast(jax.Array, sol.ts), cast(jax.Array, sol.ys)

    ts, ys = jax.vmap(run)(jnp.array([0.0, 2.0]), jnp.array([2.0, 0.0]))
    assert jnp.allclose(ts, 0.9)
    assert jnp.allclose(ys, 0.9)


def test_backward_event_condition_pytree():
    term = diffrax.ODETerm(lambda t, y, args: 1.0)
    event = diffrax.Event(
        [
            lambda t, y, args, **kwargs: t - 0.9,
            lambda t, y, args, **kwargs: t < 0.4,
        ],
        root_finder=optx.Newton(rtol=1e-9, atol=1e-9),
    )
    sol = diffrax.diffeqsolve(term, diffrax.Tsit5(), 2.0, 0.0, -0.25, 2.0, event=event)
    assert jnp.isclose(cast(jax.Array, sol.ts)[0], 0.9)
    assert sol.event_mask is not None
    assert sol.event_mask[0]
    assert not sol.event_mask[1]


@pytest.mark.parametrize("root_find", (False, True))
def test_backward_event_at_initial_time(root_find):
    term = diffrax.ODETerm(lambda t, y, args: 1.0)
    root_finder = optx.Newton(rtol=1e-9, atol=1e-9) if root_find else None
    event = diffrax.Event(lambda t, y, args, **kwargs: t > 1.0, root_finder)
    sol = diffrax.diffeqsolve(term, diffrax.Tsit5(), 2.0, 0.0, -0.25, 2.0, event=event)
    assert sol.result == diffrax.RESULTS.event_occurred
    assert sol.stats["num_steps"] == 0
    assert jnp.isclose(cast(jax.Array, sol.ts)[0], 2.0)
    assert jnp.isclose(cast(jax.Array, sol.ys)[0], 2.0)


def test_backward_event_callable_module_gradient():
    class TimeCondition(eqx.Module):
        threshold: jax.Array

        def __call__(self, t, y, args, **kwargs):
            return t - self.threshold

    @eqx.filter_jit
    def run(condition):
        term = diffrax.ODETerm(lambda t, y, args: t)
        event = diffrax.Event(condition, optx.Newton(rtol=1e-9, atol=1e-9))
        sol = diffrax.diffeqsolve(
            term, diffrax.Tsit5(), 2.0, 0.0, -0.25, 2.0, event=event
        )
        return cast(jax.Array, sol.ts)[0], cast(jax.Array, sol.ys)[0]

    condition = TimeCondition(jnp.array(0.9))
    time, value = run(condition)
    grad_time, grad_value = eqx.filter_jacrev(run)(condition)
    assert jnp.isclose(time, 0.9)
    assert jnp.isclose(value, 0.405)
    assert jnp.isclose(grad_time.threshold, 1.0)
    assert jnp.isclose(grad_value.threshold, 0.9)


def test_backward_event_vector_field_parameter_gradient():
    class LinearTimeVectorField(eqx.Module):
        rate: jax.Array

        def __call__(self, t, y, args):
            return self.rate * t

    class VectorFieldCondition(eqx.Module):
        threshold: jax.Array

        def __call__(self, t, y, args, *, terms, solver, **kwargs):
            return solver.func(terms, t, y, args) - self.threshold

    def run(rate, threshold):
        term = diffrax.ODETerm(LinearTimeVectorField(rate))
        event = diffrax.Event(
            VectorFieldCondition(threshold), optx.Newton(rtol=1e-9, atol=1e-9)
        )
        sol = diffrax.diffeqsolve(
            term, diffrax.Tsit5(), 2.0, 0.0, -0.25, 2 * rate, event=event
        )
        return cast(jax.Array, sol.ts)[0], cast(jax.Array, sol.ys)[0]

    rate, threshold = jnp.array(1.5), jnp.array(1.2)
    time, value = run(rate, threshold)
    (grad_time_rate, grad_time_threshold), (grad_value_rate, grad_value_threshold) = (
        jax.jacrev(run, argnums=(0, 1))(rate, threshold)
    )
    assert jnp.isclose(time, threshold / rate)
    assert jnp.isclose(value, 0.5 * threshold**2 / rate)
    assert jnp.isclose(grad_time_rate, -threshold / rate**2)
    assert jnp.isclose(grad_time_threshold, 1 / rate)
    assert jnp.isclose(grad_value_rate, -0.5 * threshold**2 / rate**2)
    assert jnp.isclose(grad_value_threshold, threshold / rate)


@pytest.mark.parametrize("steady_state", (False, True))
def test_backward_legacy_event_physical_time(steady_state):
    term = diffrax.ODETerm(lambda t, y, args: t - 1.0)
    if steady_state:
        event = diffrax.SteadyStateEvent(rtol=0.0, atol=1e-12)
    else:
        event = diffrax.DiscreteTerminatingEvent(
            lambda state, **kwargs: state.tprev <= 1.0
        )
    with pytest.warns(match="discrete_terminating_event"):
        sol = diffrax.diffeqsolve(
            term,
            diffrax.Tsit5(),
            2.0,
            0.0,
            -0.25,
            0.0,
            discrete_terminating_event=event,
        )
    assert sol.result == diffrax.RESULTS.event_occurred
    assert jnp.isclose(cast(jax.Array, sol.ts)[0], 1.0)
    assert jnp.isclose(cast(jax.Array, sol.ys)[0], -0.5)


@pytest.mark.parametrize("root_find", (False, True))
def test_event_equal_initial_final_time(root_find):
    term = diffrax.ODETerm(lambda t, y, args: 1.0)
    root_finder = optx.Newton(rtol=1e-9, atol=1e-9) if root_find else None
    event = diffrax.Event(lambda t, y, args, **kwargs: t >= 2.0, root_finder)
    sol = diffrax.diffeqsolve(term, diffrax.Tsit5(), 2.0, 2.0, 0.25, 0.4, event=event)
    assert sol.result == diffrax.RESULTS.event_occurred
    assert sol.stats["num_steps"] == 0
    assert jnp.isclose(cast(jax.Array, sol.ts)[0], 2.0)
    assert jnp.isclose(cast(jax.Array, sol.ys)[0], 0.4)
