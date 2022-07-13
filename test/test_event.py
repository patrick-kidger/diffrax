import diffrax
import jax.numpy as jnp


def test_discrete_terminate1():
    term = diffrax.ODETerm(lambda t, y, args: y)
    solver = diffrax.Tsit5()
    t0 = 0
    t1 = jnp.inf
    dt0 = 1
    y0 = 1.0
    event = diffrax.DiscreteTerminatingEvent(lambda state, **kwargs: state.y > 10)
    sol = diffrax.diffeqsolve(
        term, solver, t0, t1, dt0, y0, discrete_terminating_event=event
    )
    assert jnp.all(sol.ys > 10)


def test_discrete_terminate2():
    term = diffrax.ODETerm(lambda t, y, args: y)
    solver = diffrax.Tsit5()
    t0 = 0
    t1 = jnp.inf
    dt0 = 1
    y0 = 1.0
    event = diffrax.DiscreteTerminatingEvent(lambda state, **kwargs: state.tprev > 10)
    sol = diffrax.diffeqsolve(
        term, solver, t0, t1, dt0, y0, discrete_terminating_event=event
    )
    assert jnp.all(sol.ts > 10)


# diffrax.SteadyStateEvent tested as part of test_adjoint.py::test_implicit
