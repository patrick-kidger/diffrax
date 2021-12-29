import diffrax
import jax.numpy as jnp
import pytest

from helpers import shaped_allclose


def test_step_to_location():
    def f(t, y, args):
        return -y

    solver = diffrax.euler(f)
    y0 = jnp.array(1.0)
    ts = jnp.linspace(0.0, 2.0, 3)
    sol = diffrax.diffeqsolve(
        solver,
        t0=0,
        t1=2,
        y0=y0,
        dt0=None,
        stepsize_controller=diffrax.StepToLocation(ts=ts),
        saveat=diffrax.SaveAt(ts=ts),
    )
    true_ys = jnp.array([1.0, 0.0, 0.0])
    assert shaped_allclose(sol.ys, true_ys)

    ts = jnp.linspace(0.0, 2.0, 5)
    sol = diffrax.diffeqsolve(
        solver,
        t0=0,
        t1=2,
        y0=y0,
        dt0=None,
        stepsize_controller=diffrax.StepToLocation(ts=ts),
        saveat=diffrax.SaveAt(ts=ts),
    )
    true_ys = jnp.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    assert shaped_allclose(sol.ys, true_ys)

    ts = jnp.linspace(0.1, 2.0, 5)
    with pytest.raises(ValueError):
        sol = diffrax.diffeqsolve(
            solver,
            t0=0,
            t1=2,
            y0=y0,
            dt0=None,
            stepsize_controller=diffrax.StepToLocation(ts=ts),
        )

    ts = jnp.linspace(0.0, 2.1, 5)
    with pytest.raises(ValueError):
        sol = diffrax.diffeqsolve(
            solver,
            t0=0,
            t1=2,
            y0=y0,
            dt0=None,
            stepsize_controller=diffrax.StepToLocation(ts=ts),
        )
