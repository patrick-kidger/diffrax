import diffrax
import jax.numpy as jnp
import pytest

from .helpers import shaped_allclose


def _test(ts, flip=False):
    def f(t, y, args):
        return -y

    y0 = jnp.array(1.0)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(f),
        solver=diffrax.Euler(),
        t0=2 if flip else 0,
        t1=0 if flip else 2,
        dt0=None,
        y0=y0,
        stepsize_controller=diffrax.StepTo(ts=ts),
        saveat=diffrax.SaveAt(ts=ts),
    )
    return sol


def test_step_to_location():
    ts = jnp.linspace(0.0, 2.0, 3)
    sol = _test(ts)
    true_ys = jnp.array([1.0, 0.0, 0.0])
    assert shaped_allclose(sol.ys, true_ys)

    ts = jnp.linspace(0.0, 2.0, 5)
    sol = _test(ts)
    true_ys = jnp.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    assert shaped_allclose(sol.ys, true_ys)

    # ts[0] != t0
    ts = jnp.linspace(0.1, 2.0, 5)
    with pytest.raises(RuntimeError):
        _test(ts)

    # ts[-1] != t1
    ts = jnp.linspace(0.0, 2.1, 5)
    with pytest.raises(RuntimeError):
        _test(ts)

    # Not monotonic
    ts = jnp.array([0.0, 2.3, 2.0])
    with pytest.raises(RuntimeError):
        _test(ts)

    # Reverse time
    ts = jnp.linspace(2.0, 0.0, 5)
    with pytest.raises(RuntimeError):
        _test(ts)
    sol = _test(ts, flip=True)
    true_ys = jnp.array([1.0, 1.5, 2.25, 3.375, 5.0625])
    assert shaped_allclose(sol.ys, true_ys)
