import diffrax
import jax
import jax.numpy as jnp

from .helpers import tree_allclose


def test_local_linear_interpolation():
    t0 = 2.0
    t1 = 3.3
    t0_ = 2.8
    t1_ = 2.9
    for y0 in (2.1, jnp.array(2.1), jnp.array([2.1, 3.1])):
        y1 = y0 + 0.1
        interp = diffrax.LocalLinearInterpolation(t0=t0, t1=t1, y0=y0, y1=y1)

        # evaluate position
        pred = interp.evaluate(t0_)
        true = jnp.array(y0 + (y1 - y0) * (t0_ - t0) / (t1 - t0))
        assert tree_allclose(pred, true)

        _, pred = jax.jvp(interp.evaluate, (t0_,), (jnp.ones_like(t0_),))
        true = jnp.array((y1 - y0) / (t1 - t0))
        assert tree_allclose(pred, true)

        # evaluate increment
        pred = interp.evaluate(t0_, t1_)
        true = jnp.array((y1 - y0) * (t1_ - t0_) / (t1 - t0))
        assert tree_allclose(pred, true)

        _, pred = jax.jvp(
            interp.evaluate, (t0_, t1_), (jnp.ones_like(t0_), jnp.ones_like(t1_))
        )
        assert tree_allclose(pred, jnp.zeros_like(pred))

        # evaluate over zero-length interval. Note t1=t0.
        interp = diffrax.LocalLinearInterpolation(t0=t0, t1=t0, y0=y0, y1=y1)
        pred = interp.evaluate(t0)
        true, _ = jnp.broadcast_arrays(y0, y1)
        assert tree_allclose(pred, true)

        _, pred = jax.jvp(interp.evaluate, (t0,), (jnp.ones_like(t0),))
        assert tree_allclose(pred, jnp.zeros_like(pred))


def test_third_order_hermite():
    t0 = 2.0
    t1 = 3.9

    def y(t):
        return jnp.asarray(0.4 + 0.7 * t - 1.1 * t**2 + 0.4 * t**3)

    y0, f0 = jax.jvp(y, (t0,), (1.0,))
    y1, f1 = jax.jvp(y, (t1,), (1.0,))
    k0 = f0 * (t1 - t0)
    k1 = f1 * (t1 - t0)
    interp = diffrax.ThirdOrderHermitePolynomialInterpolation(
        t0=t0, t1=t1, y0=y0, y1=y1, k0=k0, k1=k1
    )
    assert tree_allclose(interp.evaluate(2.6), y(2.6))
