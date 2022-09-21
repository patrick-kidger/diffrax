import diffrax.local_interpolation
import jax
import jax.numpy as jnp

from .helpers import shaped_allclose


def test_local_linear_interpolation():
    t0 = 2.0
    t1 = 3.3
    t0_ = 2.8
    t1_ = 2.9
    for y0 in (2.1, jnp.array(2.1), jnp.array([2.1, 3.1])):
        for y1 in (2.2, jnp.array(2.2), jnp.array([2.2, 3.2])):
            interp = diffrax.local_interpolation.LocalLinearInterpolation(
                t0=t0, t1=t1, y0=y0, y1=y1
            )

            # evaluate position
            pred = interp.evaluate(t0_)
            true = y0 + (y1 - y0) * (t0_ - t0) / (t1 - t0)
            assert shaped_allclose(pred, true)

            _, pred = jax.jvp(interp.evaluate, (t0_,), (jnp.ones_like(t0_),))
            true = (y1 - y0) / (t1 - t0)
            assert shaped_allclose(pred, true)

            # evaluate increment
            pred = interp.evaluate(t0_, t1_)
            true = (y1 - y0) * (t1_ - t0_) / (t1 - t0)
            assert shaped_allclose(pred, true)

            _, pred = jax.jvp(
                interp.evaluate, (t0_, t1_), (jnp.ones_like(t0_), jnp.ones_like(t1_))
            )
            assert shaped_allclose(pred, jnp.zeros_like(pred))

            # evaluate over zero-length interval. Note t1=t0.
            interp = diffrax.local_interpolation.LocalLinearInterpolation(
                t0=t0, t1=t0, y0=y0, y1=y1
            )
            pred = interp.evaluate(t0)
            true, _ = jnp.broadcast_arrays(y0, y1)
            assert shaped_allclose(pred, true)

            _, pred = jax.jvp(interp.evaluate, (t0,), (jnp.ones_like(t0),))
            assert shaped_allclose(pred, jnp.zeros_like(pred))
