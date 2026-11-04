import diffrax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

from .helpers import implicit_tol, tree_allclose


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


@pytest.mark.parametrize(
    "solver",
    [
        diffrax.Euler(),
        diffrax.Midpoint(),
        diffrax.Dopri5(),
        diffrax.Tsit5(),
        diffrax.Dopri8(),
        diffrax.KenCarp3(),
    ],
)
def test_local_pytree_increment(solver):
    dtype = jnp.float64
    solver = implicit_tol(solver)
    slopes = {"a": jnp.array(2.0, dtype), "b": (jnp.array([-1.0, 3.0], dtype),)}
    y0 = jtu.tree_map(jnp.zeros_like, slopes)
    term = diffrax.ODETerm(lambda t, y, args: slopes)
    if isinstance(solver, diffrax.KenCarp3):
        half = diffrax.ODETerm(
            lambda t, y, args: jtu.tree_map(lambda s: 0.5 * s, slopes)
        )
        term = diffrax.MultiTerm(half, diffrax.ODETerm(half.vector_field))
    start, end = jnp.array(0.0, dtype), jnp.array(1.0, dtype)
    state = solver.init(term, start, end, y0, None)
    _, _, dense_info, _, _ = solver.step(term, start, end, y0, None, state, False)
    interpolation = solver.interpolation_cls(t0=start, t1=end, **dense_info)
    for t0, t1 in [(0.25, 0.75), (0.75, 0.25), (0.5, 0.5)]:
        t0, t1 = jnp.array(t0, dtype), jnp.array(t1, dtype)
        expected = jtu.tree_map(lambda s: (t1 - t0) * s, slopes)
        fn = lambda start, end: interpolation.evaluate(start, end)
        assert tree_allclose(fn(t0, t1), expected)
        assert tree_allclose(jax.jit(fn)(t0, t1), expected)
        _, tangent = jax.jvp(fn, (t0, t1), (jnp.ones_like(t0), jnp.zeros_like(t1)))
        assert tree_allclose(tangent, jtu.tree_map(lambda s: -s, slopes))
