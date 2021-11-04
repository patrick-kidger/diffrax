import diffrax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest


@pytest.mark.parametrize(
    "stepsize_controller", (diffrax.ConstantStepSize(), diffrax.IController())
)
def test_vmap_y0(stepsize_controller):
    t0 = 0
    t1 = 1
    dt0 = 0.1

    key = jrandom.PRNGKey(5678)

    y0 = jrandom.normal(key, (10, 2))
    a = jnp.array([[-0.2, 1], [1, -0.2]])

    def f(t, y, args):
        return a @ y

    solver = diffrax.heun(f)

    saveat = diffrax.SaveAt(t0=True)
    sol = jax.vmap(
        lambda y0i: diffrax.diffeqsolve(
            solver,
            t0,
            t1,
            y0i,
            dt0,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
        )
    )(y0)
    assert jnp.array_equal(sol.t0, jnp.full((10,), t0))
    assert jnp.array_equal(sol.t1, jnp.full((10,), t1))
    assert jnp.array_equal(sol.ts, jnp.full((10, 1), t0))
    assert sol.ys.shape == (10, 1, 2)

    saveat = diffrax.SaveAt(t1=True)
    sol = jax.vmap(
        lambda y0i: diffrax.diffeqsolve(
            solver,
            t0,
            t1,
            y0i,
            dt0,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
        )
    )(y0)
    assert jnp.array_equal(sol.t0, jnp.full((10,), t0))
    assert jnp.array_equal(sol.t1, jnp.full((10,), t1))
    assert jnp.array_equal(sol.ts, jnp.full((10, 1), t1))
    assert sol.ys.shape == (10, 1, 2)

    _t = [0, 0.3, 0.7, 1]
    saveat = diffrax.SaveAt(ts=_t)
    sol = jax.vmap(
        lambda y0i: diffrax.diffeqsolve(
            solver,
            t0,
            t1,
            y0i,
            dt0,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
        )
    )(y0)
    assert jnp.array_equal(sol.t0, jnp.full((10,), t0))
    assert jnp.array_equal(sol.t1, jnp.full((10,), t1))
    assert jnp.array_equal(sol.ts, jnp.broadcast_to(_t, (10, 4)))
    assert sol.ys.shape == (10, 4, 2)

    saveat = diffrax.SaveAt(steps=True)
    sol = jax.vmap(
        lambda y0i: diffrax.diffeqsolve(
            solver,
            t0,
            t1,
            y0i,
            dt0,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
        )
    )(y0)
    num_steps = sol.stats["num_steps"]
    if not isinstance(stepsize_controller, diffrax.ConstantStepSize):
        # not the same number of steps for every batch element
        assert len(set(num_steps)) > 1
    assert jnp.array_equal(sol.t0, jnp.full((10,), t0))
    assert jnp.array_equal(sol.t1, jnp.full((10,), t1))
    assert sol.ts.shape == (10, max(num_steps))
    assert sol.ys.shape == (10, max(num_steps), 2)

    saveat = diffrax.SaveAt(dense=True)
    sol = jax.vmap(
        lambda y0i: diffrax.diffeqsolve(
            solver,
            t0,
            t1,
            y0i,
            dt0,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
        )
    )(y0)
    assert jnp.array_equal(sol.t0, jnp.full((10,), t0))
    assert jnp.array_equal(sol.t1, jnp.full((10,), t1))
    assert sol.ts is None
    assert sol.ys is None
    assert jax.vmap(lambda sol: sol.evaluate(0.5))(sol).shape == (10, 2)
    assert jax.vmap(lambda sol: sol.derivative(0.5))(sol).shape == (10, 2)
