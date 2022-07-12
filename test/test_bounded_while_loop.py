# TODO:
# - Test forward times
# - Test grad time
# - Test compile time

import functools as ft

import diffrax
import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from .helpers import shaped_allclose, time_fn


def test_functional_no_vmap_no_inplace():
    def cond_fun(val):
        x, step = val
        return step < 5

    def body_fun(val, _):
        x, step = val
        return (x + 0.1, step + 1)

    init_val = (jnp.array([0.3]), 0)

    val = diffrax.misc.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=0)
    assert shaped_allclose(val[0], jnp.array([0.3])) and val[1] == 0

    val = diffrax.misc.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=1)
    assert shaped_allclose(val[0], jnp.array([0.4])) and val[1] == 1

    val = diffrax.misc.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=2)
    assert shaped_allclose(val[0], jnp.array([0.5])) and val[1] == 2

    val = diffrax.misc.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=4)
    assert shaped_allclose(val[0], jnp.array([0.7])) and val[1] == 4

    val = diffrax.misc.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=8)
    assert shaped_allclose(val[0], jnp.array([0.8])) and val[1] == 5

    val = diffrax.misc.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=None)
    assert shaped_allclose(val[0], jnp.array([0.8])) and val[1] == 5


def test_functional_no_vmap_inplace():
    def cond_fun(val):
        x, step = val
        return step < 5

    def body_fun(val, inplace):
        x, step = val
        x = inplace(x).at[jnp.minimum(step + 1, 4)].set(x[step] + 0.1)
        step = inplace(step).at[()].set(step + 1)
        x = diffrax.misc.HadInplaceUpdate(x)
        step = diffrax.misc.HadInplaceUpdate(step)
        return x, step

    init_val = (jnp.array([0.3, 0.3, 0.3, 0.3, 0.3]), 0)

    val = diffrax.misc.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=0)
    assert shaped_allclose(val[0], jnp.array([0.3, 0.3, 0.3, 0.3, 0.3])) and val[1] == 0

    val = diffrax.misc.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=1)
    assert shaped_allclose(val[0], jnp.array([0.3, 0.4, 0.3, 0.3, 0.3])) and val[1] == 1

    val = diffrax.misc.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=2)
    assert shaped_allclose(val[0], jnp.array([0.3, 0.4, 0.5, 0.3, 0.3])) and val[1] == 2

    val = diffrax.misc.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=4)
    assert shaped_allclose(val[0], jnp.array([0.3, 0.4, 0.5, 0.6, 0.7])) and val[1] == 4

    val = diffrax.misc.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=8)
    assert shaped_allclose(val[0], jnp.array([0.3, 0.4, 0.5, 0.6, 0.8])) and val[1] == 5

    val = diffrax.misc.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=None)
    assert shaped_allclose(val[0], jnp.array([0.3, 0.4, 0.5, 0.6, 0.8])) and val[1] == 5


def test_functional_vmap_no_inplace():
    def cond_fun(val):
        x, step = val
        return step < 5

    def body_fun(val, _):
        x, step = val
        return (x + 0.1, step + 1)

    init_val = (jnp.array([[0.3], [0.4]]), jnp.array([0, 3]))

    val = jax.vmap(
        lambda v: diffrax.misc.bounded_while_loop(cond_fun, body_fun, v, max_steps=0)
    )(init_val)
    assert shaped_allclose(val[0], jnp.array([[0.3], [0.4]])) and jnp.array_equal(
        val[1], jnp.array([0, 3])
    )

    val = jax.vmap(
        lambda v: diffrax.misc.bounded_while_loop(cond_fun, body_fun, v, max_steps=1)
    )(init_val)
    assert shaped_allclose(val[0], jnp.array([[0.4], [0.5]])) and jnp.array_equal(
        val[1], jnp.array([1, 4])
    )

    val = jax.vmap(
        lambda v: diffrax.misc.bounded_while_loop(cond_fun, body_fun, v, max_steps=2)
    )(init_val)
    assert shaped_allclose(val[0], jnp.array([[0.5], [0.6]])) and jnp.array_equal(
        val[1], jnp.array([2, 5])
    )

    val = jax.vmap(
        lambda v: diffrax.misc.bounded_while_loop(cond_fun, body_fun, v, max_steps=4)
    )(init_val)
    assert shaped_allclose(val[0], jnp.array([[0.7], [0.6]])) and jnp.array_equal(
        val[1], jnp.array([4, 5])
    )

    val = jax.vmap(
        lambda v: diffrax.misc.bounded_while_loop(cond_fun, body_fun, v, max_steps=8)
    )(init_val)
    assert shaped_allclose(val[0], jnp.array([[0.8], [0.6]])) and jnp.array_equal(
        val[1], jnp.array([5, 5])
    )

    val = jax.vmap(
        lambda v: diffrax.misc.bounded_while_loop(cond_fun, body_fun, v, max_steps=None)
    )(init_val)
    assert shaped_allclose(val[0], jnp.array([[0.8], [0.6]])) and jnp.array_equal(
        val[1], jnp.array([5, 5])
    )


def test_functional_vmap_inplace():
    def cond_fun(val):
        x, step, max_step = val
        return step < max_step

    def body_fun(val, inplace):
        x, step, max_step = val
        x = inplace(x).at[jnp.minimum(step + 1, 4)].set(x[step] + 0.1)
        step = inplace(step).at[()].set(step + 1)
        x = diffrax.misc.HadInplaceUpdate(x)
        step = diffrax.misc.HadInplaceUpdate(step)
        return x, step, max_step

    init_val = (
        jnp.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4]]),
        jnp.array([0, 1]),
        jnp.array([5, 3]),
    )

    val = jax.vmap(
        lambda v: diffrax.misc.bounded_while_loop(cond_fun, body_fun, v, max_steps=0)
    )(init_val)
    assert shaped_allclose(
        val[0], jnp.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4]])
    ) and jnp.array_equal(val[1], jnp.array([0, 1]))

    val = jax.vmap(
        lambda v: diffrax.misc.bounded_while_loop(cond_fun, body_fun, v, max_steps=1)
    )(init_val)
    assert shaped_allclose(
        val[0], jnp.array([[0.3, 0.4, 0.3, 0.3, 0.3], [0.4, 0.4, 0.5, 0.4, 0.4]])
    ) and jnp.array_equal(val[1], jnp.array([1, 2]))

    val = jax.vmap(
        lambda v: diffrax.misc.bounded_while_loop(cond_fun, body_fun, v, max_steps=2)
    )(init_val)
    assert shaped_allclose(
        val[0], jnp.array([[0.3, 0.4, 0.5, 0.3, 0.3], [0.4, 0.4, 0.5, 0.6, 0.4]])
    ) and jnp.array_equal(val[1], jnp.array([2, 3]))

    val = jax.vmap(
        lambda v: diffrax.misc.bounded_while_loop(cond_fun, body_fun, v, max_steps=4)
    )(init_val)
    assert shaped_allclose(
        val[0], jnp.array([[0.3, 0.4, 0.5, 0.6, 0.7], [0.4, 0.4, 0.5, 0.6, 0.4]])
    ) and jnp.array_equal(val[1], jnp.array([4, 3]))

    val = jax.vmap(
        lambda v: diffrax.misc.bounded_while_loop(cond_fun, body_fun, v, max_steps=8)
    )(init_val)
    assert shaped_allclose(
        val[0], jnp.array([[0.3, 0.4, 0.5, 0.6, 0.8], [0.4, 0.4, 0.5, 0.6, 0.4]])
    ) and jnp.array_equal(val[1], jnp.array([5, 3]))

    val = jax.vmap(
        lambda v: diffrax.misc.bounded_while_loop(cond_fun, body_fun, v, max_steps=None)
    )(init_val)
    assert shaped_allclose(
        val[0], jnp.array([[0.3, 0.4, 0.5, 0.6, 0.8], [0.4, 0.4, 0.5, 0.6, 0.4]])
    ) and jnp.array_equal(val[1], jnp.array([5, 3]))


#
# Test speed. Two things are tested:
# - asymptotic computational complexity;
# - speed compared to `lax.while_loop`.
#


def _make_update(i, u, v):
    return u if i is None else v.at[i].set(u)


def _body_fun(body_fun):
    def __body_fun(val):
        update, index = body_fun(val)
        return jax.tree_map(_make_update, index, update, val)

    return __body_fun


def _quadratic_fit(x, y):
    return np.polynomial.Polynomial.fit(x, y, deg=2).convert().coef


def _test_scaling_max_steps():
    key = jrandom.PRNGKey(567)
    expensive_fn = eqx.nn.MLP(in_size=1, out_size=1, width_size=1024, depth=2, key=key)

    def cond_fun(val):
        x, step = val
        return step < 5

    def body_fun(val):
        x, step = val
        return (expensive_fn(x[step, None])[0], step + 1), (
            jnp.minimum(step + 1, 5),
            None,
        )

    init_val = (
        jnp.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4]]),
        jnp.array([0, 3]),
    )

    @ft.partial(jax.jit, static_argnums=1)
    @ft.partial(jax.vmap, in_axes=(0, None))
    def test_fun(val, max_steps):
        return diffrax.misc.bounded_while_loop(cond_fun, body_fun, val, max_steps)

    time16 = time_fn(lambda: test_fun(init_val, 16), repeat=10)
    time32 = time_fn(lambda: test_fun(init_val, 32), repeat=10)
    time64 = time_fn(lambda: test_fun(init_val, 64), repeat=10)
    time128 = time_fn(lambda: test_fun(init_val, 128), repeat=10)
    time256 = time_fn(lambda: test_fun(init_val, 256), repeat=10)
    maxtime = max(time16, time32, time64, time128, time256)

    # Rescale to fit the graph inside [0, 1] x [0, 1] so that polynomials are actually
    # a reasonable thing to use.
    _, c1, c2 = _quadratic_fit(
        [16 / 256, 32 / 256, 64 / 256, 128 / 256, 256 / 256],
        [
            time16 / maxtime,
            time32 / maxtime,
            time64 / maxtime,
            time128 / maxtime,
            time256 / maxtime,
        ],
    )
    # Runtime expected to be O(1)
    assert -0.05 < c1 < 0.05
    assert -0.05 < c2 < 0.05

    @ft.partial(jax.jit, static_argnums=1)
    @jax.vmap
    def lax_test_fun(val):
        return lax.while_loop(cond_fun, _body_fun(body_fun), val)

    lax_time = time_fn(lambda: lax_test_fun(init_val), repeat=10)

    assert maxtime < 2 * lax_time


def _test_scaling_num_steps():
    key = jrandom.PRNGKey(567)
    expensive_fn = eqx.nn.MLP(in_size=1, out_size=1, width_size=1024, depth=2, key=key)

    def cond_fun(val):
        x, step, num_steps = val
        return step < num_steps

    def body_fun(val):
        x, step, num_steps = val
        return (expensive_fn(x[step, None])[0], step + 1, num_steps), (
            jnp.minimum(step + 1, num_steps),
            None,
            None,
        )

    init_val = (jnp.array([[0.3] * 256, [0.4] * 256]), jnp.array([0, 3]))

    @ft.partial(jax.jit, static_argnums=1)
    @ft.partial(jax.vmap, in_axes=(0, None))
    def test_fun(val, num_steps):
        return diffrax.misc.bounded_while_loop(
            cond_fun, body_fun, (*val, num_steps), max_steps=256
        )

    time16 = time_fn(lambda: test_fun(init_val, 16), repeat=10)
    time32 = time_fn(lambda: test_fun(init_val, 32), repeat=10)
    time64 = time_fn(lambda: test_fun(init_val, 64), repeat=10)
    time128 = time_fn(lambda: test_fun(init_val, 128), repeat=10)
    time256 = time_fn(lambda: test_fun(init_val, 256), repeat=10)

    _, c1, c2 = _quadratic_fit(
        [16, 32, 64, 128, 256], [time16, time32, time64, time128, time256]
    )
    # Runtime expected to be O(steps taken)
    assert 0.95 < c1 < 1.05
    assert -0.05 < c2 < 0.05

    @ft.partial(jax.jit, static_argnums=1)
    @ft.partial(jax.vmap, in_axes=(0, None))
    def lax_test_fun(val, num_steps):
        return lax.while_loop(cond_fun, _body_fun(body_fun), (*val, num_steps))

    lax_time16 = time_fn(lambda: lax_test_fun(init_val, 16), repeat=10)
    lax_time32 = time_fn(lambda: lax_test_fun(init_val, 32), repeat=10)
    lax_time64 = time_fn(lambda: lax_test_fun(init_val, 64), repeat=10)
    lax_time128 = time_fn(lambda: lax_test_fun(init_val, 128), repeat=10)
    lax_time256 = time_fn(lambda: lax_test_fun(init_val, 256), repeat=10)

    assert time16 < 2 * lax_time16
    assert time32 < 2 * lax_time32
    assert time64 < 2 * lax_time64
    assert time128 < 2 * lax_time128
    assert time256 < 2 * lax_time256
