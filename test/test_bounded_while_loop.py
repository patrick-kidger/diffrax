# TODO:
# - Test jvp, grad speed
# - Test compile time
# However these tests will currently fail because of JAX bugs.

import diffrax
import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from helpers import time_fn


def test_functional_no_vmap_no_inplace():
    def cond_fun(val):
        x, step = val
        return step < 5

    def body_fun(val):
        x, step = val
        return (x + 0.1, step + 1), None

    init_val = (jnp.array([0.3]), 0)

    val = diffrax.utils.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=0)
    assert jnp.allclose(val[0], 0.3) and val[1] == 0

    val = diffrax.utils.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=1)
    assert jnp.allclose(val[0], 0.4) and val[1] == 1

    val = diffrax.utils.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=2)
    assert jnp.allclose(val[0], 0.5) and val[1] == 2

    val = diffrax.utils.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=4)
    assert jnp.allclose(val[0], 0.7) and val[1] == 4

    val = diffrax.utils.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=8)
    assert jnp.allclose(val[0], 0.8) and val[1] == 5

    val = diffrax.utils.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=None)
    assert jnp.allclose(val[0], 0.8) and val[1] == 5


def test_functional_no_vmap_inplace():
    def cond_fun(val):
        x, step = val
        return step < 5

    def body_fun(val):
        x, step = val
        return (x + 0.1, step + 1), (min(step, 4), diffrax.utils.Index(()))

    init_val = (jnp.array([0.3, 0.3, 0.3, 0.3, 0.3]), 0)

    val = diffrax.utils.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=0)
    assert jnp.array_equal(val[0], jnp.array([0.3, 0.3, 0.3, 0.3, 0.3])) and val[1] == 0

    val = diffrax.utils.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=1)
    assert jnp.array_equal(val[0], jnp.array([0.3, 0.4, 0.3, 0.3, 0.3])) and val[1] == 1

    val = diffrax.utils.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=2)
    assert jnp.array_equal(val[0], jnp.array([0.3, 0.4, 0.5, 0.3, 0.3])) and val[1] == 2

    val = diffrax.utils.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=4)
    assert jnp.array_equal(val[0], jnp.array([0.3, 0.4, 0.5, 0.6, 0.7])) and val[1] == 4

    val = diffrax.utils.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=8)
    assert jnp.array_equal(val[0], jnp.array([0.3, 0.4, 0.5, 0.6, 0.8])) and val[1] == 5

    val = diffrax.utils.bounded_while_loop(cond_fun, body_fun, init_val, max_steps=None)
    assert jnp.array_equal(val[0], jnp.array([0.3, 0.4, 0.5, 0.6, 0.8])) and val[1] == 5


def test_functional_vmap_no_inplace():
    def cond_fun(val):
        x, step = val
        return step < 5

    def body_fun(val):
        x, step = val
        return (x + 0.1, step + 1), None

    init_val = (jnp.array([[0.3], [0.4]]), jnp.array([0, 3]))

    val = jax.vmap(
        lambda v: diffrax.utils.bounded_while_loop(cond_fun, body_fun, v, max_steps=0)
    )(init_val)
    assert jnp.array_equal(val[0], jnp.array([[0.3], [0.4]])) and jnp.array_equal(
        val[1], jnp.array([0, 3])
    )

    val = jax.vmap(
        lambda v: diffrax.utils.bounded_while_loop(cond_fun, body_fun, v, max_steps=1)
    )(init_val)
    assert jnp.array_equal(val[0], jnp.array([[0.4], [0.5]])) and jnp.array_equal(
        val[1], jnp.array([1, 4])
    )

    val = jax.vmap(
        lambda v: diffrax.utils.bounded_while_loop(cond_fun, body_fun, v, max_steps=2)
    )(init_val)
    assert jnp.array_equal(val[0], jnp.array([[0.5], [0.6]])) and jnp.array_equal(
        val[1], jnp.array([2, 5])
    )

    val = jax.vmap(
        lambda v: diffrax.utils.bounded_while_loop(cond_fun, body_fun, v, max_steps=4)
    )(init_val)
    assert jnp.array_equal(val[0], jnp.array([[0.7], [0.6]])) and jnp.array_equal(
        val[1], jnp.array([4, 5])
    )

    val = jax.vmap(
        lambda v: diffrax.utils.bounded_while_loop(cond_fun, body_fun, v, max_steps=8)
    )(init_val)
    assert jnp.array_equal(val[0], jnp.array([[0.8], [0.6]])) and jnp.array_equal(
        val[1], jnp.array([5, 5])
    )

    val = jax.vmap(
        lambda v: diffrax.utils.bounded_while_loop(
            cond_fun, body_fun, v, max_steps=None
        )
    )(init_val)
    assert jnp.array_equal(val[0], jnp.array([[0.8], [0.6]])) and jnp.array_equal(
        val[1], jnp.array([5, 5])
    )


def test_functional_vmap_inplace():
    def cond_fun(val):
        x, step = val
        return step < 5

    def body_fun(val):
        x, step = val
        return (x + 0.1, step + 1), (min(step, 4), diffrax.utils.Index(()))

    init_val = (
        jnp.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4]]),
        jnp.array([0, 3]),
    )

    val = jax.vmap(
        lambda v: diffrax.utils.bounded_while_loop(cond_fun, body_fun, v, max_steps=0)
    )(init_val)
    assert jnp.array_equal(
        val[0], jnp.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4]])
    ) and jnp.array_equal(val[1], jnp.array([0, 3]))

    val = jax.vmap(
        lambda v: diffrax.utils.bounded_while_loop(cond_fun, body_fun, v, max_steps=1)
    )(init_val)
    assert jnp.array_equal(
        val[0], jnp.array([[0.3, 0.4, 0.3, 0.3, 0.3], [0.4, 0.5, 0.4, 0.4, 0.4]])
    ) and jnp.array_equal(val[1], jnp.array([1, 4]))

    val = jax.vmap(
        lambda v: diffrax.utils.bounded_while_loop(cond_fun, body_fun, v, max_steps=2)
    )(init_val)
    assert jnp.array_equal(
        val[0], jnp.array([[0.3, 0.4, 0.5, 0.3, 0.3], [0.4, 0.5, 0.6, 0.4, 0.4]])
    ) and jnp.array_equal(val[1], jnp.array([2, 5]))

    val = jax.vmap(
        lambda v: diffrax.utils.bounded_while_loop(cond_fun, body_fun, v, max_steps=4)
    )(init_val)
    assert jnp.array_equal(
        val[0], jnp.array([[0.3, 0.4, 0.5, 0.6, 0.7], [0.4, 0.5, 0.6, 0.4, 0.4]])
    ) and jnp.array_equal(val[1], jnp.array([4, 5]))

    val = jax.vmap(
        lambda v: diffrax.utils.bounded_while_loop(cond_fun, body_fun, v, max_steps=8)
    )(init_val)
    assert jnp.array_equal(
        val[0], jnp.array([[0.3, 0.4, 0.5, 0.6, 0.8], [0.4, 0.5, 0.6, 0.4, 0.4]])
    ) and jnp.array_equal(val[1], jnp.array([5, 5]))

    val = jax.vmap(
        lambda v: diffrax.utils.bounded_while_loop(
            cond_fun, body_fun, v, max_steps=None
        )
    )(init_val)
    assert jnp.array_equal(
        val[0], jnp.array([[0.3, 0.4, 0.5, 0.6, 0.8], [0.4, 0.5, 0.6, 0.4, 0.4]])
    ) and jnp.array_equal(val[1], jnp.array([5, 5]))


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


def test_scaling_max_steps():
    key = jrandom.PRNGKey(567)
    expensive_fn = eqx.nn.MLP(in_size=1, out_size=1, width_size=2048, depth=5, key=key)

    def cond_fun(val):
        x, step = val
        return step < 5

    def body_fun(val):
        x, step = val
        return (expensive_fn(x), step + 1), (min(step, 5), None)

    init_val = (
        jnp.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4]]),
        jnp.array([0, 3]),
    )

    fn = lambda: jax.jit(
        jax.vmap(
            lambda v: diffrax.utils.bounded_while_loop(
                cond_fun, body_fun, v, max_steps=16
            )
        )
    )(init_val)
    time16 = time_fn(fn, repeat=10)

    fn = lambda: jax.jit(
        jax.vmap(
            lambda v: diffrax.utils.bounded_while_loop(
                cond_fun, body_fun, v, max_steps=32
            )
        )
    )(init_val)
    time32 = time_fn(fn, repeat=10)

    fn = lambda: jax.jit(
        jax.vmap(
            lambda v: diffrax.utils.bounded_while_loop(
                cond_fun, body_fun, v, max_steps=64
            )
        )
    )(init_val)
    time64 = time_fn(fn, repeat=10)

    fn = lambda: jax.jit(
        jax.vmap(
            lambda v: diffrax.utils.bounded_while_loop(
                cond_fun, body_fun, v, max_steps=128
            )
        )
    )(init_val)
    time128 = time_fn(fn, repeat=10)

    fn = lambda: jax.jit(
        jax.vmap(
            lambda v: diffrax.utils.bounded_while_loop(
                cond_fun, body_fun, v, max_steps=256
            )
        )
    )(init_val)
    time256 = time_fn(fn, repeat=10)

    c2, c1, _ = _quadratic_fit(
        [16, 32, 64, 128, 256], [time16, time32, time64, time128, time256]
    )
    # constant scaling
    assert -0.05 < c2 < 0.05
    assert -0.05 < c1 < 0.05

    fn = lambda: jax.vmap(lambda v: lax.while_loop(cond_fun, _body_fun(body_fun), v))(
        init_val
    )
    time_lax = time_fn(fn, repeat=10)

    assert max(time16, time32, time64, time128, time256) < 2 * time_lax


def test_scaling_num_steps():
    key = jrandom.PRNGKey(567)
    expensive_fn = eqx.nn.MLP(in_size=1, out_size=1, width_size=2048, depth=5, key=key)

    def cond_fun(val):
        x, step, num_steps = val
        return step < num_steps

    def body_fun(val):
        x, step, num_steps = val
        return (expensive_fn(x), step + 1, num_steps), (
            min(step, num_steps),
            None,
            None,
        )

    init_val = (jnp.array([[0.3] * 256, [0.4] * 256]), jnp.array([0, 3]))

    fn = lambda: jax.jit(
        jax.vmap(
            lambda v: diffrax.utils.bounded_while_loop(
                cond_fun, body_fun, (*v, 16), max_steps=256
            )
        )
    )(init_val)
    time16 = time_fn(fn, repeat=10)

    fn = lambda: jax.jit(
        jax.vmap(
            lambda v: diffrax.utils.bounded_while_loop(
                cond_fun, body_fun, (*v, 32), max_steps=256
            )
        )
    )(init_val)
    time32 = time_fn(fn, repeat=10)

    fn = lambda: jax.jit(
        jax.vmap(
            lambda v: diffrax.utils.bounded_while_loop(
                cond_fun, body_fun, (*v, 64), max_steps=256
            )
        )
    )(init_val)
    time64 = time_fn(fn, repeat=10)

    fn = lambda: jax.jit(
        jax.vmap(
            lambda v: diffrax.utils.bounded_while_loop(
                cond_fun, body_fun, (*v, 128), max_steps=256
            )
        )
    )(init_val)
    time128 = time_fn(fn, repeat=10)

    fn = lambda: jax.jit(
        jax.vmap(
            lambda v: diffrax.utils.bounded_while_loop(
                cond_fun, body_fun, (*v, 256), max_steps=256
            )
        )
    )(init_val)
    time256 = time_fn(fn, repeat=10)

    c2, c1, _ = _quadratic_fit(
        [16, 32, 64, 128, 256], [time16, time32, time64, time128, time256]
    )
    # linear scaling
    assert -0.05 < c2 < 0.05
    assert 0.95 < c1 < 1.05

    fn = lambda: jax.jit(
        jax.vmap(lambda v: lax.while_loop(cond_fun, _body_fun(body_fun), (*v, 16)))
    )(init_val)
    lax_time16 = time_fn(fn, repeat=10)

    fn = lambda: jax.jit(
        jax.vmap(lambda v: lax.while_loop(cond_fun, _body_fun(body_fun), (*v, 32)))
    )(init_val)
    lax_time32 = time_fn(fn, repeat=10)

    fn = lambda: jax.jit(
        jax.vmap(lambda v: lax.while_loop(cond_fun, _body_fun(body_fun), (*v, 64)))
    )(init_val)
    lax_time64 = time_fn(fn, repeat=10)

    fn = lambda: jax.jit(
        jax.vmap(lambda v: lax.while_loop(cond_fun, _body_fun(body_fun), (*v, 128)))
    )(init_val)
    lax_time128 = time_fn(fn, repeat=10)

    fn = lambda: jax.jit(
        jax.vmap(lambda v: lax.while_loop(cond_fun, _body_fun(body_fun), (*v, 256)))
    )(init_val)
    lax_time256 = time_fn(fn, repeat=10)

    assert time16 < 2 * lax_time16
    assert time32 < 2 * lax_time32
    assert time64 < 2 * lax_time64
    assert time128 < 2 * lax_time128
    assert time256 < 2 * lax_time256
