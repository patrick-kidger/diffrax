import functools as ft
import timeit
from typing import Optional

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest
from diffrax.bounded_while_loop import bounded_while_loop

from .helpers import shaped_allclose


def test_functional_no_vmap_no_inplace():
    def cond_fun(val):
        x, step = val
        return step < 5

    def body_fun(val):
        x, step = val
        return (x + 0.1, step + 1)

    init_val = (jnp.array([0.3]), 0)

    val = bounded_while_loop(cond_fun, body_fun, init_val, max_steps=0)
    assert shaped_allclose(val[0], jnp.array([0.3])) and val[1] == 0

    val = bounded_while_loop(cond_fun, body_fun, init_val, max_steps=1)
    assert shaped_allclose(val[0], jnp.array([0.4])) and val[1] == 1

    val = bounded_while_loop(cond_fun, body_fun, init_val, max_steps=2)
    assert shaped_allclose(val[0], jnp.array([0.5])) and val[1] == 2

    val = bounded_while_loop(cond_fun, body_fun, init_val, max_steps=4)
    assert shaped_allclose(val[0], jnp.array([0.7])) and val[1] == 4

    val = bounded_while_loop(cond_fun, body_fun, init_val, max_steps=8)
    assert shaped_allclose(val[0], jnp.array([0.8])) and val[1] == 5

    val = bounded_while_loop(cond_fun, body_fun, init_val, max_steps=None)
    assert shaped_allclose(val[0], jnp.array([0.8])) and val[1] == 5


def test_functional_no_vmap_inplace():
    def cond_fun(val):
        x, step = val
        return step < 5

    def body_fun(val):
        x, step = val
        x = x.at[jnp.minimum(step + 1, 4)].set(x[step] + 0.1)
        step = step.at[()].set(step + 1)
        return x, step

    def buffers(val):
        x, step = val
        return x

    init_val = (jnp.array([0.3, 0.3, 0.3, 0.3, 0.3]), 0)

    val = bounded_while_loop(cond_fun, body_fun, init_val, max_steps=0, buffers=buffers)
    assert shaped_allclose(val[0], jnp.array([0.3, 0.3, 0.3, 0.3, 0.3])) and val[1] == 0

    val = bounded_while_loop(cond_fun, body_fun, init_val, max_steps=1, buffers=buffers)
    assert shaped_allclose(val[0], jnp.array([0.3, 0.4, 0.3, 0.3, 0.3])) and val[1] == 1

    val = bounded_while_loop(cond_fun, body_fun, init_val, max_steps=2, buffers=buffers)
    assert shaped_allclose(val[0], jnp.array([0.3, 0.4, 0.5, 0.3, 0.3])) and val[1] == 2

    val = bounded_while_loop(cond_fun, body_fun, init_val, max_steps=4, buffers=buffers)
    assert shaped_allclose(val[0], jnp.array([0.3, 0.4, 0.5, 0.6, 0.7])) and val[1] == 4

    val = bounded_while_loop(cond_fun, body_fun, init_val, max_steps=8, buffers=buffers)
    assert shaped_allclose(val[0], jnp.array([0.3, 0.4, 0.5, 0.6, 0.8])) and val[1] == 5

    val = bounded_while_loop(
        cond_fun, body_fun, init_val, max_steps=None, buffers=buffers
    )
    assert shaped_allclose(val[0], jnp.array([0.3, 0.4, 0.5, 0.6, 0.8])) and val[1] == 5


def test_functional_vmap_no_inplace():
    def cond_fun(val):
        x, step = val
        return step < 5

    def body_fun(val):
        x, step = val
        return (x + 0.1, step + 1)

    init_val = (jnp.array([[0.3], [0.4]]), jnp.array([0, 3]))

    val = jax.vmap(lambda v: bounded_while_loop(cond_fun, body_fun, v, max_steps=0))(
        init_val
    )
    assert shaped_allclose(val[0], jnp.array([[0.3], [0.4]])) and jnp.array_equal(
        val[1], jnp.array([0, 3])
    )

    val = jax.vmap(lambda v: bounded_while_loop(cond_fun, body_fun, v, max_steps=1))(
        init_val
    )
    assert shaped_allclose(val[0], jnp.array([[0.4], [0.5]])) and jnp.array_equal(
        val[1], jnp.array([1, 4])
    )

    val = jax.vmap(lambda v: bounded_while_loop(cond_fun, body_fun, v, max_steps=2))(
        init_val
    )
    assert shaped_allclose(val[0], jnp.array([[0.5], [0.6]])) and jnp.array_equal(
        val[1], jnp.array([2, 5])
    )

    val = jax.vmap(lambda v: bounded_while_loop(cond_fun, body_fun, v, max_steps=4))(
        init_val
    )
    assert shaped_allclose(val[0], jnp.array([[0.7], [0.6]])) and jnp.array_equal(
        val[1], jnp.array([4, 5])
    )

    val = jax.vmap(lambda v: bounded_while_loop(cond_fun, body_fun, v, max_steps=8))(
        init_val
    )
    assert shaped_allclose(val[0], jnp.array([[0.8], [0.6]])) and jnp.array_equal(
        val[1], jnp.array([5, 5])
    )

    val = jax.vmap(lambda v: bounded_while_loop(cond_fun, body_fun, v, max_steps=None))(
        init_val
    )
    assert shaped_allclose(val[0], jnp.array([[0.8], [0.6]])) and jnp.array_equal(
        val[1], jnp.array([5, 5])
    )


def test_functional_vmap_inplace():
    def cond_fun(val):
        x, step, max_step = val
        return step < max_step

    def body_fun(val):
        x, step, max_step = val
        x = x.at[jnp.minimum(step + 1, 4)].set(x[step] + 0.1)
        step = step.at[()].set(step + 1)
        return x, step, max_step

    def buffers(val):
        x, step, max_step = val
        return x

    init_val = (
        jnp.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4]]),
        jnp.array([0, 1]),
        jnp.array([5, 3]),
    )

    val = jax.vmap(
        lambda v: bounded_while_loop(
            cond_fun, body_fun, v, max_steps=0, buffers=buffers
        )
    )(init_val)
    assert shaped_allclose(
        val[0], jnp.array([[0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4]])
    ) and jnp.array_equal(val[1], jnp.array([0, 1]))

    val = jax.vmap(
        lambda v: bounded_while_loop(
            cond_fun, body_fun, v, max_steps=1, buffers=buffers
        )
    )(init_val)
    assert shaped_allclose(
        val[0], jnp.array([[0.3, 0.4, 0.3, 0.3, 0.3], [0.4, 0.4, 0.5, 0.4, 0.4]])
    ) and jnp.array_equal(val[1], jnp.array([1, 2]))

    val = jax.vmap(
        lambda v: bounded_while_loop(
            cond_fun, body_fun, v, max_steps=2, buffers=buffers
        )
    )(init_val)
    assert shaped_allclose(
        val[0], jnp.array([[0.3, 0.4, 0.5, 0.3, 0.3], [0.4, 0.4, 0.5, 0.6, 0.4]])
    ) and jnp.array_equal(val[1], jnp.array([2, 3]))

    val = jax.vmap(
        lambda v: bounded_while_loop(
            cond_fun, body_fun, v, max_steps=4, buffers=buffers
        )
    )(init_val)
    assert shaped_allclose(
        val[0], jnp.array([[0.3, 0.4, 0.5, 0.6, 0.7], [0.4, 0.4, 0.5, 0.6, 0.4]])
    ) and jnp.array_equal(val[1], jnp.array([4, 3]))

    val = jax.vmap(
        lambda v: bounded_while_loop(
            cond_fun, body_fun, v, max_steps=8, buffers=buffers
        )
    )(init_val)
    assert shaped_allclose(
        val[0], jnp.array([[0.3, 0.4, 0.5, 0.6, 0.8], [0.4, 0.4, 0.5, 0.6, 0.4]])
    ) and jnp.array_equal(val[1], jnp.array([5, 3]))

    val = jax.vmap(
        lambda v: bounded_while_loop(
            cond_fun, body_fun, v, max_steps=None, buffers=buffers
        )
    )(init_val)
    assert shaped_allclose(
        val[0], jnp.array([[0.3, 0.4, 0.5, 0.6, 0.8], [0.4, 0.4, 0.5, 0.6, 0.4]])
    ) and jnp.array_equal(val[1], jnp.array([5, 3]))


#
# Remaining tests copied from Equinox's tests for `checkpointed_while_loop`.
#


def _get_problem(key, *, num_steps: Optional[int]):
    valkey1, valkey2, modelkey = jr.split(key, 3)

    def cond_fun(carry):
        if num_steps is None:
            return True
        else:
            step, _, _ = carry
            return step < num_steps

    def make_body_fun(dynamic_mlp):
        mlp = eqx.combine(dynamic_mlp, static_mlp)

        def body_fun(carry):
            # A simple new_val = mlp(val) tends to converge to a fixed point in just a
            # few iterations, which implies zero gradient... which doesn't make for a
            # test that actually tests anything. Making things rotational like this
            # keeps things more interesting.
            step, val1, val2 = carry
            (theta,) = mlp(val1)
            real, imag = val1
            z = real + imag * 1j
            z = z * jnp.exp(1j * theta)
            real = jnp.real(z)
            imag = jnp.imag(z)
            val1 = jnp.stack([real, imag])
            val2 = val2.at[step % 8].set(real)
            return step + 1, val1, val2

        return body_fun

    init_val1 = jr.normal(valkey1, (2,))
    init_val2 = jr.normal(valkey2, (20,))
    mlp = eqx.nn.MLP(2, 1, 2, 2, key=modelkey)
    dynamic_mlp, static_mlp = eqx.partition(mlp, eqx.is_array)

    return cond_fun, make_body_fun, init_val1, init_val2, dynamic_mlp


def _while_as_scan(cond, body, init_val, max_steps):
    def f(val, _):
        val2 = lax.cond(cond(val), body, lambda x: x, val)
        return val2, None

    final_val, _ = lax.scan(f, init_val, xs=None, length=max_steps)
    return final_val


@pytest.mark.parametrize("buffer", (False, True))
def test_forward(buffer, getkey):
    cond_fun, make_body_fun, init_val1, init_val2, mlp = _get_problem(
        getkey(), num_steps=5
    )
    body_fun = make_body_fun(mlp)
    true_final_carry = lax.while_loop(cond_fun, body_fun, (0, init_val1, init_val2))
    if buffer:
        buffer_fn = lambda i: i[2]
    else:
        buffer_fn = None
    final_carry = bounded_while_loop(
        cond_fun,
        body_fun,
        (0, init_val1, init_val2),
        max_steps=16,
        buffers=buffer_fn,
    )
    assert shaped_allclose(final_carry, true_final_carry)


@pytest.mark.parametrize("buffer", (False, True))
def test_backward(buffer, getkey):
    cond_fun, make_body_fun, init_val1, init_val2, mlp = _get_problem(
        getkey(), num_steps=None
    )

    @jax.jit
    @jax.value_and_grad
    def true_run(arg):
        init_val1, init_val2, mlp = arg
        body_fun = make_body_fun(mlp)
        _, true_final_val1, true_final_val2 = _while_as_scan(
            cond_fun, body_fun, (0, init_val1, init_val2), max_steps=14
        )
        return jnp.sum(true_final_val1) + jnp.sum(true_final_val2)

    @jax.jit
    @jax.value_and_grad
    def run(arg):
        init_val1, init_val2, mlp = arg
        if buffer:
            buffer_fn = lambda i: i[2]
        else:
            buffer_fn = None
        body_fun = make_body_fun(mlp)
        _, final_val1, final_val2 = bounded_while_loop(
            cond_fun,
            body_fun,
            (0, init_val1, init_val2),
            max_steps=14,
            buffers=buffer_fn,
        )
        return jnp.sum(final_val1) + jnp.sum(final_val2)

    true_value, true_grad = true_run((init_val1, init_val2, mlp))
    value, grad = run((init_val1, init_val2, mlp))
    assert shaped_allclose(value, true_value)
    assert shaped_allclose(grad, true_grad, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("buffer", (False, True))
def test_vmap_primal_unbatched_cond(buffer, getkey):
    cond_fun, make_body_fun, init_val1, init_val2, mlp = _get_problem(
        getkey(), num_steps=14
    )

    @jax.jit
    @ft.partial(jax.vmap, in_axes=((0, 0, None),))
    @jax.value_and_grad
    def true_run(arg):
        init_val1, init_val2, mlp = arg
        body_fun = make_body_fun(mlp)
        _, true_final_val1, true_final_val2 = _while_as_scan(
            cond_fun, body_fun, (0, init_val1, init_val2), max_steps=14
        )
        return jnp.sum(true_final_val1) + jnp.sum(true_final_val2)

    @jax.jit
    @ft.partial(jax.vmap, in_axes=((0, 0, None),))
    @jax.value_and_grad
    def run(arg):
        init_val1, init_val2, mlp = arg
        if buffer:
            buffer_fn = lambda i: i[2]
        else:
            buffer_fn = None
        body_fun = make_body_fun(mlp)
        _, final_val1, final_val2 = bounded_while_loop(
            cond_fun,
            body_fun,
            (0, init_val1, init_val2),
            max_steps=16,
            buffers=buffer_fn,
        )
        return jnp.sum(final_val1) + jnp.sum(final_val2)

    init_val1, init_val2 = jtu.tree_map(
        lambda x: jr.normal(getkey(), (3,) + x.shape, x.dtype), (init_val1, init_val2)
    )
    true_value, true_grad = true_run((init_val1, init_val2, mlp))
    value, grad = run((init_val1, init_val2, mlp))
    assert shaped_allclose(value, true_value)
    assert shaped_allclose(grad, true_grad)


@pytest.mark.parametrize("buffer", (False, True))
def test_vmap_primal_batched_cond(buffer, getkey):
    cond_fun, make_body_fun, init_val1, init_val2, mlp = _get_problem(
        getkey(), num_steps=14
    )

    @jax.jit
    @ft.partial(jax.vmap, in_axes=((0, 0, None), 0))
    @jax.value_and_grad
    def true_run(arg, init_step):
        init_val1, init_val2, mlp = arg
        body_fun = make_body_fun(mlp)
        _, true_final_val1, true_final_val2 = _while_as_scan(
            cond_fun, body_fun, (init_step, init_val1, init_val2), max_steps=14
        )
        return jnp.sum(true_final_val1) + jnp.sum(true_final_val2)

    @jax.jit
    @ft.partial(jax.vmap, in_axes=((0, 0, None), 0))
    @jax.value_and_grad
    def run(arg, init_step):
        init_val1, init_val2, mlp = arg
        if buffer:
            buffer_fn = lambda i: i[2]
        else:
            buffer_fn = None
        body_fun = make_body_fun(mlp)
        _, final_val1, final_val2 = bounded_while_loop(
            cond_fun,
            body_fun,
            (init_step, init_val1, init_val2),
            max_steps=16,
            buffers=buffer_fn,
        )
        return jnp.sum(final_val1) + jnp.sum(final_val2)

    init_step = jnp.array([0, 1, 2, 3, 5, 10])
    init_val1, init_val2 = jtu.tree_map(
        lambda x: jr.normal(getkey(), (6,) + x.shape, x.dtype), (init_val1, init_val2)
    )
    true_value, true_grad = true_run((init_val1, init_val2, mlp), init_step)
    value, grad = run((init_val1, init_val2, mlp), init_step)
    assert shaped_allclose(value, true_value, rtol=1e-4, atol=1e-4)
    assert shaped_allclose(grad, true_grad, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("buffer", (False, True))
def test_vmap_cotangent(buffer, getkey):
    cond_fun, make_body_fun, init_val1, init_val2, mlp = _get_problem(
        getkey(), num_steps=14
    )

    @jax.jit
    @jax.jacrev
    def true_run(arg):
        init_val1, init_val2, mlp = arg
        body_fun = make_body_fun(mlp)
        _, true_final_val1, true_final_val2 = _while_as_scan(
            cond_fun, body_fun, (0, init_val1, init_val2), max_steps=14
        )
        return true_final_val1, true_final_val2

    @jax.jit
    @jax.jacrev
    def run(arg):
        init_val1, init_val2, mlp = arg
        if buffer:
            buffer_fn = lambda i: i[2]
        else:
            buffer_fn = None
        body_fun = make_body_fun(mlp)
        _, final_val1, final_val2 = bounded_while_loop(
            cond_fun,
            body_fun,
            (0, init_val1, init_val2),
            max_steps=16,
            buffers=buffer_fn,
        )
        return final_val1, final_val2

    true_jac = true_run((init_val1, init_val2, mlp))
    jac = run((init_val1, init_val2, mlp))
    assert shaped_allclose(jac, true_jac, rtol=1e-4, atol=1e-4)


# This tests the possible failure mode of "the buffer doesn't do anything".
# This test takes O(1e-3) seconds with buffer.
# This test takes O(10) seconds without buffer.
# This speed improvement is precisely the reason that buffer exists.
def test_speed_buffer_while():
    size = 16**4

    @jax.jit
    @jax.vmap
    def f(init_step, init_xs):
        def cond(carry):
            step, xs = carry
            return step < size

        def body(carry):
            step, xs = carry
            xs = xs.at[step].set(1)
            return step + 1, xs

        def loop(init_xs):
            return bounded_while_loop(
                cond,
                body,
                (init_step, init_xs),
                max_steps=size,
                buffers=lambda i: i[1],
            )

        # Linearize so that we save residuals
        return jax.linearize(loop, init_xs)

    # nontrivial batch size is important to ensure that the `.at[].set()` is really a
    # scatter, and that XLA doesn't optimise it into a dynamic_update_slice. (Which
    # can be switched with `select` in the compiler.)
    args = jnp.array([0, 1]), jnp.zeros((2, size))
    f(*args)  # compile

    speed = timeit.timeit(lambda: f(*args), number=1)
    assert speed < 0.1


# This isn't testing any particular failure mode: just that things generally work.
def test_speed_grad_checkpointed_while(getkey):
    mlp = eqx.nn.MLP(2, 1, 2, 2, key=getkey())

    @jax.jit
    @jax.vmap
    @jax.grad
    def f(init_val, init_step):
        def cond(carry):
            step, _ = carry
            return step < 8 * 16**3

        def body(carry):
            step, val = carry
            (theta,) = mlp(val)
            real, imag = val
            z = real + imag * 1j
            z = z * jnp.exp(1j * theta)
            real = jnp.real(z)
            imag = jnp.imag(z)
            return step + 1, jnp.stack([real, imag])

        _, final_xs = bounded_while_loop(
            cond,
            body,
            (init_step, init_val),
            max_steps=16**3,
        )
        return jnp.sum(final_xs)

    init_step = jnp.array([0, 10])
    init_val = jr.normal(getkey(), (2, 2))

    f(init_val, init_step)  # compile
    speed = timeit.timeit(lambda: f(init_val, init_step), number=1)
    # Should take ~0.001 seconds
    assert speed < 0.01


# This is deliberately meant to emulate the pattern of saving used in
# `diffrax.diffeqsolve(..., saveat=SaveAt(ts=...))`.
def test_nested_loops(getkey):
    @ft.partial(jax.jit, static_argnums=5)
    @ft.partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, None))
    def run(step, vals, ts, final_step, cotangents, true):
        value, vjp_fn = jax.vjp(
            lambda *v: outer_loop(step, v, ts, true, final_step), *vals
        )
        cotangents = vjp_fn(cotangents)
        return value, cotangents

    def outer_loop(step, vals, ts, true, final_step):
        def cond(carry):
            step, _ = carry
            return step < final_step

        def body(carry):
            step, (val1, val2, val3, val4) = carry
            mul = 1 + 0.05 * jnp.sin(105 * val1 + 1)
            val1 = val1 * mul
            return inner_loop(step, (val1, val2, val3, val4), ts, true)

        def buffers(carry):
            _, (_, val2, val3, _) = carry
            return val2, val3

        if true:
            while_loop = ft.partial(_while_as_scan, max_steps=50)
        else:
            while_loop = ft.partial(bounded_while_loop, max_steps=50, buffers=buffers)
        _, out = while_loop(cond, body, (step, vals))
        return out

    def inner_loop(step, vals, ts, true):
        ts_done = jnp.floor(ts[step] + 1)

        def cond(carry):
            step, _ = carry
            return ts[step] < ts_done

        def body(carry):
            step, (val1, val2, val3, val4) = carry
            mul = 1 + 0.05 * jnp.sin(100 * val1 + 3)
            val1 = val1 * mul
            val2 = val2.at[step].set(val1)
            val3 = val3.at[step].set(val1)
            val4 = val4.at[step].set(val1)
            return step + 1, (val1, val2, val3, val4)

        def buffers(carry):
            _, (_, _, val3, val4) = carry
            return val3, val4

        if true:
            while_loop = ft.partial(_while_as_scan, max_steps=10)
        else:
            while_loop = ft.partial(bounded_while_loop, max_steps=10, buffers=buffers)
        return while_loop(cond, body, (step, vals))

    step = jnp.array([0, 5])
    val1 = jr.uniform(getkey(), shape=(2,), minval=0.1, maxval=0.7)
    val2 = val3 = val4 = jnp.zeros((2, 47))
    ts = jnp.stack([jnp.linspace(0, 19, 47), jnp.linspace(0, 13, 47)])
    final_step = jnp.array([46, 43])
    cotangents = (
        jr.normal(getkey(), (2,)),
        jr.normal(getkey(), (2, 47)),
        jr.normal(getkey(), (2, 47)),
        jr.normal(getkey(), (2, 47)),
    )

    value, grads = run(
        step, (val1, val2, val3, val4), ts, final_step, cotangents, False
    )
    true_value, true_grads = run(
        step, (val1, val2, val3, val4), ts, final_step, cotangents, True
    )

    assert shaped_allclose(value, true_value)
    assert shaped_allclose(grads, true_grads, rtol=1e-4, atol=1e-5)
