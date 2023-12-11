import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optimistix as optx

from .helpers import tree_allclose


# Basic test
def _fn1(x, args):
    return x + 2


# PyTree-valued iteration
def _fn2(x, args):
    a, b = x
    return a - b, b


# Nontrivial interactions between inputs
@jax.jit
def _fn3(x, args):
    mlp = eqx.nn.MLP(4, 4, 256, 2, key=jr.PRNGKey(678))
    return mlp(x) - x


# PyTree-valued argument
def _fn4(x, args):
    a, b = args
    return a * x + b


def test_very_chord():
    tol = 1e-6
    solver = diffrax.VeryChord(rtol=tol, atol=tol)

    x1 = jnp.array(-1.0)
    args1 = None
    grads1 = None

    x2 = (jnp.array(1.0), -0.1)
    args2 = None
    grads2 = None

    x3 = jnp.array([0.1, 0.1, -0.1, -0.05])
    args3 = None
    grads3 = None

    x4 = jnp.array(-2.0)
    args4 = (jnp.array(0.1), jnp.array(2.0))
    grads4 = (jnp.array(200.0), jnp.array(-10.0))

    for fn, x, args, grads in (
        (_fn1, x1, args1, grads1),
        (_fn2, x2, args2, grads2),
        (_fn3, x3, args3, grads3),
        (_fn4, x4, args4, grads4),
    ):
        # Make sure the test is valid
        out = fn(x, args)
        assert jtu.tree_structure(out) == jtu.tree_structure(x)

        def _assert_shape(a, b):
            assert jnp.shape(a) == jnp.shape(b)

        jtu.tree_map(_assert_shape, out, x)

        # Chord method
        zero = jtu.tree_map(jnp.zeros_like, x)
        sol = optx.root_find(fn, solver, x, args, max_steps=100)
        assert sol.result == optx.RESULTS.successful
        assert tree_allclose(fn(sol.value, args), zero, rtol=tol, atol=tol)

        # Very chord method
        init_state = optx.root_find(fn, solver, x, args, max_steps=0, throw=False).state
        sol = optx.root_find(
            fn, solver, x, args, max_steps=100, options=dict(init_state=init_state)
        )
        assert sol.result == optx.RESULTS.successful
        assert tree_allclose(fn(sol.value, args), zero, rtol=tol, atol=tol)
