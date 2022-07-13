import operator

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from .helpers import shaped_allclose


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
    mlp = eqx.nn.MLP(4, 4, 256, 2, key=jrandom.PRNGKey(678))
    return mlp(x) - x


# Has an argument
def _fn4(x, args):
    return (x - args) ** 2


# PyTree-valued argument
def _fn5(x, args):
    a, b = args
    return a * x + b


def test_newton_solver():
    tol = 1e-6
    solver = diffrax.NewtonNonlinearSolver(max_steps=100, rtol=tol, atol=tol)
    mega_solver = diffrax.NewtonNonlinearSolver(max_steps=100000000, rtol=tol, atol=tol)

    x1 = jnp.array(-1.0)
    args1 = None
    grads1 = None

    x2 = (jnp.array(1.0), -0.1)
    args2 = None
    grads2 = None

    x3 = jnp.array([0.1, 0.1, -0.1, -0.05])
    args3 = None
    grads3 = None

    x4 = jnp.array([3.0, 4.0])
    args4 = jnp.array([5.0])
    grads4 = jnp.array([2.0])

    x5 = jnp.array(-2.0)
    args5 = (jnp.array(0.1), jnp.array(2.0))
    grads5 = (jnp.array(200.0), jnp.array(-10.0))

    for fn, x, args, grads in (
        (_fn1, x1, args1, grads1),
        (_fn2, x2, args2, grads2),
        (_fn3, x3, args3, grads3),
        (_fn4, x4, args4, grads4),
        (_fn5, x5, args5, grads5),
    ):
        # Make sure the test is valid
        out = fn(x, args)
        assert jax.tree_structure(out) == jax.tree_structure(x)

        def _assert_shape(a, b):
            assert jnp.shape(a) == jnp.shape(b)

        jax.tree_map(_assert_shape, out, x)

        # Newton's method
        zero = jax.tree_map(jnp.zeros_like, x)
        sol = solver(fn, x, args)
        assert sol.result == 0
        assert shaped_allclose(fn(sol.root, args), zero, rtol=tol, atol=tol)

        # Chord method
        jac = solver.jac(fn, x, args)
        if fn is _fn4:
            # Need a looooot of steps for this one.
            sol = mega_solver(fn, x, args, jac)
        else:
            sol = solver(fn, x, args, jac)
        assert sol.result == 0
        assert shaped_allclose(fn(sol.root, args), zero, rtol=tol, atol=tol)

        def _fn(y, a):
            sol = solver(fn, y, a)
            root = jax.tree_map(jnp.sum, sol.root)
            return jax.tree_util.tree_reduce(operator.add, root)

        x_grads, args_grads = jax.grad(_fn, argnums=(0, 1))(x, args)
        assert shaped_allclose(x_grads, zero, rtol=tol, atol=tol)
        assert shaped_allclose(args_grads, grads, rtol=tol, atol=tol)
