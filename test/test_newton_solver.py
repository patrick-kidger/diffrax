import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

import helpers


def _fn1(x):
    return x + 2


def _fn2(x):
    a, b = x
    return a - b


@jax.jit
def _fn3(x):
    mlp = eqx.nn.MLP(4, 4, 256, 2, key=jrandom.PRNGKey(678))
    return mlp(x)


def _fn4(x, y):
    return jnp.sum((x - y) ** 2)


def _fn5(x, y):
    a, b = y
    return a * x + b


def test_newton_solver():
    tol = 1e-6
    solver = diffrax.NewtonNonlinearSolver(max_steps=100, rtol=tol, atol=tol)

    x1 = jnp.array(-1.0)
    args1 = ()
    grads1 = ()

    x2 = (jnp.array(1.0), -0.1)
    args2 = ()
    grads2 = ()

    x3 = jnp.array([0.1, 0.1, -0.1, -0.05])
    args3 = ()
    grads3 = ()

    x4 = jnp.array([3.0, 4.0])
    args4 = jnp.array([[[5.0]]])
    grads4 = jnp.array([[[1.0]]])

    x5 = jnp.array(-2.0)
    args5 = (jnp.array(0.1), jnp.array(2.0))
    grads5 = (jnp.array(-10.0), jnp.array(200.0))

    for fn, x, args, grads in (
        (_fn1, x1, args1, grads1),
        (_fn2, x2, args2, grads2),
        (_fn3, x3, args3, grads3),
        (_fn4, x4, args4, grads4),
        (_fn5, x5, args5, grads5),
    ):
        out = solver(fn, x, args)
        assert jnp.allclose(fn(out, *args), 0, rtol=tol, atol=tol)
        jac = solver.jac(fn, x, args)
        out = solver(fn, x, args, jac)
        assert jnp.allclose(fn(out, *args), 0, rtol=tol, atol=tol)

        def _fn(y, a):
            return jnp.sum(solver(fn, y, a))

        x_grads, args_grads = jax.grad(_fn)(x, args)
        assert jnp.allclose(x_grads, 0, rtol=tol, atol=tol)
        assert helpers.tree_allclose(args_grads, grads, rtol=tol, atol=tol)
