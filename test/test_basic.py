import diffrax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import operator

from helpers import random_pytree, treedefs


key = jrandom.PRNGKey(56789)


def test_basic():
    for solver_ctr in (diffrax.euler,):
        for t_dtype in (int, float, jnp.int32, jnp.float32):
            for treedef in treedefs:
                _test_basic(solver_ctr, t_dtype, treedef)


def _test_basic(solver_ctr, t_dtype, treedef):
    def f(t, y):
        return jax.tree_map(operator.neg, y)

    solver = solver_ctr(f)
    if t_dtype is int:
        t0 = 0
        t1 = 2
        dt0 = 1
    elif t_dtype is float:
        t0 = 0.
        t1 = 2.
        dt0 = 1.
    elif t_dtype is jnp.int32:
        t0 = jnp.array(0)
        t1 = jnp.array(2)
        dt0 = jnp.array(1)
    elif t_dtype is jnp.float32:
        t0 = jnp.array(0.)
        t1 = jnp.array(2.)
        dt0 = jnp.array(1.)
    else:
        raise ValueError
    y0 = random_pytree(key, treedef)
    diffrax.diffeqint(solver, t0, t1, y0, dt0)
