import diffrax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import operator

from helpers import random_pytree, treedefs


key = jrandom.PRNGKey(56789)


def test_basic():
    for solver_ctr in (diffrax.bosh3, diffrax.dopri5, diffrax.dopri8, diffrax.euler, diffrax.fehlberg2, diffrax.heun):
        for t_dtype in (int, float, jnp.int32, jnp.float32):
            for treedef in treedefs:
                for stepsize_controller in (diffrax.ConstantStepSize(), diffrax.IController()):
                    if solver_ctr is diffrax.euler and isinstance(stepsize_controller, diffrax.IController):
                        continue
                    _test_basic(solver_ctr, t_dtype, treedef, stepsize_controller)


def _test_basic(solver_ctr, t_dtype, treedef, stepsize_controller):
    def f(t, y, args):
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
    diffrax.diffeqint(solver, t0, t1, y0, dt0, stepsize_controller=stepsize_controller)
