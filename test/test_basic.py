import operator

import jax
import jax.numpy as jnp
import jax.random as jrandom
from helpers import random_pytree, treedefs
import pytest

import diffrax


@pytest.mark.parametrize("solver_ctr", (diffrax.bosh3, diffrax.dopri5, diffrax.dopri8, diffrax.euler, diffrax.fehlberg2, diffrax.heun, diffrax.tsit5))
@pytest.mark.parametrize("t_dtype", (int, float, jnp.int32, jnp.float32))
@pytest.mark.parametrize("treedef", treedefs)
@pytest.mark.parametrize("stepsize_controller", (diffrax.ConstantStepSize(), diffrax.IController()))
@pytest.mark.parametrize("jit", (False, True))
def test_basic(solver_ctr, t_dtype, treedef, stepsize_controller, jit, getkey):
    if solver_ctr is diffrax.euler and isinstance(
        stepsize_controller, diffrax.IController
    ):
        return


    def f(t, y, args):
        return jax.tree_map(operator.neg, y)

    solver = solver_ctr(f)
    if t_dtype is int:
        t0 = 0
        t1 = 2
        dt0 = 1
    elif t_dtype is float:
        t0 = 0.0
        t1 = 2.0
        dt0 = 1.0
    elif t_dtype is jnp.int32:
        t0 = jnp.array(0)
        t1 = jnp.array(2)
        dt0 = jnp.array(1)
    elif t_dtype is jnp.float32:
        t0 = jnp.array(0.0)
        t1 = jnp.array(2.0)
        dt0 = jnp.array(1.0)
    else:
        raise ValueError
    y0 = random_pytree(getkey(), treedef)
    diffrax.diffeqsolve(
        solver, t0, t1, y0, dt0, stepsize_controller=stepsize_controller, jit=jit
    )
