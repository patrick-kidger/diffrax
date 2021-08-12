import operator

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest
import scipy.stats
from helpers import all_ode_solvers, random_pytree, treedefs

import diffrax


@pytest.mark.parametrize("solver_ctr", all_ode_solvers)
@pytest.mark.parametrize("t_dtype", (int, float, jnp.int32, jnp.float32))
@pytest.mark.parametrize("treedef", treedefs)
@pytest.mark.parametrize(
    "stepsize_controller", (diffrax.ConstantStepSize(), diffrax.IController())
)
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


@pytest.mark.parametrize("solver_ctr", all_ode_solvers)
def test_order(solver_ctr):
    key = jrandom.PRNGKey(5678)
    akey, ykey = jrandom.split(key, 2)

    A = jrandom.normal(akey, (10, 10), dtype=jnp.float64) * 0.5

    def f(t, y, args):
        return A @ y

    solver = solver_ctr(f)
    t0 = 0
    t1 = 4
    y0 = jrandom.normal(ykey, (10,), dtype=jnp.float64)

    true_yT = jax.scipy.linalg.expm((t1 - t0) * A) @ y0
    exponents = []
    errors = []
    for exponent in [0, -1, -2, -3, -4, -6, -8, -12]:
        dt0 = 2 ** exponent
        sol = diffrax.diffeqsolve(solver, t0, t1, y0, dt0)
        yT = sol.ys[-1]
        error = jnp.sum(jnp.abs(yT - true_yT))
        if error < 2 ** -28:
            break
        exponents.append(exponent)
        errors.append(jnp.log2(error))

    order = scipy.stats.linregress(exponents, errors).slope
    assert solver.order - 0.8 < order < solver.order + 0.8


# Step size deliberately chosen not to divide the time interval
@pytest.mark.parametrize(
    "solver_ctr,dt0",
    ((diffrax.euler, -0.3), (diffrax.tsit5, -0.3), (diffrax.tsit5, None)),
)
@pytest.mark.parametrize(
    "saveat",
    (
        diffrax.SaveAt(t0=True),
        diffrax.SaveAt(t1=True),
        diffrax.SaveAt(t=[3.5, 0.7]),
        diffrax.SaveAt(steps=True),
        diffrax.SaveAt(dense=True),
    ),
)
def test_reverse_time(solver_ctr, dt0, saveat, getkey):
    key = getkey()
    y0 = jrandom.normal(key, (2, 2))
    stepsize_controller = (
        diffrax.IController() if dt0 is None else diffrax.ConstantStepSize()
    )

    def f(t, y, args):
        return -y

    solver = solver_ctr(f)
    t0 = 4
    t1 = 0.3
    sol1 = diffrax.diffeqsolve(
        solver, t0, t1, y0, dt0, stepsize_controller=stepsize_controller, saveat=saveat
    )

    def f(t, y, args):
        return y

    solver = solver_ctr(f)
    t0 = -4
    t1 = -0.3
    negdt0 = None if dt0 is None else -dt0
    if saveat.t is not None:
        saveat = diffrax.SaveAt(t=[-ti for ti in saveat.t])
    sol2 = diffrax.diffeqsolve(
        solver,
        t0,
        t1,
        y0,
        negdt0,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
    )

    if saveat.t0 or saveat.t1 or saveat.t is not None or saveat.steps:
        assert jnp.allclose(sol1.ts, -sol2.ts[::-1])
        assert jnp.allclose(sol1.ys, sol2.ys)
    if saveat.dense:
        t = jnp.linspace(4, 0.3, 20)
        for ti in t:
            assert jnp.allclose(sol1.evaluate(ti), sol2.evaluate(-ti))
            if solver_ctr is not diffrax.tsit5:
                # derivative not implemented for Tsit5
                assert jnp.allclose(sol1.derivative(ti), -sol2.derivative(-ti))
