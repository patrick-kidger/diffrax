import math
import operator

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest
import scipy.stats

from helpers import all_ode_solvers, random_pytree, shaped_allclose, treedefs


@pytest.mark.parametrize("solver_ctr", all_ode_solvers)
@pytest.mark.parametrize("t_dtype", (int, float, jnp.int32, jnp.float32))
@pytest.mark.parametrize("treedef", treedefs)
@pytest.mark.parametrize(
    "stepsize_controller", (diffrax.ConstantStepSize(), diffrax.IController())
)
def test_basic(solver_ctr, t_dtype, treedef, stepsize_controller, getkey):
    if not issubclass(solver_ctr, diffrax.AbstractAdaptiveSolver) and isinstance(
        stepsize_controller, diffrax.IController
    ):
        return

    def f(t, y, args):
        return jax.tree_map(operator.neg, y)

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
    try:
        diffrax.diffeqsolve(
            diffrax.ODETerm(f),
            t0,
            t1,
            y0,
            dt0,
            solver=solver_ctr(),
            stepsize_controller=stepsize_controller,
        )
    except RuntimeError as e:
        if isinstance(stepsize_controller, diffrax.ConstantStepSize) and str(
            e
        ).startswith("Implicit"):
            # Implicit method failed to converge. A very normal thing to have happen;
            # usually we'd use adaptive timestepping to handle it.
            pass
        else:
            raise


@pytest.mark.parametrize("solver_ctr", all_ode_solvers)
def test_order(solver_ctr):
    key = jrandom.PRNGKey(5678)
    akey, ykey = jrandom.split(key, 2)

    A = jrandom.normal(akey, (10, 10), dtype=jnp.float64) * 0.5

    def f(t, y, args):
        return A @ y

    t0 = 0
    t1 = 4
    y0 = jrandom.normal(ykey, (10,), dtype=jnp.float64)

    true_yT = jax.scipy.linalg.expm((t1 - t0) * A) @ y0
    exponents = []
    errors = []
    for exponent in [0, -1, -2, -3, -4, -6, -8, -12]:
        dt0 = 2 ** exponent
        sol = diffrax.diffeqsolve(diffrax.ODETerm(f), t0, t1, y0, dt0, solver_ctr())
        yT = sol.ys[-1]
        error = jnp.sum(jnp.abs(yT - true_yT))
        if error < 2 ** -28:
            break
        exponents.append(exponent)
        errors.append(jnp.log2(error))

    order = scipy.stats.linregress(exponents, errors).slope
    # We accept quite a wide range. Improving this test would be nice.
    assert -0.9 < order - solver_ctr.order < 0.9


# Step size deliberately chosen not to divide the time interval
@pytest.mark.parametrize(
    "solver_ctr,dt0",
    ((diffrax.Euler, -0.3), (diffrax.Tsit5, -0.3), (diffrax.Tsit5, None)),
)
@pytest.mark.parametrize(
    "saveat",
    (
        diffrax.SaveAt(t0=True),
        diffrax.SaveAt(t1=True),
        diffrax.SaveAt(ts=[3.5, 0.7]),
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

    t0 = 4
    t1 = 0.3
    sol1 = diffrax.diffeqsolve(
        diffrax.ODETerm(f),
        t0,
        t1,
        y0,
        dt0,
        solver_ctr(),
        stepsize_controller=stepsize_controller,
        saveat=saveat,
    )
    assert shaped_allclose(sol1.t0, 4)
    assert shaped_allclose(sol1.t1, 0.3)

    def f(t, y, args):
        return y

    t0 = -4
    t1 = -0.3
    negdt0 = None if dt0 is None else -dt0
    if saveat.ts is not None:
        saveat = diffrax.SaveAt(ts=[-ti for ti in saveat.ts])
    sol2 = diffrax.diffeqsolve(
        diffrax.ODETerm(f),
        t0,
        t1,
        y0,
        negdt0,
        solver_ctr(),
        stepsize_controller=stepsize_controller,
        saveat=saveat,
    )
    assert shaped_allclose(sol2.t0, -4)
    assert shaped_allclose(sol2.t1, -0.3)

    if saveat.t0 or saveat.t1 or saveat.ts is not None or saveat.steps:
        assert shaped_allclose(sol1.ts, -sol2.ts, equal_nan=True)
        assert shaped_allclose(sol1.ys, sol2.ys, equal_nan=True)
    if saveat.dense:
        t = jnp.linspace(0.3, 4, 20)
        for ti in t:
            assert shaped_allclose(sol1.evaluate(ti), sol2.evaluate(-ti))
            if solver_ctr is not diffrax.Tsit5:
                # derivative not implemented for Tsit5
                assert shaped_allclose(sol1.derivative(ti), -sol2.derivative(-ti))


@pytest.mark.parametrize(
    "solver_ctr,stepsize_controller,dt0",
    (
        (diffrax.Tsit5, diffrax.ConstantStepSize(), 0.3),
        (diffrax.Tsit5, diffrax.IController(rtol=1e-8, atol=1e-8), None),
        (diffrax.Kvaerno3, diffrax.IController(rtol=1e-8, atol=1e-8), None),
    ),
)
@pytest.mark.parametrize("treedef", treedefs)
def test_pytree_state(solver_ctr, stepsize_controller, dt0, treedef, getkey):
    term = diffrax.ODETerm(lambda t, y, args: jax.tree_map(operator.neg, y))
    y0 = random_pytree(getkey(), treedef)
    sol = diffrax.diffeqsolve(
        term,
        t0=0,
        t1=1,
        y0=y0,
        dt0=dt0,
        solver=solver_ctr(),
        stepsize_controller=stepsize_controller,
    )
    y1 = sol.ys
    true_y1 = jax.tree_map(lambda x: (x * math.exp(-1))[None], y0)
    assert shaped_allclose(y1, true_y1)


def test_semi_implicit_euler():
    term1 = diffrax.ODETerm(lambda t, y, args: -y)
    term2 = diffrax.ODETerm(lambda t, y, args: y)
    y0 = (1.0, -0.5)
    dt0 = 0.00001
    sol1 = diffrax.diffeqsolve(
        (term1, term2),
        0,
        1,
        y0,
        dt0,
        solver=diffrax.SemiImplicitEuler(),
        max_steps=100000,
    )
    term_combined = diffrax.ODETerm(lambda t, y, args: (-y[1], y[0]))
    sol2 = diffrax.diffeqsolve(term_combined, 0, 1, y0, 0.001, solver=diffrax.Tsit5())
    assert shaped_allclose(sol1.ys, sol2.ys)
