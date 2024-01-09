from typing import cast

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array

from .helpers import tree_allclose


def test_step_ts():
    term = diffrax.ODETerm(lambda t, y, args: -0.2 * y)
    solver = diffrax.Dopri5()
    t0 = 0
    t1 = 5
    dt0 = None
    y0 = 1.0
    stepsize_controller = diffrax.PIDController(rtol=1e-4, atol=1e-6, step_ts=[3, 4])
    saveat = diffrax.SaveAt(steps=True)
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
    )
    assert 3 in cast(Array, sol.ts)
    assert 4 in cast(Array, sol.ts)


def test_jump_ts():
    # Tests no regression of https://github.com/patrick-kidger/diffrax/issues/58

    def vector_field(t, y, args):
        x, v = y
        force = jnp.where(t < 7.5, 10, -10)
        return v, -4 * jnp.pi**2 * x - 4 * jnp.pi * 0.05 * v + force

    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Dopri5()
    t0 = 0
    t1 = 15
    dt0 = None
    y0 = 1.5, 0
    saveat = diffrax.SaveAt(steps=True)

    def run(**kwargs):
        stepsize_controller = diffrax.PIDController(rtol=1e-4, atol=1e-6, **kwargs)
        return diffrax.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            y0,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
        )

    sol_no_jump_ts = run()
    sol_with_jump_ts = run(jump_ts=[7.5])
    assert sol_no_jump_ts.stats["num_steps"] > sol_with_jump_ts.stats["num_steps"]
    assert sol_with_jump_ts.result == diffrax.RESULTS.successful

    sol = run(jump_ts=[7.5], step_ts=[7.5])
    assert sol.result == diffrax.RESULTS.successful
    sol = run(jump_ts=[7.5], step_ts=[3.5, 8])
    assert sol.result == diffrax.RESULTS.successful
    assert 3.5 in cast(Array, sol.ts)
    assert 8 in cast(Array, sol.ts)


def test_backprop():
    @eqx.filter_jit
    @eqx.filter_grad
    def run(ys, controller, state):
        y0, y1_candidate, y_error = ys
        _, tprev, tnext, _, state, _ = controller.adapt_step_size(
            0, 1, y0, y1_candidate, None, y_error, 5, state
        )
        with jax.numpy_dtype_promotion("standard"):
            return tprev + tnext + sum(jnp.sum(x) for x in jtu.tree_leaves(state))

    y0 = jnp.array(1.0)
    y1_candidate = jnp.array(2.0)
    term = diffrax.ODETerm(lambda t, y, args: -y)
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(rtol=1e-4, atol=1e-4)
    _, state = stepsize_controller.init(term, 0, 1, y0, 0.1, None, solver.func, 5)

    for y_error in (jnp.array(0.0), jnp.array(3.0), jnp.array(jnp.inf)):
        ys = (y0, y1_candidate, y_error)
        grads = run(ys, stepsize_controller, state)
        assert not any(jnp.isnan(grad).any() for grad in grads)


def test_grad_of_discontinuous_forcing():
    def vector_field(t, y, forcing):
        y, _ = y
        dy = -y + forcing(t)
        dsum = y
        return dy, dsum

    def run(t):
        term = diffrax.ODETerm(vector_field)
        solver = diffrax.Tsit5()
        t0 = 0
        t1 = 1
        dt0 = None
        y0 = 1.0
        stepsize_controller = diffrax.PIDController(
            rtol=1e-8, atol=1e-8, step_ts=t[None]
        )

        def forcing(s):
            return jnp.where(s < t, 0, 1)

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            (y0, 0),
            args=forcing,
            stepsize_controller=stepsize_controller,
        )
        _, sum = cast(Array, sol.ys)
        (sum,) = sum
        return sum

    r = jax.jit(run)
    eps = 1e-5
    finite_diff = (r(0.5) - r(0.5 - eps)) / eps
    autodiff = jax.jit(jax.grad(run))(0.5)
    assert tree_allclose(finite_diff, autodiff)
