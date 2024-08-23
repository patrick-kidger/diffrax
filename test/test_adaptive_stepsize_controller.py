from typing import cast

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest
from jaxtyping import Array

from .helpers import tree_allclose


def test_step_ts():
    term = diffrax.ODETerm(lambda t, y, args: -0.2 * y)
    solver = diffrax.Dopri5()
    t0 = 0
    t1 = 5
    dt0 = None
    y0 = 1.0
    pid_controller = diffrax.PIDController(rtol=1e-4, atol=1e-6)
    stepsize_controller = diffrax.JumpStepWrapper(pid_controller, step_ts=[3, 4])
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
        pid_controller = diffrax.PIDController(rtol=1e-4, atol=1e-6)
        stepsize_controller = diffrax.JumpStepWrapper(pid_controller, **kwargs)
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


def test_revisit_steps():
    t0 = 0
    t1 = 5
    dt0 = 0.5
    y0 = 1.0
    drift = diffrax.ODETerm(lambda t, y, args: -0.2 * y)

    def diffusion_vf(t, y, args):
        return jnp.ones((), dtype=y.dtype)

    bm = diffrax.VirtualBrownianTree(t0, t1, 2**-8, (), jr.key(0))
    diffusion = diffrax.ControlTerm(diffusion_vf, bm)
    term = diffrax.MultiTerm(drift, diffusion)
    solver = diffrax.Heun()
    pid_controller = diffrax.PIDController(
        rtol=0, atol=1e-3, dtmin=2**-7, pcoeff=0.5, icoeff=0.8
    )

    rejected_ts = []

    def callback_fun(keep_step, t1):
        if not keep_step:
            rejected_ts.append(t1)

    stepsize_controller = diffrax.JumpStepWrapper(
        pid_controller,
        step_ts=[3, 4],
        rejected_step_buffer_len=10,
        _callback_on_reject=callback_fun,
    )
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

    assert len(rejected_ts) > 10
    # check if all rejected ts are in the array sol.ts
    assert all([t in sol.ts for t in rejected_ts])
    assert 3 in cast(Array, sol.ts)
    assert 4 in cast(Array, sol.ts)


@pytest.mark.parametrize("use_jump_step", [True, False])
def test_backprop(use_jump_step):
    t0 = jnp.asarray(0, dtype=jnp.float64)
    t1 = jnp.asarray(1, dtype=jnp.float64)

    @eqx.filter_jit
    @eqx.filter_grad
    def run(ys, controller, state):
        y0, y1_candidate, y_error = ys
        _, tprev, tnext, _, state, _ = controller.adapt_step_size(
            t0, t1, y0, y1_candidate, None, y_error, 5, state
        )
        with jax.numpy_dtype_promotion("standard"):
            return tprev + tnext + sum(jnp.sum(x) for x in jtu.tree_leaves(state))

    y0 = jnp.array(1.0)
    y1_candidate = jnp.array(2.0)
    term = diffrax.ODETerm(lambda t, y, args: -y)
    solver = diffrax.Tsit5()
    controller = diffrax.PIDController(rtol=1e-4, atol=1e-4)
    if use_jump_step:
        controller = diffrax.JumpStepWrapper(
            controller, step_ts=[0.5], rejected_step_buffer_len=20
        )
    _, state = controller.init(term, t0, t1, y0, 0.1, None, solver.func, 5)

    for y_error in (jnp.array(0.0), jnp.array(3.0), jnp.array(jnp.inf)):
        ys = (y0, y1_candidate, y_error)
        grads = run(ys, controller, state)
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
        pid_controller = diffrax.PIDController(
            rtol=1e-8,
            atol=1e-8,
        )
        stepsize_controller = diffrax.JumpStepWrapper(pid_controller, step_ts=t[None])

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


def test_pid_meta():
    ts = jnp.array([3, 4], dtype=jnp.float64)
    pid1 = diffrax.PIDController(rtol=1e-4, atol=1e-6)
    pid2 = diffrax.PIDController(rtol=1e-4, atol=1e-6, step_ts=ts)
    pid3 = diffrax.PIDController(rtol=1e-4, atol=1e-6, step_ts=ts, jump_ts=ts)
    assert not isinstance(pid1, diffrax.JumpStepWrapper)
    assert isinstance(pid1, diffrax.PIDController)
    assert isinstance(pid2, diffrax.JumpStepWrapper)
    assert isinstance(pid3, diffrax.JumpStepWrapper)
    assert all(pid2.step_ts == ts)
    assert all(pid3.step_ts == ts)
    assert all(pid3.jump_ts == ts)
