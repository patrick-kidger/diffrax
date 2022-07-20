import math
from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import pytest

from .helpers import shaped_allclose


def test_no_adjoint():
    def fn(y0):
        term = diffrax.ODETerm(lambda t, y, args: -y)
        t0 = 0
        t1 = 1
        dt0 = 0.1
        solver = diffrax.Dopri5()
        adjoint = diffrax.NoAdjoint()
        sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, adjoint=adjoint)
        return jnp.sum(sol.ys)

    with pytest.raises(RuntimeError):
        jax.grad(fn)(1.0)

    primal, dual = jax.jvp(fn, (1.0,), (1.0,))
    e_inv = 1 / math.e
    assert shaped_allclose(primal, e_inv)
    assert shaped_allclose(dual, e_inv)


class _VectorField(eqx.Module):
    nondiff_arg: int
    diff_arg: float

    def __call__(self, t, y, args):
        assert y.shape == (2,)
        diff_arg, nondiff_arg = args
        dya = diff_arg * y[0] + nondiff_arg * y[1]
        dyb = self.nondiff_arg * y[0] + self.diff_arg * y[1]
        return jnp.stack([dya, dyb])


def test_backsolve(getkey):
    y0 = jnp.array([0.9, 5.4])
    args = (0.1, -1)
    term = diffrax.ODETerm(_VectorField(nondiff_arg=1, diff_arg=-0.1))
    y0__args__term = (y0, args, term)
    del y0, args, term

    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)

    def _run(y0__args__term, saveat, adjoint):
        y0, args, term = y0__args__term
        return jnp.sum(
            diffrax.diffeqsolve(
                term,
                solver,
                0.3,
                9.5,
                None,
                y0,
                args,
                stepsize_controller=stepsize_controller,
                saveat=saveat,
                adjoint=adjoint,
            ).ys
        )

    diff, nondiff = eqx.partition(y0__args__term, eqx.is_inexact_array)
    _run_grad = eqx.filter_jit(
        jax.grad(
            lambda d, saveat, adjoint: _run(eqx.combine(d, nondiff), saveat, adjoint)
        )
    )
    _run_grad_int = eqx.filter_jit(jax.grad(_run, allow_int=True))

    # Yep, test that they're not implemented. We can remove these checks if we ever
    # do implement them.
    # Until that day comes, it's worth checking that things don't silently break.
    with pytest.raises(NotImplementedError):
        _run_grad_int(
            y0__args__term, diffrax.SaveAt(steps=True), diffrax.BacksolveAdjoint()
        )
    with pytest.raises(NotImplementedError):
        _run_grad_int(
            y0__args__term, diffrax.SaveAt(dense=True), diffrax.BacksolveAdjoint()
        )

    def _convert_float0(x):
        # bool also to work around JAX issue #11238
        if x.dtype in (jax.dtypes.float0, jnp.dtype("bool")):
            return 0
        else:
            return x

    for t0 in (True, False):
        for t1 in (True, False):
            for ts in (None, [0.3], [2.0], [9.5], [1.0, 7.0], [0.3, 7.0, 9.5]):
                if t0 is False and t1 is False and ts is None:
                    continue
                saveat = diffrax.SaveAt(t0=t0, t1=t1, ts=ts)
                true_grads = _run_grad_int(
                    y0__args__term, saveat, diffrax.RecursiveCheckpointAdjoint()
                )
                backsolve_grads = _run_grad_int(
                    y0__args__term, saveat, diffrax.BacksolveAdjoint()
                )
                true_grads = jax.tree_map(_convert_float0, true_grads)
                backsolve_grads = jax.tree_map(_convert_float0, backsolve_grads)
                assert shaped_allclose(true_grads, backsolve_grads)

                true_grads = _run_grad(
                    diff, saveat, diffrax.RecursiveCheckpointAdjoint()
                )
                backsolve_grads = _run_grad(diff, saveat, diffrax.BacksolveAdjoint())
                assert shaped_allclose(true_grads, backsolve_grads)


def test_adjoint_seminorm():
    vector_field = lambda t, y, args: -y
    term = diffrax.ODETerm(vector_field)

    def solve(y0):
        adjoint = diffrax.BacksolveAdjoint(
            stepsize_controller=diffrax.PIDController(
                rtol=1e-3, atol=1e-6, norm=diffrax.adjoint_rms_seminorm
            )
        )
        sol = diffrax.diffeqsolve(
            term,
            diffrax.Tsit5(),
            0,
            1,
            None,
            y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            adjoint=adjoint,
        )
        return jnp.sum(sol.ys)

    jax.grad(solve)(2.0)


def test_closure_errors():
    mlp = eqx.nn.MLP(1, 1, 8, 2, key=jrandom.PRNGKey(0))

    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def run(model):
        def f(t, y, args):
            return model(y)

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(f),
            diffrax.Euler(),
            0,
            1,
            0.1,
            jnp.array([1.0]),
            adjoint=diffrax.BacksolveAdjoint(),
        )
        return jnp.sum(sol.ys)

    with pytest.raises(jax.interpreters.ad.CustomVJPException):
        run(mlp)


def test_closure_fixed():
    mlp = eqx.nn.MLP(1, 1, 8, 2, key=jrandom.PRNGKey(0))

    class VectorField(eqx.Module):
        model: eqx.Module

        def __call__(self, t, y, args):
            return self.model(y)

    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def run(model):
        f = VectorField(model)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(f),
            diffrax.Euler(),
            0,
            1,
            0.1,
            jnp.array([1.0]),
            adjoint=diffrax.BacksolveAdjoint(),
        )
        return jnp.sum(sol.ys)

    run(mlp)


def test_implicit():
    class ExponentialDecayToSteadyState(eqx.Module):
        steady_state: float
        non_jax_type: Any

        def __call__(self, t, y, args):
            return self.steady_state - y

    def loss(model, target_steady_state):
        term = diffrax.ODETerm(model)
        solver = diffrax.Tsit5()
        t0 = 0
        t1 = jnp.inf
        dt0 = None
        y0 = 1.0
        max_steps = None
        controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        event = diffrax.SteadyStateEvent()
        adjoint = diffrax.ImplicitAdjoint()
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            y0,
            max_steps=max_steps,
            stepsize_controller=controller,
            discrete_terminating_event=event,
            adjoint=adjoint,
        )
        (y1,) = sol.ys
        return (y1 - target_steady_state) ** 2

    model = ExponentialDecayToSteadyState(jnp.array(0.0), object())
    target_steady_state = jnp.array(0.76)
    optim = optax.sgd(1e-2, momentum=0.7, nesterov=True)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(model, opt_state, target_steady_state):
        grads = eqx.filter_grad(loss)(model, target_steady_state)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state

    for step in range(100):
        model, opt_state = make_step(model, opt_state, target_steady_state)
    assert shaped_allclose(
        model.steady_state, target_steady_state, rtol=1e-2, atol=1e-2
    )
