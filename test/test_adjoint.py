from collections.abc import Callable
from typing import Any, cast

import diffrax
import equinox as eqx
import jax
import jax.interpreters.ad
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax
import pytest
from jaxtyping import Array

from .helpers import tree_allclose


class _VectorField(eqx.Module):
    nondiff_arg: int
    diff_arg: float

    def __call__(self, t, y, args):
        assert y.shape == (2,)
        diff_arg, nondiff_arg = args
        dya = diff_arg * y[0] + nondiff_arg * y[1]
        dyb = self.nondiff_arg * y[0] + self.diff_arg * y[1]
        return jnp.stack([dya, dyb])


@pytest.mark.slow
def test_against(getkey):
    y0 = jnp.array([0.9, 5.4])
    args = (0.1, -1)
    term = diffrax.ODETerm(_VectorField(nondiff_arg=1, diff_arg=-0.1))
    y0__args__term = (y0, args, term)
    del y0, args, term

    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)

    def _run(y0__args__term, saveat, adjoint):
        y0, args, term = y0__args__term
        ys = diffrax.diffeqsolve(
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
        return jnp.sum(cast(Array, ys))

    # Only does gradients with respect to y0
    def _run_finite_diff(y0__args__term, saveat, adjoint):
        y0, args, term = y0__args__term
        y0_a = y0 + jnp.array([1e-5, 0])
        y0_b = y0 + jnp.array([0, 1e-5])
        val = _run((y0, args, term), saveat, adjoint)
        val_a = _run((y0_a, args, term), saveat, adjoint)
        val_b = _run((y0_b, args, term), saveat, adjoint)
        out_a = (val_a - val) / 1e-5
        out_b = (val_b - val) / 1e-5
        return jnp.stack([out_a, out_b])

    inexact, static = eqx.partition(y0__args__term, eqx.is_inexact_array)

    def _run_inexact(inexact, saveat, adjoint):
        return _run(eqx.combine(inexact, static), saveat, adjoint)

    _run_grad = eqx.filter_jit(jax.grad(_run_inexact))
    _run_grad_int = eqx.filter_jit(jax.grad(_run, allow_int=True))

    twice_inexact = jtu.tree_map(lambda *x: jnp.stack(x), inexact, inexact)

    @eqx.filter_jit
    def _run_vmap_grad(twice_inexact, saveat, adjoint):
        f = jax.vmap(jax.grad(_run_inexact), in_axes=(0, None, None))
        return f(twice_inexact, saveat, adjoint)

    # @eqx.filter_jit
    # def _run_vmap_finite_diff(twice_inexact, saveat, adjoint):
    #     @jax.vmap
    #     def _run_impl(inexact):
    #         y0__args__term = eqx.combine(inexact, static)
    #         return _run_finite_diff(y0__args__term, saveat, adjoint)
    #     return _run_impl(twice_inexact)

    @eqx.filter_jit
    def _run_grad_vmap(twice_inexact, saveat, adjoint):
        @jax.grad
        def _run_impl(twice_inexact):
            f = jax.vmap(_run_inexact, in_axes=(0, None, None))
            out = f(twice_inexact, saveat, adjoint)
            assert out.shape == (2,)
            return jnp.sum(out)

        return _run_impl(twice_inexact)

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

                fd_grads = _run_finite_diff(
                    y0__args__term, saveat, diffrax.RecursiveCheckpointAdjoint()
                )
                direct_grads = _run_grad(inexact, saveat, diffrax.DirectAdjoint())
                recursive_grads = _run_grad(
                    inexact, saveat, diffrax.RecursiveCheckpointAdjoint()
                )
                backsolve_grads = _run_grad(inexact, saveat, diffrax.BacksolveAdjoint())
                assert tree_allclose(fd_grads, direct_grads[0])
                assert tree_allclose(direct_grads, recursive_grads, atol=1e-5)
                assert tree_allclose(direct_grads, backsolve_grads, atol=1e-5)

                direct_grads = _run_grad_int(
                    y0__args__term, saveat, diffrax.DirectAdjoint()
                )
                recursive_grads = _run_grad_int(
                    y0__args__term, saveat, diffrax.RecursiveCheckpointAdjoint()
                )
                backsolve_grads = _run_grad_int(
                    y0__args__term, saveat, diffrax.BacksolveAdjoint()
                )
                direct_grads = jtu.tree_map(_convert_float0, direct_grads)
                recursive_grads = jtu.tree_map(_convert_float0, recursive_grads)
                backsolve_grads = jtu.tree_map(_convert_float0, backsolve_grads)
                assert tree_allclose(fd_grads, direct_grads[0])
                assert tree_allclose(direct_grads, recursive_grads, atol=1e-5)
                assert tree_allclose(direct_grads, backsolve_grads, atol=1e-5)

                fd_grads = jtu.tree_map(lambda *x: jnp.stack(x), fd_grads, fd_grads)
                direct_grads = _run_vmap_grad(
                    twice_inexact, saveat, diffrax.DirectAdjoint()
                )
                recursive_grads = _run_vmap_grad(
                    twice_inexact, saveat, diffrax.RecursiveCheckpointAdjoint()
                )
                backsolve_grads = _run_vmap_grad(
                    twice_inexact, saveat, diffrax.BacksolveAdjoint()
                )
                assert tree_allclose(fd_grads, direct_grads[0])
                assert tree_allclose(direct_grads, recursive_grads, atol=1e-5)
                assert tree_allclose(direct_grads, backsolve_grads, atol=1e-5)

                direct_grads = _run_grad_vmap(
                    twice_inexact, saveat, diffrax.DirectAdjoint()
                )
                recursive_grads = _run_grad_vmap(
                    twice_inexact, saveat, diffrax.RecursiveCheckpointAdjoint()
                )
                backsolve_grads = _run_grad_vmap(
                    twice_inexact, saveat, diffrax.BacksolveAdjoint()
                )
                assert tree_allclose(fd_grads, direct_grads[0])
                assert tree_allclose(direct_grads, recursive_grads, atol=1e-5)
                assert tree_allclose(direct_grads, backsolve_grads, atol=1e-5)


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
        return jnp.sum(cast(Array, sol.ys))

    jax.grad(solve)(2.0)


def test_closure_errors():
    mlp = eqx.nn.MLP(1, 1, 8, 2, key=jr.PRNGKey(0))

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
        return jnp.sum(cast(Array, sol.ys))

    with pytest.raises(jax.interpreters.ad.CustomVJPException):
        run(mlp)


def test_closure_fixed():
    mlp = eqx.nn.MLP(1, 1, 8, 2, key=jr.PRNGKey(0))

    class VectorField(eqx.Module):
        model: Callable

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
        return jnp.sum(cast(Array, sol.ys))

    run(mlp)


def test_implicit():
    class ExponentialDecayToSteadyState(eqx.Module):
        steady_state: Array
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
        (y1,) = cast(Array, sol.ys)
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
    assert tree_allclose(model.steady_state, target_steady_state, rtol=1e-2, atol=1e-2)


def test_backprop_ts(getkey):
    mlp = eqx.nn.MLP(1, 1, 8, 2, key=jr.PRNGKey(0))

    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def run(model):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, args: model(y)),
            diffrax.Euler(),
            0,
            1,
            0.1,
            jnp.array([1.0]),
            saveat=diffrax.SaveAt(ts=jnp.linspace(0, 1, 5)),
        )
        return jnp.sum(cast(Array, sol.ys))

    run(mlp)


def test_sde_against(getkey):
    def f(t, y, args):
        k0, _ = args
        return -k0 * y

    def g(t, y, args):
        _, k1 = args
        return k1 * y

    t0 = 0
    t1 = 1
    dt0 = 0.001
    tol = 1e-5
    shape = (2,)
    bm = diffrax.VirtualBrownianTree(t0, t1, tol, shape, key=getkey())
    drift = diffrax.ODETerm(f)
    diffusion = diffrax.WeaklyDiagonalControlTerm(g, bm)
    terms = diffrax.MultiTerm(drift, diffusion)
    solver = diffrax.Heun()

    @eqx.filter_jit
    @jax.grad
    def run(y0__args, adjoint):
        y0, args = y0__args
        sol = diffrax.diffeqsolve(terms, solver, t0, t1, dt0, y0, args, adjoint=adjoint)
        return jnp.sum(cast(Array, sol.ys))

    y0 = jnp.array([1.0, 2.0])
    args = (0.5, 0.1)
    grads1 = run((y0, args), diffrax.DirectAdjoint())
    grads2 = run((y0, args), diffrax.BacksolveAdjoint())
    grads3 = run((y0, args), diffrax.RecursiveCheckpointAdjoint())
    assert tree_allclose(grads1, grads2, rtol=1e-3, atol=1e-3)
    assert tree_allclose(grads1, grads3, rtol=1e-3, atol=1e-3)


def test_implicit_runge_kutta_direct_adjoint():
    diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: -y),
        diffrax.Kvaerno5(),
        0,
        1,
        0.01,
        1.0,
        adjoint=diffrax.DirectAdjoint(),
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
    )
