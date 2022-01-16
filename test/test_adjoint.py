import math
import types

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from helpers import shaped_allclose


def test_ode_adjoint_term(getkey):
    vector_field = lambda t, y, args: -y
    term = diffrax.ODETerm(vector_field)
    adjoint_term = diffrax.term.AdjointTerm(term)
    t, y, a_y, dt = jrandom.normal(getkey(), (4,))
    aug = (y, a_y, None, diffrax.ODETerm(None))
    args = None
    vf_prod1 = adjoint_term.vf_prod(t, aug, args, dt)
    vf = adjoint_term.vf(t, aug, args)
    vf_prod2 = adjoint_term.prod(vf, dt)
    assert shaped_allclose(vf_prod1, vf_prod2)


def test_cde_adjoint_term(getkey):
    class VF(eqx.Module):
        mlp: eqx.nn.MLP

        def __call__(self, t, y, args):
            in_ = jnp.concatenate([t[None], y, *args])
            out = self.mlp(in_)
            return out.reshape(2, 3)

    mlp = eqx.nn.MLP(in_size=5, out_size=6, width_size=3, depth=1, key=getkey())
    vector_field = VF(mlp)
    control = types.SimpleNamespace(
        evaluate=lambda t0, t1: jrandom.normal(getkey(), (3,))
    )
    term = diffrax.ControlTerm(vector_field, control)
    adjoint_term = diffrax.term.AdjointTerm(term)
    t = jrandom.normal(getkey(), ())
    y = jrandom.normal(getkey(), (2,))
    args = (jrandom.normal(getkey(), (1,)), jrandom.normal(getkey(), (1,)))
    a_y = jrandom.normal(getkey(), (2,))
    a_args = (jrandom.normal(getkey(), (1,)), jrandom.normal(getkey(), (1,)))
    randlike = lambda a: jrandom.normal(getkey(), a.shape)
    a_term = jax.tree_map(randlike, eqx.filter(term, eqx.is_array))
    aug = (y, a_y, a_args, a_term)
    dt = adjoint_term.contr(t, t + 1)

    vf_prod1 = adjoint_term.vf_prod(t, aug, args, dt)
    vf = adjoint_term.vf(t, aug, args)
    vf_prod2 = adjoint_term.prod(vf, dt)
    assert shaped_allclose(vf_prod1, vf_prod2)


def test_no_adjoint():
    def fn(y0):
        term = diffrax.ODETerm(lambda t, y, args: -y)
        t0 = 0
        t1 = 1
        dt0 = 0.1
        solver = diffrax.Dopri5()
        adjoint = diffrax.NoAdjoint()
        sol = diffrax.diffeqsolve(term, t0, t1, y0, dt0, solver, adjoint=adjoint)
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
    stepsize_controller = diffrax.IController(rtol=1e-8, atol=1e-8)

    def _run(y0__args__term, saveat, adjoint):
        y0, args, term = y0__args__term
        return jnp.sum(
            diffrax.diffeqsolve(
                term,
                0.3,
                9.5,
                y0,
                None,
                solver,
                args=args,
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
        if x.dtype is jax.dtypes.float0:
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
