from typing import cast

import diffrax
import equinox as eqx
import jax
import jax._src.interpreters.ad
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest
from jaxtyping import Array

from .helpers import tree_allclose


@pytest.mark.slow
def test_direct_brownian(getkey):
    key = getkey()
    key, subkey = jax.random.split(key)
    driftkey, diffusionkey, ykey = jr.split(subkey, 3)
    drift_mlp = eqx.nn.MLP(
        in_size=2,
        out_size=2,
        width_size=8,
        depth=2,
        activation=jax.nn.swish,
        final_activation=jnp.tanh,
        key=driftkey,
    )
    diffusion_mlp = eqx.nn.MLP(
        in_size=2,
        out_size=2,
        width_size=8,
        depth=2,
        activation=jax.nn.swish,
        final_activation=jnp.tanh,
        key=diffusionkey,
    )

    class Field(eqx.Module):
        force: eqx.nn.MLP

        def __call__(self, t, y, args):
            return self.force(y)

    class DiffusionField(eqx.Module):
        force: eqx.nn.MLP

        def __call__(self, t, y, args):
            return lx.DiagonalLinearOperator(self.force(y))

    y0 = jr.normal(ykey, (2,))

    k1, k2, k3 = jax.random.split(key, 3)

    vbt = diffrax.VirtualBrownianTree(
        0.3, 9.5, 1e-4, (2,), k1, levy_area=diffrax.SpaceTimeLevyArea
    )
    dbp = diffrax.UnsafeBrownianPath((2,), k2, levy_area=diffrax.SpaceTimeLevyArea)
    dbp_pre = diffrax.UnsafeBrownianPath(
        (2,), k3, levy_area=diffrax.SpaceTimeLevyArea, precompute=int(9.5 / 0.1)
    )

    vbt_terms = diffrax.MultiTerm(
        diffrax.ODETerm(Field(drift_mlp)),
        diffrax.ControlTerm(DiffusionField(diffusion_mlp), vbt),
    )
    dbp_terms = diffrax.MultiTerm(
        diffrax.ODETerm(Field(drift_mlp)),
        diffrax.ControlTerm(DiffusionField(diffusion_mlp), dbp),
    )
    dbp_pre_terms = diffrax.MultiTerm(
        diffrax.ODETerm(Field(drift_mlp)),
        diffrax.ControlTerm(DiffusionField(diffusion_mlp), dbp_pre),
    )

    solver = diffrax.Heun()

    y0_args_term0 = (y0, None, vbt_terms)
    y0_args_term1 = (y0, None, dbp_terms)
    y0_args_term2 = (y0, None, dbp_pre_terms)

    def _run(y0__args__term, saveat, adjoint):
        y0, args, term = y0__args__term
        ys = diffrax.diffeqsolve(
            term,
            solver,
            0.3,
            9.5,
            0.1,
            y0,
            args,
            saveat=saveat,
            adjoint=adjoint,
            max_steps=250,  # see note below
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

    for t0 in (True, False):
        for t1 in (True, False):
            for ts in (None, [0.3], [2.0], [9.5], [1.0, 7.0], [0.3, 7.0, 9.5]):
                for i, y0__args__term in enumerate(
                    (y0_args_term0, y0_args_term1, y0_args_term2)
                ):
                    if t0 is False and t1 is False and ts is None:
                        continue

                    saveat = diffrax.SaveAt(t0=t0, t1=t1, ts=ts)

                    inexact, static = eqx.partition(
                        y0__args__term, eqx.is_inexact_array
                    )

                    def _run_inexact(inexact, saveat, adjoint):
                        return _run(eqx.combine(inexact, static), saveat, adjoint)

                    _run_grad = eqx.filter_jit(jax.grad(_run_inexact))
                    _run_fwd_grad = eqx.filter_jit(jax.jacfwd(_run_inexact))

                    fd_grads = _run_finite_diff(
                        y0__args__term, saveat, diffrax.RecursiveCheckpointAdjoint()
                    )
                    recursive_grads = _run_grad(
                        inexact, saveat, diffrax.RecursiveCheckpointAdjoint()
                    )
                    if i == 0:
                        backsolve_grads = _run_grad(
                            inexact, saveat, diffrax.BacksolveAdjoint()
                        )
                        assert tree_allclose(fd_grads, backsolve_grads[0], atol=1e-3)

                    forward_grads = _run_fwd_grad(
                        inexact, saveat, diffrax.ForwardMode()
                    )
                    # TODO: fix via https://github.com/patrick-kidger/equinox/issues/923
                    # turns out this actually only fails for steps >256. Which is weird,
                    # because thats means 3 vs 2 calls in the base 16. But idk why that
                    # matter and yields some opaque assertion error. Maybe something to
                    # do with shapes? AssertionError
                    #    ...
                    #    assert all(all(map(core.typematch,
                    # j.out_avals, branches_known[0].out_avals))
                    #    for j in branches_known[1:])
                    direct_grads = _run_grad(inexact, saveat, diffrax.DirectAdjoint())
                    assert tree_allclose(fd_grads, direct_grads[0], atol=1e-3)
                    assert tree_allclose(fd_grads, recursive_grads[0], atol=1e-3)
                    assert tree_allclose(fd_grads, forward_grads[0], atol=1e-3)
