from typing import cast

import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import Array

from .helpers import tree_allclose


class VectorField(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, in_size, out_size, width_size, depth, key):
        self.mlp = eqx.nn.MLP(in_size, out_size, width_size, depth, key=key)

    def __call__(self, t, y, args):
        return args * self.mlp(y)


@eqx.filter_value_and_grad
def _loss(y0__args__term, solver, saveat, adjoint, stepsize_controller, dual_y0):
    y0, args, term = y0__args__term

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=5,
        dt0=0.01,
        y0=y0,
        args=args,
        saveat=saveat,
        max_steps=4096,
        adjoint=adjoint,
        stepsize_controller=stepsize_controller,
    )
    if dual_y0:
        y1 = sol.ys[0]  # pyright: ignore
    else:
        y1 = sol.ys
    return jnp.sum(cast(Array, y1))


def _compare_grads(y0__args__term, solver, saveat, stepsize_controller, dual_y0):
    loss, grads_base = _loss(
        y0__args__term,
        solver,
        saveat,
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        stepsize_controller=stepsize_controller,
        dual_y0=dual_y0,
    )
    loss, grads_reversible = _loss(
        y0__args__term,
        solver,
        saveat,
        adjoint=diffrax.ReversibleAdjoint(),
        stepsize_controller=stepsize_controller,
        dual_y0=dual_y0,
    )
    assert tree_allclose(grads_base, grads_reversible, atol=1e-5)


@pytest.mark.parametrize(
    "saveat",
    [
        diffrax.SaveAt(t0=True, t1=True),
        diffrax.SaveAt(t0=True, ts=jnp.linspace(0, 5, 10), t1=True),
    ],
)
def test_semi_implicit_euler(saveat):
    n = 10
    y0 = jnp.linspace(1, 10, num=n)
    key = jr.PRNGKey(10)
    fkey, gkey = jr.split(key, 2)
    f = VectorField(n, n, n, depth=4, key=fkey)
    g = VectorField(n, n, n, depth=4, key=gkey)
    terms = (diffrax.ODETerm(f), diffrax.ODETerm(g))
    y0 = (y0, y0)
    args = jnp.linspace(0, 1, n)
    solver = diffrax.SemiImplicitEuler()
    stepsize_controller = diffrax.ConstantStepSize()

    _compare_grads((y0, args, terms), solver, saveat, stepsize_controller, dual_y0=True)


@pytest.mark.parametrize(
    "stepsize_controller",
    [diffrax.ConstantStepSize(), diffrax.PIDController(rtol=1e-8, atol=1e-8)],
)
@pytest.mark.parametrize(
    "saveat",
    [
        diffrax.SaveAt(t0=True, t1=True),
        diffrax.SaveAt(t0=True, ts=jnp.linspace(0, 5, 10), t1=True),
    ],
)
def test_reversible_heun_ode(stepsize_controller, saveat):
    n = 10
    y0 = jnp.linspace(1, 10, num=n)
    key = jr.PRNGKey(10)
    f = VectorField(n, n, n, depth=4, key=key)
    terms = diffrax.ODETerm(f)
    y0 = y0
    args = jnp.linspace(0, 1, n)
    solver = diffrax.ReversibleHeun()

    _compare_grads(
        (y0, args, terms), solver, saveat, stepsize_controller, dual_y0=False
    )


@pytest.mark.parametrize(
    "saveat",
    [
        diffrax.SaveAt(t0=True, t1=True),
        diffrax.SaveAt(t0=True, ts=jnp.linspace(0, 5, 10), t1=True),
    ],
)
def test_reversible_heun_sde(saveat):
    n = 10
    y0 = jnp.linspace(1, 10, num=n)
    key = jr.PRNGKey(10)
    fkey, Wkey = jr.split(key, 2)
    f = VectorField(n, n, n, depth=4, key=fkey)
    g = lambda t, y, args: jnp.ones((n,))
    W = diffrax.VirtualBrownianTree(t0=0, t1=5, tol=1e-3, shape=(n,), key=Wkey)
    terms = diffrax.MultiTerm(diffrax.ODETerm(f), diffrax.ControlTerm(g, W))
    y0 = y0
    args = jnp.linspace(0, 1, n)
    solver = diffrax.ReversibleHeun()
    stepsize_controller = diffrax.ConstantStepSize()

    _compare_grads(
        (y0, args, terms), solver, saveat, stepsize_controller, dual_y0=False
    )


@pytest.mark.parametrize(
    "saveat",
    [
        diffrax.SaveAt(t0=True, t1=True),
        diffrax.SaveAt(t0=True, ts=jnp.linspace(0, 5, 10), t1=True),
    ],
)
def test_leapfrog_midpoint(saveat):
    n = 10
    y0 = jnp.linspace(1, 10, num=n)
    key = jr.PRNGKey(10)
    f = VectorField(n, n, n, depth=4, key=key)
    terms = diffrax.ODETerm(f)
    y0 = y0
    args = jnp.linspace(0, 1, n)
    solver = diffrax.LeapfrogMidpoint()
    stepsize_controller = diffrax.ConstantStepSize()

    _compare_grads(
        (y0, args, terms), solver, saveat, stepsize_controller, dual_y0=False
    )
