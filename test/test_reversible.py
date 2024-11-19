from typing import cast

import diffrax
import equinox as eqx
import jax.numpy as jnp
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


@eqx.filter_value_and_grad
def _loss(y0__args__term, solver, adjoint):
    y0, args, term = y0__args__term

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=5,
        dt0=0.01,
        y0=y0,
        args=args,
        adjoint=adjoint,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
    )
    y1 = sol.ys
    return jnp.sum(cast(Array, y1))


def test_constant_stepsizes():
    y0 = jnp.array([0.9, 5.4])
    args = (0.1, -1)
    term = diffrax.ODETerm(_VectorField(nondiff_arg=1, diff_arg=-0.1))

    base_solver = diffrax.Tsit5()
    reversible_solver = diffrax.Reversible(base_solver, l=0.999)

    # Base
    sol = diffrax.diffeqsolve(
        term,
        base_solver,
        t0=0,
        t1=5,
        dt0=0.01,
        y0=y0,
        args=args,
    )
    y1_base = sol.ys

    # Reversible
    sol = diffrax.diffeqsolve(
        term,
        reversible_solver,
        t0=0,
        t1=5,
        dt0=0.01,
        y0=y0,
        args=args,
    )
    y1_rev = sol.ys

    assert tree_allclose(y1_base, y1_rev, atol=1e-5)


def test_adaptive_stepsizes():
    y0 = jnp.array([0.9, 5.4])
    args = (0.1, -1)
    term = diffrax.ODETerm(_VectorField(nondiff_arg=1, diff_arg=-0.1))

    base_solver = diffrax.Tsit5()
    reversible_solver = diffrax.Reversible(base_solver, l=0.999)
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)

    # Base
    sol = diffrax.diffeqsolve(
        term,
        base_solver,
        t0=0,
        t1=5,
        dt0=0.01,
        y0=y0,
        args=args,
        stepsize_controller=stepsize_controller,
    )
    y1_base = sol.ys

    # Reversible
    sol = diffrax.diffeqsolve(
        term,
        reversible_solver,
        t0=0,
        t1=5,
        dt0=0.01,
        y0=y0,
        args=args,
        stepsize_controller=stepsize_controller,
    )
    y1_rev = sol.ys

    assert tree_allclose(y1_base, y1_rev, atol=1e-5)


def test_reversible_adjoint():
    y0 = jnp.array([0.9, 5.4])
    args = (0.1, -1)
    term = diffrax.ODETerm(_VectorField(nondiff_arg=1, diff_arg=-0.1))
    y0__args__term = (y0, args, term)
    del y0, args, term

    base_solver = diffrax.Tsit5()
    reversible_solver = diffrax.Reversible(base_solver, l=0.999)

    loss, grads_base = _loss(
        y0__args__term, base_solver, adjoint=diffrax.RecursiveCheckpointAdjoint()
    )
    loss, grads_reversible = _loss(
        y0__args__term, reversible_solver, adjoint=diffrax.ReversibleAdjoint()
    )

    assert tree_allclose(grads_base, grads_reversible, atol=1e-5)
