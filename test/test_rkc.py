"""Tests for the RKC (Runge--Kutta--Chebyshev) stabilised explicit solver."""

import diffrax
import jax
import jax.numpy as jnp
import pytest

from .helpers import tree_allclose


def test_rkc_num_stages_validation():
    with pytest.raises(ValueError, match="num_stages"):
        diffrax.RKC(num_stages=1)
    with pytest.raises(ValueError, match="num_stages"):
        diffrax.RKC(num_stages=0)
    # Should not raise.
    diffrax.RKC(num_stages=2)


def test_rkc_order():
    solver = diffrax.RKC(num_stages=4)
    term = diffrax.ODETerm(lambda t, y, args: -y)
    assert solver.order(term) == 1


def test_rkc_exponential_decay():
    """Solve dy/dt = -y, y(0) = 1.  Exact solution: exp(-1)."""
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: -y),
        diffrax.RKC(num_stages=3),
        t0=0.0,
        t1=1.0,
        dt0=0.01,
        y0=jnp.array(1.0),
    )
    assert tree_allclose(sol.ys[-1], jnp.exp(-1.0), atol=1e-2, rtol=1e-2)


def test_rkc_convergence_order():
    """Verify first-order convergence by halving the step size."""

    def f(t, y, args):
        return -y

    t0, t1 = 0.0, 1.0
    y0 = jnp.array(1.0)
    true_y1 = jnp.exp(-1.0)
    solver = diffrax.RKC(num_stages=4)

    errors = []
    for dt in [0.1, 0.05, 0.025]:
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(f), solver, t0, t1, dt, y0, max_steps=None
        )
        errors.append(float(jnp.abs(sol.ys[-1] - true_y1)))

    # Each halving should at least halve the error (order >= 1).
    # With multiple Chebyshev stages the effective order can exceed 1 on simple
    # problems, so we allow a wide upper bound.
    ratio_1 = errors[0] / errors[1]
    ratio_2 = errors[1] / errors[2]
    assert 1.5 < ratio_1 < 8.0
    assert 1.5 < ratio_2 < 8.0


def test_rkc_stability():
    """RKC with enough stages should remain stable where Euler would diverge.

    For dy/dt = -200*y with dt=0.05, Euler has |1 + lambda*dt| = |1 - 10| = 9 > 1
    so it diverges.  RKC(num_stages=8) has a stability limit ~0.65*64 ≈ 41.6,
    so |lambda|*dt = 10 is within its stability region.
    """
    term = diffrax.ODETerm(lambda t, y, args: -200.0 * y)
    y0 = jnp.array(1.0)

    # Euler diverges.
    euler_sol = diffrax.diffeqsolve(
        term, diffrax.Euler(), t0=0.0, t1=1.0, dt0=0.05, y0=y0
    )
    assert jnp.abs(euler_sol.ys[-1]) > 1e10

    # RKC stays bounded.
    rkc_sol = diffrax.diffeqsolve(
        term, diffrax.RKC(num_stages=8), t0=0.0, t1=1.0, dt0=0.05, y0=y0
    )
    assert jnp.abs(rkc_sol.ys[-1]) < 1.0


def test_rkc_system():
    """Solve a 2D linear system."""

    def vf(t, y, args):
        return jnp.array([-0.5 * y[0] + 0.1 * y[1], 0.1 * y[0] - 0.5 * y[1]])

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vf),
        diffrax.RKC(num_stages=4),
        t0=0.0,
        t1=2.0,
        dt0=0.02,
        y0=jnp.array([1.0, 0.5]),
    )
    # Just check it didn't diverge and is decaying.
    assert jnp.all(jnp.abs(sol.ys[-1]) < 1.0)


def test_rkc_pytree_state():
    """RKC should work with PyTree-valued states."""

    def vf(t, y, args):
        return {"a": -y["a"], "b": -2.0 * y["b"]}

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vf),
        diffrax.RKC(num_stages=3),
        t0=0.0,
        t1=1.0,
        dt0=0.01,
        y0={"a": jnp.array(1.0), "b": jnp.array(2.0)},
    )
    assert tree_allclose(sol.ys["a"][-1], jnp.exp(-1.0), atol=1e-2, rtol=1e-2)
    assert tree_allclose(
        sol.ys["b"][-1], 2.0 * jnp.exp(-2.0), atol=5e-2, rtol=5e-2
    )


def test_rkc_differentiable():
    """The solver should be differentiable through diffeqsolve."""

    def loss(y0_val):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, args: -y),
            diffrax.RKC(num_stages=3),
            t0=0.0,
            t1=1.0,
            dt0=0.05,
            y0=y0_val,
        )
        return jnp.sum(sol.ys[-1] ** 2)

    grad_val = jax.grad(loss)(jnp.array(1.0))
    # Gradient should be approximately 2 * exp(-1) * exp(-1) ≈ 0.2707
    assert tree_allclose(grad_val, 2.0 * jnp.exp(-2.0), atol=5e-2, rtol=5e-2)


def test_rkc_half_solver():
    """RKC wrapped in HalfSolver should work with adaptive stepping."""
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: -y),
        diffrax.HalfSolver(diffrax.RKC(num_stages=4)),
        t0=0.0,
        t1=1.0,
        dt0=0.1,
        y0=jnp.array(1.0),
        stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
    )
    assert tree_allclose(sol.ys[-1], jnp.exp(-1.0), atol=1e-3, rtol=1e-3)


def test_rkc_different_stage_counts():
    """Higher stage counts should give comparable accuracy but allow larger steps."""
    term = diffrax.ODETerm(lambda t, y, args: -10.0 * y)
    y0 = jnp.array(1.0)
    exact = jnp.exp(-10.0)

    for s in [2, 4, 6, 8]:
        sol = diffrax.diffeqsolve(
            term, diffrax.RKC(num_stages=s), t0=0.0, t1=1.0, dt0=0.005, y0=y0
        )
        assert tree_allclose(sol.ys[-1], exact, atol=0.1, rtol=0.1)
