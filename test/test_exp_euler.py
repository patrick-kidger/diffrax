import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from diffrax import (
    ControlTerm,
    Dopri5,
    EulerHeun,
    ExponentialEuler,
    HalfSolver,
    MultiTerm,
    ODETerm,
    PIDController,
    SaveAt,
    VirtualBrownianTree,
    diffeqsolve,
    LinearODETerm
  
)



def test_linear():
    """Linear equation should be exact, even at large timesteps."""
    linear_term = lx.DiagonalLinearOperator(-jnp.ones((1,)))  # noqa: E731
    non_linear_term = lambda t, y, args: jnp.zeros_like(y)  # noqa: E731
    term = MultiTerm(LinearODETerm(linear_term), ODETerm(non_linear_term))

    saveat = SaveAt(ts=jnp.linspace(0, 3, 2))
    sol = diffeqsolve(
        term,
        ExponentialEuler(),
        t0=0,
        t1=3,
        dt0=1.0,
        y0=jnp.ones((1,)),
        saveat=saveat,
    )
    assert jnp.allclose(sol.ys, jnp.array([[jnp.exp(0.0)], [jnp.exp(-3)]]))


def test_non_linear():
    """Non linear, comparison to Dopri5"""
    A = -jnp.abs(jr.normal(jr.key(0), (10,)))
    y0 = jr.normal(jr.key(42), (10,))

    linear_term = lx.DiagonalLinearOperator(A)  # noqa: E731
    non_linear_term = lambda t, y, args: 2 * jnp.cos(y) ** 3  # noqa: E731
    term = MultiTerm(LinearODETerm(linear_term), ODETerm(non_linear_term))

    saveat = SaveAt(ts=jnp.linspace(0, 3, 100))

    # Exponential solver
    sol_exp = diffeqsolve(
        term,
        ExponentialEuler(),
        t0=0,
        t1=3,
        dt0=1e-3,
        y0=y0,
        saveat=saveat,
    )

    # Baseline solver
    solver = Dopri5()
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
    sol_baseline = diffeqsolve(
        term,
        solver,
        t0=0,
        t1=3,
        dt0=1e-3,
        y0=y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )
    assert jnp.allclose(sol_exp.ys, sol_baseline.ys, rtol=2e-3, atol=2e-3)


def test_error():
    """Testing if we can use the HalfSolver to get adaptive steps."""
    A = -jnp.abs(jr.normal(jr.key(0), (10,)))
    y0 = jr.normal(jr.key(42), (10,))

    linear_term = lx.DiagonalLinearOperator(A)  # noqa: E731
    non_linear_term = lambda t, y, args: 2 * jnp.cos(y) ** 3  # noqa: E731
    term = MultiTerm(LinearODETerm(linear_term), ODETerm(non_linear_term))
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
    saveat = SaveAt(ts=jnp.linspace(0, 3, 100))

    # Exponential solver
    sol = diffeqsolve(
        term,
        ExponentialEuler(),
        t0=0,
        t1=3,
        dt0=1e-4,
        y0=y0,
        saveat=saveat,
        max_steps=100000,
    )

    # Exponential solver
    sol_adapt = diffeqsolve(
        term,
        HalfSolver(ExponentialEuler()),
        t0=0,
        t1=3,
        dt0=1.0,  # larger stepsize, should be adjusted
        y0=y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )
    assert jnp.allclose(sol.ys, sol_adapt.ys, rtol=2e-3, atol=3e-3)


def test_sde():
    "Test using an SDE."
    A = -jnp.abs(jr.normal(jr.key(0), (10,)))
    y0 = jr.normal(jr.key(42), (10,))

    linear_term = lx.DiagonalLinearOperator(A)  # noqa: E731
    non_linear_term = lambda t, y, args: 2 * jnp.cos(y) ** 3  # noqa: E731
    diffusion_term = lambda t, y, args: lx.DiagonalLinearOperator(jnp.full((10,), 0.1))  # noqa: E731
    brownian_motion = VirtualBrownianTree(
        0, 3, tol=1e-3, shape=(10,), key=jr.PRNGKey(0)
    )
    term = MultiTerm(
        LinearODETerm(linear_term),
        ODETerm(non_linear_term),
        ControlTerm(diffusion_term, brownian_motion),
    )
    saveat = SaveAt(ts=jnp.linspace(0, 3, 100))

    # Exponential solver
    sol_exp = diffeqsolve(
        term,
        ExponentialEuler(),
        t0=0,
        t1=3,
        dt0=1e-3,
        y0=y0,
        saveat=saveat,
    )

    # Shark solver
    linear_term_exp = lambda t, y, args: A * y + 2 * jnp.cos(y) ** 3  # noqa: E731
    term_exp = MultiTerm(
        ODETerm(linear_term_exp),
        ControlTerm(diffusion_term, brownian_motion),
    )
    sol_euler = diffeqsolve(
        term_exp,
        EulerHeun(),
        t0=0,
        t1=3,
        dt0=1e-3,
        y0=y0,
        saveat=saveat,
    )
    assert jnp.allclose(sol_exp.ys, sol_euler.ys, rtol=1e-3, atol=1e-3)


def test_diagonal_matrix_exponential():
    """Compare result of diagonal specialisation to full matrix exponential."""
    A = -jnp.abs(jr.normal(jr.key(0), (10,)))
    y0 = jr.normal(jr.key(42), (10,))

    linear_term_diag = lx.DiagonalLinearOperator(A)
    linear_term_full = lx.MatrixLinearOperator(jnp.diag(A))
    non_linear_term = lambda t, y, args: 2 * jnp.cos(y) ** 3  # noqa: E731
    term_diag = MultiTerm(LinearODETerm(linear_term_diag), ODETerm(non_linear_term))
    term_full = MultiTerm(LinearODETerm(linear_term_full), ODETerm(non_linear_term))
    saveat = SaveAt(ts=jnp.linspace(0, 3, 100))

    # Diagonal approach
    sol_diag = diffeqsolve(
        term_diag,
        ExponentialEuler(),
        t0=0,
        t1=3,
        dt0=1e-3,
        y0=y0,
        saveat=saveat,
    )
    # Full matrix exponential
    sol_full = diffeqsolve(
        term_full,
        ExponentialEuler(),
        t0=0,
        t1=3,
        dt0=1e-3,
        y0=y0,
        saveat=saveat,
    )
    assert jnp.allclose(sol_diag.ys, sol_full.ys, rtol=1e-4, atol=1e-4)


def test_matrix_exponential():
    """Test if normal solver accept our term structure, and make sure results are the same."""
    A = -jnp.abs(jr.normal(jr.key(0), (10, 10)))
    y0 = jr.normal(jr.key(42), (10,))

    linear_term = lx.MatrixLinearOperator(A)
    non_linear_term = lambda t, y, args: 2 * jnp.cos(y) ** 3  # noqa: E731
    term = MultiTerm(LinearODETerm(linear_term), ODETerm(non_linear_term))
    saveat = SaveAt(ts=jnp.linspace(0, 3, 100))

    # Exponential solver
    sol_exp = diffeqsolve(
        term,
        ExponentialEuler(),
        t0=0,
        t1=3,
        dt0=1e-3,
        y0=y0,
        saveat=saveat,
    )

    solver = Dopri5()
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
    sol_dopri = diffeqsolve(
        term,
        solver,
        t0=0,
        t1=3,
        dt0=1e-3,
        y0=y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )

    jnp.allclose(sol_exp.ys, sol_dopri.ys, rtol=2e-3, atol=2e-3)
