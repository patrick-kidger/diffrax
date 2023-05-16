import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest


def test_half_solver():
    term = diffrax.ODETerm(lambda t, y, args: -y)
    t0 = 0
    t1 = 1
    y0 = 1.0
    dt0 = None
    solver = diffrax.HalfSolver(diffrax.Euler())
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
    diffrax.diffeqsolve(
        term, solver, t0, t1, dt0, y0, stepsize_controller=stepsize_controller
    )


def test_instance_check():
    assert isinstance(diffrax.HalfSolver(diffrax.Euler()), diffrax.Euler)
    assert not isinstance(diffrax.HalfSolver(diffrax.Euler()), diffrax.Heun)


def test_implicit_euler_adaptive():
    term = diffrax.ODETerm(lambda t, y, args: -10 * y**3)
    solver1 = diffrax.ImplicitEuler(
        nonlinear_solver=diffrax.NewtonNonlinearSolver(rtol=1e-5, atol=1e-5)
    )
    solver2 = diffrax.ImplicitEuler()
    t0 = 0
    t1 = 1
    dt0 = 1
    y0 = 1.0
    stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)
    out1 = diffrax.diffeqsolve(term, solver1, t0, t1, dt0, y0, throw=False)
    out2 = diffrax.diffeqsolve(
        term,
        solver2,
        t0,
        t1,
        dt0,
        y0,
        stepsize_controller=stepsize_controller,
        throw=False,
    )
    assert out1.result == diffrax.RESULTS.implicit_nonconvergence
    assert out2.result == diffrax.RESULTS.successful


def test_multiple_tableau1():
    class DoubleDopri5(diffrax.AbstractRungeKutta):
        tableau = diffrax.MultiButcherTableau(
            diffrax.Dopri5.tableau, diffrax.Dopri5.tableau
        )
        calculate_jacobian = diffrax.CalculateJacobian.never

        def interpolation_cls(self, *, k, **kwargs):
            return diffrax.LocalLinearInterpolation(**kwargs)

    mlp1 = eqx.nn.MLP(2, 2, 32, 1, key=jr.PRNGKey(0))
    mlp2 = eqx.nn.MLP(2, 2, 32, 1, key=jr.PRNGKey(1))

    term1 = diffrax.ODETerm(lambda t, y, args: mlp1(y))
    term2 = diffrax.ODETerm(lambda t, y, args: mlp2(y))
    t0 = 0
    t1 = 1
    dt0 = 0.1
    y0 = jnp.array([1.0, 2.0])
    out_a = diffrax.diffeqsolve(
        diffrax.MultiTerm(term1, term2),
        diffrax.Dopri5(),
        t0,
        t1,
        dt0,
        y0,
    )
    out_b = diffrax.diffeqsolve(
        diffrax.MultiTerm(term1, term2),
        DoubleDopri5(),
        t0,
        t1,
        dt0,
        y0,
    )
    assert jnp.allclose(out_a.ys, out_b.ys, rtol=1e-8, atol=1e-8)

    with pytest.raises(ValueError):
        diffrax.diffeqsolve(
            (term1, term2),
            DoubleDopri5(),
            t0,
            t1,
            dt0,
            y0,
        )


def test_multiple_tableau2():
    # Different number of stages
    with pytest.raises(ValueError):

        class X(diffrax.AbstractRungeKutta):
            tableau = diffrax.MultiButcherTableau(
                diffrax.Dopri5.tableau, diffrax.Bosh3.tableau
            )
            calculate_jacobian = diffrax.CalculateJacobian.never

            def interpolation_cls(self, *, k, **kwargs):
                return diffrax.LocalLinearInterpolation(**kwargs)

    # Multiple implicit
    with pytest.raises(ValueError):

        class Y(diffrax.AbstractRungeKutta):
            tableau = diffrax.MultiButcherTableau(
                diffrax.Kvaerno3.tableau, diffrax.Kvaerno3.tableau
            )
            calculate_jacobian = diffrax.CalculateJacobian.never

            def interpolation_cls(self, *, k, **kwargs):
                return diffrax.LocalLinearInterpolation(**kwargs)

    class Z(diffrax.AbstractRungeKutta):
        tableau = diffrax.MultiButcherTableau(
            diffrax.Bosh3.tableau, diffrax.Kvaerno3.tableau
        )
        calculate_jacobian = diffrax.CalculateJacobian.never

        def interpolation_cls(self, *, k, **kwargs):
            return diffrax.LocalLinearInterpolation(**kwargs)
