import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
import scipy.stats

from .helpers import all_ode_solvers, all_split_solvers, implicit_tol, tree_allclose


def _test_path_derivative(path, name):
    for percentage in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        t = path.t0 + percentage * (path.t1 - path.t0)
        _, x = jax.jvp(path.evaluate, (t,), (jnp.ones_like(t),))
        y = path.derivative(t)
        assert tree_allclose(x, y, rtol=1e-3, atol=1e-4)


def _test_path_endpoints(path, name, y0, y1):
    assert tree_allclose(y0, path.evaluate(path.t0))
    assert tree_allclose(y1, path.evaluate(path.t1))


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_derivative(dtype, getkey):
    ts = jnp.linspace(0, 5, 8)
    ys = jr.normal(getkey(), (8, 4), dtype=dtype)

    paths = []

    # global interpolation

    linear_interp = diffrax.LinearInterpolation(ts=ts, ys=ys)
    paths.append((linear_interp, "linear", ys[0], ys[-1]))

    cubic_coeffs = diffrax.backward_hermite_coefficients(ts, ys)
    cubic_interp = diffrax.CubicInterpolation(ts=ts, coeffs=cubic_coeffs)
    paths.append((cubic_interp, "cubic", ys[0], ys[-1]))

    y0 = jr.normal(getkey(), (3,), dtype=dtype)
    dense_interp = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, p: -y),
        diffrax.Euler(),
        0,
        1,
        0.01,
        y0,
        saveat=diffrax.SaveAt(dense=True, t1=True),
    )
    y1 = dense_interp.ys[-1]  # pyright: ignore
    paths.append((dense_interp, "dense", y0, y1))

    # local interpolation

    local_linear_interp = diffrax.LocalLinearInterpolation(
        t0=ts[0], t1=ts[-1], y0=ys[0], y1=ys[-1]
    )
    paths.append((local_linear_interp, "local linear", ys[0], ys[-1]))

    for solver in all_ode_solvers:
        solver = implicit_tol(solver)
        y0 = jr.normal(getkey(), (3,), dtype=dtype)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, p: -y),
            solver,
            0,
            1,
            0.01,
            y0,
            saveat=diffrax.SaveAt(dense=True, t1=True),
        )
        y1 = solution.ys[-1]  # pyright: ignore
        paths.append((solution, type(solver).__name__, y0, y1))

    for solver in all_split_solvers:
        solver = implicit_tol(solver)
        y0 = jr.normal(getkey(), (3,), dtype=dtype)

        solution = diffrax.diffeqsolve(
            diffrax.MultiTerm(
                diffrax.ODETerm(lambda t, y, p: -0.7 * y),
                diffrax.ODETerm(lambda t, y, p: -0.3 * y),
            ),
            solver,
            0,
            1,
            0.01,
            y0,
            saveat=diffrax.SaveAt(dense=True, t1=True),
        )
        y1 = solution.ys[-1]  # pyright: ignore
        paths.append((solution, type(solver).__name__, y0, y1))

    # actually do tests

    for path, name, y0, y1 in paths:
        _test_path_derivative(path, name)
        _test_path_endpoints(path, name, y0, y1)


def test_dopri8_interpolation_order():
    # Dopri8 is 8th order at the end of each step, but its dense output is only 5th
    # order, so its O(dt0^6) local error dominates in between steps.
    term = diffrax.ODETerm(lambda t, y, args: y * jnp.cos(t))
    t0 = 0
    t1 = 4
    y0 = jnp.array(1.0)
    exponents = [0, -1, -2, -3]
    step_errors = []
    mid_errors = []
    for exponent in exponents:
        dt0 = 2**exponent
        saveat = diffrax.SaveAt(dense=True)
        sol = diffrax.diffeqsolve(
            term, diffrax.Dopri8(), t0, t1, dt0, y0, saveat=saveat
        )
        step_ts = jnp.arange(t0 + dt0, t1 + dt0 / 2, dt0)
        mid_ts = step_ts - dt0 / 2
        for ts, errors in ((step_ts, step_errors), (mid_ts, mid_errors)):
            error = jnp.max(jnp.abs(jax.vmap(sol.evaluate)(ts) - jnp.exp(jnp.sin(ts))))
            errors.append(jnp.log2(error))
    step_order = scipy.stats.linregress(exponents, step_errors).slope  # pyright: ignore
    mid_order = scipy.stats.linregress(exponents, mid_errors).slope  # pyright: ignore
    assert -0.5 < step_order - 8 < 0.5
    assert -0.5 < mid_order - 6 < 0.5
