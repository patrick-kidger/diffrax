# These are a collection of non-stiff reference problems.
#
# Problems are taken from
# Hull, Enright, Fellen and Sedgwick, 1972
# "Comparing numerical methods for ordinary differential equations"
# SIAM Journal of Numerical Analysis 9(4) pp. 603--637
#
# See also
# https://github.com/rtqichen/torchdiffeq/tree/master/tests/DETEST
# for a PyTorch implementation.

import math

import diffrax
import jax.numpy as jnp
import pytest
import scipy.integrate as integrate

from helpers import all_ode_solvers


#
# Class A: Single equations
#


def _a1():
    diffeq = lambda t, y, args: -y
    init = 1
    return diffeq, init


def _a2():
    diffeq = lambda t, y, args: -0.5 * y ** 3
    init = 1
    return diffeq, init


def _a3():
    diffeq = lambda t, y, args: y * jnp.cos(t)
    init = 1
    return diffeq, init


def _a4():
    diffeq = lambda t, y, args: 0.25 * y * (1 - 0.05 * y)
    init = 1
    return diffeq, init


def _a5():
    diffeq = lambda t, y, args: (y - t) / (y + t)
    init = 4
    return diffeq, init


#
# class B: Small systems
#


def _b1():
    def diffeq(t, y, args):
        y1, y2 = y
        dy1 = 2 * (y1 - y1 * y2)
        dy2 = -y2 + y1 * y2
        return dy1, dy2

    init = (1, 3)
    return diffeq, init


def _b2():
    def diffeq(t, y, args):
        y1, y2, y3 = y
        dy1 = -y1 + y2
        dy2 = y1 - 2 * y2 + y3
        dy3 = y2 - y3
        return dy1, dy2, dy3

    init = (2, 0, 1)
    return diffeq, init


def _b3():
    def diffeq(t, y, args):
        y1, y2, y3 = y
        dy1 = -y1
        dy2 = y1 - y2 ** 2
        dy3 = y2 ** 2
        return dy1, dy2, dy3

    init = (1, 0, 0)
    return diffeq, init


def _b4():
    def diffeq(t, y, args):
        y1, y2, y3 = y
        r = jnp.sqrt(y1 ** 2 + y2 ** 2)
        dy1 = -y2 - y1 * y3 / r
        dy2 = y1 - y2 * y3 / r
        dy3 = y1 / r
        return dy1, dy2, dy3

    init = (3, 0, 0)
    return diffeq, init


def _b5():
    def diffeq(t, y, args):
        y1, y2, y3 = y
        dy1 = y2 * y3
        dy2 = -y1 * y3
        dy3 = -0.51 * y1 * y2
        return dy1, dy2, dy3

    init = (0, 1, 1)
    return diffeq, init


#
# Class C: Moderate systems
#


def _c1():
    A = (
        jnp.zeros((10, 10))
        .at[jnp.arange(9), jnp.arange(9)]
        .set(-1)
        .at[jnp.arange(1, 10), jnp.arange(9)]
        .set(1)
    )

    def diffeq(t, y, args):
        return A @ y

    init = jnp.zeros(10).at[0].set(1)
    return diffeq, init


def _c2():
    A = (
        jnp.zeros((10, 10))
        .at[jnp.arange(9), jnp.arange(9)]
        .set(-jnp.arange(1, 10))
        .at[jnp.arange(1, 10), jnp.arange(9)]
        .set(jnp.arange(1, 10))
    )

    def diffeq(t, y, args):
        return A @ y

    init = jnp.zeros(10).at[0].set(1)
    return diffeq, init


def _c3():
    A = (
        jnp.zeros((10, 10))
        .at[jnp.arange(9), jnp.arange(9)]
        .set(-2)
        .at[jnp.arange(1, 10), jnp.arange(9)]
        .set(1)
        .at[jnp.arange(9), jnp.arange(1, 10)]
        .set(1)
    )

    def diffeq(t, y, args):
        return A @ y

    init = jnp.zeros(10).at[0].set(1)
    return diffeq, init


def _c4():
    A = (
        jnp.zeros((51, 51))
        .at[jnp.arange(50), jnp.arange(50)]
        .set(-2)
        .at[jnp.arange(1, 51), jnp.arange(50)]
        .set(1)
        .at[jnp.arange(50), jnp.arange(1, 51)]
        .set(1)
    )

    def diffeq(t, y, args):
        return A @ y

    init = jnp.zeros(51).at[0].set(1)
    return diffeq, init


def _c5():
    k2 = 2.95912208286
    m0 = 1.00000597682
    m = jnp.array(
        [
            0.000954786104043,
            0.000285583733151,
            0.0000437273164546,
            0.0000517759138449,
            0.00000277777777778,
        ]
    )

    def diffeq(t, y, args):
        v, dv = y
        assert v.shape == (3, 5)
        r_cubed = jnp.sum(v ** 2, axis=0) ** 1.5
        d_cubed = jnp.sum((v[:, :, None] - v[:, None, :]) ** 2, axis=0) ** 1.5

        term1 = -v @ ((m0 + m) / r_cubed)
        term2 = (v[:, None, :] - v[:, :, None]) / d_cubed
        term3 = v @ (1 / r_cubed)
        term4 = m * (term2 - term3[:, None, None])
        term5 = jnp.sum(term4, axis=2) - jnp.diagonal(term4, axis1=1, axis2=2)

        ddv = k2 * (term1 + term5)
        return dv, ddv

    v0 = jnp.array(
        [
            [
                3.42947415189,
                6.64145542550,
                11.2630437207,
                -30.1552268759,
                -21.1238353380,
            ],
            [3.35386959711, 5.97156957878, 14.6952576794, 1.65699966404, 28.4465098142],
            [1.35494901715, 2.18231499728, 6.27960525067, 1.43785752721, 15.3882659679],
        ]
    )

    dv0 = jnp.array(
        [
            [
                -0.557160570446,
                -0.415570776342,
                -0.325325669158,
                -0.024047625417,
                -0.176860753121,
            ],
            [
                0.505696783289,
                0.365682722812,
                0.189706021964,
                -0.287659532608,
                -0.216393453025,
            ],
            [
                0.230578543901,
                0.169143213293,
                0.087726532278,
                -0.117219543175,
                -0.014864789309,
            ],
        ]
    )
    init = v0, dv0
    return diffeq, init


#
# class D: Orbit equations
#


def _make_d(ε):
    def diffeq(t, y, args):
        y1, y2, y3, y4 = y
        r_cubed = (y1 ** 2 + y2 ** 2) ** (1.5)
        dy1 = y3
        dy2 = y4
        dy3 = -y1 / r_cubed
        dy4 = -y2 / r_cubed
        return dy1, dy2, dy3, dy4

    init = (1 - ε, 0, 0, math.sqrt((1 + ε) / (1 - ε)))
    return diffeq, init


_d1 = lambda: _make_d(0.1)
_d2 = lambda: _make_d(0.3)
_d3 = lambda: _make_d(0.5)
_d4 = lambda: _make_d(0.7)
_d5 = lambda: _make_d(0.9)


#
# class E: Higher order equations
#


def _e1():
    def diffeq(t, y, args):
        y1, y2 = y
        dy1 = y2
        dy2 = -(y2 / (t + 1) + y1 * (1 - 0.25 / (t + 1) ** 2))
        return dy1, dy2

    init = (0.6713967071418030, 0.09540051444747446)
    return diffeq, init


def _e2():
    def diffeq(t, y, args):
        y1, y2 = y
        dy1 = y2
        dy2 = (1 - y1 ** 2) * y2 - y1
        return dy1, dy2

    init = (2, 0)
    return diffeq, init


def _e3():
    def diffeq(t, y, args):
        y1, y2 = y
        dy1 = y2
        dy2 = y1 ** 3 / 6 - y1 + 2 * jnp.sin(2.78535 * t)
        return dy1, dy2

    init = (0, 0)
    return diffeq, init


def _e4():
    # This one is a bit weird: y1 doesn't affect y2.
    def diffeq(t, y, args):
        y1, y2 = y
        dy1 = y2
        dy2 = 0.032 - 0.4 * y2 ** 2
        return dy1, dy2

    init = (30, 0)
    return diffeq, init


def _e5():
    # Again kind of weird: y1 doesn't affect y2.
    def diffeq(t, y, args):
        y1, y2 = y
        dy1 = y2
        dy2 = jnp.sqrt(1 + y2 ** 2) / (25 - t)
        return dy1, dy2

    init = (0, 0)
    return diffeq, init


@pytest.mark.parametrize("solver_ctr", all_ode_solvers)
def test_a(solver_ctr):
    _test(solver_ctr, [_a1, _a2, _a3, _a4, _a5])


@pytest.mark.parametrize("solver_ctr", all_ode_solvers)
def test_b(solver_ctr):
    _test(solver_ctr, [_b1, _b2, _b3, _b4, _b5])


@pytest.mark.parametrize("solver_ctr", all_ode_solvers)
def test_c(solver_ctr):
    _test(solver_ctr, [_c1, _c2, _c3, _c4, _c5])


@pytest.mark.parametrize("solver_ctr", all_ode_solvers)
def test_d(solver_ctr):
    _test(solver_ctr, [_d1, _d2, _d3, _d4, _d5])


@pytest.mark.parametrize("solver_ctr", all_ode_solvers)
def test_e(solver_ctr):
    _test(solver_ctr, [_e1, _e2, _e3, _e4, _e5])


def _test(solver_ctr, problems):
    for problem in problems:
        vector_field, init = problem()
        solver = solver_ctr(vector_field)
        sol = diffrax.diffeqsolve(solver, t0=0, t1=20, y0=init, dt0=0.01)
        y1 = sol.ys[0]

        scipy_y0, unravel = diffrax.utils.ravel_pytree(init)
        scipy_y0 = scipy_y0.to_py()

        def scipy_fn(t, y):
            y = unravel(y)
            dy = vector_field(t, y, None)
            dy, _ = diffrax.utils.ravel_pytree(dy)
            return dy.to_py()

        scipy_sol = integrate.solve_ivp(
            scipy_fn,
            (0, 20),
            scipy_y0,
            method="DOP853",
            rtol=1e-12,
            atol=1e-12,
            t_eval=[20],
        )
        scipy_y1 = unravel(scipy_sol.y[0])

        assert jnp.allclose(y1, scipy_y1)
