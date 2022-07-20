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
import jax
import jax.flatten_util as fu
import jax.numpy as jnp
import pytest
import scipy.integrate as integrate

from .helpers import all_ode_solvers, implicit_tol, shaped_allclose


#
# Class A: Single equations
#


def _a1():
    diffeq = lambda t, y, args: -y
    init = 1.0
    return diffeq, init


def _a2():
    diffeq = lambda t, y, args: -0.5 * y**3
    init = 1.0
    return diffeq, init


def _a3():
    diffeq = lambda t, y, args: y * jnp.cos(t)
    init = 1.0
    return diffeq, init


def _a4():
    diffeq = lambda t, y, args: 0.25 * y * (1 - 0.05 * y)
    init = 1.0
    return diffeq, init


def _a5():
    diffeq = lambda t, y, args: (y - t) / (y + t)
    init = 4.0
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

    init = (1.0, 3.0)
    return diffeq, init


def _b2():
    def diffeq(t, y, args):
        y1, y2, y3 = y
        dy1 = -y1 + y2
        dy2 = y1 - 2 * y2 + y3
        dy3 = y2 - y3
        return dy1, dy2, dy3

    init = (2.0, 0.0, 1.0)
    return diffeq, init


def _b3():
    def diffeq(t, y, args):
        y1, y2, y3 = y
        dy1 = -y1
        dy2 = y1 - y2**2
        dy3 = y2**2
        return dy1, dy2, dy3

    init = (1.0, 0.0, 0.0)
    return diffeq, init


def _b4():
    def diffeq(t, y, args):
        y1, y2, y3 = y
        r = jnp.sqrt(y1**2 + y2**2)
        dy1 = -y2 - y1 * y3 / r
        dy2 = y1 - y2 * y3 / r
        dy3 = y1 / r
        return dy1, dy2, dy3

    init = (3.0, 0.0, 0.0)
    return diffeq, init


def _b5():
    def diffeq(t, y, args):
        y1, y2, y3 = y
        dy1 = y2 * y3
        dy2 = -y1 * y3
        dy3 = -0.51 * y1 * y2
        return dy1, dy2, dy3

    init = (0.0, 1.0, 1.0)
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
    m_j = m_k = jnp.array(
        [
            0.000954786104043,
            0.000285583733151,
            0.0000437273164546,
            0.0000517759138449,
            0.00000277777777778,
        ]
    )

    def diffeq(t, y, args):
        y_ij, dy_ij = y
        y_ik = y_ij
        assert y_ij.shape == (3, 5)
        r_cubed_j = r_cubed_k = jnp.sum(y_ij**2, axis=0) ** 1.5
        d_cubed_jk = jnp.sum((y_ij[:, :, None] - y_ij[:, None, :]) ** 2, axis=0) ** 1.5

        term1_ij = -(m0 + m_j) * y_ij / r_cubed_j
        term2_ijk = (y_ij[:, None, :] - y_ij[:, :, None]) / d_cubed_jk
        term3_ik = y_ik / r_cubed_k
        term4_ijk = m_k * (term2_ijk - term3_ik[:, None])
        term4_ijk = term4_ijk.at[:, jnp.arange(5), jnp.arange(5)].set(0)
        term5_ij = jnp.sum(term4_ijk, axis=-1)

        ddy_ij = k2 * (term1_ij + term5_ij)
        return dy_ij, ddy_ij

    y0_ij = jnp.array(
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

    dy0_ij = jnp.array(
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
    init = y0_ij, dy0_ij
    return diffeq, init


#
# class D: Orbit equations
#


def _make_d(ε):
    def diffeq(t, y, args):
        y1, y2, y3, y4 = y
        r_cubed = (y1**2 + y2**2) ** (1.5)
        dy1 = y3
        dy2 = y4
        dy3 = -y1 / r_cubed
        dy4 = -y2 / r_cubed
        return dy1, dy2, dy3, dy4

    init = (1 - ε, 0.0, 0.0, math.sqrt((1 + ε) / (1 - ε)))
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
        dy2 = (1 - y1**2) * y2 - y1
        return dy1, dy2

    init = (2.0, 0.0)
    return diffeq, init


def _e3():
    def diffeq(t, y, args):
        y1, y2 = y
        dy1 = y2
        dy2 = y1**3 / 6 - y1 + 2 * jnp.sin(2.78535 * t)
        return dy1, dy2

    init = (0.0, 0.0)
    return diffeq, init


def _e4():
    # This one is a bit weird: y1 doesn't affect y2.
    def diffeq(t, y, args):
        y1, y2 = y
        dy1 = y2
        dy2 = 0.032 - 0.4 * y2**2
        return dy1, dy2

    init = (30.0, 0.0)
    return diffeq, init


def _e5():
    # Again kind of weird: y1 doesn't affect y2.
    def diffeq(t, y, args):
        y1, y2 = y
        dy1 = y2
        dy2 = jnp.sqrt(1 + y2**2) / (25 - t)
        return dy1, dy2

    init = (0.0, 0.0)
    return diffeq, init


@pytest.mark.parametrize("solver", all_ode_solvers)
def test_a(solver):
    if isinstance(solver, (diffrax.Euler, diffrax.ImplicitEuler)):
        # Euler is pretty bad at solving things, so only do some simple tests.
        _test(solver, [_a1, _a2], higher=False)
    else:
        _test(solver, [_a1, _a2, _a3, _a4, _a5], higher=False)


@pytest.mark.parametrize("solver", all_ode_solvers)
def test_b(solver):
    _test(solver, [_b1, _b2, _b3, _b4, _b5], higher=True)


@pytest.mark.parametrize("solver", all_ode_solvers)
def test_c(solver):
    _test(solver, [_c1, _c2, _c3, _c4, _c5], higher=True)


@pytest.mark.parametrize("solver", all_ode_solvers)
def test_d(solver):
    _test(solver, [_d1, _d2, _d3, _d4, _d5], higher=True)


@pytest.mark.parametrize("solver", all_ode_solvers)
def test_e(solver):
    _test(solver, [_e1, _e2, _e3, _e4, _e5], higher=True)


def _test(solver, problems, higher):
    for problem in problems:
        vector_field, init = problem()
        term = diffrax.ODETerm(vector_field)
        if higher and solver.order(term) < 4:
            # Too difficult to get accurate solutions with a low-order solver
            return
        max_steps = 16**4
        if not isinstance(solver, diffrax.AbstractAdaptiveSolver):
            solver = implicit_tol(solver)
            dt0 = 0.01
            if type(solver) is diffrax.LeapfrogMidpoint:
                # This is an *awful* long-time-horizon solver.
                # It gets decent results to begin with, but then the oscillations
                # build up by t=20.
                # Teeny-tiny steps fix this.
                dt0 = 0.000001
                max_steps = 20_000_001
            stepsize_controller = diffrax.ConstantStepSize()
        elif type(solver) is diffrax.ReversibleHeun and problem is _a1:
            # ReversibleHeun is a bit like LeapfrogMidpoint, and therefore bad over
            # long time horizons. (It develops very large oscillations over long time
            # horizons.)
            # Unlike LeapfrogMidpoint, however, ReversibleHeun offers adaptive step
            # sizing... which picks up on the problem, and tries to take teeny-tiny
            # steps to compensate. In practice this means the solve does not terminate
            # even for very large values of max_steps.
            # Just for this one problem, therefore, we switch to using a constant step
            # size. (To avoid the adaptive step sizing sabotaging us.)
            dt0 = 0.001
            stepsize_controller = diffrax.ConstantStepSize()
        else:
            dt0 = None
            if solver.order(term) < 4:
                rtol = 1e-6
                atol = 1e-6
            else:
                rtol = 1e-8
                atol = 1e-8
            stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
        sol = diffrax.diffeqsolve(
            term,
            solver=solver,
            t0=0.0,
            t1=20.0,
            dt0=dt0,
            y0=init,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
        )
        y1 = jax.tree_map(lambda yi: yi[0], sol.ys)

        scipy_y0, unravel = fu.ravel_pytree(init)
        scipy_y0 = scipy_y0.to_py()

        def scipy_fn(t, y):
            y = unravel(y)
            dy = vector_field(t, y, None)
            dy, _ = fu.ravel_pytree(dy)
            return dy.to_py()

        scipy_sol = integrate.solve_ivp(
            scipy_fn,
            (0, 20),
            scipy_y0,
            method="DOP853",
            rtol=1e-8,
            atol=1e-8,
            t_eval=[20],
        )
        scipy_y1 = unravel(scipy_sol.y[:, 0])

        if solver.order(term) < 4:
            rtol = 1e-3
            atol = 1e-3
        else:
            rtol = 4e-5
            atol = 4e-5

        assert shaped_allclose(y1, scipy_y1, rtol=rtol, atol=atol)
