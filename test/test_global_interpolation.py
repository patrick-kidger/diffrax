import operator

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from helpers import all_ode_solvers, tree_allclose


# TODO:
# - Decide how to handle weakly increasing times in interpolation routines


@pytest.mark.parametrize("mode", ["linear", "cubic"])
def test_interpolation_coeffs(mode):
    # Data is linear so both linear and cubic interpolation should produce the same
    # results where there is missing data.
    ts = ys = jnp.linspace(0.0, 9.0, 10)
    nan_ys = ys.at[jnp.array([0, 3, 4, 6, 9])].set(jnp.nan)
    nan_ys = nan_ys[:, None]

    def _interp(tree, **kwargs):
        if tree:
            to_interp = (nan_ys,)
        else:
            to_interp = nan_ys
        if mode == "linear":
            return diffrax.linear_interpolation(ts, to_interp, **kwargs)
        elif mode == "cubic":
            coeffs = diffrax.backward_hermite_coefficients(ts, to_interp, **kwargs)
            interp = diffrax.CubicInterpolation(ts, coeffs)
            return jax.vmap(interp.evaluate)(ts)

    interp_ys = _interp(tree=False)
    true_ys = ys.at[jnp.array([0, 9])].set(jnp.nan)[:, None]
    assert jnp.allclose(interp_ys, true_ys, equal_nan=True)
    (interp_ys,) = _interp(tree=True)
    assert jnp.allclose(interp_ys, true_ys, equal_nan=True)

    interp_ys = _interp(tree=False, fill_forward_nans_at_end=True)
    true_ys = ys.at[0].set(jnp.nan).at[9].set(8.0)[:, None]
    assert jnp.allclose(interp_ys, true_ys, equal_nan=True)
    (interp_ys,) = _interp(tree=True, fill_forward_nans_at_end=True)
    assert jnp.allclose(interp_ys, true_ys, equal_nan=True)

    interp_ys = _interp(tree=False, replace_nans_at_start=5.5)
    true_ys = ys.at[0].set(5.5).at[9].set(jnp.nan)[:, None]
    assert jnp.allclose(interp_ys, true_ys, equal_nan=True)
    (interp_ys,) = _interp(tree=True, replace_nans_at_start=(5.5,))
    assert jnp.allclose(interp_ys, true_ys, equal_nan=True)


def test_rectilinear_interpolation_coeffs():
    ts = jnp.linspace(0.0, 9.0, 10)
    ys = jnp.array(
        [jnp.nan, 0.2, 0.1, jnp.nan, jnp.nan, 0.5, jnp.nan, 0.8, 0.1, jnp.nan]
    )[:, None]

    interp_ys = diffrax.rectilinear_interpolation(ts, ys)
    true_ys = jnp.array(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
            [
                jnp.nan,
                jnp.nan,
                0.2,
                0.2,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.5,
                0.5,
                0.5,
                0.5,
                0.8,
                0.8,
                0.1,
                0.1,
                0.1,
                0.1,
            ],
        ]
    )
    assert jnp.allclose(interp_ys, true_ys, equal_nan=True)
    (interp_ys,) = diffrax.rectilinear_interpolation(ts, (ys,))
    assert jnp.allclose(interp_ys, true_ys, equal_nan=True)

    interp_ys = diffrax.rectilinear_interpolation(ts, ys, replace_nans_at_start=5.5)
    true_ys = jnp.array(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
            [
                5.5,
                5.5,
                0.2,
                0.2,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.5,
                0.5,
                0.5,
                0.5,
                0.8,
                0.8,
                0.1,
                0.1,
                0.1,
                0.1,
            ],
        ]
    )
    assert jnp.allclose(interp_ys, true_ys, equal_nan=True)
    (interp_ys,) = diffrax.rectilinear_interpolation(ts, (ys,))
    assert jnp.allclose(interp_ys, true_ys, equal_nan=True)


def test_cubic_interpolation_no_deriv0():
    ts = jnp.array([-0.5, 0, 1.0])
    ys = jnp.array([[0.1], [0.5], [-0.2]])
    coeffs = diffrax.backward_hermite_coefficients(ts, ys)
    interp = diffrax.CubicInterpolation(ts, coeffs)

    # First piece is linear

    points = jnp.linspace(-0.5, 0, 10)

    interp_ys = jax.vmap(interp.evaluate)(points)
    true_ys = 0.1 + 0.4 * jnp.linspace(0, 1, 10)[:, None]
    assert jnp.allclose(interp_ys, true_ys)

    derivs = jax.vmap(interp.derivative)(points)
    true_derivs = 0.8
    assert jnp.allclose(derivs, true_derivs)

    # Second piece is cubic

    points = jnp.linspace(0, 1.0, 10)

    interp_ys = jax.vmap(interp.evaluate)(points)
    true_ys = jax.vmap(lambda p: jnp.polyval(jnp.array([1.5, -3, 0.8, 0.5]), p))(
        points
    )[:, None]
    assert jnp.allclose(interp_ys, true_ys)

    derivs = jax.vmap(interp.derivative)(points)
    true_derivs = jax.vmap(lambda p: jnp.polyval(jnp.array([4.5, -6, 0.8]), p))(points)[
        :, None
    ]
    assert jnp.allclose(derivs, true_derivs)


def test_cubic_interpolation_deriv0():
    ts = jnp.array([-0.5, 0, 1.0])
    ys = jnp.array([[0.1], [0.5], [-0.2]])
    coeffs = diffrax.backward_hermite_coefficients(ts, ys, deriv0=jnp.array([0.4]))
    interp = diffrax.CubicInterpolation(ts, coeffs)

    # First piece is cubic

    points = jnp.linspace(-0.5, 0, 10)

    interp_ys = jax.vmap(interp.evaluate)(points)
    true_ys = jax.vmap(lambda p: jnp.polyval(jnp.array([-1.6, -0.8, 0.8, 0.5]), p))(
        points
    )[:, None]
    assert jnp.allclose(interp_ys, true_ys)

    derivs = jax.vmap(interp.derivative)(points)
    true_derivs = jax.vmap(lambda p: jnp.polyval(jnp.array([-4.8, -1.6, 0.8]), p))(
        points
    )[:, None]
    assert jnp.allclose(derivs, true_derivs)

    # Second piece is cubic

    points = jnp.linspace(0, 1.0, 10)

    interp_ys = jax.vmap(interp.evaluate)(points)
    true_ys = jax.vmap(lambda p: jnp.polyval(jnp.array([1.5, -3, 0.8, 0.5]), p))(
        points
    )[:, None]
    assert jnp.allclose(interp_ys, true_ys)

    derivs = jax.vmap(interp.derivative)(points)
    true_derivs = jax.vmap(lambda p: jnp.polyval(jnp.array([4.5, -6, 0.8]), p))(points)[
        :, None
    ]
    assert jnp.allclose(derivs, true_derivs)


@pytest.mark.parametrize("mode", ["linear", "cubic"])
def test_interpolation_classes(mode, getkey):
    length = 8
    num_channels = 3
    ts_ = [
        jnp.linspace(0, 10, length),
        jnp.array([0.0, 2.0, 3.0, 3.1, 4.0, 4.0, 5.0, 5.0]),
    ]
    _make = lambda: jrandom.normal(getkey(), (length, num_channels))
    ys_ = [
        _make(),
        [_make(), {"a": _make(), "b": _make()}],
    ]
    for ts in ts_:
        assert len(ts) == length
        for ys in ys_:
            if mode == "linear":
                interp = diffrax.LinearInterpolation(ts, ys)
            elif mode == "cubic":
                coeffs = diffrax.backward_hermite_coefficients(ts, ys)
                interp = diffrax.CubicInterpolation(ts, coeffs)
            else:
                raise RuntimeError

            assert jnp.array_equal(interp.t0, ts[0])
            assert jnp.array_equal(interp.t1, ts[-1])
            true_ys = []
            prev_ti = None
            prev_yi = None
            for i, ti in enumerate(ts):
                yi = jax.tree_map(operator.itemgetter(i), ys)

                # It's important any time ti == prev_ti that the associated yi is
                # treated as "junk data" and ignored.
                if ti == prev_ti:
                    true_ys.append(prev_yi)
                else:
                    true_ys.append(yi)
                prev_ti = ti
                prev_yi = yi
            true_ys = diffrax.misc.stack_pytrees(true_ys)
            pred_ys = jax.vmap(interp.evaluate)(ts)
            assert tree_allclose(pred_ys, true_ys)

            if mode == "linear":
                for i, (t0, t1) in enumerate(zip(ts[:-1], ts[1:])):
                    if t0 == t1:
                        continue
                    y0 = jax.tree_map(operator.itemgetter(i), ys)
                    y1 = jax.tree_map(operator.itemgetter(i + 1), ys)
                    points = jnp.linspace(t0, t1, 10)
                    firstval = interp.evaluate(t0, left=False)
                    vals = jax.vmap(interp.evaluate)(points[1:])

                    def _test(firstval, vals, y0, y1):
                        vals = jnp.concatenate([firstval[None], vals])
                        true_vals = y0 + ((points - t0) / (t1 - t0))[:, None] * (
                            y1 - y0
                        )
                        assert jnp.allclose(vals, true_vals)

                    jax.tree_map(_test, firstval, vals, y0, y1)
                    firstderiv = interp.derivative(t0, left=False)
                    derivs = jax.vmap(interp.derivative)(points[1:])

                    def _test(firstderiv, derivs, y0, y1):
                        derivs = jnp.concatenate([firstderiv[None], derivs])
                        true_derivs = (y1 - y0) / (t1 - t0)
                        assert jnp.allclose(derivs, true_derivs)

                    jax.tree_map(_test, firstderiv, derivs, y0, y1)


# TODO: test around vmap -- that it should handle repeated times correctly.
@pytest.mark.parametrize("solver_ctr", all_ode_solvers)
def test_dense_interpolation(solver_ctr, getkey):
    y0 = jrandom.uniform(getkey(), (), minval=0.4, maxval=2)
    solver = solver_ctr(lambda t, y, args: -y)
    sol = diffrax.diffeqsolve(
        solver, t0=0, t1=1, y0=y0, dt0=0.0001, saveat=diffrax.SaveAt(dense=True)
    )
    points = jnp.linspace(0, 1, 1000)  # finer resolution than the step size
    vals = jax.vmap(sol.evaluate)(points)
    true_vals = jnp.exp(-points) * y0
    assert jnp.allclose(vals, true_vals, atol=1e-3)

    # Tsit5 derivative is not yet implemented.
    if solver_ctr is not diffrax.tsit5:
        derivs = jax.vmap(sol.derivative)(points)
        true_derivs = -true_vals
        assert jnp.allclose(derivs, true_derivs, atol=1e-3)
