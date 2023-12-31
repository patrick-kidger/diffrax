import contextlib
import math
from typing import Literal
from typing_extensions import TypeAlias

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest
import scipy.stats as stats


_Spline: TypeAlias = Literal["quad", "sqrt", "zero"]
_vals = {
    int: [0, 2],
    float: [0.0, 2.0],
    jnp.int32: [jnp.array(0, dtype=jnp.int32), jnp.array(2, dtype=jnp.int32)],
    jnp.float32: [jnp.array(0.0, dtype=jnp.float32), jnp.array(2.0, dtype=jnp.float32)],
}


def _make_struct(shape, dtype):
    if dtype is None:
        dtype = jnp.float64
    return jax.ShapeDtypeStruct(shape, dtype)


@pytest.mark.parametrize(
    "ctr", [diffrax.UnsafeBrownianPath, diffrax.VirtualBrownianTree]
)
@pytest.mark.parametrize("levy_area", ["", "space-time"])
@pytest.mark.parametrize("use_levy", (False, True))
def test_shape_and_dtype(ctr, levy_area, use_levy, getkey):
    t0 = 0.0
    t1 = 2.0

    shapes = (
        (),
        (0,),
        (1, 0),
        (2,),
        (3, 4),
        (1, 2, 3, 4),
        {
            "a": (1,),
            "b": (2, 3),
        },
        (
            (1, 2),
            (
                (3, 4),
                (5, 6),
            ),
        ),
    )

    dtypes = (
        None,
        None,
        None,
        jnp.float16,
        jnp.float32,
        jnp.float64,
        {"a": None, "b": jnp.float64},
        (jnp.float16, (jnp.float32, jnp.float64)),
    )

    def is_tuple_of_ints(obj):
        return isinstance(obj, tuple) and all(isinstance(x, int) for x in obj)

    for shape, dtype in zip(shapes, dtypes):
        # Shape to pass as input
        if dtype is not None:
            shape = jtu.tree_map(_make_struct, shape, dtype, is_leaf=is_tuple_of_ints)

        if ctr is diffrax.UnsafeBrownianPath:
            path = ctr(shape, getkey(), levy_area=levy_area)
            assert path.t0 == -jnp.inf
            assert path.t1 == jnp.inf
        elif ctr is diffrax.VirtualBrownianTree:
            tol = 2**-3
            path = ctr(t0, t1, tol, shape, getkey(), levy_area=levy_area)
            assert path.t0 == 0
            assert path.t1 == 2
        else:
            assert False

        # Expected output shape
        if dtype is None:
            shape = jtu.tree_map(_make_struct, shape, dtype, is_leaf=is_tuple_of_ints)

        for t0_dtype, (t0, _) in _vals.items():
            for t1_dtype, (_, t2) in _vals.items():
                if all(x in (float, jnp.float32) for x in (t0_dtype, t1_dtype)):
                    context = contextlib.nullcontext()
                else:
                    context = jax.numpy_dtype_promotion("standard")
                with context:
                    out = path.evaluate(t0, t1, use_levy=use_levy)
                if use_levy:
                    assert isinstance(out, diffrax.LevyVal)
                    w = out.W
                    h = out.H
                    if levy_area == "":
                        assert h is None
                    else:
                        assert eqx.filter_eval_shape(lambda: h) == shape
                else:
                    w = out
                assert eqx.filter_eval_shape(lambda: w) == shape


@pytest.mark.parametrize(
    "ctr", [diffrax.UnsafeBrownianPath, diffrax.VirtualBrownianTree]
)
@pytest.mark.parametrize("levy_area", ["", "space-time"])
@pytest.mark.parametrize("use_levy", (False, True))
def test_statistics(ctr, levy_area, use_levy):
    # Deterministic key for this test; not using getkey()
    key = jr.PRNGKey(5678)
    keys = jr.split(key, 10000)

    def _eval(key):
        if ctr is diffrax.UnsafeBrownianPath:
            path = ctr(shape=(), key=key, levy_area=levy_area)
        elif ctr is diffrax.VirtualBrownianTree:
            path = ctr(t0=0, t1=5, tol=2**-5, shape=(), key=key, levy_area=levy_area)
        else:
            assert False
        return path.evaluate(0, 5, use_levy=use_levy)

    values = jax.vmap(_eval)(keys)
    if use_levy:
        assert isinstance(values, diffrax.LevyVal)
        w = values.W
        h = values.H
        if levy_area == "":
            assert h is None
        else:
            assert h is not None
            assert h.shape == (10000,)
            ref_dist = stats.norm(loc=0, scale=math.sqrt(5 / 12))
            _, pval = stats.kstest(h, ref_dist.cdf)
            assert pval > 0.1
    else:
        w = values
    assert w.shape == (10000,)
    ref_dist = stats.norm(loc=0, scale=math.sqrt(5))
    _, pval = stats.kstest(w, ref_dist.cdf)
    assert pval > 0.1


def conditional_statistics(
    levy_area, use_levy: bool, tol, spacing, spline: _Spline, min_num_points
):
    key = jr.PRNGKey(5678)
    bm_key, sample_key, permute_key = jr.split(key, 3)
    # Get some randomly selected points; not too close to avoid discretisation error.
    t0 = 0.3
    t1 = 8.7
    ts = jr.uniform(sample_key, shape=(100,), minval=t0, maxval=t1)
    sorted_ts = jnp.sort(ts)
    ts = []
    prev_ti = sorted_ts[0]
    ts.append(prev_ti)
    for ti in sorted_ts[1:]:
        if ti < prev_ti + spacing:
            continue
        prev_ti = ti
        ts.append(ti)
    ts = jnp.stack(ts)
    assert len(ts) > min_num_points
    ts = jr.permutation(permute_key, ts)

    # Get some random paths
    bm_keys = jr.split(bm_key, 10000)

    path = jax.vmap(
        lambda k: diffrax.VirtualBrownianTree(
            t0=t0, t1=t1, shape=(), tol=tol, key=k, levy_area=levy_area, _spline=spline
        )
    )(bm_keys)

    # Sample some points
    out = []
    for ti in ts:
        vals = jax.vmap(lambda p: p.evaluate(t0, ti, use_levy=use_levy))(path)
        out.append((ti, vals))
    out = sorted(out, key=lambda x: x[0])

    pvals_w1 = []
    pvals_w2 = []
    pvals_h = []

    # Test their conditional statistics
    for i in range(1, len(ts) - 1):
        s, bm_s = out[i - 1]
        r, bm_r = out[i]
        u, bm_u = out[i + 1]
        if use_levy:
            w_s = bm_s.W
            w_r = bm_r.W
            w_u = bm_u.W
            h_s = bm_s.H
            h_r = bm_r.H
            h_u = bm_u.H
        else:
            w_s = bm_s
            w_r = bm_r
            w_u = bm_u
            h_s = None
            h_r = None
            h_u = None

        # Check w_r|(w_s, w_u)
        w_mean1 = w_s + (w_u - w_s) * ((r - s) / (u - s))
        w_var1 = (u - r) * (r - s) / (u - s)
        w_std1 = math.sqrt(w_var1)
        normalised_w1 = (w_r - w_mean1) / w_std1
        _, pval_w1 = stats.kstest(normalised_w1, stats.norm.cdf)
        # Raise if the failure is statistically significant at 10%, subject to
        # multiple-testing correction.
        pvals_w1.append(pval_w1)

        if levy_area == "space-time" and use_levy:
            assert h_s is not None
            assert h_r is not None
            assert h_u is not None
            s = s - t0
            r = r - t0
            u = u - t0
            su = u - s
            sr = r - s
            ru = u - r
            d = math.sqrt(sr**3 + ru**3)
            a = (1 / (2 * su * d)) * sr ** (7 / 2) * math.sqrt(ru)
            b = (1 / (2 * su * d)) * ru ** (7 / 2) * math.sqrt(sr)
            c = (1.0 / (math.sqrt(12) * d)) * sr ** (3 / 2) * ru ** (3 / 2)
            h_su = (1.0 / su) * (u * h_u - s * h_s - u / 2 * w_s + s / 2 * w_u)

            # Check w_r|(w_s, w_u, h_s, h_u)
            w_mean2 = w_s + (sr / su) * (w_u - w_s) + (6 * sr * ru / (su**2)) * h_su
            w_std2 = 2 * (a + b) / su
            normalised_w2 = (w_r - w_mean2) / w_std2
            _, pval_w2 = stats.kstest(normalised_w2, stats.norm.cdf)
            pvals_w2.append(pval_w2)

            # Check h_r|(w_s, w_u, h_s, h_u)
            h_mean = (
                (s / r) * h_s
                + (sr**3 / (r * su**2)) * h_su
                + 0.5 * w_s
                - s / (2 * r) * w_mean2
            )
            h_var = (c / r) ** 2 + ((a * u + s * b) / (r * su)) ** 2
            h_std = math.sqrt(h_var)
            normalised_hh = (h_r - h_mean) / h_std
            _, pval_h = stats.kstest(normalised_hh, stats.norm.cdf)
            pvals_h.append(pval_h)
    return jnp.array(pvals_w1), jnp.array(pvals_w2), jnp.array(pvals_h)


@pytest.mark.parametrize("levy_area", ["", "space-time"])
@pytest.mark.parametrize("use_levy", (False, True))
def test_conditional_statistics(levy_area, use_levy):
    pvals_w1, pvals_w2, pvals_h = conditional_statistics(
        levy_area,
        use_levy,
        tol=2**-10,
        spacing=2**-8,
        spline="sqrt",
        min_num_points=90,
    )
    assert jnp.all(pvals_w1 > 0.1 / pvals_w1.shape[0])
    if levy_area == "space-time" and use_levy:
        assert jnp.all(pvals_w2 > 0.1 / pvals_w2.shape[0])
        assert jnp.all(pvals_h > 0.1 / pvals_h.shape[0])
    else:
        assert len(pvals_w2) == 0
        assert len(pvals_h) == 0


def _levy_area_spline():
    for levy_area in ("", "space-time"):
        for spline in ("quad", "sqrt", "zero"):
            if levy_area == "space-time" and spline == "quad":
                continue
            yield levy_area, spline


@pytest.mark.parametrize("levy_area,spline", _levy_area_spline())
@pytest.mark.parametrize("use_levy", (False, True))
def test_spline(levy_area, use_levy, spline):
    pvals_w1, pvals_w2, pvals_h = conditional_statistics(
        levy_area,
        use_levy=use_levy,
        tol=2**-4,
        spacing=2**-5,
        spline=spline,
        min_num_points=50,
    )

    if spline == "sqrt":
        # For the correct spline, make sure that all p-values are above
        # 0.1 (subject to multiple-testing correction) and the average
        # p-value is above 0.3.
        def pred(pvals):
            return jnp.min(pvals) > 0.1 / pvals.shape[0] and jnp.mean(pvals) > 0.3

    else:
        # make sure that for incorrect splines at least one p-value is
        # below 0.001 (subject to multiple-testing correction) and the
        # average p-value is below 0.03 (i.e. at least 10x smaller than
        # for the correct spline).
        def pred(pvals):
            return jnp.min(pvals) < 0.001 / pvals.shape[0] and jnp.mean(pvals) < 0.03

    if levy_area == "space-time" and use_levy:
        assert pred(pvals_w2)
        assert pred(pvals_h)
    else:
        assert len(pvals_w2) == 0
        assert len(pvals_h) == 0
        if levy_area == "":
            assert pred(pvals_w1)
        elif spline == "sqrt":  # levy_area == "space-time"
            assert pred(pvals_w1)
        else:  # levy_area == "space-time" and spline == "zero"
            # We need a milder upper bound on jnp.mean(pvals_w1) because
            # the presence of space-time Levy area gives W_r (i.e. the output
            # of the Brownian path) a variance very close to the correct one,
            # even when the spline is wrong. In pvals_w2 the influence of the
            # Levy area is subtracted in the mean, so we can use a stricter test.
            n = pvals_w1.shape[0]
            assert jnp.min(pvals_w1) < 0.01 / n and jnp.mean(pvals_w1) < 0.2


def test_levy_area_reverse_time():
    key = jr.PRNGKey(5678)
    bm_key, sample_key = jr.split(key, 2)
    bm = diffrax.VirtualBrownianTree(
        t0=0, t1=5, tol=2**-5, shape=(), key=bm_key, levy_area="space-time"
    )

    ts = jr.uniform(sample_key, shape=(100,), minval=0, maxval=5)

    vec_eval = jax.vmap(lambda t_prev, t: bm.evaluate(t_prev, t))

    fwd_increments = vec_eval(ts[:-1], ts[1:])
    back_increments = vec_eval(ts[1:], ts[:-1])

    assert jtu.tree_map(
        lambda fwd, bck: jnp.allclose(fwd, -bck), fwd_increments, back_increments
    )
