import math

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest
import scipy.stats as stats


_vals = {
    int: [0, 2],
    float: [0.0, 2.0],
    jnp.int32: [jnp.array(0, dtype=jnp.int32), jnp.array(2, dtype=jnp.int32)],
    jnp.float32: [jnp.array(0.0, dtype=jnp.float32), jnp.array(2.0, dtype=jnp.float32)],
}


def _make_struct(shape, dtype):
    dtype = jax.dtypes.canonicalize_dtype(dtype)
    return jax.ShapeDtypeStruct(shape, dtype)


@pytest.mark.parametrize(
    "ctr", [diffrax.UnsafeBrownianPath, diffrax.VirtualBrownianTree]
)
def test_shape_and_dtype(ctr, getkey):
    t0 = 0
    t1 = 2

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
            path = ctr(shape, getkey(), levy_area="space-time")
            assert path.t0 == -jnp.inf
            assert path.t1 == jnp.inf
        elif ctr is diffrax.VirtualBrownianTree:
            tol = 2**-3
            path = ctr(t0, t1, tol, shape, getkey(), levy_area="space-time")
            assert path.t0 == 0
            assert path.t1 == 2
        else:
            assert False

        # Expected output shape
        if dtype is None:
            shape = jtu.tree_map(_make_struct, shape, dtype, is_leaf=is_tuple_of_ints)

        for _t0 in _vals.values():
            for _t1 in _vals.values():
                t0, _ = _t0
                _, t1 = _t1
                bm = path.evaluate(t0, t1, use_levy=True)
                out_w = bm.W
                out_hh = bm.H
                out_w_shape = jtu.tree_map(
                    lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), out_w
                )
                out_hh_shape = jtu.tree_map(
                    lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), out_hh
                )
                assert out_hh_shape == shape
                assert out_w_shape == shape


@pytest.mark.parametrize(
    "ctr", [diffrax.UnsafeBrownianPath, diffrax.VirtualBrownianTree]
)
def test_statistics(ctr):
    # Deterministic key for this test; not using getkey()
    key = jr.PRNGKey(5678)
    keys = jr.split(key, 10000)

    def _eval(key):
        if ctr is diffrax.UnsafeBrownianPath:
            path = ctr(shape=(), key=key, levy_area="space-time")
        elif ctr is diffrax.VirtualBrownianTree:
            path = ctr(t0=0, t1=5, tol=2**-5, shape=(), key=key, levy_area="space-time")
        else:
            assert False
        return path.evaluate(0, 5, use_levy=True)

    bm_inc = jax.vmap(_eval)(keys)
    values_w = bm_inc.W
    values_h = bm_inc.H
    assert values_w.shape == (10000,) and values_h.shape == (10000,)
    ref_dist_w = stats.norm(loc=0, scale=math.sqrt(5))
    _, pval_w = stats.kstest(values_w, ref_dist_w.cdf)
    ref_dist_h = stats.norm(loc=0, scale=math.sqrt(5 / 12))
    _, pval_h = stats.kstest(values_h, ref_dist_h.cdf)
    assert pval_w > 0.1
    assert pval_h > 0.1


def test_conditional_statistics():
    key = jr.PRNGKey(5678)
    bm_key, sample_key, permute_key = jr.split(key, 3)

    # Get >80 randomly selected points; not too close to avoid discretisation error.
    t0 = 0.3
    t1 = 8.7
    ts = jr.uniform(sample_key, shape=(100,), minval=t0, maxval=t1)
    sorted_ts = jnp.sort(ts)
    prev_ti = sorted_ts[0]
    ts = [prev_ti]
    for ti in sorted_ts[1:]:
        if ti < prev_ti + 2**-10:
            continue
        prev_ti = ti
        ts.append(ti)
    ts = jnp.stack(ts)
    assert len(ts) > 80
    ts = jr.permutation(permute_key, ts)

    # Get some random paths
    bm_keys = jr.split(bm_key, 10000)
    path = jax.vmap(
        lambda k: diffrax.VirtualBrownianTree(
            t0=t0, t1=t1, shape=(), tol=2**-13, key=k, levy_area="space-time"
        )
    )(bm_keys)

    # Sample some points
    out = []
    for ti in ts:
        vals = jax.vmap(lambda p: p.evaluate(t0, ti, use_levy=True))(path)
        out.append((ti, vals))
    out = sorted(out, key=lambda x: x[0])

    # Test their conditional statistics
    for i in range(1, len(ts) - 1):
        s, bm_s = out[i - 1]
        r, bm_r = out[i]
        u, bm_u = out[i + 1]

        w_s, hh_s = bm_s.W, bm_s.H
        w_r, hh_r = bm_r.W, bm_r.H
        w_u, hh_u = bm_u.W, bm_u.H

        s = s - t0
        r = r - t0
        u = u - t0
        su = u - s
        sr = r - s
        ru = u - r
        d = jnp.sqrt(jnp.power(sr, 3) + jnp.power(ru, 3))
        a = (1 / (2 * su * d)) * jnp.power(sr, 7 / 2) * jnp.sqrt(ru)
        b = (1 / (2 * su * d)) * jnp.power(ru, 7 / 2) * jnp.sqrt(sr)
        c = (1.0 / (jnp.sqrt(12) * d)) * jnp.power(sr, 3 / 2) * jnp.power(ru, 3 / 2)

        hh_su = (1.0 / su) * (u * hh_u - s * hh_s - u / 2 * w_s + s / 2 * w_u)

        w_mean = w_s + (sr / su) * (w_u - w_s) + (6 * sr * ru / jnp.square(su)) * hh_su
        w_std = 2 * (a + b) / su
        normalised_w = (w_r - w_mean) / w_std
        hh_mean = (
            (s / r) * hh_s
            + (jnp.power(sr, 3) / (r * jnp.square(su))) * hh_su
            + 0.5 * w_s
            - s / (2 * r) * w_mean
        )
        hh_var = jnp.square(c / r) + jnp.square((a * u + s * b) / (r * su))
        hh_std = jnp.sqrt(hh_var)
        normalised_hh = (hh_r - hh_mean) / hh_std

        _, pval_w = stats.kstest(normalised_w, stats.norm.cdf)
        _, pval_hh = stats.kstest(normalised_hh, stats.norm.cdf)

        # Raise if the failure is statistically significant at 10%, subject to
        # multiple-testing correction.
        assert pval_w > 0.001
        assert pval_hh > 0.001


def test_reverse_time():
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
