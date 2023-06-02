import math

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jrandom
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
            path = ctr(shape, getkey())
            assert path.t0 is None
            assert path.t1 is None
        elif ctr is diffrax.VirtualBrownianTree:
            tol = 2**-5
            path = ctr(t0, t1, tol, shape, getkey())
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
                out = path.evaluate(t0, t1)
                out_shape = jtu.tree_map(
                    lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), out
                )
                assert out_shape == shape


@pytest.mark.parametrize(
    "ctr", [diffrax.UnsafeBrownianPath, diffrax.VirtualBrownianTree]
)
def test_statistics(ctr):
    # Deterministic key for this test; not using getkey()
    key = jrandom.PRNGKey(5678)
    keys = jrandom.split(key, 10000)

    def _eval(key):
        if ctr is diffrax.UnsafeBrownianPath:
            path = ctr(shape=(), key=key)
        elif ctr is diffrax.VirtualBrownianTree:
            path = ctr(t0=0, t1=5, tol=2**-5, shape=(), key=key)
        else:
            assert False
        return path.evaluate(0, 5)

    values = jax.vmap(_eval)(keys)
    assert values.shape == (10000,)
    ref_dist = stats.norm(loc=0, scale=math.sqrt(5))
    _, pval = stats.kstest(values, ref_dist.cdf)
    assert pval > 0.1


def test_conditional_statistics():
    key = jrandom.PRNGKey(5678)
    bm_key, sample_key, permute_key = jrandom.split(key, 3)

    # Get >80 randomly selected points; not too close to avoid discretisation error.
    t0 = 0.3
    t1 = 8.7
    ts = jrandom.uniform(sample_key, shape=(100,), minval=t0, maxval=t1)
    sorted_ts = jnp.sort(ts)
    ts = []
    prev_ti = sorted_ts[0]
    for ti in sorted_ts[1:]:
        if ti < prev_ti + 2**-10:
            continue
        prev_ti = ti
        ts.append(ti)
    ts = jnp.stack(ts)
    assert len(ts) > 80
    ts = jrandom.permutation(permute_key, ts)

    # Get some random paths
    bm_keys = jrandom.split(bm_key, 100000)
    path = jax.vmap(
        lambda k: diffrax.VirtualBrownianTree(
            t0=t0, t1=t1, shape=(), tol=2**-12, key=k
        )
    )(bm_keys)

    # Sample some points
    out = []
    for ti in ts:
        vals = jax.vmap(lambda p: p.evaluate(t0, ti))(path)
        out.append((ti, vals))
    out = sorted(out, key=lambda x: x[0])

    # Test their conditional statistics
    for i in range(1, 98):
        prev_t, prev_vals = out[i - 1]
        this_t, this_vals = out[i]
        next_t, next_vals = out[i + 1]
        mean = prev_vals + (next_vals - prev_vals) * (
            (this_t - prev_t) / (next_t - prev_t)
        )
        var = (next_t - this_t) * (this_t - prev_t) / (next_t - prev_t)
        std = math.sqrt(var)
        normalised_vals = (this_vals - mean) / std
        _, pval = stats.kstest(normalised_vals, stats.norm.cdf)

        # Raise if the failure is statistically significant at 10%, subject to
        # multiple-testing correction.
        assert pval > 0.001
