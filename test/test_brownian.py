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


_levy_areas = (
    diffrax.BrownianIncrement,
    diffrax.SpaceTimeLevyArea,
    diffrax.SpaceTimeTimeLevyArea,
)
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
@pytest.mark.parametrize("levy_area", _levy_areas)
@pytest.mark.parametrize("use_levy", (False, True))
def test_shape_and_dtype(ctr, levy_area, use_levy, getkey):
    t0 = 0.0
    t1 = 2.0

    shapes_dtypes1 = (
        ((), None),
        ((0,), None),
        ((1, 0), None),
        ((3, 4), jnp.float32),
        ((1, 2, 3, 4), jnp.float64),
        ({"a": (1,), "b": (2, 3)}, {"a": None, "b": jnp.float64}),
        ((2,), jnp.float16),
        (((1, 2), ((3, 4), (5, 6))), (jnp.float16, (jnp.float32, jnp.float64))),
    )

    shapes_dtypes2 = (
        ((1, 2, 3, 4), jnp.complex128),
        ({"a": (1,), "b": (2, 3)}, {"a": jnp.float64, "b": jnp.complex128}),
    )

    if (
        ctr is diffrax.VirtualBrownianTree
        and levy_area is diffrax.SpaceTimeTimeLevyArea
    ):
        # VBT with STTLA does not support complex dtypes
        # because it uses jax.random.multivariate_normal
        shapes_dtypes = shapes_dtypes1

    else:
        shapes_dtypes = shapes_dtypes1 + shapes_dtypes2

    def is_tuple_of_ints(obj):
        return isinstance(obj, tuple) and all(isinstance(x, int) for x in obj)

    for shape, dtype in shapes_dtypes:
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
                    assert isinstance(out, diffrax.AbstractBrownianIncrement)
                    w = out.W
                    if isinstance(out, diffrax.SpaceTimeLevyArea):
                        h = out.H
                        assert eqx.filter_eval_shape(lambda: h) == shape
                        if isinstance(out, diffrax.SpaceTimeTimeLevyArea):
                            k = out.K
                            assert eqx.filter_eval_shape(lambda: k) == shape
                else:
                    w = out
                assert eqx.filter_eval_shape(lambda: w) == shape


@pytest.mark.parametrize(
    "ctr", [diffrax.VirtualBrownianTree, diffrax.UnsafeBrownianPath]
)
@pytest.mark.parametrize("levy_area", _levy_areas)
@pytest.mark.parametrize("use_levy", (True, False))
def test_statistics(ctr, levy_area, use_levy):
    # Deterministic key for this test; not using getkey()
    key = jr.PRNGKey(5678)
    num_samples = 600000
    keys = jr.split(key, num_samples)
    t0, t1 = 0.0, 5.0
    dt = t1 - t0

    def _eval(key):
        if ctr is diffrax.UnsafeBrownianPath:
            path = ctr(shape=(), key=key, levy_area=levy_area)
        elif ctr is diffrax.VirtualBrownianTree:
            path = ctr(t0=0, t1=5, tol=2**-5, shape=(), key=key, levy_area=levy_area)
        else:
            assert False
        return path.evaluate(t0, t1, use_levy=use_levy)

    values = jax.vmap(_eval)(keys)
    if use_levy:
        assert isinstance(values, diffrax.AbstractBrownianIncrement)
        w = values.W

        if isinstance(values, diffrax.SpaceTimeLevyArea):
            h = values.H
            assert h is not None
            assert h.shape == (num_samples,)
            ref_dist_hh = stats.norm(loc=0, scale=math.sqrt(dt / 12))
            _, pval = stats.kstest(h, ref_dist_hh.cdf)
            assert pval > 0.1
            # Check independence of h and w
            assert jnp.mean(h * w) < 0.01

        elif isinstance(values, diffrax.SpaceTimeTimeLevyArea):
            h = values.H
            k = values.K
            assert h is not None
            assert k is not None
            assert h.shape == (num_samples,)
            assert k.shape == (num_samples,)
            ref_dist_hh = stats.norm(loc=0, scale=math.sqrt(dt / 12))
            _, pval = stats.kstest(h, ref_dist_hh.cdf)
            assert pval > 0.1

            ref_dist_kk = stats.norm(loc=0, scale=math.sqrt(dt / 720))
            _, pval = stats.kstest(k, ref_dist_kk.cdf)
            assert pval > 0.1

            # Check independence of w, h and k
            assert jnp.abs(jnp.mean(w * h)) < 0.01
            assert jnp.abs(jnp.mean(h * k)) < 0.01
            assert jnp.abs(jnp.mean(k * w)) < 0.01

    else:
        w = values
    assert w.shape == (num_samples,)
    ref_dist = stats.norm(loc=0, scale=math.sqrt(dt))
    _, pval = stats.kstest(w, ref_dist.cdf)
    assert pval > 0.1


def _true_cond_stats_wh(bm_s, bm_u, s, r, u):
    w_s = bm_s.W
    w_u = bm_u.W
    h_s = bm_s.H
    h_u = bm_u.H
    su = u - s
    sr = r - s
    ru = u - r
    sr3 = jnp.power(sr, 3)
    ru3 = jnp.power(ru, 3)
    su2 = jnp.power(su, 2)
    sr_ru_half = jnp.sqrt(sr * ru)
    d = math.sqrt(sr3 + ru3)
    a = (1 / (2 * su * d)) * sr3 * sr_ru_half
    b = (1 / (2 * su * d)) * ru3 * sr_ru_half
    c = (1.0 / (math.sqrt(12) * d)) * sr ** (3 / 2) * ru ** (3 / 2)
    h_su = (1.0 / su) * (u * h_u - s * h_s - u / 2 * w_s + s / 2 * w_u)

    # Check w_r|(w_s, w_u, h_s, h_u)
    w_mean = w_s + (sr / su) * (w_u - w_s) + (6 * sr * ru / (su2)) * h_su
    w_std = 2 * (a + b) / su

    # Check h_r|(w_s, w_u, h_s, h_u)
    h_mean = (s / r) * h_s + (sr3 / (r * su2)) * h_su + 0.5 * w_s - s / (2 * r) * w_mean
    h_var = (c / r) ** 2 + ((a * u + s * b) / (r * su)) ** 2

    h_std = jnp.sqrt(h_var)

    return w_mean, w_std, h_mean, h_std


def _true_cond_stats_whk(bm_s, bm_u, s, r, u):
    su = u - s
    sr = r - s
    ru = u - r
    sr3 = jnp.power(sr, 3)
    su3 = jnp.power(su, 3)
    w_s = bm_s.W
    w_u = bm_u.W
    h_s = bm_s.H
    h_u = bm_u.H
    k_s = bm_s.K
    k_u = bm_u.K

    su5 = jnp.power(su, 5)
    sr2 = jnp.square(sr)
    ru2 = jnp.square(ru)
    bh_s = s * h_s
    bh_u = u * h_u
    bk_s = s**2 * k_s
    bk_u = u**2 * k_u
    # u_bb_s := u * brownian bridge on [0,u] evaluated at s
    u_bb_s = u * w_s - s * w_u

    # Chen's relation for H
    bh_su = bh_u - bh_s - 0.5 * u_bb_s

    # Chen's relation for \bar{K}_{s,u} := (u-s)^2 * K_{s,u}
    bk_su = bk_u - bk_s - su / 2 * bh_s + s / 2 * bh_su - ((u - 2 * s) / 12) * u_bb_s

    # compute the mean of (W_sr, H_sr, K_sr) conditioned on
    # (W_s, H_s, K_s, W_u, H_u, K_u)
    bb_mean = (6 * sr * ru / su3) * bh_su + (
        120 * sr * ru * (su / 2 - sr) / su5
    ) * bk_su
    mean_w = (sr / su) * (w_u - w_s) + bb_mean
    mean_hh = (sr2 / su3) * bh_su + (30 * sr2 * ru / su5) * bk_su
    mean_kk = (sr3 / su5) * bk_su

    mean_whk = jnp.stack([mean_w, mean_hh, mean_kk], axis=0)

    # now compute the covariance matrix of (W_sr, H_sr, K_sr) conditioned on
    # (W_s, H_s, K_s, W_u, H_u, K_u)
    # note that the covariance matrix is independent of the values of
    # (W_s, H_s, K_s, W_u, H_u, K_u), since those are already represented
    # in the mean.
    sr5 = jnp.power(sr, 5)

    ww_cov = (sr * ru * ((sr - ru) ** 4 + 4 * (sr2 * ru2))) / su5
    wh_cov = -(sr3 * ru * (sr2 - 3 * sr * ru + 6 * ru2)) / (2 * su5)
    wk_cov = (sr**4 * ru * (sr - ru)) / (12 * su5)
    hh_cov = sr / 12 * (1 - (sr3 * (sr2 + 2 * sr * ru + 16 * ru2)) / su5)
    hk_cov = -(sr5 * ru) / (24 * su5)
    kk_cov = sr / 720 * (1 - sr5 / su5)

    cov = jnp.array(
        [
            [ww_cov, wh_cov, wk_cov],
            [wh_cov, hh_cov, hk_cov],
            [wk_cov, hk_cov, kk_cov],
        ]
    )

    return mean_whk, cov


def _conditional_statistics(
    levy_area, use_levy: bool, tol, spacing, spline: _Spline, min_num_points
):
    key = jr.PRNGKey(5680)
    bm_key, sample_key, permute_key = jr.split(key, 3)
    # Get some randomly selected points; not too close to avoid discretisation error.
    t0 = 0.0
    t1 = 8.7
    boundary = 0.1
    ts = jr.uniform(
        sample_key, shape=(10000,), minval=t0 + boundary, maxval=t1 - boundary
    )
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
    num_paths = 20000
    bm_keys = jr.split(bm_key, num_paths)

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
    pvals_k = []

    total_mean_err = jnp.zeros((3,), dtype=jnp.float64)
    total_cov_err = jnp.zeros((3, 3), dtype=jnp.float64)

    # Test their conditional statistics
    for i in range(1, len(ts) - 1):
        s, bm_s = out[i - 1]
        r, bm_r = out[i]
        u, bm_u = out[i + 1]
        if use_levy:
            w_s = bm_s.W
            w_r = bm_r.W
            w_u = bm_u.W
            if issubclass(levy_area, diffrax.AbstractSpaceTimeLevyArea):
                h_s = bm_s.H
                h_r = bm_r.H
            else:
                h_s = None
                h_r = None
            if issubclass(levy_area, diffrax.AbstractSpaceTimeTimeLevyArea):
                k_s = bm_s.K
                k_r = bm_r.K
            else:
                k_s = None
                k_r = None
        else:
            w_s = bm_s
            w_r = bm_r
            w_u = bm_u
            h_s = None
            h_r = None
            k_s = None
            k_r = None

        # Check w_r|(w_s, w_u)
        w_mean1 = w_s + (w_u - w_s) * ((r - s) / (u - s))
        w_var1 = (u - r) * (r - s) / (u - s)
        w_std1 = math.sqrt(w_var1)
        normalised_w1 = (w_r - w_mean1) / w_std1
        _, pval_w1 = stats.kstest(normalised_w1, stats.norm.cdf)
        # Raise if the failure is statistically significant at 10%, subject to
        # multiple-testing correction.
        pvals_w1.append(pval_w1)

        s = s - t0
        r = r - t0
        u = u - t0
        sr = r - s
        if levy_area == diffrax.SpaceTimeTimeLevyArea and use_levy:
            assert bm_s.H is not None
            assert bm_r.H is not None
            assert bm_u.H is not None
            assert bm_s.K is not None
            assert bm_r.K is not None
            assert bm_u.K is not None
            sr2 = jnp.square(sr)
            bh_s = s * h_s
            bh_r = r * h_r
            bk_s = s**2 * k_s
            bk_r = r**2 * k_r

            # Compute the target conditional mean and covariance
            true_mean_whk, true_cov = _true_cond_stats_whk(bm_s, bm_u, s, r, u)

            # now compute the values of (W_sr, H_sr, K_sr), which are to be tested
            # against the normal distribution N(mean, cov)
            w_sr = w_r - w_s
            r_bb_s = r * w_s - s * w_r
            bh_sr = bh_r - bh_s - 0.5 * r_bb_s
            h_sr = bh_sr / sr
            bk_sr = (
                bk_r
                - bk_s
                - 0.5 * (sr * bh_s - s * bh_sr)
                - ((r - 2 * s) / 12) * r_bb_s
            )
            k_sr = bk_sr / sr2

            y = jnp.stack([w_sr, h_sr, k_sr], axis=0)

            # now we have to confirm that (w_centred, h_centred, k_centred) have
            # zero mean and covariance matrix cov

            hat_y = y - true_mean_whk
            tilde_mean = jnp.mean(true_mean_whk, axis=1)
            y_mean = jnp.mean(y, axis=1)
            mean_diff = y_mean - tilde_mean
            emp_cov = jnp.cov(hat_y)

            mean_err = jnp.abs(mean_diff)
            cov_err = jnp.abs(emp_cov - true_cov)
            total_mean_err += mean_err / (len(ts) - 2)
            total_cov_err += cov_err / (len(ts) - 2)

            hat_w, hat_h, hat_k = hat_y
            w_var2, h_var2, k_var2 = jnp.diag(true_cov)

            normalised_w2 = hat_w / math.sqrt(w_var2)
            _, pval_w2 = stats.kstest(normalised_w2, stats.norm.cdf)
            pvals_w2.append(pval_w2)

            normalised_h = hat_h / math.sqrt(h_var2)
            _, pval_h = stats.kstest(normalised_h, stats.norm.cdf)
            pvals_h.append(pval_h)

            normalised_k = hat_k / math.sqrt(k_var2)
            _, pval_k = stats.kstest(normalised_k, stats.norm.cdf)
            pvals_k.append(pval_k)

        elif levy_area == diffrax.SpaceTimeLevyArea and use_levy:
            assert bm_s.H is not None
            assert bm_r.H is not None
            assert bm_u.H is not None

            # Compute the true conditional statistics for W and H
            w_mean2, w_std2, h_mean, h_std = _true_cond_stats_wh(bm_s, bm_u, s, r, u)

            # Check w_r|(w_s, w_u, h_s, h_u)
            normalised_w2 = (w_r - w_mean2) / w_std2
            _, pval_w2 = stats.kstest(normalised_w2, stats.norm.cdf)
            pvals_w2.append(pval_w2)

            # Check h_r|(w_s, w_u, h_s, h_u)
            normalised_hh = (h_r - h_mean) / h_std
            _, pval_h = stats.kstest(normalised_hh, stats.norm.cdf)
            pvals_h.append(pval_h)
    return (
        jnp.array(pvals_w1),
        jnp.array(pvals_w2),
        jnp.array(pvals_h),
        jnp.array(pvals_k),
        total_mean_err,
        total_cov_err,
    )


@pytest.mark.parametrize("levy_area", _levy_areas)
@pytest.mark.parametrize("use_levy", (True, False))
def test_conditional_statistics(levy_area, use_levy):
    pvals_w1, pvals_w2, pvals_h, pvals_k, mean_err, cov_err = _conditional_statistics(
        levy_area,
        use_levy,
        tol=2**-10,
        spacing=2**-8,
        spline="sqrt",
        min_num_points=90,
    )
    assert jnp.all(pvals_w1 > 0.1 / pvals_w1.shape[0])

    if levy_area is diffrax.SpaceTimeTimeLevyArea and use_levy:
        assert jnp.sum(mean_err) < 0.005
        assert jnp.sum(cov_err) < 0.001
        assert jnp.all(pvals_w2 > 0.1 / pvals_w2.shape[0])
        assert jnp.all(pvals_h > 0.1 / pvals_h.shape[0])
        assert jnp.all(pvals_k > 0.1 / pvals_k.shape[0])

    elif levy_area is diffrax.SpaceTimeLevyArea and use_levy:
        assert jnp.all(pvals_w2 > 0.1 / pvals_w2.shape[0])
        assert jnp.all(pvals_h > 0.1 / pvals_h.shape[0])
        assert len(pvals_k) == 0
        assert jnp.all(mean_err == 0.0)
        assert jnp.all(cov_err == 0.0)

    else:
        assert len(pvals_w2) == 0
        assert len(pvals_h) == 0
        assert len(pvals_k) == 0


def _levy_area_spline():
    levy_areas = (
        diffrax.BrownianIncrement,
        diffrax.SpaceTimeLevyArea,
        diffrax.SpaceTimeTimeLevyArea,
    )
    for levy_area in levy_areas:
        for spline in ("quad", "sqrt", "zero"):
            if (
                issubclass(levy_area, diffrax.AbstractSpaceTimeLevyArea)
                and spline == "quad"
            ):
                # The quad spline is not defined for space-time and
                # space-time-time Lévy area
                continue
            yield levy_area, spline


@pytest.mark.parametrize("levy_area,spline", _levy_area_spline())
@pytest.mark.parametrize("use_levy", (True, False))
def test_spline(levy_area, use_levy, spline: _Spline):
    pvals_w1, pvals_w2, pvals_h, pvals_k, mean_err, cov_err = _conditional_statistics(
        levy_area,
        use_levy=use_levy,
        tol=2**-3,
        spacing=2**-3,
        spline=spline,
        min_num_points=20,
    )

    if spline == "sqrt":
        # For the correct spline, make sure that all p-values are above
        # 0.1 (subject to multiple-testing correction) and the average
        # p-value is above 0.3.
        def pred(pvals):
            return jnp.min(pvals) > 0.1 / pvals.shape[0] and jnp.mean(pvals) > 0.3

        def pred_sttla(_mean_err, _cov_err):
            return jnp.sum(_mean_err) < 0.005 and jnp.sum(_cov_err) < 0.002

    else:
        # make sure that for incorrect splines at least one p-value is
        # below 0.001 (subject to multiple-testing correction) and the
        # average p-value is below 0.03 (i.e. at least 10x smaller than
        # for the correct spline).
        def pred(pvals):
            return jnp.min(pvals) < 0.01 / pvals.shape[0] and jnp.mean(pvals) < 0.03

        def pred_sttla(_mean_err, _cov_err):
            return jnp.sum(_cov_err) > 0.005

    if levy_area is diffrax.SpaceTimeTimeLevyArea and use_levy:
        assert pred(pvals_w2)
        assert pred(pvals_h)
        assert pred_sttla(mean_err, cov_err)

    elif levy_area is diffrax.SpaceTimeLevyArea and use_levy:
        assert pred(pvals_w2)
        assert pred(pvals_h)
        assert len(pvals_k) == 0

    else:
        assert len(pvals_w2) == 0
        assert len(pvals_h) == 0
        assert len(pvals_k) == 0
        if levy_area is diffrax.BrownianIncrement:
            assert pred(pvals_w1)
        elif spline == "sqrt":  # levy_area == SpaceTimeLevyArea
            assert pred(pvals_w1)
        elif levy_area == diffrax.SpaceTimeLevyArea and spline == "zero":
            # We need a milder upper bound on jnp.mean(pvals_w1) because
            # the presence of space-time Lévy area gives W_r (i.e. the output
            # of the Brownian path) a variance very close to the correct one,
            # even when the spline is wrong. In pvals_w2 the influence of the
            # Lévy area is subtracted in the mean, so we can use a stricter test.
            n = pvals_w1.shape[0]
            assert jnp.min(pvals_w1) < 0.03 / n and jnp.mean(pvals_w1) < 0.2
        else:
            pass


@pytest.mark.parametrize(
    "tol,spline", [(100.0, "sqrt"), (0.9, "sqrt"), (2**-15, "zero")]
)
def test_whk_interpolation(tol, spline):
    key = jr.key(5678)
    r_key, bm_key = jr.split(key, 2)
    s = jnp.array(0.0, dtype=jnp.float64)
    u = jnp.array(5.7, dtype=jnp.float64)
    bound = 0.0
    rs = jr.uniform(
        r_key, (1000,), dtype=jnp.float64, minval=s + bound, maxval=u - bound
    )
    path = diffrax.VirtualBrownianTree(
        t0=s,
        t1=u,
        shape=(10000,),
        tol=tol,
        key=bm_key,
        levy_area=diffrax.SpaceTimeTimeLevyArea,
        _spline=spline,
    )

    @jax.jit
    def eval_paths(t):
        # return jax.vmap(lambda p: p.evaluate(t, use_levy=True))(paths)
        return path.evaluate(t, use_levy=True)

    bm_s = eval_paths(s)
    bm_u = eval_paths(u)

    total_mean_err = 0.0
    total_cov_err = 0.0
    _pvals_w = []
    _pvals_h = []
    _pvals_k = []
    for r in rs:
        bm_r = eval_paths(r)
        w_s = bm_s.W
        w_r = bm_r.W
        h_s = bm_s.H
        h_r = bm_r.H
        k_s = bm_s.K
        k_r = bm_r.K
        sr = r - s
        sr2 = jnp.square(sr)
        bh_s = s * h_s
        bh_r = r * h_r
        bk_s = s**2 * k_s
        bk_r = r**2 * k_r

        # Compute the target conditional mean and covariance
        true_mean_whk, true_cov = _true_cond_stats_whk(bm_s, bm_u, s, r, u)

        # now compute the values of (W_sr, H_sr, K_sr), which are to be tested
        # against the normal distribution N(mean, cov)
        w_sr = w_r - w_s
        r_bb_s = r * w_s - s * w_r
        bh_sr = bh_r - bh_s - 0.5 * r_bb_s
        h_sr = bh_sr / sr
        bk_sr = (
            bk_r - bk_s - 0.5 * (sr * bh_s - s * bh_sr) - ((r - 2 * s) / 12) * r_bb_s
        )
        k_sr = bk_sr / sr2

        y = jnp.stack([w_sr, h_sr, k_sr], axis=0)

        # now we have to confirm that (w_centred, h_centred, k_centred) have
        # zero mean and covariance matrix cov

        hat_y = y - true_mean_whk
        tilde_mean = jnp.mean(true_mean_whk, axis=1)
        y_mean = jnp.mean(y, axis=1)
        mean_diff = y_mean - tilde_mean
        emp_cov = jnp.cov(hat_y)

        mean_err = mean_diff
        cov_err = emp_cov - true_cov
        total_mean_err += jnp.abs(mean_err) / (len(rs))
        total_cov_err += jnp.abs(cov_err) / (len(rs))

        hat_w, hat_h, hat_k = hat_y
        w_var2, h_var, k_var = jnp.diag(true_cov)

        normalised_w2 = hat_w / math.sqrt(w_var2)
        _, pval_w2 = stats.kstest(normalised_w2, stats.norm.cdf)
        _pvals_w.append(pval_w2)

        normalised_h = hat_h / math.sqrt(h_var)
        _, pval_h = stats.kstest(normalised_h, stats.norm.cdf)
        _pvals_h.append(pval_h)

        normalised_k = hat_k / math.sqrt(k_var)
        _, pval_k = stats.kstest(normalised_k, stats.norm.cdf)
        _pvals_k.append(pval_k)

    _pvals_w = jnp.array(_pvals_w)
    _pvals_h = jnp.array(_pvals_h)
    _pvals_k = jnp.array(_pvals_k)
    assert jnp.all(_pvals_w > 0.1 / _pvals_w.shape[0])
    assert jnp.all(_pvals_h > 0.1 / _pvals_h.shape[0])
    assert jnp.all(_pvals_k > 0.1 / _pvals_k.shape[0])
    assert jnp.all(jnp.abs(total_mean_err) < 0.01)
    assert jnp.all(jnp.abs(total_cov_err) < 0.01)


def test_levy_area_reverse_time():
    key = jr.PRNGKey(5678)
    bm_key, sample_key = jr.split(key, 2)
    bm = diffrax.VirtualBrownianTree(
        t0=0, t1=5, tol=2**-5, shape=(), key=bm_key, levy_area=diffrax.SpaceTimeLevyArea
    )

    ts = jr.uniform(sample_key, shape=(100,), minval=0, maxval=5)

    vec_eval = jax.vmap(lambda t_prev, t: bm.evaluate(t_prev, t))

    fwd_increments = vec_eval(ts[:-1], ts[1:])
    back_increments = vec_eval(ts[1:], ts[:-1])

    assert jtu.tree_map(
        lambda fwd, bck: jnp.allclose(fwd, -bck), fwd_increments, back_increments
    )
