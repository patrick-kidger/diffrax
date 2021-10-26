import math

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import scipy.stats as stats


_vals = {
    int: [0, 2],
    float: [0.0, 2.0],
    jnp.int32: [jnp.array(0, dtype=jnp.int32), jnp.array(2, dtype=jnp.int32)],
    jnp.float32: [jnp.array(0.0, dtype=jnp.float32), jnp.array(2.0, dtype=jnp.float32)],
}


def test_unsafe_brownian_shape(getkey):
    for shape in ((0,), (1, 0), (2,), (3, 4), (1, 2, 3, 4)):
        path = diffrax.UnsafeBrownianPath(getkey(), shape)
        assert path.t0 is None
        assert path.t1 is None
        for _t0 in _vals.values():
            for _t1 in _vals.values():
                t0, _ = _t0
                _, t1 = _t1
                out = path.evaluate(t0, t1)
                assert out.shape == shape


def test_unsafe_brownian_statistics():
    # Deterministic key for this test; not using getkey()
    key = jrandom.PRNGKey(5678)
    keys = jrandom.split(key, 10000)

    def _eval(key):
        path = diffrax.UnsafeBrownianPath(shape=(), key=key)
        return path.evaluate(0, 5)

    values = jax.vmap(_eval)(keys)
    assert values.shape == (10000,)
    ref_dist = stats.norm(
        loc=np.zeros_like(values), scale=np.full_like(values, math.sqrt(5))
    )
    _, pval = stats.kstest(values, ref_dist.cdf)
    assert pval > 0.1
