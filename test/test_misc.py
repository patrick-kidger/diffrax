import diffrax
import jax.numpy as jnp

from .helpers import shaped_allclose


def test_fill_forward():
    in_ = jnp.array([jnp.nan, 0.0, 1.0, jnp.nan, jnp.nan, 2.0, jnp.nan])
    out_ = jnp.array([jnp.nan, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0])
    fill_in = diffrax.misc.fill_forward(in_[:, None])
    assert shaped_allclose(fill_in, out_[:, None], equal_nan=True)
