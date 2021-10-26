import diffrax
import jax.numpy as jnp


def test_fill_forward():
    in_ = jnp.array([jnp.nan, 0.0, 1.0, jnp.nan, jnp.nan, 2.0, jnp.nan])
    out_ = jnp.array([jnp.nan, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0])
    fill_in = diffrax.utils.fill_forward(in_[:, None])
    assert jnp.allclose(fill_in, out_[:, None], equal_nan=True)
