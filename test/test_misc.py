import diffrax
import jax.numpy as jnp

from .helpers import tree_allclose


def test_fill_forward():
    in_ = jnp.array([jnp.nan, 0.0, 1.0, jnp.nan, jnp.nan, 2.0, jnp.nan])
    out_ = jnp.array([jnp.nan, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0])
    fill_in = diffrax._misc.fill_forward(in_[:, None])
    assert tree_allclose(fill_in, out_[:, None], equal_nan=True)


def test_force_bitcast_convert_type():
    val_1 = jnp.float64(1e6)
    val_2 = jnp.float64(1e6 + 1e-4)

    # Val_1 and val_2 are different as float64,
    # but would be the same if naively downcast to float32.
    assert val_1 != val_2
    assert val_1.astype(jnp.int32) == val_2.astype(jnp.int32)

    val_1_cast = diffrax._misc.force_bitcast_convert_type(val_1, jnp.int32)
    val_2_cast = diffrax._misc.force_bitcast_convert_type(val_2, jnp.int32)

    assert val_1_cast.dtype == jnp.int32
    assert val_2_cast.dtype == jnp.int32

    # Bitcasted values should be different in the smaller type
    assert val_1_cast != val_2_cast
