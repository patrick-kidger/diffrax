import equinox.internal as eqxi
import jax
import pytest


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_numpy_rank_promotion", "raise")
jax.config.update("jax_numpy_dtype_promotion", "strict")


@pytest.fixture()
def getkey():
    return eqxi.GetKey()
