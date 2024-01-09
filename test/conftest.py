import equinox.internal as eqxi
import jax.config
import pytest


jax.config.update("jax_enable_x64", True)  # pyright: ignore
jax.config.update("jax_numpy_rank_promotion", "raise")  # pyright: ignore
jax.config.update("jax_numpy_dtype_promotion", "strict")  # pyright: ignore


@pytest.fixture()
def getkey():
    return eqxi.GetKey()
