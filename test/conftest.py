import equinox.internal as eqxi
import pytest
from jax import config


config.update("jax_enable_x64", True)
config.update("jax_numpy_rank_promotion", "raise")
config.update("jax_numpy_dtype_promotion", "strict")


@pytest.fixture()
def getkey():
    return eqxi.GetKey()
