import equinox.internal as eqxi
import jax.config
import pytest


jax.config.update("jax_enable_x64", True)


@pytest.fixture()
def getkey():
    return eqxi.GetKey()
