import random

import jax.config
import jax.random as jrandom
import pytest


jax.config.update("jax_enable_x64", True)


@pytest.fixture()
def getkey():
    def _getkey():
        # Not sure what the maximum actually is but this will do
        return jrandom.PRNGKey(random.randint(0, 2 ** 31 - 1))

    return _getkey
