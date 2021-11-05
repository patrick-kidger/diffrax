from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom

from ..misc import check_no_derivative, force_bitcast_convert_type
from .base import AbstractBrownianPath


class UnsafeBrownianPath(AbstractBrownianPath):
    """Brownian simulation that is only suitable for certain cases.

    This is a very quick way to simulate Brownian motion, but can only be used when all
    of the following are true:

    1. You are using a fixed step size controller. (Not an adaptive one.)

    2. If you are backpropagating, you are doing it with discretise-then-optimise.

    3. You do not need deterministic solutions. (This is susceptible to small variations
       in floating-point arithmetic.)

    Internally this operates by just sampling a fresh normal random variable over every
    interval, ignoring the correlation between samples exhibited in true Brownian
    motion. Hence the restrictions above. (They are when the correlation structure
    isn't needed.)

    **Arguments:**

    - **shape** (`tuple[int]`): What shape each individual Brownian sample should be.
    - **key** (`jax.random.PRNGKey`): A random key.
    """

    shape: Tuple[int] = eqx.static_field()
    key: jrandom.PRNGKey

    @property
    def t0(self):
        return None

    @property
    def t1(self):
        return None

    def evaluate(self, t0, t1):
        r"""Return a Brownian increment
        $w(t_1) - w(t_0) \sim \mathcal{N}(0, t_1 - t_0)$.

        **Arguments:**

        - **t0** (`Scalar`)
        - **t1** (`Scalar`)

        **Returns:**

        A JAX array of shape `shape`.
        """
        check_no_derivative(t0, "t0")
        check_no_derivative(t1, "t1")
        t0_ = force_bitcast_convert_type(t0, jnp.int32)
        t1_ = force_bitcast_convert_type(t1, jnp.int32)
        key = jrandom.fold_in(self.key, t0_)
        key = jrandom.fold_in(key, t1_)
        return jrandom.normal(key, self.shape) * jnp.sqrt(t1 - t0)
