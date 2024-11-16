from typing import cast

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Scalar, ScalarLike

from .._custom_types import Y
from .._search import AbstractSearch, FunctionInfo
from .._solution import RESULTS


class LearningRate(AbstractSearch[Y, FunctionInfo, FunctionInfo, None], strict=True):
    """Move downhill by taking a step of the fixed size `learning_rate`."""

    learning_rate: ScalarLike = eqx.field(converter=jnp.asarray)

    def init(self, y: Y, f_info_struct: FunctionInfo) -> None:
        return None

    def step(
        self,
        first_step: Bool[Array, ""],
        y: Y,
        y_eval: Y,
        f_info: FunctionInfo,
        f_eval_info: FunctionInfo,
        state: None,
    ) -> tuple[Scalar, Bool[Array, ""], RESULTS, None]:
        del first_step, y, y_eval, f_info, f_eval_info, state
        learning_rate = cast(Array, self.learning_rate)
        return learning_rate, jnp.array(True), RESULTS.successful, None


LearningRate.__init__.__doc__ = """**Arguments:**

- `learning_rate`: The fixed step-size used at each step.
"""
