import jax
import jax.numpy as jnp
from typing import Optional, Tuple

from .custom_types import Array, PyTree, Scalar
from .jax_tricks import tree_dataclass
from .path import AbstractPath


@tree_dataclass
class AbstractInterpolation(AbstractPath):
    ts: Array["times"]  # noqa: F821

    requested_state = frozenset()

    def _interpret_t(self, t: Scalar) -> Tuple[Scalar, Scalar]:
        maxlen = self.ts.shape[0] - 2
        index = jnp.searchsorted(self.ts, t)
        index = jnp.clip(index - 1, a_min=0, a_max=maxlen)
        # Will never access the final element of `ts`; this is correct behaviour.
        fractional_part = t - self.ts[index]
        return index, fractional_part


@tree_dataclass
class LinearInterpolation(AbstractInterpolation):
    ys: PyTree

    def derivative(self, t: Scalar) -> PyTree:
        index, _ = self._interpret_t(t)
        return jax.tree_map(lambda _ys: (_ys[index + 1] - _ys[index]) / (self.ts[index + 1] - self.ts[index]), self.ys)

    def evaluate(self, t0: Scalar, t1: Optional[Scalar] = None) -> PyTree:
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)
        index, fractional_part = self._interpret_t(t0)
        prev_ys = jax.tree_map(lambda _ys: _ys[index], self.ys)
        next_ys = jax.tree_map(lambda _ys: _ys[index + 1], self.ys)
        prev_t = self.ts[index]
        next_t = self.ts[index + 1]
        diff_t = next_t - prev_t
        return jax.tree_map(
            lambda _prev_ys, _next_ys: _prev_ys + fractional_part * (_next_ys - _prev_ys) / diff_t, prev_ys, next_ys
        )


@tree_dataclass
class FourthOrderPolynomialInterpolation(AbstractInterpolation):
    ys: PyTree

    requested_state = frozenset({"k"})

    def derivative(self, t: Scalar) -> PyTree:
        ...

    def evaluate(self, t0: Scalar, t1: Optional[Scalar] = None) -> PyTree:
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)
        ...
