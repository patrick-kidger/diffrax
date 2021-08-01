from typing import Optional, Tuple, Type

import jax
import jax.numpy as jnp

from .custom_types import Array, DenseInfo, PyTree, Scalar, SquashTreeDef
from .local_interpolation import AbstractLocalInterpolation
from .misc import tree_unsquash
from .path import AbstractPath


class AbstractGlobalInterpolation(AbstractPath):
    ts: Array["times"]  # noqa: F821

    def _interpret_t(self, t: Scalar, left: bool) -> Tuple[Scalar, Scalar]:
        maxlen = self.ts.shape[0] - 2
        index = jnp.searchsorted(self.ts, t, side="left" if left else "right")
        index = jnp.clip(index - 1, a_min=0, a_max=maxlen)
        # Will never access the final element of `ts`; this is correct behaviour.
        fractional_part = t - self.ts[index]
        return index, fractional_part


class LinearInterpolation(AbstractGlobalInterpolation):
    ys: PyTree

    def derivative(self, t: Scalar, left: bool = True) -> PyTree:
        index, _ = self._interpret_t(t, left)
        return jax.tree_map(
            lambda _ys: (_ys[index + 1] - _ys[index])
            / (self.ts[index + 1] - self.ts[index]),
            self.ys,
        )

    def evaluate(
        self, t0: Scalar, t1: Optional[Scalar] = None, left: bool = True
    ) -> PyTree:
        if t1 is not None:
            return self.evaluate(t1, left=left) - self.evaluate(t0, left=left)
        index, fractional_part = self._interpret_t(t0, left)
        prev_ys = jax.tree_map(lambda _ys: _ys[index], self.ys)
        next_ys = jax.tree_map(lambda _ys: _ys[index + 1], self.ys)
        prev_t = self.ts[index]
        next_t = self.ts[index + 1]
        diff_t = next_t - prev_t
        return jax.tree_map(
            lambda _prev_ys, _next_ys: _prev_ys
            + fractional_part * (_next_ys - _prev_ys) / diff_t,
            prev_ys,
            next_ys,
        )


class DenseInterpolation(AbstractGlobalInterpolation):
    interpolation_cls: Type[AbstractLocalInterpolation]
    infos: DenseInfo
    y_treedef: SquashTreeDef

    def _get_local_interpolation(self, t: Scalar, left: bool):
        index, _ = self._interpret_t(t, left)
        prev_t = self.ts[index]
        next_t = self.ts[index + 1]
        infos = jax.tree_map(lambda _d: _d[index], self.infos)
        return self.interpolation_cls(t0=prev_t, t1=next_t, **infos)

    def derivative(self, t: Scalar, left: bool = True) -> PyTree:
        # Passing `left` doesn't matter on a local interpolation, which is globally
        # continuous.
        return tree_unsquash(
            self.y_treedef, self._get_local_interpolation(t, left).derivative(t)
        )

    def evaluate(
        self, t0: Scalar, t1: Optional[Scalar] = None, left: bool = True
    ) -> PyTree:
        if t1 is not None:
            return self.evaluate(t1, left=left) - self.evaluate(t0, left=left)
        # Passing `left` doesn't matter on a local interpolation, which is globally
        # continuous.
        return tree_unsquash(
            self.y_treedef, self._get_local_interpolation(t0, left).evaluate(t0)
        )
