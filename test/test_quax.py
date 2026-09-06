"""`diffeqsolve` under `quax.quaxify`.

Quax dispatches on *operations*. An array conjured from a shape rather than
from an operand -- `jnp.full(shape, ...)` -- gives it nothing to dispatch on,
so a wrapped `y0` is silently erased the moment a saved value round-trips
through the `SaveAt` buffer. These tests pin the buffer as an operation on
the value being saved.
"""

from collections.abc import Sequence

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.core import ShapedArray
from jax.extend.core import Primitive


# `quaxify(diffeqsolve)` additionally needs quax's `custom_vjp` support, which
# diffrax's buffered loops go through; that landed in quax 0.5.0.
quax = pytest.importorskip("quax", minversion="0.5.0")


class Boxed(quax.ArrayValue):
    """Propagates itself through any primitive it is an operand of.

    Boxes *inexact* outputs only. Boxing integers too would put the box on the
    solver's own loop counters and save indices, at which point essentially
    every primitive has a boxed operand and the buffer gets re-boxed on its
    first write -- which hides exactly the bug under test (and is why
    `sol.ts` came back wrapped in diffrax#438).
    """

    array: jax.Array = eqx.field(converter=jnp.asarray)

    def aval(self) -> ShapedArray:
        return ShapedArray(jnp.shape(self.array), jnp.result_type(self.array))

    def materialise(self) -> jax.Array:
        return self.array

    @staticmethod
    def default(primitive: Primitive, values: Sequence, params: dict):
        raw = [v.array if isinstance(v, Boxed) else v for v in values]
        out = primitive.bind(*raw, **params)

        def box(x):
            inexact = eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.inexact)
            return Boxed(x) if inexact else x

        return [box(x) for x in out] if primitive.multiple_results else box(out)


def _solve(y0, saveat):
    return diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: -0.5 * y),
        diffrax.Euler(),
        t0=0.0,
        t1=1.0,
        dt0=0.1,
        y0=y0,
        saveat=saveat,
        # `throw=True` erases the type again via equinox's `error_if`:
        # patrick-kidger/equinox#1257. Not a diffrax issue.
        throw=False,
    )


@pytest.mark.parametrize(
    "saveat",
    [
        diffrax.SaveAt(t1=True),
        diffrax.SaveAt(t0=True, t1=True),
        diffrax.SaveAt(ts=[0.0, 0.5, 1.0]),
        diffrax.SaveAt(steps=True),
    ],
    ids=["t1", "t0t1", "ts", "steps"],
)
def test_saveat_buffer_preserves_quax_type(saveat):
    y0 = jnp.array([1.0])
    expected = _solve(y0, saveat).ys
    assert expected is not None

    got = quax.quaxify(_solve)(Boxed(y0), saveat).ys

    assert isinstance(got, Boxed), f"`sol.ys` came back as {type(got).__name__}"
    assert jnp.array_equal(got.array, expected, equal_nan=True)
