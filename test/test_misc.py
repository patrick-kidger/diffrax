import diffrax
import jax.numpy as jnp
import pytest

from .helpers import tree_allclose


def test_fill_forward():
    in_ = jnp.array([jnp.nan, 0.0, 1.0, jnp.nan, jnp.nan, 2.0, jnp.nan])
    out_ = jnp.array([jnp.nan, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0])
    fill_in = diffrax._misc.fill_forward(in_[:, None])
    assert tree_allclose(fill_in, out_[:, None], equal_nan=True)


def test_weaklydiagonal_deprecate():
    with pytest.warns(
        PendingDeprecationWarning,
        match="WeaklyDiagonalControlTerm is pending deprecation",
    ):
        _ = diffrax.WeaklyDiagonalControlTerm(
            lambda t, y, args: 0.0, lambda t0, t1: jnp.array(t1 - t0)
        )
