import diffrax
import jax
import jax.numpy as jnp
import pytest

from .helpers import random_pytree, shaped_allclose, treedefs


def test_fill_forward():
    in_ = jnp.array([jnp.nan, 0.0, 1.0, jnp.nan, jnp.nan, 2.0, jnp.nan])
    out_ = jnp.array([jnp.nan, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0])
    fill_in = diffrax.misc.fill_forward(in_[:, None])
    assert shaped_allclose(fill_in, out_[:, None], equal_nan=True)


def test_ω_add_mul(getkey):
    # ω(...) initialisation
    ω = diffrax.misc.ω
    a = [0, 1]
    b = [1, 2]
    c = (ω(a) + ω(b)).ω
    assert c == [1, 3]

    # ...**ω initialisation
    for treedef in treedefs:
        a = b = c = random_pytree(getkey(), treedef)

        e1 = (a**ω * 2 + b**ω * c**ω - 3).ω
        e2 = jax.tree_map(lambda ai, bi, ci: ai * 2 + bi * ci - 3, a, b, c)
        assert shaped_allclose(e1, e2)


def test_ω_inplace(getkey):
    ω = diffrax.misc.ω
    for treedef in treedefs:
        a = random_pytree(getkey(), treedef)
        b1 = ω(a).at[()].set(3).ω
        b2 = jax.tree_map(lambda ai: ai.at[()].set(3), a)
        assert shaped_allclose(b1, b2)

        a2 = jax.tree_map(lambda x: x + 1, a)

        b3 = ω(a).at[()].set(ω(a2)).ω
        b4 = jax.tree_map(lambda ai, a2i: ai.at[()].set(a2i[()]), a, a2)
        assert shaped_allclose(b3, b4)


def test_ω_is_leaf(getkey):
    ω = diffrax.misc.ω
    for treedef in treedefs:
        a = b = random_pytree(getkey(), treedef)
        with pytest.raises(ValueError):
            ω(a) + ω(b, is_leaf=lambda x: isinstance(x, int))
        with pytest.raises(ValueError):
            ω(a, is_leaf=lambda x: isinstance(x, int)) + ω(b)
        with pytest.raises(ValueError):
            ω(a, is_leaf=lambda x: isinstance(x, int)) + ω(
                b, is_leaf=lambda x: isinstance(x, (int, str))
            )

        out = ω(a, is_leaf=lambda x: isinstance(x, int)) + ω(
            b, is_leaf=lambda x: isinstance(x, int)
        )
        assert out.is_leaf(4)
        assert not out.is_leaf("hi")

        b = ω(a, is_leaf=lambda x: isinstance(x, int)).at[()].set(3)
        assert out.is_leaf(4)
        assert not out.is_leaf("hi")

        a2 = jax.tree_map(lambda x: x + 1, a)

        c = (
            ω(a, is_leaf=lambda x: isinstance(x, int))
            .at[()]
            .set(ω(a2, is_leaf=lambda x: isinstance(x, int)))
        )
        assert c.is_leaf(4)
        assert not c.is_leaf("hi")

        with pytest.raises(ValueError):
            ω(a, is_leaf=lambda x: isinstance(x, int)).at[()].set(ω(a2))
        with pytest.raises(ValueError):
            ω(a).at[()].set(ω(a2, is_leaf=lambda x: isinstance(x, int)))


def test_unvmap():
    unvmap_all = diffrax.misc.unvmap_all
    unvmap_any = diffrax.misc.unvmap_any
    jit_unvmap_all = jax.jit(unvmap_all)
    jit_unvmap_any = jax.jit(unvmap_any)
    vmap_unvmap_all = jax.vmap(unvmap_all, out_axes=None)
    vmap_unvmap_any = jax.vmap(unvmap_any, out_axes=None)

    tt = jnp.array([True, True])
    tf = jnp.array([True, False])
    ff = jnp.array([False, False])

    assert jnp.array_equal(unvmap_all(tt), jnp.array(True))
    assert jnp.array_equal(unvmap_all(tf), jnp.array(False))
    assert jnp.array_equal(unvmap_all(ff), jnp.array(False))
    assert jnp.array_equal(unvmap_any(tt), jnp.array(True))
    assert jnp.array_equal(unvmap_any(tf), jnp.array(True))
    assert jnp.array_equal(unvmap_any(ff), jnp.array(False))

    assert jnp.array_equal(jit_unvmap_all(tt), jnp.array(True))
    assert jnp.array_equal(jit_unvmap_all(tf), jnp.array(False))
    assert jnp.array_equal(jit_unvmap_all(ff), jnp.array(False))
    assert jnp.array_equal(jit_unvmap_any(tt), jnp.array(True))
    assert jnp.array_equal(jit_unvmap_any(tf), jnp.array(True))
    assert jnp.array_equal(jit_unvmap_any(ff), jnp.array(False))

    assert jnp.array_equal(vmap_unvmap_all(tt), jnp.array(True))
    assert jnp.array_equal(vmap_unvmap_all(tf), jnp.array(False))
    assert jnp.array_equal(vmap_unvmap_all(ff), jnp.array(False))
    assert jnp.array_equal(vmap_unvmap_any(tt), jnp.array(True))
    assert jnp.array_equal(vmap_unvmap_any(tf), jnp.array(True))
    assert jnp.array_equal(vmap_unvmap_any(ff), jnp.array(False))

    unvmap_max = diffrax.misc.unvmap_max
    jit_unvmap_max = jax.jit(unvmap_max)
    vmap_unvmap_max = jax.vmap(unvmap_max, out_axes=None)

    _21 = jnp.array([2, 1])
    _11 = jnp.array([1, 1])

    assert jnp.array_equal(unvmap_max(_21), jnp.array(2))
    assert jnp.array_equal(unvmap_max(_11), jnp.array(1))

    assert jnp.array_equal(jit_unvmap_max(_21), jnp.array(2))
    assert jnp.array_equal(jit_unvmap_max(_11), jnp.array(1))

    assert jnp.array_equal(vmap_unvmap_max(_21), jnp.array(2))
    assert jnp.array_equal(vmap_unvmap_max(_11), jnp.array(1))


def test_nondifferentiable_input():
    ndi = lambda x: diffrax.misc.nondifferentiable_input(x, "name")
    ndi(
        jnp.array(
            2,
        )
    )  # no error
    with pytest.raises(ValueError):
        jax.jvp(ndi, (jnp.array(2.0),), (jnp.array(1.0),))
    with pytest.raises(ValueError):
        jax.grad(ndi)(jnp.array(2.0))


def test_nondifferentiable_output():
    ndo = diffrax.misc.nondifferentiable_output
    ndo(jnp.array(2.0))  # no error
    jax.jvp(ndo, (jnp.array(2.0),), (jnp.array(1.0),))  # no error
    with pytest.raises(RuntimeError):
        jax.grad(ndo)(jnp.array(2.0))
