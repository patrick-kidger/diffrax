import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import lineax as lx
import pytest
from diffrax import ControlTerm, MultiTerm, ODETerm, WeaklyDiagonalControlTerm


def _solvers():
    yield diffrax.SPaRK
    yield diffrax.GeneralShARK
    yield diffrax.SlowRK
    yield diffrax.ShARK
    yield diffrax.SRA1
    yield diffrax.SEA


# Define the SDE
def dict_drift(t, y, args):
    pytree, _ = args
    return jtu.tree_map(lambda _, x: -0.5 * x, pytree, y)


def dict_diffusion(t, y, args):
    pytree, additive = args

    def get_matrix(y_leaf):
        if additive:
            return 2.0 * jnp.ones(y_leaf.shape + (3,), dtype=jnp.float64)
        else:
            return 2.0 * jnp.broadcast_to(
                jnp.expand_dims(y_leaf, axis=y_leaf.ndim), y_leaf.shape + (3,)
            )

    return jtu.tree_map(get_matrix, y)


@pytest.mark.parametrize("shape", [(), (5, 2)])
@pytest.mark.parametrize("solver_ctr", _solvers())
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_sde_solver_shape(shape, solver_ctr, dtype):
    pytree = ({"a": 0, "b": [0, 0]}, 0, 0)
    key = jr.PRNGKey(0)
    y0 = jtu.tree_map(lambda _: jr.normal(key, shape, dtype=dtype), pytree)
    t0, t1, dt0 = 0.0, 1.0, 0.3

    # Some solvers only work with additive noise
    additive = solver_ctr in [diffrax.ShARK, diffrax.SRA1, diffrax.SEA]
    args = (pytree, additive)
    solver = solver_ctr()
    bmkey = jr.key(1)
    struct = jax.ShapeDtypeStruct((3,), dtype)
    bm_shape = jtu.tree_map(lambda _: struct, pytree)
    bm = diffrax.VirtualBrownianTree(
        t0, t1, 0.1, bm_shape, bmkey, diffrax.SpaceTimeLevyArea
    )
    terms = MultiTerm(ODETerm(dict_drift), ControlTerm(dict_diffusion, bm))
    solution = diffrax.diffeqsolve(
        terms, solver, t0, t1, dt0, y0, args, saveat=diffrax.SaveAt(t1=True)
    )
    assert jtu.tree_structure(solution.ys) == jtu.tree_structure(y0)
    for leaf in jtu.tree_leaves(solution.ys):
        assert leaf[0].shape == shape


def _weakly_diagonal_noise_helper(solver, dtype):
    w_shape = (3,)
    args = (0.5, 1.2)

    def _diffusion(t, y, args):
        a, b = args
        return jnp.array([b, t, 1 / (t + 1.0)], dtype=dtype)

    def _drift(t, y, args):
        a, b = args
        return -a * y

    y0 = jnp.ones(w_shape, dtype)

    bm = diffrax.VirtualBrownianTree(
        0.0, 1.0, 0.05, w_shape, jr.key(0), diffrax.SpaceTimeLevyArea
    )

    with pytest.warns(match="`WeaklyDiagonalControlTerm` is now deprecated"):
        diffusion = WeaklyDiagonalControlTerm(_diffusion, bm)
    terms = MultiTerm(ODETerm(_drift), diffusion)
    saveat = diffrax.SaveAt(t1=True)
    solution = diffrax.diffeqsolve(
        terms, solver, 0.0, 1.0, 0.1, y0, args, saveat=saveat
    )
    assert solution.ys is not None
    assert solution.ys.shape == (1, 3)


def _lineax_weakly_diagonal_noise_helper(solver, dtype):
    w_shape = (3,)
    args = (0.5, 1.2)

    def _diffusion(t, y, args):
        a, b = args
        return lx.DiagonalLinearOperator(jnp.array([b, t, 1 / (t + 1.0)], dtype=dtype))

    def _drift(t, y, args):
        a, b = args
        return -a * y

    y0 = jnp.ones(w_shape, dtype)

    bm = diffrax.VirtualBrownianTree(
        0.0, 1.0, 0.05, w_shape, jr.PRNGKey(0), diffrax.SpaceTimeLevyArea
    )

    terms = MultiTerm(ODETerm(_drift), ControlTerm(_diffusion, bm))
    saveat = diffrax.SaveAt(t1=True)
    solution = diffrax.diffeqsolve(
        terms, solver, 0.0, 1.0, 0.1, y0, args, saveat=saveat
    )
    assert solution.ys is not None
    assert solution.ys.shape == (1, 3)


@pytest.mark.parametrize("solver_ctr", _solvers())
@pytest.mark.parametrize(
    "dtype",
    (jnp.float64, jnp.complex128),
)
@pytest.mark.parametrize(
    "weak_type",
    ("old", "lineax"),
)
def test_weakly_diagonal_noise(solver_ctr, dtype, weak_type):
    if weak_type == "old":
        _weakly_diagonal_noise_helper(solver_ctr(), dtype)
    elif weak_type == "lineax":
        _lineax_weakly_diagonal_noise_helper(solver_ctr(), dtype)
    else:
        raise ValueError("Invalid weak_type")


@pytest.mark.parametrize(
    "dtype",
    (jnp.float64, jnp.complex128),
)
@pytest.mark.parametrize(
    "weak_type",
    ("old", "lineax"),
)
def test_halfsolver_term_compatible(dtype, weak_type):
    if weak_type == "old":
        _weakly_diagonal_noise_helper(diffrax.HalfSolver(diffrax.SPaRK()), dtype)
    elif weak_type == "lineax":
        _lineax_weakly_diagonal_noise_helper(diffrax.HalfSolver(diffrax.SPaRK()), dtype)
    else:
        raise ValueError("Invalid weak_type")
