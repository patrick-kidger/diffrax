import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from .helpers import tree_allclose


def test_ode_term():
    vector_field = lambda t, y, args: -y
    term = diffrax.ODETerm(vector_field)
    dt = term.contr(0, 1)
    vf = term.vf(0, 1, None)
    vf_prod = term.vf_prod(0, 1, None, dt)
    assert tree_allclose(vf_prod, term.prod(vf, dt))


def test_control_term(getkey):
    vector_field = lambda t, y, args: jr.normal(args, (3, 2))
    derivkey = getkey()

    class Control(diffrax.AbstractPath):
        t0 = 0
        t1 = 1

        def evaluate(self, t0, t1=None, left=True):
            return jr.normal(getkey(), (2,))

        def derivative(self, t, left=True):
            return jr.normal(derivkey, (2,))

    control = Control()  # pyright: ignore
    term = diffrax.ControlTerm(vector_field, control)
    args = getkey()
    dx = term.contr(0, 1)
    y = jnp.array([1.0, 2.0, 3.0])
    vf = term.vf(0, y, args)
    vf_prod = term.vf_prod(0, y, args, dx)
    assert dx.shape == (2,)
    assert vf.shape == (3, 2)
    assert vf_prod.shape == (3,)
    assert tree_allclose(vf_prod, term.prod(vf, dx))

    term = term.to_ode()
    dt = term.contr(0, 1)
    vf = term.vf(0, y, args)
    vf_prod = term.vf_prod(0, y, args, dt)
    assert vf.shape == (3,)
    assert vf_prod.shape == (3,)
    assert tree_allclose(vf_prod, term.prod(vf, dt))


def test_weakly_diagional_control_term(getkey):
    vector_field = lambda t, y, args: jr.normal(args, (3,))
    derivkey = getkey()

    class Control(diffrax.AbstractPath):
        t0 = 0
        t1 = 1

        def evaluate(self, t0, t1=None, left=True):
            return jr.normal(getkey(), (3,))

        def derivative(self, t, left=True):
            return jr.normal(derivkey, (3,))

    control = Control()  # pyright: ignore
    term = diffrax.WeaklyDiagonalControlTerm(vector_field, control)
    args = getkey()
    dx = term.contr(0, 1)
    y = jnp.array([1.0, 2.0, 3.0])
    vf = term.vf(0, y, args)
    vf_prod = term.vf_prod(0, y, args, dx)
    assert dx.shape == (3,)
    assert vf.shape == (3,)
    assert vf_prod.shape == (3,)
    assert tree_allclose(vf_prod, term.prod(vf, dx))

    term = term.to_ode()
    dt = term.contr(0, 1)
    vf = term.vf(0, y, args)
    vf_prod = term.vf_prod(0, y, args, dt)
    assert vf.shape == (3,)
    assert vf_prod.shape == (3,)
    assert tree_allclose(vf_prod, term.prod(vf, dt))


def test_ode_adjoint_term(getkey):
    vector_field = lambda t, y, args: -y
    term = diffrax.ODETerm(vector_field)
    adjoint_term = diffrax._term.AdjointTerm(term)
    t, y, a_y, dt = jr.normal(getkey(), (4,))
    ode_term = diffrax.ODETerm(lambda t, y, args: None)
    ode_term = eqx.tree_at(lambda m: m.vector_field, ode_term, None)
    aug = (y, a_y, None, ode_term)
    args = None
    vf_prod1 = adjoint_term.vf_prod(t, aug, args, dt)
    vf = adjoint_term.vf(t, aug, args)
    vf_prod2 = adjoint_term.prod(vf, dt)
    assert tree_allclose(vf_prod1, vf_prod2)


def test_cde_adjoint_term(getkey):
    class VF(eqx.Module):
        mlp: eqx.nn.MLP

        def __call__(self, t, y, args):
            (y,) = y
            in_ = jnp.concatenate([t[None], y, *args])
            out = self.mlp(in_)
            return (out.reshape(2, 3),)

    mlp = eqx.nn.MLP(in_size=5, out_size=6, width_size=3, depth=1, key=getkey())
    vector_field = VF(mlp)
    control = lambda t0, t1: (jr.normal(getkey(), (3,)),)
    term = diffrax.ControlTerm(vector_field, control)
    adjoint_term = diffrax._term.AdjointTerm(term)
    t = jr.normal(getkey(), ())
    y = (jr.normal(getkey(), (2,)),)
    args = (jr.normal(getkey(), (1,)), jr.normal(getkey(), (1,)))
    a_y = (jr.normal(getkey(), (2,)),)
    a_args = (jr.normal(getkey(), (1,)), jr.normal(getkey(), (1,)))
    randlike = lambda a: jr.normal(getkey(), a.shape)
    a_term = jtu.tree_map(randlike, eqx.filter(term, eqx.is_array))
    aug = (y, a_y, a_args, a_term)
    dt = adjoint_term.contr(t, t + 1)

    vf_prod1 = adjoint_term.vf_prod(t, aug, args, dt)
    vf = adjoint_term.vf(t, aug, args)
    vf_prod2 = adjoint_term.prod(vf, dt)
    assert tree_allclose(vf_prod1, vf_prod2)
