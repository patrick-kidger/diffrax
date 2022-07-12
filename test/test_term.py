import types

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from .helpers import shaped_allclose


def test_ode_term():
    vector_field = lambda t, y, args: -y
    term = diffrax.ODETerm(vector_field)
    dt = term.contr(0, 1)
    vf = term.vf(0, 1, None)
    vf_prod = term.vf_prod(0, 1, None, dt)
    assert shaped_allclose(vf_prod, term.prod(vf, dt))


def test_control_term(getkey):
    vector_field = lambda t, y, args: jrandom.normal(args, (3, 2))
    derivkey = getkey()

    class Control(diffrax.AbstractPath):
        t0 = 0
        t1 = 1

        def evaluate(self, t0, t1):
            return jrandom.normal(getkey(), (2,))

        def derivative(self, t):
            return jrandom.normal(derivkey, (2,))

    control = Control()
    term = diffrax.ControlTerm(vector_field, control)
    args = getkey()
    dx = term.contr(0, 1)
    vf = term.vf(0, None, args)
    vf_prod = term.vf_prod(0, None, args, dx)
    assert dx.shape == (2,)
    assert vf.shape == (3, 2)
    assert vf_prod.shape == (3,)
    assert shaped_allclose(vf_prod, term.prod(vf, dx))

    term = term.to_ode()
    dt = term.contr(0, 1)
    vf = term.vf(0, None, args)
    vf_prod = term.vf_prod(0, None, args, dt)
    assert vf.shape == (3,)
    assert vf_prod.shape == (3,)
    assert shaped_allclose(vf_prod, term.prod(vf, dt))


def test_weakly_diagional_control_term(getkey):
    vector_field = lambda t, y, args: jrandom.normal(args, (3,))
    derivkey = getkey()

    class Control(diffrax.AbstractPath):
        t0 = 0
        t1 = 1

        def evaluate(self, t0, t1):
            return jrandom.normal(getkey(), (3,))

        def derivative(self, t):
            return jrandom.normal(derivkey, (3,))

    control = Control()
    term = diffrax.WeaklyDiagonalControlTerm(vector_field, control)
    args = getkey()
    dx = term.contr(0, 1)
    vf = term.vf(0, None, args)
    vf_prod = term.vf_prod(0, None, args, dx)
    assert dx.shape == (3,)
    assert vf.shape == (3,)
    assert vf_prod.shape == (3,)
    assert shaped_allclose(vf_prod, term.prod(vf, dx))

    term = term.to_ode()
    dt = term.contr(0, 1)
    vf = term.vf(0, None, args)
    vf_prod = term.vf_prod(0, None, args, dt)
    assert vf.shape == (3,)
    assert vf_prod.shape == (3,)
    assert shaped_allclose(vf_prod, term.prod(vf, dt))


def test_ode_adjoint_term(getkey):
    vector_field = lambda t, y, args: -y
    term = diffrax.ODETerm(vector_field)
    adjoint_term = diffrax.term.AdjointTerm(term)
    t, y, a_y, dt = jrandom.normal(getkey(), (4,))
    aug = (y, a_y, None, diffrax.ODETerm(None))
    args = None
    vf_prod1 = adjoint_term.vf_prod(t, aug, args, dt)
    vf = adjoint_term.vf(t, aug, args)
    vf_prod2 = adjoint_term.prod(vf, dt)
    assert shaped_allclose(vf_prod1, vf_prod2)


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
    control = types.SimpleNamespace(
        evaluate=lambda t0, t1: (jrandom.normal(getkey(), (3,)),)
    )
    term = diffrax.ControlTerm(vector_field, control)
    adjoint_term = diffrax.term.AdjointTerm(term)
    t = jrandom.normal(getkey(), ())
    y = (jrandom.normal(getkey(), (2,)),)
    args = (jrandom.normal(getkey(), (1,)), jrandom.normal(getkey(), (1,)))
    a_y = (jrandom.normal(getkey(), (2,)),)
    a_args = (jrandom.normal(getkey(), (1,)), jrandom.normal(getkey(), (1,)))
    randlike = lambda a: jrandom.normal(getkey(), a.shape)
    a_term = jax.tree_map(randlike, eqx.filter(term, eqx.is_array))
    aug = (y, a_y, a_args, a_term)
    dt = adjoint_term.contr(t, t + 1)

    vf_prod1 = adjoint_term.vf_prod(t, aug, args, dt)
    vf = adjoint_term.vf(t, aug, args)
    vf_prod2 = adjoint_term.prod(vf, dt)
    assert shaped_allclose(vf_prod1, vf_prod2)
