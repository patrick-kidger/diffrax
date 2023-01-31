import diffrax
import jax.numpy as jnp
import jax.random as jrandom
from diffrax.misc.sde_kl_divergence import _kl_block_diffusion, sde_kl_divergence
from diffrax.term import ControlTerm


def test_weakly_diagonal(getkey):

    key = getkey()
    n = 2
    y0 = jnp.ones(shape=(n,))
    drift1 = lambda t, y, args: -y
    drift2 = lambda t, y, args: y
    diffusion = lambda t, y, args: jnp.ones(shape=(n,))
    bm = diffrax.UnsafeBrownianPath(shape=(n,), key=key)
    control = diffrax.WeaklyDiagonalControlTerm(diffusion, bm)
    aug_sde, aug_y0 = sde_kl_divergence(
        drift1=diffrax.ODETerm(drift1),
        drift2=diffrax.ODETerm(drift2),
        diffusion=control,
        y0=y0,
    )

    solver = diffrax.Euler()
    saveat = diffrax.SaveAt(ts=[1.0])
    sol = diffrax.diffeqsolve(
        aug_sde,
        solver=solver,
        saveat=saveat,
        t0=0.0,
        t1=1.0,
        y0=aug_y0,
        dt0=1e-2,
        adjoint=diffrax.NoAdjoint(),
    )
    assert len(sol.ys) == 2


def test_weakly_diagonal_small_diffusion(getkey):

    key = getkey()
    n = 2
    y0 = jnp.ones(shape=(n,))
    drift1 = lambda t, y, args: -y
    drift2 = lambda t, y, args: y
    diffusion = lambda t, y, args: jnp.ones(shape=(n,)) * 1e-8
    bm = diffrax.UnsafeBrownianPath(shape=(n,), key=key)
    control = diffrax.WeaklyDiagonalControlTerm(diffusion, bm)
    aug_sde, aug_y0 = sde_kl_divergence(
        drift1=diffrax.ODETerm(drift1),
        drift2=diffrax.ODETerm(drift2),
        diffusion=control,
        y0=y0,
    )
    solver = diffrax.Euler()
    saveat = diffrax.SaveAt(ts=[1.0])
    sol = diffrax.diffeqsolve(
        aug_sde,
        solver=solver,
        saveat=saveat,
        t0=0.0,
        t1=1.0,
        y0=aug_y0,
        dt0=1e-2,
        adjoint=diffrax.NoAdjoint(),
    )
    assert len(sol.ys) == 2


def test_general_diffusion(getkey):

    key = getkey()
    n, m = 2, 3
    y0 = jnp.ones(shape=(2,))
    drift1 = lambda t, y, args: -y
    drift2 = lambda t, y, args: y
    diffusion = lambda t, y, args: jnp.ones(shape=(n, m))
    bm = diffrax.UnsafeBrownianPath(shape=(m,), key=key)
    control = diffrax.ControlTerm(diffusion, bm)
    aug_sde, aug_y0 = sde_kl_divergence(
        drift1=diffrax.ODETerm(drift1),
        drift2=diffrax.ODETerm(drift2),
        diffusion=control,
        y0=y0,
    )
    solver = diffrax.Euler()
    saveat = diffrax.SaveAt(ts=[1.0])
    sol = diffrax.diffeqsolve(
        aug_sde,
        solver=solver,
        saveat=saveat,
        t0=0.0,
        t1=1.0,
        y0=aug_y0,
        dt0=1e-2,
        adjoint=diffrax.NoAdjoint(),
    )
    assert len(sol.ys) == 2


def test_block_matrix(getkey):

    key = getkey()

    y0 = (jnp.ones((2,)), jnp.ones((3,)))
    drift1 = lambda t, y, args: (jnp.ones((2,)), jnp.ones((3,)))
    drift2 = lambda t, y, args: (jnp.ones((2,)), jnp.ones((3,)))
    diffusion = lambda t, y, args: (jnp.ones((2,)), jnp.ones((3, 4)))

    # Here we define new type of AbstractPath
    class NewPath(diffrax.AbstractPath):

        path1: diffrax.AbstractPath
        path2: diffrax.AbstractPath

        def evaluate(self, t0, t1=None, left: bool = True):
            ret1 = self.path1.evaluate(t0, t1, left)
            ret2 = self.path2.evaluate(t0, t1, left)
            return (ret1, ret2)

    # Control term is also changed accordingly
    class NewControl(ControlTerm):
        @staticmethod
        def prod(vf, control):
            return (vf[0] * control[0], vf[1] @ control[1])

        def vf_prod(self, t, y, args, control):
            vf = self.vf(t, y)
            return (vf[0] * control[0], vf[1] @ control[1])

    keys = jrandom.split(key)
    path1 = diffrax.UnsafeBrownianPath(shape=(2,), key=keys[0])
    path2 = diffrax.UnsafeBrownianPath(shape=(4,), key=keys[1])
    path = NewPath(path1=path1, path2=path2)
    control = NewControl(diffusion, path)
    aug_sde, aug_y0 = sde_kl_divergence(
        drift1=diffrax.ODETerm(drift1),
        drift2=diffrax.ODETerm(drift2),
        diffusion=control,
        y0=y0,
    )
    solver = diffrax.Euler()
    saveat = diffrax.SaveAt(ts=[1.0])
    diffrax.diffeqsolve(
        aug_sde,
        solver=solver,
        saveat=saveat,
        t0=0.0,
        t1=1.0,
        y0=aug_y0,
        dt0=1e-2,
        adjoint=diffrax.NoAdjoint(),
    )


def test_block_diffusion():

    # case 1
    drift = (jnp.zeros((2,)), jnp.zeros((2,)))
    diffusion = (jnp.ones((2,)), jnp.ones((2, 3)))

    result = _kl_block_diffusion(drift=drift, diffusion=diffusion)
    assert result == 0.0

    # case 2
    drift = {
        "block1": jnp.zeros((2,)),
        "block2": jnp.zeros((2,)),
        "block3": jnp.zeros((3,)),
    }
    diffusion = {
        "block1": jnp.ones((2,)),
        "block2": jnp.ones((2, 3)),
        "block3": jnp.ones((3, 4)),
    }
    result = _kl_block_diffusion(drift=drift, diffusion=diffusion)
    assert result == 0.0

    # case 3
    drift = [jnp.zeros((2,)), jnp.zeros((2,)), {"block": jnp.zeros((3,))}]

    diffusion = [jnp.ones((2,)), jnp.ones((2, 3)), {"block": jnp.zeros((3, 4))}]

    result = _kl_block_diffusion(drift=drift, diffusion=diffusion)
    assert result == 0.0
