import diffrax
import jax.numpy as jnp


def test_weakly_diagonal(getkey):

    key = getkey()
    n = 2
    y0 = jnp.ones(shape=(n,))
    drift1 = lambda t, y, args: -y
    drift2 = lambda t, y, args: y
    diffusion = lambda t, y, args: jnp.ones(shape=(n,))
    bm = diffrax.UnsafeBrownianPath(shape=(n,), key=key)
    control = diffrax.WeaklyDiagonalControlTerm(diffusion, bm)
    sde1 = diffrax.MultiTerm(diffrax.ODETerm(drift1), control)
    sde2 = diffrax.MultiTerm(diffrax.ODETerm(drift2), control)
    aug_sde, aug_y0 = diffrax.misc.sde_kl_divergence(
        sde1=sde1, sde2=sde2, context=None, y0=y0
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
    sde1 = diffrax.MultiTerm(diffrax.ODETerm(drift1), control)
    sde2 = diffrax.MultiTerm(diffrax.ODETerm(drift2), control)
    aug_sde, aug_y0 = diffrax.misc.sde_kl_divergence(
        sde1=sde1, sde2=sde2, context=None, y0=y0
    )
    solver = diffrax.Euler()
    saveat = diffrax.SaveAt(ts=[1.0])
    sol = diffrax.diffeqsolve(
        aug_sde, solver=solver, saveat=saveat, t0=0.0, t1=1.0, y0=aug_y0, dt0=1e-2
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
    sde1 = diffrax.MultiTerm(diffrax.ODETerm(drift1), control)
    sde2 = diffrax.MultiTerm(diffrax.ODETerm(drift2), control)
    aug_sde, aug_y0 = diffrax.misc.sde_kl_divergence(
        sde1=sde1, sde2=sde2, context=None, y0=y0
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
