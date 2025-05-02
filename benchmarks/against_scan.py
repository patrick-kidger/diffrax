# This benchmark should be ran on the GPU.

import timeit

import diffrax as dfx
import jax
import jax.lax as lax
import jax.numpy as jnp


# SETUP

N = 256
N_steps = 2000
ts = jnp.linspace(0, 1, N_steps + 1)
u0, v0 = jnp.zeros((N, N)), jnp.zeros((N, N)).at[32, 32].set(1.0)
fields = (u0, v0)
du = lambda t, v, args: -(v**2)
dv = lambda t, u, args: -jnp.fft.irfft(jnp.sin(jnp.fft.rfft(u)))
sample = lambda t, y, args: y[0][64, 64]  # Some arbitrary sampling function


def speedtest(fn, name):
    fwd = jax.jit(fn)
    bwd = jax.jit(jax.grad(fn))
    integration_times = timeit.repeat(
        lambda: jax.block_until_ready(fwd(fields, ts)), number=1, repeat=10
    )
    print(f"{name} fwd: {min(integration_times)}")
    grad_times = timeit.repeat(
        lambda: jax.block_until_ready(bwd(fields, ts)), number=1, repeat=10
    )
    print(f"{name} fwd+bwd: {min(grad_times)}")


# INTEGRATE WITH scan


@jax.checkpoint  # pyright: ignore
def body(carry, t):
    u, v, dt = carry
    u = u + du(t, v, None) * dt
    v = v + dv(t, u, None) * dt
    return (u, v, dt), sample(t, (u, v), None)


def scan_fn(fields, t):
    dt = t[1] - t[0]
    carry = (fields[0], fields[1], dt)
    _, values = lax.scan(body, carry, t[:-1])
    return jnp.mean(values**2)


speedtest(scan_fn, "scan")


# INTEGRATE WITH SemiImplicitEuler


@jax.jit
def dfx_fn(fields, t):
    return dfx.diffeqsolve(
        terms=(dfx.ODETerm(du), dfx.ODETerm(dv)),
        solver=dfx.SemiImplicitEuler(),
        t0=t[0],
        t1=t[-1],
        dt0=None,
        y0=fields,
        args=None,
        saveat=dfx.SaveAt(steps=True, fn=sample, dense=False),
        stepsize_controller=dfx.StepTo(ts),
        adjoint=dfx.RecursiveCheckpointAdjoint(checkpoints=N_steps),
        max_steps=N_steps,
        throw=False,
    ).ys


speedtest(dfx_fn, "SemiImplicitEuler")
