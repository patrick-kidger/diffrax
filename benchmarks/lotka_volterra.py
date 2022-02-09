# Based on https://benchmarks.sciml.ai/html/MultiLanguage/wrapper_packages.html

import timeit

import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, Dopri8, ODETerm, PIDController, Tsit5


jax.config.update("jax_enable_x64", True)


def vector_field(t, y, args):
    a = 1.5
    b = -1
    c = -3
    d = 1
    x = y[0]
    y = y[1]
    return jnp.stack([a * x + b * x * y, c * y + d * x * y])


terms = ODETerm(vector_field)
t0 = 0
t1 = 10
y0 = jnp.array([1.0, 1.0])
dt0 = None

ref_sol = diffeqsolve(
    terms,
    Dopri8(),
    t0,
    t1,
    dt0,
    y0,
    stepsize_controller=PIDController(rtol=1e-14, atol=1e-14),
)


@jax.jit
def run(rtol, atol):
    sol = diffeqsolve(
        terms,
        Tsit5(),
        t0,
        t1,
        dt0,
        y0,
        stepsize_controller=PIDController(rtol=rtol, atol=atol),
    )
    return sol.ys, sol.stats["num_steps"]


def time_run(rtol, atol):
    ys, num_steps = run(rtol, atol)  # compile and get solution
    error = jnp.sqrt(jnp.sum((ys - ref_sol.ys) ** 2)).item()
    num_steps = num_steps.item()
    time = min(timeit.repeat(lambda: run(rtol, atol), repeat=100, number=1))
    print(f"error={error} time={time} num_steps={num_steps}")


rtols = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10)
atols = (1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13)
for rtol, atol in zip(rtols, atols):
    time_run(rtol, atol)
