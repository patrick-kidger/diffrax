"""Benchmarks the effect of `diffrax.AbstractRungeKutta(scan_stages=...)`.

On my CPU-only machine:
```
bash> python scan_stages_cnf.py --scan_stages=False --backsolve=False
Compile+run time 79.18114789901301
Run time 0.16631506383419037

bash> python scan_stages_cnf.py --scan_stages=False --backsolve=True
Compile+run time 28.233896102989092
Run time 0.021237157052382827

bash> python scan_stages_cnf.py --scan_stages=True --backsolve=False
Compile+run time 37.9795492868870
Run time 0.16300765215419233

bash> python scan_stages_cnf.py --scan_stages=True --backsolve=True
Compile+run time 12.199542510090396
Run time 0.024600893026217818
```

(Not forgetting that --backsolve=True produces only approximate gradients, so the fact
that it obtains better compile time and run time doesn't mean it's always the best
choice.)
"""

# This benchmark is adapted from
# https://github.com/patrick-kidger/diffrax/issues/94#issuecomment-1140527134

import functools as ft
import timeit

import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp


def vector_field_prob(t, input, model):
    y, _ = input
    f, vjp_fn = jax.vjp(model, y)
    (size,) = y.shape
    eye = jnp.eye(size)
    (dfdy,) = jax.vmap(vjp_fn)(eye)
    logp = jnp.trace(dfdy)
    return f, logp


@eqx.filter_vmap(in_axes=(None, 0, None, None))
def log_prob(model, y0, scan_stages, backsolve):
    term = diffrax.ODETerm(vector_field_prob)
    solver = diffrax.Dopri5(scan_stages=scan_stages)
    stepsize_controller = diffrax.PIDController(rtol=1.4e-8, atol=1.4e-8)
    if backsolve:
        adjoint = diffrax.BacksolveAdjoint()
    else:
        adjoint = diffrax.RecursiveCheckpointAdjoint()
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=0.5,
        dt0=0.05,
        y0=(y0, 0.0),
        args=model,
        stepsize_controller=stepsize_controller,
        adjoint=adjoint,
    )
    (y1,), (log_prob,) = sol.ys
    return log_prob + jsp.stats.norm.logpdf(y1).sum(0)


@eqx.filter_jit
@eqx.filter_grad
def solve(model, inputs, scan_stages, backsolve):
    return -log_prob(model, inputs, scan_stages, backsolve).mean()


def run(scan_stages, backsolve):
    mkey, dkey = jr.split(jr.PRNGKey(0), 2)
    model = eqx.nn.MLP(2, 2, 10, 2, activation=jnn.gelu, key=mkey)
    x = jr.normal(dkey, (256, 2))
    solve2 = ft.partial(solve, model, x, scan_stages, backsolve)
    print(f"scan_stages={scan_stages}, backsolve={backsolve}")
    print("Compile+run time", timeit.timeit(solve2, number=1))
    print("Run time", timeit.timeit(solve2, number=1))
    print()


run(scan_stages=False, backsolve=False)
run(scan_stages=False, backsolve=True)
run(scan_stages=True, backsolve=False)
run(scan_stages=True, backsolve=True)
