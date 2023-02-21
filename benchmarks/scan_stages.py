"""Benchmarks the effect of `diffrax.AbstractRungeKutta(scan_stages=...)`.

On my relatively beefy CPU-only machine:
```
scan_stages=True
Compile+run time 1.8253102810122073
Run time 0.00017526978626847267

scan_stages=False
Compile+run time 10.679616351146251
Run time 0.00021236995235085487
```
"""

import functools as ft
import timeit

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr


def _weight(in_, out, key):
    return [[w_ij for w_ij in w_i] for w_i in jr.normal(key, (out, in_))]


class VectorField(eqx.Module):
    weights: list

    def __init__(self, in_, out, width, depth, *, key):
        keys = jr.split(key, depth + 1)
        self.weights = [_weight(in_, width, keys[0])]
        for i in range(1, depth):
            self.weights.append(_weight(width, width, keys[i]))
        self.weights.append(_weight(width, out, keys[depth]))

    def __call__(self, t, y, args):
        # Inefficient computation graph to make a toy example more expensive.
        y = [y_i for y_i in y]
        for w in self.weights:
            y = [sum(w_ij * y_j for w_ij, y_j in zip(w_i, y)) for w_i in w]
        return jnp.stack(y)


def run(scan_stages):
    vf = VectorField(1, 1, 16, 2, key=jr.PRNGKey(0))
    term = dfx.ODETerm(vf)
    solver = dfx.Dopri8(scan_stages=scan_stages)
    stepsize_controller = dfx.PIDController(rtol=1e-3, atol=1e-6)
    t0 = 0
    t1 = 1
    dt0 = None

    @eqx.filter_jit
    def solve(y0):
        return dfx.diffeqsolve(
            term, solver, t0, t1, dt0, y0, stepsize_controller=stepsize_controller
        )

    solve_ = ft.partial(solve, jnp.array([1.0]))
    print(f"scan_stages={scan_stages}")
    print("Compile+run time", timeit.timeit(solve_, number=1))
    print("Run time", timeit.timeit(solve_, number=1))


run(scan_stages=True)
print()
run(scan_stages=False)
