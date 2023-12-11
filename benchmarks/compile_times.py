import functools as ft
import timeit
from typing import cast

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array


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


def run(inline: bool, grad: bool, adjoint_name: str):
    if adjoint_name == "direct":
        adjoint = dfx.DirectAdjoint()
    elif adjoint_name == "recursive":
        adjoint = dfx.RecursiveCheckpointAdjoint()
    elif adjoint_name == "backsolve":
        adjoint = dfx.BacksolveAdjoint()
    else:
        raise ValueError
    if grad:
        grad_decorator = jax.grad
    else:
        grad_decorator = lambda x: x

    vf = VectorField(1, 1, 16, 2, key=jr.PRNGKey(0))
    if not inline:
        vf = eqx.internal.noinline(vf)
    term = dfx.ODETerm(vf)
    solver = dfx.Dopri8()
    stepsize_controller = dfx.PIDController(rtol=1e-3, atol=1e-6)
    t0 = 0
    t1 = 1
    dt0 = 0.01

    @jax.jit
    @grad_decorator
    def solve(y0):
        sol = dfx.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            y0,
            stepsize_controller=stepsize_controller,
            adjoint=adjoint,
            max_steps=16**2,
        )
        return jnp.sum(cast(Array, sol.ys))

    solve_ = ft.partial(solve, jnp.array([1.0]))
    compile_time = timeit.timeit(solve_, number=1)
    print(f"{inline=}, {grad=}, adjoint={adjoint_name}, {compile_time=}")


run(inline=False, grad=False, adjoint_name="direct")
run(inline=False, grad=False, adjoint_name="recursive")
run(inline=False, grad=False, adjoint_name="backsolve")

run(inline=False, grad=True, adjoint_name="direct")
run(inline=False, grad=True, adjoint_name="recursive")
run(inline=False, grad=True, adjoint_name="backsolve")

run(inline=True, grad=False, adjoint_name="direct")
run(inline=True, grad=False, adjoint_name="recursive")
run(inline=True, grad=False, adjoint_name="backsolve")

run(inline=True, grad=True, adjoint_name="direct")
run(inline=True, grad=True, adjoint_name="recursive")
run(inline=True, grad=True, adjoint_name="backsolve")
