"""Benchmarks Diffrax vs torchdiffeq vs jax.experimental.ode.odeint"""

import gc
import time
from typing import cast

import diffrax
import equinox as eqx
import jax
import jax.experimental.ode as experimental
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import torch  # pyright: ignore
import torchdiffeq  # pyright: ignore
from jaxtyping import Array


class FuncTorch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.func = torch.jit.script(  # pyright: ignore
            torch.nn.Sequential(
                torch.nn.Linear(4, 32),
                torch.nn.Softplus(),
                torch.nn.Linear(32, 4),
                torch.nn.Tanh(),
            )
        )

    def forward(self, t, y):
        return self.func(y)  # pyright: ignore


class FuncJax(eqx.Module):
    func: eqx.nn.MLP

    def __init__(self):
        self.func = eqx.nn.MLP(
            in_size=4,
            out_size=4,
            width_size=32,
            depth=1,
            activation=jnn.softplus,
            final_activation=jnn.tanh,
            key=jr.PRNGKey(0),
        )

    def __call__(self, t, y, args):
        return jax.vmap(self.func)(y)


class NeuralODETorch(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.func = FuncTorch()
        self.multiple = multiple

    def forward(self, y0, t1):
        if self.multiple:
            t = torch.linspace(0, t1, 100, device=y0.device)
        else:
            t = torch.tensor([0.0, t1], device=y0.device)
        y = torchdiffeq.odeint(self.func, y0, t, method="dopri5", rtol=1e-6, atol=1e-6)
        return y.sum()


class NeuralODEDiffrax(eqx.Module):
    func: FuncJax
    stepsize_controller: diffrax.AbstractStepSizeController
    multiple: bool

    def __init__(self, multiple):
        self.func = FuncJax()
        self.stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-6)
        self.multiple = multiple

    def __call__(self, y0, t1):
        term = diffrax.ODETerm(self.func)
        if self.multiple:
            saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, t1, 100))
        else:
            saveat = diffrax.SaveAt(t0=True, t1=True)
        sol = diffrax.diffeqsolve(
            term,
            t0=0.0,
            t1=t1,
            y0=y0,
            dt0=None,
            solver=diffrax.Dopri5(),
            stepsize_controller=self.stepsize_controller,
            saveat=saveat,
            adjoint=diffrax.BacksolveAdjoint(),
        )
        return jnp.sum(cast(Array, sol.ys))


class NeuralODEExperimental(eqx.Module):
    func: FuncJax
    multiple: bool

    def __init__(self, multiple):
        self.func = FuncJax()
        self.multiple = multiple

    def __call__(self, y0, t1):
        if self.multiple:
            t = jnp.linspace(0, t1, 100)
        else:
            t = jnp.array([0.0, t1])

        args, static = eqx.partition(self.func, eqx.is_array)

        def _experimental_func(_y, _t, _args):
            _func = eqx.combine(_args, static)
            return _func(_t, _y, None)

        out = experimental.odeint(_experimental_func, y0, t, args, rtol=1e-6, atol=1e-6)
        return jnp.sum(out)


def timed(fn):
    def _timer(*args, **kwargs):
        gcold = gc.isenabled()
        if gcold:
            gc.collect()
        gc.disable()
        start = time.perf_counter_ns()
        try:
            fn(*args, **kwargs)
            end = time.perf_counter_ns()
        finally:
            if gcold:
                gc.enable()
                gc.collect()
        return end - start

    return _timer


@timed
def time_torch(neural_ode_torch, y0, t1, grad):
    if grad:
        y0 = y0.detach().requires_grad_()
        neural_ode_torch(y0, t1).backward()
    else:
        neural_ode_torch(y0, t1)


@eqx.filter_jit
@eqx.filter_value_and_grad
def _grad_jax(arg, t1):
    neural_ode_jax, y0 = arg
    return neural_ode_jax(y0, t1)


@eqx.filter_jit
def _eval_jax(neural_ode_jax, y0, t1):
    return neural_ode_jax(y0, t1)


@timed
def time_jax(neural_ode_jax, y0, t1, grad):
    if grad:
        arg = (neural_ode_jax, y0)
        _grad_jax(arg, t1)
    else:
        _eval_jax(neural_ode_jax, y0, t1)


def run(multiple, grad, batch_size=64, t1=100):
    neural_ode_torch = NeuralODETorch(multiple)
    neural_ode_diffrax = NeuralODEDiffrax(multiple)
    neural_ode_experimental = NeuralODEExperimental(multiple)

    with torch.no_grad():
        func_jax = neural_ode_diffrax.func.func
        func_torch = neural_ode_torch.func.func
        func_torch[0].weight.copy_(torch.tensor(np.asarray(func_jax.layers[0].weight)))  # pyright: ignore
        func_torch[0].bias.copy_(torch.tensor(np.asarray(func_jax.layers[0].bias)))  # pyright: ignore
        func_torch[2].weight.copy_(torch.tensor(np.asarray(func_jax.layers[1].weight)))  # pyright: ignore
        func_torch[2].bias.copy_(torch.tensor(np.asarray(func_jax.layers[1].bias)))  # pyright: ignore

    y0_jax = jr.normal(jr.PRNGKey(1), (batch_size, 4))
    y0_torch = torch.tensor(np.asarray(y0_jax))

    time_torch(neural_ode_torch, y0_torch, t1, grad)
    torch_time = time_torch(neural_ode_torch, y0_torch, t1, grad)

    time_jax(neural_ode_diffrax, jnp.copy(y0_jax), t1, grad)
    diffrax_time = time_jax(neural_ode_diffrax, jnp.copy(y0_jax), t1, grad)

    time_jax(neural_ode_experimental, jnp.copy(y0_jax), t1, grad)
    experimental_time = time_jax(neural_ode_experimental, jnp.copy(y0_jax), t1, grad)

    print(
        f"""  multiple={multiple}, grad={grad}
       torch_time={torch_time}
     diffrax_time={diffrax_time}
experimental_time={experimental_time}
    """
    )


if __name__ == "__main__":
    run(multiple=False, grad=False)
    run(multiple=True, grad=False)
    run(multiple=False, grad=True)
    run(multiple=True, grad=True)
