###########
#
# This example trains a Neural ODE to reproduce a toy dataset of spirals.
#
###########

import functools as ft
import math
import time

import diffrax
import equinox as eqx
import fire
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy as jsp
import optax


###########
# We use Equinox as a convenient choice of neural network libary.
#
# It offers easy-to-use syntax without being a framework -- i.e. it interacts with
# normal JAX without any surprises -- so it's a trustworthy choice when trying to do
# something complicated, like interacting with a differential equation solve.
###########
class Func(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, in_size, out_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            key=key,
        )

    def __call__(self, t, y, args):
        return self.mlp(y)


class NeuralODE(eqx.Module):
    solver: diffrax.AbstractSolver
    stepsize_controller: diffrax.AbstractStepSizeController

    def __init__(self, in_size, out_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        fkey, ykey = jrandom.split(key, 2)
        self.solver = diffrax.tsit5(
            Func(in_size, out_size, width_size, depth, key=fkey)
        )
        ###########
        # unvmap_dt makes a whole batch use the same timestep sizes.
        # (Rather than per-batch-element adaptive time stepping.)
        #
        # This breaks the `vmap` abstraction slightly, but is a bit quicker.
        # Turn it off at the end of training if you like, but it offers a nice speedup
        # usually worth using at the start of training.
        ###########
        self.stepsize_controller = diffrax.IController(unvmap_dt=True)

    def __call__(self, y0, t):
        solution = diffrax.diffeqsolve(
            self.solver,
            t0=t[0],
            t1=t[-1],
            y0=y0,
            dt0=t[1] - t[0],
            stepsize_controller=self.stepsize_controller,
            ###########
            # Not saved as an attribute as we don't want to differentiate `t`.
            # (For simplicity our training loop differentiates all floating-point
            # arrays, corresponding to parameters of the model.)
            ###########
            saveat=diffrax.SaveAt(t=t),
        )
        return solution.ys


###########
# Toy dataset of spirals
###########
def get_data(dataset_size, *, key):
    theta = jrandom.uniform(key, (dataset_size,), minval=0, maxval=2 * math.pi)
    y0 = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    t = jnp.linspace(0, 25, 100)
    matrix = jnp.array([[-0.3, 2], [-2, -0.3]])
    y = jax.vmap(
        lambda y0i: jax.vmap(lambda ti: jsp.linalg.expm(ti * matrix) @ y0i)(t)
    )(y0)
    return t, y


def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def main(
    dataset_size=256,
    batch_size=32,
    learning_rate=3e-4,
    steps=100,
    width_size=8,
    depth=1,
    seed=5678,
):
    key = jrandom.PRNGKey(seed)
    data_key, model_key, loader_key = jrandom.split(key, 3)

    t, y = get_data(dataset_size, key=data_key)
    in_size = out_size = y.shape[-1]

    model = NeuralODE(in_size, out_size, width_size, depth, key=model_key)

    ###########
    # Training loop like normal.
    ###########

    @ft.partial(eqx.value_and_grad_f, filter_fn=eqx.is_inexact_array)
    def loss(model, y, t):
        # Setting an explicit axis_name works around a JAX bug that triggers
        # unnecessary re-JIT-ing in JAX version <= 0.2.18
        y_pred = jax.vmap(model, in_axes=(0, None), axis_name="")(y[:, 0], t)
        return jnp.mean((y - y_pred) ** 2)

    optim = optax.adam(learning_rate)
    opt_state = optim.init(
        jax.tree_map(lambda leaf: leaf if eqx.is_inexact_array(leaf) else None, model)
    )
    for step, (yi,) in zip(range(steps), dataloader((y,), batch_size, key=loader_key)):
        start = time.time()
        value, grads = loss(model, yi, t)
        end = time.time()
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        print(f"Step: {step}, Loss: {value}, Computation time: {end - start}")
    return value  # Final loss


if __name__ == "__main__":
    fire.Fire(main)
