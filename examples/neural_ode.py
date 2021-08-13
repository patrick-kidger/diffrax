import functools as ft
import math

import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy as jsp
import optax


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
        self.stepsize_controller = diffrax.ConstantStepSize()

    def __call__(self, y0, t):
        sol = diffrax.diffeqsolve(
            self.solver,
            t0=t[0],
            t1=t[-1],
            y0=y0,
            dt0=t[1] - t[0],
            stepsize_controller=self.stepsize_controller,
            saveat=diffrax.SaveAt(t=t),
        )
        return sol.ys


def get_data(key, dataset_size):
    theta = jrandom.uniform(key, (dataset_size,), minval=0, maxval=2 * math.pi)
    y0 = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    t = jnp.linspace(0, 25, 100)
    matrix = jnp.array([[-0.3, 2], [-2, -0.3]])
    y = jax.vmap(
        lambda y0i: jax.vmap(lambda ti: jsp.linalg.expm(ti * matrix) @ y0i)(t)
    )(y0)
    return t, y


# Simple dataloader
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
    steps=1000,
    width_size=8,
    depth=1,
    seed=5678,
):
    key = jrandom.PRNGKey(seed)
    dkey, mkey, lkey = jrandom.split(key, 3)

    t, y = get_data(dkey, dataset_size)
    in_size = out_size = y.shape[-1]

    model = NeuralODE(in_size, out_size, width_size, depth, key=mkey)

    @ft.partial(eqx.value_and_grad_f, filter_fn=eqx.is_inexact_array)
    def loss(model, y, t):
        y_pred = jax.vmap(model, in_axes=(0, None))(y[:, 0], t)
        return jnp.mean((y - y_pred) ** 2)

    optim = optax.adam(learning_rate)
    opt_state = optim.init(
        jax.tree_map(lambda leaf: leaf if eqx.is_inexact_array(leaf) else None, model)
    )
    for step, (y,) in zip(range(steps), dataloader((y,), batch_size, key=lkey)):
        value, grads = loss(model, y, t)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        print(step, value)
    return value


if __name__ == "__main__":
    main()
