import math
import pathlib
import time
from typing import List

import diffrax
import equinox as eqx
import fire
import imageio
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import optax


here = pathlib.Path(__file__).resolve().parent


def normal_log_likelihood(y):
    # Omits the constant factor
    return -0.5 * (y.size * math.log(2 * math.pi) + jnp.sum(y ** 2))


def approx_logp_wrapper(t, y, args):
    y, _ = y
    *args, eps, func = args
    fn = lambda y: func(t, y, args)
    f, vjp_fn = jax.vjp(fn, y)
    (eps_dfdy,) = vjp_fn(eps)
    logp = jnp.sum(eps_dfdy * eps)
    return f, logp


def exact_logp_wrapper(t, y, args):
    y, _ = y
    *args, _, func = args
    fn = lambda y: func(t, y, args)
    f, vjp_fn = jax.vjp(fn, y)
    (size,) = y.shape  # this implementation only works for 1D input
    eye = jnp.eye(size)
    (dfdy,) = jax.vmap(vjp_fn, axis_name='')(eye)
    logp = jnp.trace(dfdy)
    return f, logp


# This layer taken from the FFJORD repo.
class ConcatSquash(eqx.Module):
    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear
    lin3: eqx.nn.Linear

    def __init__(self, *, in_size, out_size, key, **kwargs):
        super().__init__(**kwargs)
        key1, key2, key3 = jrandom.split(key, 3)
        self.lin1 = eqx.nn.Linear(in_size, out_size, key=key)
        self.lin2 = eqx.nn.Linear(1, out_size, key=key)
        self.lin3 = eqx.nn.Linear(1, out_size, use_bias=False, key=key)

    def __call__(self, t, y):
        return self.lin1(y) * jnn.sigmoid(self.lin2(t)) + self.lin3(t)


class ConcatSquashMLP(eqx.Module):
    layers: List[eqx.nn.Linear]

    def __init__(self, *, data_size, width_size, depth, key, **kwargs):
        super().__init__(**kwargs)
        keys = jrandom.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(ConcatSquash(in_size=data_size, out_size=data_size, key=keys[0]))
        else:
            layers.append(ConcatSquash(in_size=data_size, out_size=width_size, key=keys[0]))
            for i in range(depth - 1):
                layers.append(ConcatSquash(in_size=width_size, out_size=width_size, key=keys[i + 1]))
            layers.append(ConcatSquash(in_size=width_size, out_size=data_size, key=keys[-1]))
        self.layers = layers

    def __call__(self, t, y):
        for layer in self.layers[:-1]:
            y = layer(t, y)
            y = jnn.softplus(y)
        y = self.layers[-1](t, y)
        return y



class Func(eqx.Module):
    layer: eqx.Module
    layer_type: str

    def __init__(
        self, *, data_size, width_size, depth, layer_type, key, **kwargs
    ):
        super().__init__(**kwargs)
        self.layer_type = layer_type
        if layer_type == "mlp":
            self.layer = eqx.nn.MLP(
                in_size=data_size + 1,
                out_size=data_size,
                width_size=width_size,
                depth=depth,
                activation=jnn.softplus,
                key=key,
            )
        elif layer_type == "concatsquash":
            self.layer = ConcatSquashMLP(data_size=data_size, width_size=width_size, depth=depth, key=key)
        elif layer_type == "stableflow":
            self.layer = eqx.nn.MLP(
                in_size=data_size + 1,
                out_size=1,
                width_size=width_size,
                depth=depth,
                activation=jnn.softplus,
                key=key,
            )
        else:
            raise ValueError

    def __call__(self, t, y, args):
        t = jnp.asarray(t)[None]
        if self.layer_type == "mlp":
            inp = jnp.concatenate([t, y])
            return self.layer(inp)
        elif self.layer_type == "concatsquash":
            return self.layer(t, y)
        elif self.layer_type == "stableflow":
            fn = lambda y: jnp.sqrt(1 + self.layer(jnp.concatenate([t, y]))[0] ** 2)
            return jax.grad(fn)(y)
        else:
            raise RuntimeError


class CNF(eqx.Module):
    funcs: List[Func]
    data_size: int
    exact_logp: bool
    t0: float
    t1: float
    dt0: float

    def __init__(
        self,
        *,
        data_size,
        exact_logp,
        num_blocks,
        width_size,
        depth,
        layer_type,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)
        keys = jrandom.split(key, num_blocks)
        self.funcs = [
            Func(
                data_size=data_size,
                width_size=width_size,
                depth=depth,
                layer_type=layer_type,
                key=k,
            )
            for k in keys
        ]
        self.data_size = data_size
        self.exact_logp = exact_logp
        self.t0 = 0.0
        self.t1 = 1.0
        self.dt0 = 0.1

    def train(self, y, *, key):
        solver = diffrax.tsit5(
            exact_logp_wrapper if self.exact_logp else approx_logp_wrapper
        )
        eps = jrandom.normal(key, y.shape)
        delta_log_likelihood = 0.0
        for func in reversed(self.funcs):
            y = (y, delta_log_likelihood)
            sol = diffrax.diffeqsolve(
                solver, self.t1, self.t0, y, -self.dt0, (eps, func)
            )
            (y,), (delta_log_likelihood,) = sol.ys
        return delta_log_likelihood + normal_log_likelihood(y)

    def sample(self, *, key):
        y = jrandom.normal(key, (self.data_size,))
        for func in self.funcs:
            solver = diffrax.tsit5(func)
            sol = diffrax.diffeqsolve(solver, self.t0, self.t1, y, self.dt0)
            (y,) = sol.ys
        return y


def get_data(path):
    # integer array of shape (height, width, channels) with values in {0, ..., 255}
    img = jnp.asarray(imageio.imread(path))
    height, width, channels = img.shape
    assert channels == 3
    # Convert to greyscale for simplicity.
    img = img @ jnp.array([0.2989, 0.5870, 0.1140])
    img = jnp.transpose(img)[:, ::-1]  # (width, height)
    x = jnp.arange(width, dtype=jnp.float32)
    y = jnp.arange(height, dtype=jnp.float32)
    x, y = jnp.broadcast_arrays(x[:, None], y[None, :])
    weights = 1 - img.reshape(-1).astype(jnp.float32) / jnp.max(img)
    dataset = jnp.stack(
        [x.reshape(-1), y.reshape(-1)], axis=-1
    )  # shape (dataset_size, 2)
    # For efficiency we don't bother with the particles that will have weight zero.
    cond = img.reshape(-1) < 254
    dataset = dataset[cond]
    weights = weights[cond]
    mean = jnp.mean(dataset, axis=0)
    std = jnp.std(dataset, axis=0) + 1e-6
    dataset = (dataset - mean) / std

    return dataset, weights, mean, std, width, height


def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while start < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def main(
    in_path=here / "cnf_in.png",
    out_path=here / "cnf_out.png",
    batch_size=500,
    virtual_batches=2,
    optim_name="adam",
    lr=1e-4,
    steps=10000,
    lookahead=False,
    exact_logp=True,
    num_blocks=2,
    width_size=2048,
    depth=2,
    layer_type="concatsquash",
    seed=5678,
):
    key = jrandom.PRNGKey(seed)
    model_key, loader_key, train_key, sample_key = jrandom.split(key, 4)

    dataset, weights, mean, std, width, height = get_data(in_path)
    dataset_size, data_size = dataset.shape
    data_iter = iter(dataloader((dataset, weights), batch_size, key=loader_key))

    model = CNF(
        data_size=data_size,
        exact_logp=exact_logp,
        num_blocks=num_blocks,
        width_size=width_size,
        depth=depth,
        layer_type=layer_type,
        key=model_key,
    )
    params, static, which, treedef = eqx.split(model, filter_fn=eqx.is_inexact_array)

    @jax.value_and_grad
    def loss(params, data, weight, key):
        noise_key, train_key = jax.vmap(jrandom.split, out_axes=1, axis_name='')(key)
        data = data + jrandom.normal(noise_key[0], data.shape) * 0.5 / std
        model = eqx.merge(params, static, which, treedef)
        # Setting an explicit axis_name works around a JAX bug that triggers
        # unnecessary re-JIT-ing in JAX version <= 0.2.18
        log_likelihood = jax.vmap(model.train, axis_name="")(data, key=train_key)
        return -jnp.mean(weight * log_likelihood)

    optim = getattr(optax, optim_name)(lr)
    if lookahead:
        optim = optax.lookahead(optim, sync_period=5, slow_step_size=0.1)
        params = optax.LookaheadParams.init_synced(params)
        fast = lambda x: x.fast
        slow = lambda x: x.slow
    else:
        fast = lambda x: x
        slow = lambda x: x
    opt_state = optim.init(params)

    best_value = jnp.inf
    best_params = params
    train_key = jrandom.split(train_key, batch_size)
    for step in range(steps):
        start = time.time()
        if virtual_batches == 1:
            data, weight = next(data_iter)
            value, grads = loss(fast(params), data, weight, train_key[: data.shape[0]])
            train_key = jax.vmap(lambda k: jrandom.split(k, 1)[0], axis_name='')(train_key)
        else:
            value = 0
            grads = jax.tree_map(lambda _: 0.0, params)
            for _ in range(virtual_batches):
                data, weight = next(data_iter)
                value_, grads_ = loss(fast(params), data, weight, train_key[: data.shape[0]])
                train_key = jax.vmap(lambda k: jrandom.split(k, 1)[0], axis_name='')(train_key)
                value = value + value_
                grads = jax.tree_map(lambda a, b: a + b, grads, grads_)
            value = value / virtual_batches
            grads = jax.tree_map(lambda a: a / virtual_batches, grads)
        end = time.time()
        if value < best_value:
            best_value = value
            best_params = params
        updates, opt_state = optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        print(f"Step: {step}, Loss: {value}, Computation time: {end - start}")

    print(f"Best value: {best_value}")
    best_model = eqx.merge(slow(best_params), static, which, treedef)

    num_samples = min(20 * dataset_size, 5000)
    sample_key = jrandom.split(sample_key, num_samples)
    samples = jax.vmap(best_model.sample, axis_name='')(key=sample_key)
    samples = samples * std + mean
    x = samples[:, 0]
    y = samples[:, 1]
    plt.scatter(x, y)
    plt.xlim(-1, width + 1)
    plt.ylim(-1, height + 1)
    plt.savefig(out_path)


if __name__ == "__main__":
    fire.Fire(main)
