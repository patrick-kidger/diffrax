import math
import pathlib
import time
from typing import List

import diffrax
import equinox as eqx
import fire
import imageio
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import optax


here = pathlib.Path(__file__).resolve().parent


def normal_log_likelihood(y):
    # Omits the constant factor
    return -0.5 * (math.log(2 * math.pi) + jnp.sum(y ** 2))


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
    (dfdy,) = jax.vmap(vjp_fn)(eye)
    logp = jnp.trace(dfdy)
    return f, logp


class Func(eqx.Module):
    autonomous: bool
    mlp: eqx.Module

    def __init__(
        self, *, data_size, width_size, depth, autonomous, key, **kwargs
    ):
        super().__init__(**kwargs)
        in_size = data_size if autonomous else data_size + 1
        self.autonomous = autonomous
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            key=key,
        )

    def __call__(self, t, y, args):
        if self.autonomous:
            inp = y
        else:
            t = jnp.asarray(t)[None]
            inp = jnp.concatenate([t, y])
        return self.mlp(inp)


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
        autonomous,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_size = data_size
        self.exact_logp = exact_logp

        keys = jrandom.split(key, num_blocks)
        self.funcs = [
            Func(
                data_size=data_size,
                width_size=width_size,
                depth=depth,
                autonomous=autonomous,
                key=k,
            )
            for k in keys
        ]
        self.t0 = 0.0
        self.t1 = 1.0
        self.dt0 = 0.1

    def train(self, y, *, key):
        solver = diffrax.tsit5(
            exact_logp_wrapper if self.exact_logp else approx_logp_wrapper
        )
        eps = jrandom.normal(key, y.shape)
        delta_log_likelihood = 0.0
        for func in self.funcs:
            y = (y, delta_log_likelihood)
            sol = diffrax.diffeqsolve(
                solver, self.t1, self.t0, y, -self.dt0, (eps, func)
            )
            (y,), (delta_log_likelihood,) = sol.ys
        return delta_log_likelihood + normal_log_likelihood(y)

    def sample(self, *, key):
        y = jrandom.normal(key, (self.data_size,))
        for func in reversed(self.funcs):
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
    std = jnp.std(dataset, axis=0)
    dataset = (dataset - mean) / (std + 1e-6)
    return dataset, weights, mean, std, (0, 0, width, height)


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
    data_path=here / "cnf_in.png",
    out_path=here / "cnf_out.png",
    batch_size=None,
    optim_name="sgd",
    learning_rate=1e-3,
    steps=3000,
    lookahead=False,
    exact_logp=True,
    num_blocks=2,
    width_size=2048,
    depth=2,
    autonomous=False,
    seed=5678,
):
    key = jrandom.PRNGKey(seed)
    model_key, loader_key, train_key, sample_key = jrandom.split(key, 4)

    dataset, weights, mean, std, (xmin, ymin, xmax, ymax) = get_data(data_path)
    dataset_size, data_size = dataset.shape

    if batch_size is None:
        batch_size = dataset_size
    else:
        batch_size = min(batch_size, dataset_size)

    model = CNF(
        data_size=data_size,
        exact_logp=exact_logp,
        num_blocks=num_blocks,
        width_size=width_size,
        depth=depth,
        autonomous=autonomous,
        key=model_key,
    )
    params, static, which, treedef = eqx.split(model, filter_fn=eqx.is_inexact_array)

    @jax.value_and_grad
    def loss(params, data, weight, key):
        # Setting an explicit axis_name works around a JAX bug that triggers
        # unnecessary re-JIT-ing in JAX version <= 0.2.18
        model = eqx.merge(params, static, which, treedef)
        log_likelihood = jax.vmap(model.train, axis_name="")(data, key=key)
        kl_divergence = jnp.mean(weight * (jnp.log(weight) - log_likelihood))
        return kl_divergence

    optim = getattr(optax, optim_name)(learning_rate)
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
    for step, (data, weight) in zip(
        range(steps), dataloader((dataset, weights), batch_size, key=loader_key)
    ):
        start = time.time()
        value, grads = loss(fast(params), data, weight, train_key[: data.shape[0]])
        end = time.time()
        if value < best_value:
            best_value = value
            best_params = params
        updates, opt_state = optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        train_key = jax.vmap(lambda k: jrandom.split(k, 1)[0])(train_key)
        print(f"Step: {step}, Loss: {value}, Computation time: {end - start}")

    print(f"Best value: {best_value}")
    best_model = eqx.merge(slow(best_params), static, which, treedef)

    num_samples = 20 * dataset_size
    sample_key = jrandom.split(sample_key, num_samples)
    samples = jax.vmap(best_model.sample)(key=sample_key)
    samples = samples * (1e-6 + std) + mean
    x = samples[:, 0]
    y = samples[:, 1]
    plt.scatter(x, y)
    plt.xlim(xmin - 1, xmax + 1)
    plt.ylim(ymin - 1, ymax + 1)
    plt.savefig(out_path)


if __name__ == "__main__":
    fire.Fire(main)
