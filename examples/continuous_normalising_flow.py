###########
#
# This example is a bit of fun! It constructs a continuous normalising flow (CNF)
# (https://arxiv.org/abs/1810.01367)
# to learn a distribution specified by a (greyscale) image. That is, the target
# distribution is over R^2, and the image specifies the (unnormalised) density at each
# point.
#
# You can specify your own images, and learn your own flows.
#
# Some examples which work quite well:
#
# python continuous_normalising_flow.py --in_path="../imgs/cat.png"
# python continuous_normalising_flow.py --in_path="../imgs/butterfly.png" --num_blocks=3
# python continuous_normalising_flow.py --in_path="../imgs/target.png" --width_size=128
#
###########

import functools as ft
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
import scipy.stats as stats


here = pathlib.Path(__file__).resolve().parent


def normal_log_likelihood(y):
    return -0.5 * (y.size * math.log(2 * math.pi) + jnp.sum(y ** 2))


###########
# Use Hutchinson's trace estimator to estimate the divergence of the vector field.
# (As introduced in FFJORD.)
###########
def approx_logp_wrapper(t, y, args):
    y, _ = y
    *args, eps, func = args
    fn = lambda y: func(t, y, args)
    f, vjp_fn = jax.vjp(fn, y)
    (eps_dfdy,) = vjp_fn(eps)
    logp = jnp.sum(eps_dfdy * eps)
    return f, logp


###########
# Alternatively, compute the divergence exactly.
###########
def exact_logp_wrapper(t, y, args):
    y, _ = y
    *args, _, func = args
    fn = lambda y: func(t, y, args)
    f, vjp_fn = jax.vjp(fn, y)
    (size,) = y.shape  # this implementation only works for 1D input
    eye = jnp.eye(size)
    (dfdy,) = jax.vmap(vjp_fn, axis_name="")(eye)
    logp = jnp.trace(dfdy)
    return f, logp


###########
# Credit: this layer, and some of the default hyperparameters below, are taken from the
# FFJORD repo.
###########
class ConcatSquash(eqx.Module):
    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear
    lin3: eqx.nn.Linear

    def __init__(self, *, in_size, out_size, key, **kwargs):
        super().__init__(**kwargs)
        key1, key2, key3 = jrandom.split(key, 3)
        self.lin1 = eqx.nn.Linear(in_size, out_size, key=key1)
        self.lin2 = eqx.nn.Linear(1, out_size, key=key2)
        self.lin3 = eqx.nn.Linear(1, out_size, use_bias=False, key=key3)

    def __call__(self, t, y):
        return self.lin1(y) * jnn.sigmoid(self.lin2(t)) + self.lin3(t)


###########
# Basically just an MLP, using tanh as the activation function and ConcatSquash instead
# of linear layers.
# This is the vector field on the right hand side of the ODE.
###########
class Func(eqx.Module):
    layers: List[eqx.nn.Linear]

    def __init__(self, *, data_size, width_size, depth, key, **kwargs):
        super().__init__(**kwargs)
        keys = jrandom.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(
                ConcatSquash(in_size=data_size, out_size=data_size, key=keys[0])
            )
        else:
            layers.append(
                ConcatSquash(in_size=data_size, out_size=width_size, key=keys[0])
            )
            for i in range(depth - 1):
                layers.append(
                    ConcatSquash(
                        in_size=width_size, out_size=width_size, key=keys[i + 1]
                    )
                )
            layers.append(
                ConcatSquash(in_size=width_size, out_size=data_size, key=keys[-1])
            )
        self.layers = layers

    @jax.jit
    def __call__(self, t, y, args):
        t = jnp.asarray(t)[None]
        for layer in self.layers[:-1]:
            y = layer(t, y)
            y = jnn.tanh(y)
        y = self.layers[-1](t, y)
        return y


###########
# Wrap up the differential equation solve into a model.
###########
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
                key=k,
            )
            for k in keys
        ]
        self.data_size = data_size
        self.exact_logp = exact_logp
        self.t0 = 0.0
        self.t1 = 0.5
        self.dt0 = 0.05

    ###########
    # Runs backward-in-time to train the CNF.
    ###########
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

    ###########
    # Runs forward-in-time to draw samples from the CNF.
    ###########
    def sample(self, *, key):
        y = jrandom.normal(key, (self.data_size,))
        for func in self.funcs:
            solver = diffrax.tsit5(func)
            sol = diffrax.diffeqsolve(solver, self.t0, self.t1, y, self.dt0)
            (y,) = sol.ys
        return y

    ###########
    # By way of illustration, we have a variant sample method we can query to see the
    # evolution of the samples during the forward solve.
    ###########
    def sample_flow(self, *, key):
        t_so_far = self.t0
        t_end = self.t0 + (self.t1 - self.t0) * len(self.funcs)
        save_times = jnp.linspace(self.t0, t_end, 6)
        y = jrandom.normal(key, (self.data_size,))
        out = []
        for i, func in enumerate(self.funcs):
            if i == len(self.funcs) - 1:
                save_ts = save_times[t_so_far <= save_times] - t_so_far
            else:
                save_ts = (
                    save_times[
                        (t_so_far <= save_times)
                        & (save_times < t_so_far + self.t1 - self.t0)
                    ]
                    - t_so_far
                )
                t_so_far = t_so_far + self.t1 - self.t0
            saveat = diffrax.SaveAt(ts=save_ts)
            solver = diffrax.tsit5(func)
            sol = diffrax.diffeqsolve(
                solver, self.t0, self.t1, y, self.dt0, saveat=saveat
            )
            out.append(sol.ys)
            y = sol.ys[-1]
        out = jnp.concatenate(out)
        assert len(out) == 6  # number of points we saved at
        return out


###########
# Converts the input image into data.
###########
def get_data(path):
    # integer array of shape (height, width, channels) with values in {0, ..., 255}
    img = jnp.asarray(imageio.imread(path))
    if img.shape[-1] == 4:
        img = img[..., :-1]  # ignore alpha channel
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

    return dataset, weights, mean, std, img, width, height


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
    in_path,
    out_path=None,
    batch_size=500,
    virtual_batches=2,
    lr=1e-3,
    weight_decay=1e-5,
    steps=10000,
    exact_logp=True,
    num_blocks=2,
    width_size=64,
    depth=3,
    seed=5678,
):
    if out_path is None:
        out_path = here / "cnf_results" / pathlib.Path(in_path).name
    out_path = pathlib.Path(out_path)
    out_path.resolve().parent.mkdir(parents=True, exist_ok=True)

    key = jrandom.PRNGKey(seed)
    model_key, loader_key, train_key, sample_key = jrandom.split(key, 4)

    dataset, weights, mean, std, img, width, height = get_data(in_path)
    dataset_size, data_size = dataset.shape
    data_iter = iter(dataloader((dataset, weights), batch_size, key=loader_key))

    model = CNF(
        data_size=data_size,
        exact_logp=exact_logp,
        num_blocks=num_blocks,
        width_size=width_size,
        depth=depth,
        key=model_key,
    )

    @ft.partial(eqx.value_and_grad_f, filter_fn=eqx.is_inexact_array)
    def loss(model, data, weight, key):
        batch_size, _ = data.shape
        noise_key, train_key = jrandom.split(key, 2)
        train_key = jrandom.split(key, batch_size)
        data = data + jrandom.normal(noise_key, data.shape) * 0.5 / std
        # Setting an explicit axis_name works around a JAX bug that triggers
        # unnecessary re-JIT-ing in JAX version <= 0.2.19.
        log_likelihood = jax.vmap(model.train, axis_name="")(data, key=train_key)
        return -jnp.mean(weight * log_likelihood)

    optim = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = optim.init(
        jax.tree_map(lambda x: x if eqx.is_inexact_array(x) else None, model)
    )

    for step in range(steps):
        start = time.time()
        value = 0
        grads = jax.tree_map(lambda _: 0.0, model)
        for _ in range(virtual_batches):
            data, weight = next(data_iter)
            value_, grads_ = loss(model, data, weight, train_key)
            (train_key,) = jrandom.split(train_key, 1)
            value = value + value_
            grads = jax.tree_map(
                lambda a, b: None if b is None else a + b, grads, grads_
            )
        value = value / virtual_batches
        grads = jax.tree_map(lambda a: a / virtual_batches, grads)
        end = time.time()

        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        print(f"Step: {step}, Loss: {value}, Computation time: {end - start}")

    print(f"Best value: {value}")
    num_samples = 5000
    sample_key = jrandom.split(sample_key, num_samples)
    samples = jax.vmap(model.sample, axis_name="")(key=sample_key)
    sample_flows = jax.vmap(model.sample_flow, axis_name="", out_axes=-1)(
        key=sample_key
    )
    fig, (*axs, ax, axtrue) = plt.subplots(
        1,
        2 + len(sample_flows),
        figsize=((2 + len(sample_flows)) * 10 * height / width, 10),
    )

    samples = samples * std + mean
    x = samples[:, 0]
    y = samples[:, 1]
    ax.scatter(x, y, c="black", s=2)
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_aspect(height / width)
    ax.set_xticks([])
    ax.set_yticks([])

    axtrue.imshow(img.T, origin="lower", cmap="gray")
    axtrue.set_aspect(height / width)
    axtrue.set_xticks([])
    axtrue.set_yticks([])

    x_resolution = 100
    y_resolution = int(x_resolution * (height / width))
    sample_flows = sample_flows * std[:, None] + mean[:, None]
    x_pos, y_pos = jnp.broadcast_arrays(
        jnp.linspace(-1, width + 1, x_resolution)[:, None],
        jnp.linspace(-1, height + 1, y_resolution)[None, :],
    )
    positions = jnp.stack([jnp.ravel(x_pos), jnp.ravel(y_pos)])
    densities = [stats.gaussian_kde(samples)(positions) for samples in sample_flows]
    for i, (ax, density) in enumerate(zip(axs, densities)):
        density = jnp.reshape(density, (x_resolution, y_resolution))
        ax.imshow(density.T, origin="lower", cmap="plasma")
        ax.set_aspect(height / width)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    fire.Fire(main)
