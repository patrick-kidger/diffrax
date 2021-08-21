import functools as ft
import time

import diffrax as dfx
import equinox as eqx
import fire
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import optax


class Func(eqx.Module):
    mlp: eqx.nn.MLP

    @ft.partial(eqx.jitf, filter_fn=eqx.is_array)
    def __call__(self, t, y, args):
        return self.mlp(y)


class LatentODE(eqx.Module):
    solver: dfx.AbstractSolver
    rnn_cell: eqx.nn.GRUCell
    hidden_to_latent: eqx.nn.Linear
    latent_to_hidden: eqx.nn.MLP
    hidden_to_data: eqx.nn.Linear
    hidden_size: int
    latent_size: int

    def __init__(
        self, *, data_size, hidden_size, latent_size, width_size, depth, key, **kwargs
    ):
        super().__init__(**kwargs)

        mkey, gkey, lkey = jrandom.split(key, 3)
        mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            key=mkey,
        )
        self.solver = dfx.dopri5(Func(mlp))
        self.rnn_cell = eqx.nn.GRUCell(data_size + 1, hidden_size, key=gkey)
        self.hidden_to_latent = eqx.nn.Linear(hidden_size, 2 * latent_size, key=lkey)
        self.latent_to_hidden = eqx.nn.MLP(
            latent_size, hidden_size, width_size=width_size, depth=depth, key=lkey
        )
        self.hidden_to_data = eqx.nn.Linear(hidden_size, data_size, key=lkey)
        self.hidden_size = hidden_size
        self.latent_size = latent_size

    @ft.partial(eqx.jitf, filter_fn=eqx.is_array)
    def _latent(self, ts, ys, key):
        data = jnp.concatenate([ts[:, None], ys], axis=1)
        hidden = jnp.zeros((self.hidden_size,))
        for data_i in reversed(data):
            hidden = self.rnn_cell(data_i, hidden)
        context = self.hidden_to_latent(hidden)
        mean, logstd = context[: self.latent_size], context[self.latent_size :]
        std = jnp.exp(logstd)
        latent = mean + jrandom.normal(key, (self.latent_size,)) * std
        return latent, mean, std

    def _sample(self, ts, latent):
        # TODO
        y0 = self.latent_to_hidden(latent)
        sol = dfx.diffeqsolve(
            self.solver, ts[0], ts[-1], y0, dt0=0.375, saveat=dfx.SaveAt(t=ts)
        )
        return jax.vmap(self.hidden_to_data, axis_name="")(sol.ys)

    @staticmethod
    @jax.jit
    def _loss(ys, pred_ys, mean, std):
        # -log p_θ with Gaussian p_θ
        reconstruction_loss = 0.5 * jnp.sum((ys - pred_ys) ** 2)
        # KL(N(mean, std^2) || N(0, 1))
        variational_loss = 0.5 * jnp.sum(mean ** 2 + std ** 2 - 2 * jnp.log(std) - 1)
        return reconstruction_loss + variational_loss

    def train(self, ts, ys, *, key):
        latent, mean, std = self._latent(ts, ys, key)
        pred_ys = self._sample(ts, latent)
        return self._loss(ys, pred_ys, mean, std)

    def sample(self, ts, *, key):
        latent = jrandom.normal(key, (self.latent_size,))
        return self._sample(ts, latent)


def get_data(dataset_size, *, key):
    ykey, tkey1, tkey2 = jrandom.split(key, 3)

    y0 = jrandom.normal(ykey, (dataset_size, 2))

    # As a way to demonstrate how diffrax can vmap over times, we'll use irregularly
    # sampled observations here.
    t0 = 0
    t1 = 2 + jrandom.uniform(tkey1, (dataset_size,), minval=0, maxval=1)
    ts = (
        jrandom.uniform(tkey2, (dataset_size, 20), minval=0, maxval=1)
        * (t1[:, None] - t0)
        + t0
    )
    ts = jnp.sort(ts)

    def func(t, y, args):
        return jnp.array([[-0.1, 1.3], [-1, -0.1]]) @ y

    def solve(ts, y0):
        sol = dfx.diffeqsolve(
            dfx.tsit5(func), ts[0], ts[-1], y0, dt0=0.1, saveat=dfx.SaveAt(t=ts)
        )
        return sol.ys

    ys = jax.vmap(solve, axis_name="")(ts, y0)
    return ts, ys


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
    dataset_size=10000,
    batch_size=256,
    lr=1e-2,
    steps=10000,
    hidden_size=32,
    latent_size=32,
    width_size=32,
    depth=2,
    seed=5678,
):
    key = jrandom.PRNGKey(seed)
    data_key, model_key, loader_key, train_key = jrandom.split(key, 4)

    ts, ys = get_data(dataset_size, key=data_key)
    batch_size = min(batch_size, dataset_size)

    model = LatentODE(
        data_size=ys.shape[-1],
        hidden_size=hidden_size,
        latent_size=latent_size,
        width_size=width_size,
        depth=depth,
        key=model_key,
    )

    @ft.partial(eqx.value_and_grad_f, filter_fn=eqx.is_inexact_array)
    def loss(model, ts, ys, *, key):
        loss = jax.vmap(model.train, axis_name="")(ts, ys, key=key)
        return jnp.mean(loss)

    optim = optax.chain(
        optax.scale_by_adam(), optax.scale(-lr), optax.clip_by_global_norm(0.1)
    )
    opt_state = optim.init(
        jax.tree_map(lambda leaf: leaf if eqx.is_inexact_array(leaf) else None, model)
    )
    train_key = jrandom.split(train_key, batch_size)
    for step, (ts, ys) in zip(
        range(steps), dataloader((ts, ys), batch_size, key=loader_key)
    ):
        _batch_size = ts.shape[0]
        start = time.time()
        value, grads = loss(model, ts, ys, key=train_key[:_batch_size])
        end = time.time()
        train_key = jax.vmap(lambda k: jrandom.split(k, 1)[0], axis_name="")(train_key)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        print(f"Step: {step}, Loss: {value}, Computation time: {end - start}")
    breakpoint()


if __name__ == "__main__":
    fire.Fire(main)
