###########
#
# This example trains a Neural ODE to reproduce a toy dataset of nonlinear oscillators.
#
###########

import time

import diffrax
import equinox as eqx
import fire
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import optax


###########
# We use Equinox as a convenient choice of neural network libary.
#
# It offers easy-to-use syntax without being a framework -- i.e. it interacts with
# normal JAX without any surprises -- so it's a good choice when trying to do something
# complicated, like interacting with a differential equation solve.
###########
# Recalling that a neural ODE is defined as
# y(t) = y(0) + \int_0^t f_\theta(s, y(s)) ds,
# then here we're defining the f_\theta that appears on that right hand side.
###########
class Func(eqx.Module):
    impl: eqx.nn.MLP

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.impl = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            key=key,
        )

    @eqx.filter_jit
    def __call__(self, t, y, args):
        return self.impl(y)


###########
# Here we wrap up the entire ODE solve into a model.
###########
class NeuralODE(eqx.Module):
    solver: diffrax.AbstractSolver
    stepsize_controller: diffrax.AbstractStepSizeController

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.solver = diffrax.tsit5(Func(data_size, width_size, depth, key=key))
        self.stepsize_controller = diffrax.IController()

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            self.solver,
            t0=ts[0],
            t1=ts[-1],
            y0=y0,
            dt0=ts[1] - ts[0],
            stepsize_controller=self.stepsize_controller,
            ###########
            # Not saved as an attribute as we don't want to differentiate `ts`.
            # (For simplicity our training loop treats all saved floating point-arrays
            # as parameters of the model.)
            ###########
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys


###########
# Toy dataset of nonlinear oscillators.
# Sample paths look like deformed sines and cosines.
###########
def _get_data(ts, *, key):
    y0 = jrandom.uniform(key, (2,), minval=-0.6, maxval=1)

    def f(t, y, args):
        x = y / (1 + y)
        return jnp.stack([x[1], -x[0]], axis=-1)

    solver = diffrax.tsit5(f)
    dt0 = 0.1
    saveat = diffrax.SaveAt(ts=ts)
    sol = diffrax.diffeqsolve(solver, ts[0], ts[-1], y0, dt0, saveat=saveat)
    ys = sol.ys
    return ys


def get_data(dataset_size, *, key):
    ts = jnp.linspace(0, 10, 100)
    key = jrandom.split(key, dataset_size)
    ys = jax.vmap(lambda key: _get_data(ts, key=key))(key)
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
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def main(
    dataset_size=256,
    batch_size=32,
    lr=3e-3,
    steps=5000,
    width_size=64,
    depth=2,
    seed=5678,
    plot=True,
):
    key = jrandom.PRNGKey(seed)
    data_key, model_key, loader_key = jrandom.split(key, 3)

    ts, ys = get_data(dataset_size, key=data_key)
    _, length_size, data_size = ys.shape

    model = NeuralODE(data_size, width_size, depth, key=model_key)

    ###########
    # Training loop like normal.
    # Only thing to notice is that up until step 500 we train on only the first 10% of
    # each time series. This is a standard trick to avoid getting caught in a local
    # minimum.
    ###########

    @eqx.filter_value_and_grad
    def loss(model, ti, yi):
        y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0])
        return jnp.mean((yi - y_pred) ** 2)

    optim = optax.adabelief(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    for step, (yi,) in zip(range(steps), dataloader((ys,), batch_size, key=loader_key)):
        start = time.time()
        if step > 500:
            ti = ts
        else:
            ti = ts[: length_size // 10]
            yi = yi[:, : length_size // 10]
        value, grads = loss(model, ti, yi)
        end = time.time()
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        print(f"Step: {step}, Loss: {value}, Computation time: {end - start}")

    if plot:
        plt.plot(ts, ys[0, :, 0], c="dodgerblue", label="Real")
        plt.plot(ts, ys[0, :, 1], c="dodgerblue")
        model_y = model(ts, ys[0, 0])
        plt.plot(ts, model_y[:, 0], c="crimson", label="Model")
        plt.plot(ts, model_y[:, 1], c="crimson")
        plt.legend()
        plt.tight_layout()
        plt.savefig("neural_ode.png")

    return ts, ys, model, value  # Model and final loss


if __name__ == "__main__":
    fire.Fire(main)


###########
# This example has assumed that the problem is Markov. Essentially, that the data `ys`
# is a complete observation of the system, and that we're not missing any channels.
# Note how the result of our model is evolving in data space. This is unlike e.g. an
# RNN, which has hidden state, and a linear map from hidden state to data.
#
# If we wanted we could generalise this to the non-Markov case: inside `NeuralODE`,
# project the initial condition into some high-dimensional latent space, do the ODE
# solve there, then take a linear map to get the output. See `latent_ode.py` for an
# example doing this as part of a generative model.
###########
