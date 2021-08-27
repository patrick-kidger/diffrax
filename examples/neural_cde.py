###########
#
# This example trains a Neural CDE to distinguish clockwise from counter-clockwise
# spirals.
# (https://arxiv.org/abs/2005.08926)
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
# A neural CDE looks like
#
# y(t) = y(0) + \int_0^t f_\theta(y(s)) dx(s)
#
# Where x is your data and f_\theta is a neural network. The right hand side is a
# matrix-vector product between them.
#
# This is the f_\theta.
###########
class Func(eqx.Module):
    mlp: eqx.nn.MLP
    data_size: int
    hidden_size: int

    def __init__(self, data_size, hidden_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size * data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            ###########
            # Note the use of a tanh final activation function. This is important to
            # stop the model blowing up. (Just like how GRUs and LSTMs constrain the
            # rate of change of their hidden states.)
            ###########
            final_activation=jnn.tanh,
            key=key,
        )

    @ft.partial(eqx.jitf, filter_fn=eqx.is_array)
    def __call__(self, t, y, args):
        return self.mlp(y).reshape(self.hidden_size, self.data_size)


###########
# Now wrap up the whole CDE solve into a model.
#
# In this case we cap the neural CDE with a linear layer and sigomid, to perform binary
# classification.
###########
class NeuralCDE(eqx.Module):
    initial: eqx.nn.MLP
    func: Func
    stepsize_controller: diffrax.AbstractStepSizeController
    linear: eqx.nn.Linear

    def __init__(self, data_size, hidden_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        ikey, fkey, lkey = jrandom.split(key, 3)
        self.initial = eqx.nn.MLP(data_size, hidden_size, width_size, depth, key=ikey)
        self.func = Func(data_size, hidden_size, width_size, depth, key=fkey)
        self.stepsize_controller = diffrax.IController()
        self.linear = eqx.nn.Linear(hidden_size, 1, key=lkey)

    def __call__(self, ts, coeffs):
        ###########
        # Each sample of data consists of some timestamps `ts`, and some `coeffs`
        # parameterising a control path.
        #
        # `control` is just the continuous-time path, parameterised by these
        # coefficients, that will be used to drive the CDE.
        #
        # `term` wraps the vector field and control together. In this case the
        # `.to_ode()` call says that we'd like to solve \int f(y(s)) dx(s) as an ODE,
        # by treating the right hand side as \int f(y(s)) dx/ds ds.
        #
        # [Not doing this corresponds to using the control just like time in an ODE
        # solver, e.g. the explicit Euler method becomes
        #
        # y(t_j) + f(y(t_j)) (x(t_{j+1}) - x(t_j)).
        #
        # It's not known whether it's generally better to use `.to_ode()` or not. Try
        # both and find out what works best for you?]
        #
        # `solver` is then just the solver we're using, like usual. Note that it's the
        # capital `Tsit5` not the lower-case `tsit5`, because we're using the more
        # advanced API. (The lower-case `tsit5` is just a shortcut for
        # `Tsit5(ODETerm(...))` for the common ODE case.)
        ###########
        control = diffrax.CubicInterpolation(ts=ts, coeffs=coeffs)
        term = diffrax.ControlTerm(vector_field=self.func, control=control).to_ode()
        solver = diffrax.Tsit5(term=term)

        y0 = self.initial(control.evaluate(ts[0]))
        solution = diffrax.diffeqsolve(
            solver,
            t0=ts[0],
            t1=ts[-1],
            y0=y0,
            dt0=None,
            stepsize_controller=self.stepsize_controller,
        )
        (prediction,) = jnn.sigmoid(self.linear(solution.ys[-1]))
        return prediction


###########
# Toy dataset of spirals.
#
# We interpolate the samples with Hermite cubic splines with backward differences,
# which were introduced in https://arxiv.org/abs/2106.11028. (And produces better
# results than the natural cubic splines used in the original neural CDE paper.)
###########
def get_data(dataset_size, *, key):
    length = 100
    theta = jrandom.uniform(key, (dataset_size,), minval=0, maxval=2 * math.pi)
    y0 = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    ts = jnp.broadcast_to(jnp.linspace(0, 4 * math.pi, length), (dataset_size, length))
    matrix = jnp.array([[-0.3, 2], [-2, -0.3]])
    ys = jax.vmap(
        lambda y0i, ti: jax.vmap(lambda tij: jsp.linalg.expm(tij * matrix) @ y0i)(ti)
    )(y0, ts)
    ys = ys.at[: dataset_size // 2, :, 0].multiply(-1)
    coeffs = jax.vmap(diffrax.hermite_cubic_with_backward_differences_coefficients)(
        ts, ys
    )
    labels = jnp.zeros((dataset_size,))
    labels = labels.at[: dataset_size // 2].set(1.0)
    _, _, data_size = ys.shape
    return ts, coeffs, labels, data_size


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
    learning_rate=1e-3,
    steps=20,
    hidden_size=8,
    width_size=128,
    depth=1,
    seed=5678,
):
    key = jrandom.PRNGKey(seed)
    data_key, model_key, loader_key = jrandom.split(key, 3)

    ts, coeffs, labels, data_size = get_data(dataset_size, key=data_key)

    model = NeuralCDE(data_size, hidden_size, width_size, depth, key=model_key)

    ###########
    # Training loop like normal.
    ###########

    @ft.partial(eqx.value_and_grad_f, filter_fn=eqx.is_inexact_array, has_aux=True)
    def loss(model, ti, label_i, coeff_i):
        # Setting an explicit axis_name works around a JAX bug that triggers
        # unnecessary re-JIT-ing in JAX version <= 0.2.19.
        pred = jax.vmap(model, axis_name="")(ti, coeff_i)
        # Binary cross-entropy
        bxe = label_i * jnp.log(pred) + (1 - label_i) * jnp.log(1 - pred)
        bxe = -jnp.mean(bxe)
        acc = jnp.mean((pred > 0.5) == (label_i == 1))
        return bxe, acc

    optim = optax.adam(learning_rate)
    opt_state = optim.init(
        jax.tree_map(lambda leaf: leaf if eqx.is_inexact_array(leaf) else None, model)
    )
    for step, (ti, label_i, *coeff_i) in zip(
        range(steps), dataloader((ts, labels) + coeffs, batch_size, key=loader_key)
    ):
        start = time.time()
        (bxe, acc), grads = loss(model, ti, label_i, coeff_i)
        end = time.time()
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        print(
            f"Step: {step}, Loss: {bxe}, Accuracy: {acc}, Computation time: "
            f"{end - start}"
        )
    return bxe, acc  # Final loss and accuracy


if __name__ == "__main__":
    fire.Fire(main)
