###########
#
# Training a small neural network by SGD, using diffrax.
#
# Stochastic gradient descent is just the Euler approximation to gradient flow.
# Recall the differential equation for gradient flow:
#
# dθ/dt(t) = -df/dθ(θ(t))
#
# where f(θ(t)) is the loss evaluated on the value of the parameters at timestep
# t of training. Then the explicit Euler method produces:
#
# θ_{t+1} = θ_t + Δt * (-df/dθ(θ(t)))
#
# which is the familiar formula for (stochastic) gradient descent. The step size Δt
# corresponds to the learning rate.
#
###########
#
# In this example we do exactly this to train a small neural network using diffrax.
# Practically speaking you should probably just use a standard optimisation library,
# but doing it via numerical differential equation solvers can be a bit of fun.
#
# For example you can switch out the Euler method for other methods like Heun, or you
# can start using adaptive step sizes (learning rates), and so on.
#
###########

import time
from typing import Tuple

import equinox as eqx
import fire
import jax
import jax.numpy as jnp
import jax.random as jrandom
from diffrax import diffeqsolve, euler


def get_data(key, dataset_size):
    x = jrandom.normal(key, (dataset_size, 1))
    y = 5 * x - 2
    return x, y


###########
# Functionally pure dataloader.
# - Wraps some state (arrays, key, batch_size).
# - Produces a pseudorandom batch of data every time it is called. (In this case random
#   sampling with replacement. A more sophisticated implementatation could follow normal
#   iteration-over-dataset behaviour.)
###########
class DataLoader(eqx.Module):
    arrays: Tuple[jnp.ndarray]
    batch_size: int
    key: jrandom.PRNGKey

    # Equinox Modules are Python dataclasses, which allow for a __post_init__.
    def __post_init__(self):
        dataset_size = self.arrays[0].shape[0]
        assert all(array.shape[0] == dataset_size for array in self.arrays)

    def __call__(self, step):
        key = jrandom.fold_in(self.key, step)
        dataset_size = self.arrays[0].shape[0]
        batch_indices = jrandom.randint(
            key, (self.batch_size,), minval=0, maxval=dataset_size
        )
        return tuple(array[batch_indices] for array in self.arrays)


def main(
    dataset_size=10000,
    batch_size=256,
    learning_rate=3e-3,
    steps=1000,
    width_size=8,
    seed=56789,
    jit=False,
    printout=True,
):
    start = time.time()
    data_key, model_key, loader_key = jrandom.split(jrandom.PRNGKey(seed), 3)

    data = get_data(data_key, dataset_size)
    dataloader = DataLoader(data, batch_size, key=loader_key)

    ###########
    # Equinox models are PyTrees of both JAX arrays and arbitrary Python objects. (e.g.
    # activation functions.)
    # Diffrax's diffeqsolve only works on PyTrees of JAX arrays, so we need to split up
    # these two pieces.
    ###########
    model = eqx.nn.MLP(
        in_size=1, out_size=1, width_size=width_size, depth=1, key=model_key
    )
    params, static = eqx.partition(model, eqx.is_inexact_array)

    ###########
    # Define the vector field as the gradient of a loss function.
    ###########
    @jax.jit
    @jax.value_and_grad
    def loss(params, x, y):
        model = eqx.combine(params, static)
        pred_y = jax.vmap(model)(x)
        return jnp.mean((y - pred_y) ** 2)

    def vector_field(step, params, _):
        x, y = dataloader(step)
        value, grad = loss(params, x, y)
        if printout:
            print(step, value)
        return jax.tree_map(lambda g: -learning_rate * g, grad)

    ###########
    # Try running this with jit=True/False. You should probably see a ~2-3x speedup
    # using JIT.
    #
    # Note that we can safely JIT because vector_field has only benign side-effects,
    # namely the print statement. (Doing so will mean that we don't get to see the
    # print statements as training progresses, however.)
    #
    # In particular this uses the fact that the dataloader is functionally pure, and
    # doesn't maintain any internal state.
    ###########
    solution = diffeqsolve(
        euler(vector_field), t0=0, t1=steps, y0=params, dt0=1, jit=jit
    )

    params = jax.tree_map(lambda x: x[0], solution.ys)
    value, _ = loss(params, *dataloader(0))
    end = time.time()
    print(f"Final loss: {value}")
    print(f"Training completed in {end - start} seconds")


if __name__ == "__main__":
    fire.Fire(main)
