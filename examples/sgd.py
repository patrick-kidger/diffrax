###########
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
from dataclasses import dataclass
from typing import Any

import fire
import jax
import jax.experimental.stax as stax
import jax.nn as jnn
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
# - Has __hash__ and __eq__ set so that the JIT is happy to use this as a static_argnum.
###########
@dataclass(frozen=True)
class dataloader:
    arrays: tuple[Any]
    key: Any
    batch_size: int

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

    def __hash__(self):
        return 0

    def __eq__(self, other):
        return self is other


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
    data_key, loader_key, init_key = jrandom.split(jrandom.PRNGKey(seed), 3)

    data = get_data(data_key, dataset_size)
    data = dataloader(arrays=data, key=loader_key, batch_size=batch_size)

    init_model, apply_model = stax.serial(
        stax.Dense(width_size), stax.elementwise(jnn.relu), stax.Dense(1)
    )
    _, params = init_model(init_key, (1,))
    vmap_apply_model = jax.vmap(apply_model, in_axes=(None, 0))

    @jax.jit
    @jax.value_and_grad
    def loss(params, x, y):
        pred_y = vmap_apply_model(params, x)
        return jnp.mean((y - pred_y) ** 2)

    def vector_field(step, params, data):
        x, y = data(step)
        value, grad = loss(params, x, y)
        if printout:
            print(step, value)
        return jax.tree_map(lambda g: -learning_rate * g, grad)

    ###########
    # Note that we can safely jit because vector_field has only benign side-effects
    # (the print statement). This will mean that we don't get to see the print
    # statements as training progresses, however.
    #
    # Try running this with jit=True/False. You should probably see a ~2x speedup
    # using JIT.
    ###########
    solution = diffeqsolve(
        euler(vector_field), t0=0, t1=steps, y0=params, dt0=1, args=data, jit=jit
    )

    params = jax.tree_map(lambda x: x[0], solution.ys)
    value, _ = loss(params, *data(0))
    end = time.time()
    print(f"Final loss: {value}")
    print(f"Training completed in {end - start} seconds")


if __name__ == "__main__":
    fire.Fire(main)
