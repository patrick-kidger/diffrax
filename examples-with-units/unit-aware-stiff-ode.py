# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import time

import brainunit as u
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.numpy as jnp

import diffrax

jax.config.update("jax_enable_x64", True)


class Robertson(eqx.Module):
    k1: float
    k2: float
    k3: float

    def __call__(self, t, y, args):
        f0 = -self.k1 * y[0] + self.k3 * y[1] * y[2]
        f1 = self.k1 * y[0] - self.k2 * y[1] ** 2 - self.k3 * y[1] * y[2]
        f2 = self.k2 * y[1] ** 2
        return jnp.stack([f0, f1, f2]) / u.ms


@jax.jit
def main(k1, k2, k3):
    robertson = Robertson(k1, k2, k3)
    terms = diffrax.ODETerm(robertson)
    t0 = 0.0 * u.ms
    t1 = 100.0 * u.ms
    y0 = jnp.array([1.0, 0.0, 0.0])
    dt0 = 0.0002 * u.ms
    solver = diffrax.Kvaerno5()
    saveat = diffrax.SaveAt(ts=jnp.array([0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]) * u.ms)
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0,
        y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=100000
    )
    return sol


main(0.04, 3e7, 1e4)

start = time.time()
sol = main(0.04, 3e7, 1e4)
end = time.time()

print("Results:")
for ti, yi in zip(sol.ts, sol.ys):
    print(f"t={ti.item()}, y={yi.tolist()}")
print(f"Took {sol.stats['num_steps']} steps in {end - start} seconds.")
