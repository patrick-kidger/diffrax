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


import os

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import time
import equinox as eqx
import brainunit as u

from diffrax import PIDController
import jax.random as jr
import diffrax
from diffrax import Euler, MultiTerm, VirtualBrownianTree
from diffrax import AbstractPath, ControlTerm, Dopri5
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
from diffrax import backward_hermite_coefficients, CubicInterpolation


def try2():
    def f(t, y, args):
        return -y / u.ms

    term = ODETerm(f)
    solver = Dopri5()
    y0 = jnp.array([2., 3.]) * u.mV
    solution = diffeqsolve(term, solver, t0=0 * u.ms, t1=1 * u.ms, dt0=0.1 * u.ms, y0=y0)
    print(solution)


def try_ode1():
    vector_field = lambda t, y, args: -y / u.ms
    term = ODETerm(vector_field)

    for solver in [
        diffrax.Euler(),
        diffrax.LeapfrogMidpoint(),
    ]:
        print(solver)

        saveat = SaveAt(ts=[0., 1., 2., 3.] * u.ms, t1=True, t0=True)

        sol = diffeqsolve(term, solver, t0=0 * u.ms, t1=3 * u.ms, dt0=0.1 * u.ms, y0=1, saveat=saveat)
        print(sol)

    for solver in [
        diffrax.Heun(),
        diffrax.Midpoint(),
        diffrax.Ralston(),
        diffrax.Bosh3(),
        diffrax.Tsit5(),
        diffrax.Dopri5(),
        diffrax.Dopri8(),

        diffrax.ImplicitEuler(),
        diffrax.Kvaerno3(),
        diffrax.Kvaerno4(),
        diffrax.Kvaerno5(),
        diffrax.ReversibleHeun(),
    ]:
        print(solver)

        saveat = SaveAt(ts=[0., 1., 2., 3.] * u.ms, t1=True, t0=True)
        stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

        sol = diffeqsolve(term, solver, t0=0 * u.ms, t1=3 * u.ms, dt0=None, y0=1, saveat=saveat,
                          stepsize_controller=stepsize_controller)

        print(sol.ts)  # DeviceArray([0.   , 1.   , 2.   , 3.    ])
        print(sol.ys)  # DeviceArray([1.   , 0.368, 0.135, 0.0498])

        sol = diffeqsolve(term, solver, t0=0 * u.ms, t1=3 * u.ms, dt0=0.1 * u.ms, y0=1, saveat=saveat,
                          stepsize_controller=stepsize_controller)

        print(sol.ts)  # DeviceArray([0.   , 1.   , 2.   , 3.    ])
        print(sol.ys)  # DeviceArray([1.   , 0.368, 0.135, 0.0498])

        sol = diffeqsolve(
            term, solver, t0=0 * u.ms, t1=3 * u.ms, dt0=0.1 * u.ms, y0=1 * u.mV, saveat=saveat,
            stepsize_controller=stepsize_controller
        )

        print(sol.ts)  # DeviceArray([0.   , 1.   , 2.   , 3.    ])
        print(sol.ys)  # DeviceArray([1.   , 0.368, 0.135, 0.0498])

        sol = diffeqsolve(
            term, solver, t0=0 * u.ms, t1=3 * u.ms, dt0=0.1 * u.ms, y0=1 * u.mV, saveat=SaveAt(dense=True),
            stepsize_controller=stepsize_controller
        )

        print(sol.ts)  # DeviceArray([0.   , 1.   , 2.   , 3.    ])
        print(sol.ys)  # DeviceArray([1.   , 0.368, 0.135, 0.0498])


def try_ode2():
    def vector_field(t, y, args):
        prey, predator = y
        α, β, γ, δ = args
        d_prey = α * prey - β * prey * predator
        d_predator = -γ * predator + δ * prey * predator
        d_y = d_prey / u.ms, d_predator / u.ms
        return d_y

    term = ODETerm(vector_field)
    solver = Tsit5()
    t0 = 0 * u.ms
    t1 = 140 * u.ms
    dt0 = 0.1 * u.ms
    y0 = (10.0, 10.0)
    args = (0.1, 0.02, 0.4, 0.02)
    saveat = SaveAt(ts=u.math.linspace(t0, t1, 100), t1=True, t0=True)
    sol = diffeqsolve(term, solver, t0, t1, dt0, y0, args=args, saveat=saveat)

    plt.plot(sol.ts, sol.ys[0], label="Prey")
    plt.plot(sol.ts, sol.ys[1], label="Predator")
    plt.legend()
    plt.show()


def try_sde1():
    # t0, t1 = 1 * u.ms, 3 * u.ms
    # drift = lambda t, y, args: -y / u.ms
    # def diffusion(t, y, args):
    #     return 0.1 * t
    #
    # brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-3 * u.ms, shape=(), key=jr.PRNGKey(0))
    # terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, brownian_motion))
    # solver = Euler()
    #
    # saveat = SaveAt(dense=True)
    # sol = diffeqsolve(terms, solver, t0, t1, dt0=0.05 * u.ms, y0=1.0, saveat=saveat)
    # print(sol.evaluate(1.1))  # DeviceArray(0.89436394)


    t0, t1 = 1 , 3
    drift = lambda t, y, args: -y
    def diffusion(t, y, args):
        return 0.1 * t

    brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-3 , shape=(), key=jr.PRNGKey(0))
    terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, brownian_motion))
    solver = Euler()

    saveat = SaveAt(dense=True)
    sol = diffeqsolve(terms, solver, t0, t1, dt0=0.05 , y0=1.0, saveat=saveat)
    print(sol.evaluate(1.1))  # DeviceArray(0.89436394)


def try_cde1():
    class QuadraticPath(AbstractPath):
        @property
        def t0(self):
            return 0

        @property
        def t1(self):
            return 3

        def evaluate(self, t0, t1=None, left=True):
            del left
            if t1 is not None:
                return self.evaluate(t1) - self.evaluate(t0)
            return t0 ** 2

    vector_field = lambda t, y, args: -y
    control = QuadraticPath()
    term = ControlTerm(vector_field, control).to_ode()
    solver = Dopri5()
    sol = diffeqsolve(term, solver, t0=0, t1=3, dt0=0.05, y0=1)

    print(sol.ts)  # DeviceArray([3.])
    print(sol.ys)  # DeviceArray([0.00012341])


def try_stiff_ode():
    jax.config.update("jax_enable_x64", True)

    class Robertson(eqx.Module):
        k1: float
        k2: float
        k3: float

        def __call__(self, t, y, args):
            f0 = -self.k1 * y[0] + self.k3 * y[1] * y[2]
            f1 = self.k1 * y[0] - self.k2 * y[1] ** 2 - self.k3 * y[1] * y[2]
            f2 = self.k2 * y[1] ** 2
            return u.math.stack([f0, f1, f2]) / u.ms

    @jax.jit
    def main(k1, k2, k3):
        robertson = Robertson(k1, k2, k3)
        terms = ODETerm(robertson)
        t0 = 0.0 * u.ms
        t1 = 100.0 * u.ms
        dt0 = 0.0002 * u.ms
        y0 = jnp.array([1.0, 0.0, 0.0])
        solver = diffrax.Kvaerno5()
        saveat = SaveAt(ts=jnp.array([0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]) * u.ms)
        stepsize_controller = PIDController(rtol=1e-7, atol=1e-7)
        sol = diffeqsolve(
            terms,
            solver,
            t0,
            t1,
            dt0,
            y0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=100000,
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


def try_ode_force_term():
    def force(t, args):
        m, c = args
        return m * t / u.ms + c

    def vector_field(t, y, args):
        return (-y + force(t, args)) / u.ms

    @jax.jit
    def solve(y0, args):
        term = ODETerm(vector_field)
        solver = Tsit5()
        t0 = 0 * u.ms
        t1 = 10 * u.ms
        dt0 = 0.1 * u.ms
        saveat = SaveAt(ts=u.math.linspace(t0, t1, 1000))
        sol = diffeqsolve(term, solver, t0, t1, dt0, y0, args=args, saveat=saveat)
        return sol

    y0 = 1.0
    args = (0.1, 0.02)
    sol = solve(y0, args)

    plt.plot(sol.ts, sol.ys)
    plt.xlabel("t")
    plt.ylabel("y")
    plt.show()


def try_ode_force_term2():
    def vector_field2(t, y, interp):
        return -y + interp.evaluate(t)

    @jax.jit
    @jax.grad
    def solve(points):
        t0 = 0
        t1 = 10
        ts = jnp.linspace(t0, t1, len(points))
        coeffs = backward_hermite_coefficients(ts, points)
        interp = CubicInterpolation(ts, coeffs)
        term = ODETerm(vector_field2)
        solver = Tsit5()
        dt0 = 0.1
        y0 = 1.0
        sol = diffeqsolve(term, solver, t0, t1, dt0, y0, args=interp)
        (y1,) = sol.ys
        return y1

    points = jnp.array([3.0, 0.5, -0.8, 1.8])
    grads = solve(points)

    print(grads)


if __name__ == '__main__':
    pass
    # try2()
    # try_ode1()
    # try_ode2()
    # try_sde1()
    try_cde1()
    # try_stiff_ode()
    # try_ode_force_term()
    # try_ode_force_term2()
