# Diffrax in a nutshell

!!! tip

    If you're new to Diffrax, then read this page first! Get started with Diffrax by adapting one of the quick examples below to your problem.

The main function in Diffrax is `diffeqsolve`. This solves the initial value problem corresponding to an ordinary/stochastic/controlled differential equation.

!!! note

    Have a look through the "Examples" on the sidebar for in-depth examples including things like training loops for neural ODEs, etc.

    The "Basic API" on the sidebar is the complete reference for everything you need to know to solve ODEs.

    The "Advanced API" on the sidebar is the extended referenec to include everything needed to solve SDEs and CDEs.

## Ordinary differential equations (ODEs)

An illustrative example is as follows, which solves the ODE

$y(0) = 1 \qquad \frac{\mathrm{d}y}{\mathrm{d}t}(t) = -y(t)$

over the interval $[0, 3]$.

```python
from diffrax import diffeqsolve, dopri5, SaveAt, IController

vector_field = lambda t, y, args: -y
solver = dopri5(vector_field)
saveat = SaveAt(ts=[0., 1., 2., 3.])
stepsize_controller = IController(rtol=1e-5, atol=1e-5)

sol = diffeqsolve(solver, t0=0, t1=3, y0=1, dt0=0.1, saveat=saveat,
                  stepsize_controller=stepsize_controller)

print(sol.ts)  # DeviceArray([0.   , 1.   , 2.   , 3.    ])
print(sol.ys)  # DeviceArray([1.   , 0.368, 0.135, 0.0498])
```

- The numerical solver used is `dopri5`. [Dormand--Prince 5(4), an explicit Runge--Kutta method.]
- The solution is saved at the times `0`, `1`, `2`, `3`.
- The initial step is of size `0.1`.
- For all later steps, an I-controller is used to dynamically adapt step sizes to match a desired error tolerance.
    - This is the standard "textbook" way of adapting step sizes in numerical ODE solvers.
    - Unlike some other differential equation solving libraries, Diffrax is more flexible by allowing you to specify the step size controller separately to the numerical update rule (the solver).

!!! note

    This example demonstrates pretty much everything you'll need to get started solving ODEs with Diffrax, just by switching things out in the obvious way.

- The numerical solver (here `dopri5`) can be switched out.
    - See the guide on [How to choose a solver](./how-to-choose-a-solver.md).
    - See the [Solvers](../api/solver.md) page for the full list of solvers.
- Where to save the result (e.g. to obtain dense output) can be adjusted by changing [`diffrax.SaveAt`][].
- Step sizes and locations can be changed.
    - The initial step size can be selected adaptively by setting `dt0=None`.
    - A constant step size can be used by setting `stepsize_controller = ConstantStepSize()`.
    - Things like solver tolerances, jumps in the vector field, etc. can be passed as arguments to the step size controller.
    - See the page on [Step size controllers](../api/stepsize_controller.md).
- Any static arguments (that do not evolve over time) can be passed as `diffeqsolve(..., args=...)`.

---

## Stochastic differential equations (SDEs)

Now let's consider solving the Itô SDE

$y(0) = 1 \qquad \mathrm{d}y(t) = -y(t)\mathrm{d}t + \frac{t}{10}\mathrm{d}w(t)$

over the interval $[0, 3]$.

```python
import jax.random as jrandom
from diffrax import diffeqsolve, euler_maruyama, SaveAt, UnsafeBrownianPath

drift = lambda t, y, args: -y
diffusion = lambda t, y, args: 0.1 * t
brownian_motion = UnsafeBrownianPath(shape=(), key=jrandom.PRNGKey(0))
solver = euler_maruyama(drift, diffusion, brownian_motion)
saveat = SaveAt(dense=True)

sol = diffeqsolve(solver, t0=0, t1=3, y0=1, dt0=0.05, saveat=saveat)
print(sol.evaluate(0.1))  # DeviceArray(0.9026031)
```

- The numerical solver used is `euler_maruyama`. That is to say just Euler's method, applied to an SDE.
    - This converges to an Itô SDE because of the choice of solver. (Whether an SDE solver converges to Itô or Stratonovich SDE is a property of the solver.)
- The solution is saved densely -- a continuous path is the output. We can then evaluate it at any point in the interval; in this case `0.1`.
- No step size controller is specified so by default a constant step size is used.

As you can see, basically nothing has changed compared to the ODE example; all the same APIs are used. The only difference is that we created an SDE solver rather than an ODE solver.

---

## Controlled differential equations (CDEs)

Under the hood, both ODEs and SDEs are actually solved in exactly the same way: by lowering them to CDEs. It is for this reason that we're able to pack so much power into a relatively small library.

Suppose we want to solve the CDE

$y(0) = 1 \qquad \mathrm{d}y(t) = -y(t) \mathrm{d}x(t)$

over the interval $[0, 3]$, subject to the control signal $x(t) = t^2$.

!!! note

    If we had had only $x(t) = t$ then this would just be an ODE like normal.

    CDEs are useful to work with directly when the control signal isn't known ahead of time. See the [Neural CDE](../examples/neural_cde.ipynb) example, which uses CDEs to perform time series classification, where the control signal $x$ depends on the input time series.

```python
from diffrax import AbstractPath, ControlTerm, diffeqsolve, Dopri5


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

    def derivative(self, t, left=True):
        del left
        return 2 * t


vector_field = lambda t, y, args: -y
control = QuadraticPath()
term = ControlTerm(vector_field, control).to_ode()
solver = Dopri5(term)

sol = diffeqsolve(solver, t0=0, t1=3, y0=1, dt0=0.05)
print(sol.ts)  # DeviceArray([3.])
print(sol.ys)  # DeviceArray([0.00012341])
```

- We specify a control by inheriting from [`diffrax.AbstractPath`][].
    - It's very common to create a control by interpolating data instead: Diffrax provides [Interpolations](../api/interpolation.md) for this.
- Note the use of [Terms](../api/terms.md), to describe both vector field and control together.
    - Likewise note the use of [`diffrax.Dopri5`][] rather than [`diffrax.dopri5`][].

!!! tip

    [`diffrax.dopri5`][] is just a convenience wrapper for `diffrax.Dopri5(diffrax.ODETerm(...))`. As stated earlier: all ODEs and SDEs are treated by reducing them to CDEs.

- The solution is saved at just the final time point `t1`. (This is the default value of the `diffeqsolve(..., saveat=...)` argument.)
- No step size controller is specified so by default a constant step size is used.
