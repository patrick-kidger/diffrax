# Getting started

!!! tip

    If you're new to Diffrax, then this page should tell you everything you need to get started. Try adapting one of the quick examples below to your problem.

    After that:

    - Have a look through the "Examples" on the sidebar for in-depth examples including things like training loops for neural ODEs, etc.
    - The "Basic API" on the sidebar is the main reference for everything you need to know to solve ODEs.
    - The "Advanced API" on the sidebar is the extended reference, in particular including everything needed to solve SDEs and CDEs.

The main function in Diffrax is [`diffrax.diffeqsolve`][]. This solves the initial value problem corresponding to an ordinary/stochastic/controlled differential equation.

## Ordinary differential equations (ODEs)

An illustrative example is as follows, which solves the ODE

$y(0) = 1 \qquad \frac{\mathrm{d}y}{\mathrm{d}t}(t) = -y(t)$

over the interval $[0, 3]$.

```python
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController

vector_field = lambda t, y, args: -y
term = ODETerm(vector_field)
solver = Dopri5()
saveat = SaveAt(ts=[0., 1., 2., 3.])
stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

sol = diffeqsolve(term, solver, t0=0, t1=3, dt0=0.1, y0=1, saveat=saveat,
                  stepsize_controller=stepsize_controller)

print(sol.ts)  # DeviceArray([0.   , 1.   , 2.   , 3.    ])
print(sol.ys)  # DeviceArray([1.   , 0.368, 0.135, 0.0498])
```

- The numerical solver used is `Dopri5`. [Dormand--Prince 5(4), an explicit Runge--Kutta method.]
- The solution is saved at the times `0`, `1`, `2`, `3`.
- The initial step is of size `0.1`.
- For all later steps, a PID controller is used to dynamically adapt step sizes to match a desired error tolerance.
    - This is a more efficient variation on the "textbook" way to adapt step sizes (which is an I controller, also known as a deadbeat controller).
    - The step size controller is specified separately to the solver.

!!! note

    This example demonstrates pretty much everything you'll need to get started solving ODEs with Diffrax, just by switching things out in the obvious way.

    - The numerical solver (here `Dopri5`) can be switched out.
        - See the guide on [How to choose a solver](./how-to-choose-a-solver.md).
        - See the [ODE solvers](../api/solvers/ode_solvers.md) page for the full list of solvers.
    - Where to save the result (e.g. to obtain dense output) can be adjusted by changing [`diffrax.SaveAt`][].
    - Step sizes and locations can be changed.
        - The initial step size can be selected adaptively by setting `dt0=None`.
        - A constant step size can be used by setting `stepsize_controller = ConstantStepSize()`. (This is also the default choice for `stepsize_controller` if you do not pass one at all.)
        - Things like solver tolerances, jumps in the vector field, etc. can be passed as arguments to the step size controller.
        - See the page on [Step size controllers](../api/stepsize_controller.md).
    - Any static arguments (that do not change during the integration) for the `vector_field` can be passed as `diffeqsolve(..., args=...)`.

---

## Stochastic differential equations (SDEs)

Now let's consider solving the Itô SDE

$y(0) = 1 \qquad \mathrm{d}y(t) = -y(t)\mathrm{d}t + \frac{t}{10}\mathrm{d}w(t)$

over the interval $[0, 3]$.

```python
import jax.random as jr
from diffrax import diffeqsolve, ControlTerm, Euler, MultiTerm, ODETerm, SaveAt, VirtualBrownianTree

t0, t1 = 1, 3
drift = lambda t, y, args: -y
diffusion = lambda t, y, args: 0.1 * t
brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-3, shape=(), key=jr.PRNGKey(0))
terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, brownian_motion))
solver = Euler()
saveat = SaveAt(dense=True)

sol = diffeqsolve(terms, solver, t0, t1, dt0=0.05, y0=1.0, saveat=saveat)
print(sol.evaluate(1.1))  # DeviceArray(0.89436394)
```

- On terms:
    - We use `ODETerm` to describe the $-y(t)\mathrm{d}t$ term.
    - We use `ControlTerm` to describe the $\frac{t}{10}\mathrm{d}w(t)$ term.
    - We use `MultiTerm` to bundle both of these terms together.
    - Find out more about terms on the [Terms](../api/terms.md) page.
- The numerical solver used is `Euler()`. (Also known as Euler--Maruyama when applied to SDEs.)
    - There's no clever hackery behind the scenes: `Euler()` for an SDE simply works in exactly the same way as `Euler()` for an ODE -- we just need to specify the extra diffusion term.
    - This converges to an Itô SDE because of the choice of solver. (Whether an SDE solver converges to Itô or Stratonovich SDE is a property of the solver.)
    - See the [SDE solvers](../api/solvers/sde_solvers.md) page for the full list of solvers.
- The solution is saved densely -- a continuous path is the output. We can then evaluate it at any point in the interval; in this case `0.1`.
- No step size controller is specified so by default a constant step size is used.

As you can see, basically nothing has changed compared to the ODE example; all the same APIs are used. The only difference is that we created an SDE solver rather than an ODE solver.


!!! info

    To do adaptive stepping with an SDE, then the typical approach is to wrap the solver like so -- and to use the following default step size controller:
    ```python
    solver = ...  # Euler, Heun etc. as usual
    solver = HalfSolver(solver)  # Computes error estimates using half-steps
    stepsize_controller = PIDController(pcoeff=0.1, icoeff=0.3, rtol=..., atol=...)
    ```
    (The values of `pcoeff` and `icoeff` affect how rapidly the controller changes step sizes. The optimal values are not well-understood for SDEs. Do try several values and see what seems most efficient for your problem.)

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


vector_field = lambda t, y, args: -y
control = QuadraticPath()
term = ControlTerm(vector_field, control).to_ode()
solver = Dopri5()
sol = diffeqsolve(term, solver, t0=0, t1=3, dt0=0.05, y0=1)

print(sol.ts)  # DeviceArray([3.])
print(sol.ys)  # DeviceArray([0.00012341])
```

- We specify a control by inheriting from [`diffrax.AbstractPath`][].
    - It's very common to create a control by interpolating data: Diffrax provides some [interpolation routines](../api/interpolation.md) for this.
- No `diffeqsolve(..., saveat=...)` argument is passed, so the default is used: saving at just the final time point `t1`.
- No step size controller is specified so by default a constant step size is used.
