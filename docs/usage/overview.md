# Overview

The main function in Diffrax is `diffeqsolve`. This solves the initial value problem corresponding to an ordinary/stochastic/controlled differential equation.

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
- The solution is saved at `t=0, 1, 2, 3`.
- The initial step is of size 0.1.
- For all later steps, an I-controller is used to dynamically adapt step sizes to match a desired error tolerance.
    - This is the standard "textbook" way of adapting step sizes in numerical ODE solvers.
    - Unlike some other differential equation solving libraries, Diffrax is more flexible by allowing you to specify the step size controller separately to the numerical update rule (the solver).

This demonstrates pretty much everything you'll need to get started solving ODEs with Diffrax, just by switching things out in the obvious way.

- Depending on the problem, you may wish to use a different solver. See the guide on [How to choose a solver](./how-to-choose-a-solver.md).
- The initial step size can be selected adaptively by setting `dt0=None`.
- A constant step size can be used by setting `stepsize_controller = ConstantStepSize()`.

## Stochastic differential equations (SDEs)

TODO

## Controlled differential equations (CDEs)

TODO
