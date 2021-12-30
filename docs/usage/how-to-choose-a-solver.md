# How to choose a solver

The full list of solvers is available on the [Solvers](../api/solver.md) page.

Note that many of the following solvers have convenience wrappers (e.g. [`diffrax.tsit5`][] for [`diffrax.Tsit5`][]) as a shortcut for writing out the [Terms](../api/terms/md) explicitly.

## Ordinary differential equations

### Non-stiff problems

For non-stiff problems then [`diffrax.Tsit5`][] is a good general-purpose solver.

!!! note
    
    For a long time the recommend default solver for many problems was [`diffrax.Dopri5`][]. This is the default solver used in [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq/), and is the solver used in MATLAB's `ode45`. However `Tsit5` is now reckoned on being slightly more efficient overall.

If you need accurate solutions at high tolerances then try [`diffrax.Dopri8`][].

If you are solving a neural differential equation, *and training via discretise-then-optimise*, then often very accurate solutions are not needed and a low-order solver will be most efficient. For example something like [`diffrax.Heun`][].

### Stiff problems

For stiff problems then try the [`diffrax.Kvaerno3`][], [`diffrax.Kvaerno4`][], [`diffrax.Kvaerno5`][] family of solvers.

In each case note that they take a [Nonlinear Solver](../api/nonlinear_solver.md) as an argument, which you should set appropriately; see also the [Stiff ODE example](../examples/stiff_ode.ipynb).

## Stochastic differential equations

### Itô

For Ito SDEs then [`diffrax.Euler`][] is a typical choice.

### Stratonovich

For Stratonovich SDEs then [`diffrax.Heun`][] is a typical choice.

## Controlled differential equations

### As an ODE

If the control is differentiable (e.g. an interpolation of discrete data) and isn't somehow "rough" (i.e. doesn't wiggle up and down at a very fine timescale that is smaller than you want to make numerical steps) then probably the best way to solve the CDE is to reduce it to an ODE:

```python
vector_field = ...
control = ...
term = ControlTerm(vector_field, control)
term.to_ode()
```

Then use any of the ODE solvers as discussed above.

### Directly discretising the control

The other option is to directly discretise the control. Given some control $x \colon [0, T] \to \mathbb{R}^d$ then this means solving $\mathrm{d}y(t) = f(y(t)) \mathrm{d} x(t)$ by treating $x$ a bit like time, and replacing the $\Delta t$ in most numerical solvers with some $\Delta x(t)$ instead.

(This is actually the principle on which many SDE solvers work.)

It is an open question what are the best solvers to use when taking this approach, but low-order solvers are typical: for example [`diffrax.Euler`][] or [`diffrax.Heun`][].
