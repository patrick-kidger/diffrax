# How to choose a solver

## Ordinary differential equations

The full list of ODE solvers is available on the [ODE solvers](../api/solvers/ode_solvers.md) page.

!!! info

    ODE problems are informally divided into "stiff" and "non-stiff" problems. "Stiffness" generally refers to how difficult an equation is to solve numerically. Non-stiff problems are quite common, and usually solved using straightforward techniques like explicit Runge--Kutta methods. Stiff problems usually require more computationally expensive techniques, like implicit Runge--Kutta methods.

### Non-stiff problems

For non-stiff problems then [`diffrax.Tsit5`][] is a good general-purpose solver.

!!! note
    
    For a long time the recommend default solver for many problems was [`diffrax.Dopri5`][]. This is the default solver used in [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq/), and is the solver used in MATLAB's `ode45`. However `Tsit5` is now reckoned on being slightly more efficient overall. (Try both if you wish.)

If you need accurate solutions at tight tolerances then try [`diffrax.Dopri8`][].

If you are solving a neural differential equation, and training via discretise-then-optimise (corresponding to `diffeqsolve(..., adjoint=RecursiveCheckpointAdjoint())`, which is the default), then accurate solutions are often not needed and a low-order solver will be most efficient. For example something like [`diffrax.Heun`][].

### Stiff problems

For stiff problems then try the [`diffrax.Kvaerno3`][], [`diffrax.Kvaerno4`][], [`diffrax.Kvaerno5`][] family of solvers. In addition you should almost always use an adaptive step size controller such as [`diffrax.PIDController`][].

See also the [Stiff ODE example](../examples/stiff_ode.ipynb).

!!! danger

    If solving a differential equation (stiff or not) to relatively high tolerances (typically $10^{-8}$ or lower) then you should make sure to [use 64-bit precision](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision), instead of JAX's default 32-bit precision. Not doing so can introduce a variety of interesting errors. For example the following are all symptoms of having failed to do this:

    - `NaN` gradients;
    - Taking many more solver steps than necessary (e.g. 8 steps -> 800 steps);
    - Wrapping with `jax.value_and_grad` or `jax.grad` actually changing the result of the primal (forward) computation.

### Split problems

For "split stiffness" problems, with one term that is stiff and another term that is non-stiff, then IMEX methods are appropriate: [`diffrax.KenCarp4`][] is recommended. In addition you should almost always use an adaptive step size controller such as [`diffrax.PIDController`][].

---

## Stochastic differential equations

SDE solvers are relatively specialised depending on the type of problem. Each solver will converge to either the Itô solution or the Stratonovich solution of the SDE. The Itô and Stratonovich solutions coincide iff the SDE has additive noise (as defined below). In addition some solvers require the SDE to have "commutative noise" or "additive noise". All of these terms are defined below.

### General (noncommutative) noise
This includes any SDE of the form $dy(t) = f(y(t), t) dt + g(y(t), t) dw(t),$ where $t \in [0, T]$, $y(t) \in \mathbb{R}^e$, and $w$ is a standard Brownian motion on $\mathbb{R}^d$. We refer to $f: \mathbb{R}^e \times [0, T] \to \mathbb{R}^e$ as the drift vector field (VF) and $g: \mathbb{R}^e \times [0, T] \to \mathbb{R}^{e \times d}$ is the diffusion matrix field with columns $g_i$ for $i = 1, \ldots, d$.


### Commutative noise
The diffusion matrix $σ$ is said to satisfy the commutativity condition if

$\sum_{i=1}^d g_{i, j} \frac{\partial g_{k, l}}{\partial y_i} = \sum_{i=1}^d g_{i, l} \frac{\partial g_{k, j}}{\partial y_i}$

For example, this holds:

- when $g$ is a diagonal operator (i.e. $g(y,t)$ is a diagonal matrix for all $y, t$ and the i-th diagonal entry depends only on $y_i$),
- when the dimension of BM is $d=1$, or
- when the noise is additive (see below).

- The solver with the highest order of convergence for commutative noise SDEs is [`diffrax.SlowRK`][]. [`diffrax.ItoMilstein`][] and [`diffrax.StratonovichMilstein`][] are alternatives which evaluate the vector field fewer times per step, but also compute its derivative.


### Additive noise
We say that the diffusion is additive when $g$ does not depend on $y(t)$ and the SDE can be written as $dy(t) = f(y(t), t) dt + g(t) dw(t)$.

Additive noise is a special case of commutative noise. For additive noise SDEs, the Itô and Stratonovich solutions conicide. Some solvers are specifically designed for additive noise SDEs, of these [`diffrax.SEA`][] is the cheapest, [`diffrax.ShARK`][] is the most accurate and [`diffrax.SRA1`][] is another alternative.

### Itô

For Itô SDEs:

- For general noise [`diffrax.Euler`][] is a typical choice.
- If the noise is commutative then [`diffrax.ItoMilstein`][] is a typical choice;

### Stratonovich

For Stratonovich SDEs:

- If cheap low-accuracy solves are desired then [`diffrax.EulerHeun`][] is a typical choice.
- For general noise, [`diffrax.GeneralShARK`][] is the most efficient choice, while [`diffrax.Heun`][] is a good cheap alternative.
- If an embedded method for adaptive step size control is desired and the noise is noncommutative then [`diffrax.SPaRK`][] is the recommended choice.
- If the noise is commutative, then [`diffrax.SlowRK`][] has the best order of convergence, but is expensive per step. [`diffrax.StratonovichMilstein`][] is a good cheaper alternative.

### More information about SDE solvers

A detailed example of how to simulate SDEs can be found in the [SDE example](../examples/sde_example.ipynb).
A table of all SDE solvers and their properties can be found in [SDE solver table](../devdocs/SDE_solver_table.md).

---

## Controlled differential equations

### As an ODE

If the control is differentiable (e.g. an interpolation of discrete data) and isn't somehow "rough" (i.e. doesn't wiggle up and down at a very fine timescale that is smaller than you want to make numerical steps) then probably the best way to solve the CDE is to reduce it to an ODE:

```python
vector_field = ...
control = ...
term = ControlTerm(vector_field, control)
term = term.to_ode()
```

Then use any of the ODE solvers as discussed above.

### Directly discretising the control

The other option is to directly discretise the control. Given some control $x \colon [0, T] \to \mathbb{R}^d$ then this means solving $\mathrm{d}y(t) = f(y(t)) \mathrm{d} x(t)$ by treating $x$ a bit like time, and replacing the $\Delta t$ in most numerical solvers with some $\Delta x(t)$ instead.

(This is actually the principle on which many SDE solvers work.)

It is an open question what are the best solvers to use when taking this approach, but low-order solvers are typical: for example [`diffrax.Euler`][] or [`diffrax.Heun`][].
