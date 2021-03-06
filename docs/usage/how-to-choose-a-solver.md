# How to choose a solver

## Ordinary differential equations

The full list of ODE solvers is available on the [ODE solvers](../api/solvers/ode_solvers.md) page.

!!! info

    ODE problems are informally divided into "stiff" and "non-stiff" problems. "Stiffness" generally refers to how difficult an equation is to solve numerically. Non-stiff problems are quite common, and usually solved using straightforward techniques like explicit Runge--Kutta methods. Stiff problems usually require more computationally expensive techniques, like implicit Runge--Kutta methods.

### Non-stiff problems

For non-stiff problems then [`diffrax.Tsit5`][] is a good general-purpose solver.

!!! note
    
    For a long time the recommend default solver for many problems was [`diffrax.Dopri5`][]. This is the default solver used in [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq/), and is the solver used in MATLAB's `ode45`. However `Tsit5` is now reckoned on being slightly more efficient overall. (Try both if you wish.)

If you need accurate solutions at high tolerances then try [`diffrax.Dopri8`][].

If you are solving a neural differential equation, and training via discretise-then-optimise (corresponding to `diffeqsolve(..., adjoint=RecursiveCheckpointAdjoint())`, which is the default), then accurate solutions are often not needed and a low-order solver will be most efficient. For example something like [`diffrax.Heun`][].

### Stiff problems

For stiff problems then try the [`diffrax.Kvaerno3`][], [`diffrax.Kvaerno4`][], [`diffrax.Kvaerno5`][] family of solvers. In addition you should almost always use an adaptive step size controller such as [`diffrax.PIDController`][].

See also the [Stiff ODE example](../examples/stiff_ode.ipynb).

!!! danger

    If solving a differential equation (stiff or not) to relatively high tolerances (typically $10^{-8}$ or lower) then you should make sure to [use 64-bit precision](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision), instead of JAX's default 32-bit precision. Not doing so can introduce a variety of interesting errors. For example the following are all symptoms of having failed to do this:

    - `NaN` gradients;
    - Taking many more solver steps than necessary (e.g. 8 steps -> 800 steps);
    - Wrapping with `jax.value_and_grad` or `jax.grad` actually changing the result of the primal (forward) computation.

---

## Stochastic differential equations

SDE solvers are relatively specialised depending on the type of problem. Each solver will converge to either the It?? solution or the Stratonovich solution. In addition some solvers require "commutative noise".

??? info "Commutative noise"

    Consider the SDE

    $\mathrm{d}y(t) = ??(t, y(t))\mathrm{d}t + ??(t, y(t))\mathrm{d}w(t)$

    then the diffusion matrix $??$ is said to satisfy the commutativity condition if

    $\sum_{i=1}^d ??_{i, j} \frac{\partial ??_{k, l}}{\partial y_i} = \sum_{i=1}^d ??_{i, l} \frac{\partial ??_{k, j}}{\partial y_i}$

    Some common special cases in which this condition is satisfied are:

    - the diffusion is additive ($??$ is independent of $y$);
    - the noise is scalar ($w$ is scalar-valued);
    - the diffusion is diagonal ($??$ is a diagonal matrix and such that the i-th
        diagonal entry depends only on $y_i$; *not* to be confused with the simpler
        but insufficient condition that $??$ is only a diagonal matrix)

### It??

For It?? SDEs:

- If the noise is commutative then [`diffrax.ItoMilstein`][] is a typical choice;
- If the noise is noncommutative then [`diffrax.Euler`][] is a typical choice.

### Stratonovich

For Stratonovich SDEs:

- If cheap low-accuracy solves are desired then [`diffrax.EulerHeun`][] is a good choice.
- Otherwise, and if the noise is commutative, then [`diffrax.StratonovichMilstein`][] is a typical choice.
- Otherwise, and if the noise is noncommutative, then [`diffrax.Heun`][] is a typical choice.

### Additive noise

Consider the SDE

$\mathrm{d}y(t) = ??(t, y(t))\mathrm{d}t + ??(t, y(t))\mathrm{d}w(t)$

Then the diffusion matrix $??$ is said to be additive if $??(t, y) = ??(t)$. That is to say if the diffusion is independent of $y$.

In this case then the It?? solution and the Stratonovich solution coincide, and mathematically speaking the choice of It?? vs Stratonovich is unimportant.

- The cheapest (but least accurate) solver is [`diffrax.Euler`][].
- Otherwise [`diffrax.Heun`][] is a good choice. It gets first-order strong convergence and second-order weak convergence.

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
