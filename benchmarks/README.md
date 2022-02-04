# Benchmarks

These benchmarks are a work-in-progress.

Currently there is:

- `small_neural_ode.py`, which either evaluates, differentiates, evaluates at multiple points, or differentiates at multiple points, a small neural ODE. Try running `python small_neural_ode.py` with or without the `--grad` and `--multiple` flags. The results are compared against torchdiffeq and `jax.experimental.ode`.
- `lotka_volterra.py`, which solves the Lotka-Volterra equations. These results should be compared against the DifferentialEquations.jl benchmarks for the same problem (see the code).

As a general rule, Diffrax seems to match the performance of `jax.experimental.ode` and DifferentialEquations.jl. Meanwhile all of these are multiple times faster than torchdiffeq.
