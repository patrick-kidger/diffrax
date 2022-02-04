# Benchmarks

Benchmark scripts can be found in the [benchmarks](https://github.com/patrick-kidger/diffrax/tree/main/benchmarks) folder.

As an overall summary, Diffrax, DifferentialEquations.jl, and `jax.experimental.ode` all get very similar performance. Meanwhile torchdiffeq is ~1.3 to ~20 times slower depending on the problem.
