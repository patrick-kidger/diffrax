# FAQ

### The solve is taking loads of steps / I'm getting NaN gradients / other weird behaviour.

Try switching to 64-bit precision. (Instead of the 32-bit that is the default in JAX.) [See here](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision).

### Diffrax seem to be slower than <some other library\>?

Questions of this form are a fairly common source of issues in the Diffrax issue tracker! In practice, Diffrax is consistently amongst the fastest ODE solvers, and these usually stem from incorrect usage (e.g. recompiling your JAX program on each invocation) or comparisons (e.g. using different solvers/tolerances in each implementation).

Here's a list of some of the things to keep in mind when performing such comparisons:

1. First of all, the usual list of JAX profiling concerns:

    a. Make sure that your JAX program is compiled only once, and not repeatedly on each invocation (for example by passing in different raw Python floats each time). Use [`equinox.debug.assert_max_traces(max_traces=1)`](https://docs.kidger.site/equinox/api/debug/#equinox.debug.assert_max_traces) to debug this.

    b. Your entire computation should be wrapped in a single `jax.jit`'d function (or equivalently `equinox.filter_jit`).

    c. Run this function in advance (to JIT-compile it), before running it again to measure its speed.

    d. Make sure not to include any code that is ran outside of the JIT'd function in your timings.

    e. Make sure to call `jax.block_until_ready` on the output of the the function.

    Typically your code should follow this template:
    ```python
    import equinox as eqx
    import jax
    import timeit

    @jax.jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def run(x):
        ...

    x = ...
    run(x)  # compile
    execution_time = min(timeit.repeat(lambda: jax.block_until_ready(run(x)), number=1, repeat=20))
    ```

2. Use the same ODE solver in both implementations to get an apples-to-apples comparison. It's not surprising that different solvers give different performance characteristics. (And if one implementation does not provide a solver that the other does, then no comparison can be made.)

3. Use the same step size control in both implementations.

    a. If using adaptive step sizes then note that tolerances (the `rtol`, `atol` in `diffeqsolve(..., stepsize_controller=PIDController(rtol=..., atol=...))`) have solver- and implementation-specific meanings, so having these be the same is not enough. Aim to have roughly the same number of steps instead. You can check the number of steps taken in Diffrax via `diffeqsolve(...).stats['num_steps']`.

    b. If using an automatic initial step size (`diffeqsolve(..., dt0=None)`) then use this (or disable this) in both implementations.

4. If comparing to other JAX implementations, then make sure to set `import os; os.environ["EQX_ON_ERROR"] = "nan"` at the top of your script (before you import Diffrax or Equinox). This will disable various runtime correctness checks performed by Diffrax that are are typically not performed by other JAX frameworks. These add a few milliseconds of overhead that typically does not matter in real-word usage but may be large enough to appear in microbenchmarks.

    a. If comparing to a loop-over-steps using `jax.lax.scan`, then the equivalent step size control in Diffrax is `diffeqsolve(..., stepsize_controller=StepTo(...))`.

5. If you'd like to be really precise, then the best way to benchmark competing implementations is with a work-precision diagram: solve your ODE once with very tight tolerances and a very accurate solver (in any implementation). Then for each implementation: vary the tolerances or step sizes, and plot the time for the solve against and the numerical difference between the solution and the very accurate solution. This isn't required but is the gold-standard for benchmark comparisons.

6. Both implementations should use the same precision (`float32` vs `float64`). Note that JAX defaults to 32-bit precision and requires a flag to enable 64-bit precision.

7. The problem being solved should be large enough (ideally at least 100 milliseconds to solve) that you are not simply measuring various small overheads in different frameworks.

Take a look at [Diffrax issue #82](https://github.com/patrick-kidger/diffrax/issues/82) for a good example of how seemingly-reasonable benchmarks can hide a few pitfalls!

If you think you have a performance issue – after checking all of the above! – then feel free to open an issue on the Diffrax issue page. You should include a code snippet that demonstrates the issue; typically this should not be more than around 50 lines long if we are going to be able to volunteer to help you debug it :-).

### How does this compare to `jax.experimental.ode.odeint`?

The equivalent solver in Diffrax is:
```python
diffeqsolve(
    ...,
    dt0=None,
    solver=Dopri5(),
    stepsize_controller=PIDController(rtol=1.4e-8, atol=1.4e-8),
    adjoint=BacksolveAdjoint(),
    max_steps=None,
)
```

In practice, [`diffrax.Tsit5`][] is usually a better solver than [`diffrax.Dopri5`][]. And the default adjoint method ([`diffrax.RecursiveCheckpointAdjoint`][]) is usually a better choice than [`diffrax.BacksolveAdjoint`][].

### I'm getting a `CustomVJPException`.

This can happen if you use [`diffrax.BacksolveAdjoint`][] incorrectly.

Gradients will be computed for:

- Everything in the `args` PyTree passed to `diffeqsolve(..., args=args)`;
- Everything in the `y0` PyTree passed to `diffeqsolve(..., y0=y0)`.
- Everything in the `terms` PyTree passed to `diffeqsolve(terms, ...)`.

Attempting to compute gradients with respect to anything else will result in this exception.

!!! example

    Here is a minimal example of **wrong** code that will raise this exception.

    ```python
    from diffrax import BacksolveAdjoint, diffeqsolve, Euler, ODETerm
    import equinox as eqx
    import jax.numpy as jnp
    import jax.random as jr

    mlp = eqx.nn.MLP(1, 1, 8, 2, key=jr.PRNGKey(0))

    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def run(model):
      def f(t, y, args):  # `model` captured via closure; is not part of the `terms` PyTree.
        return model(y)
      sol = diffeqsolve(ODETerm(f), Euler(), 0, 1, 0.1, jnp.array([1.0]),
                        adjoint=BacksolveAdjoint())
      return jnp.sum(sol.ys)

    run(mlp)
    ```

!!! example

    The corrected version of the previous example is as follows. In this case, the model is properly part of the PyTree structure of `terms`.

    ```python
    from diffrax import BacksolveAdjoint, diffeqsolve, Euler, ODETerm
    import equinox as eqx
    import jax.numpy as jnp
    import jax.random as jr

    mlp = eqx.nn.MLP(1, 1, 8, 2, key=jr.PRNGKey(0))

    class VectorField(eqx.Module):
        model: eqx.Module

        def __call__(self, t, y, args):
            return self.model(y)

    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def run(model):
      f = VectorField(model)
      sol = diffeqsolve(ODETerm(f), Euler(), 0, 1, 0.1, jnp.array([1.0]), adjoint=BacksolveAdjoint())
      return jnp.sum(sol.ys)

    run(mlp)
    ```
