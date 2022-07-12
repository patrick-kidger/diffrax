# FAQ

### The solve is taking loads of steps / I'm getting NaN gradients / other weird behaviour.

Try switching to 64-bit precision. (Instead of the 32-bit that is the default in JAX.) [See here](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision).

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
