# FAQ

## The solve is taking loads of steps / I'm getting NaN gradients / other weird behaviour.

Try switching to 64-bit precision. (Instead of the 32-bit that is the default in JAX.) [See here](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision).

## I'm not getting any gradient computed for one of my model parameters.

This can happen if you use `diffeqsolve(..., adjoint=BacksolveAdjoint())` incorrectly.

Gradients will be computed for:

- Everything in the `args` PyTree passed to `diffeqsolve(..., args=args)`;
- Everything in the `y0` PyTree passed to `diffeqsolve(..., y0=y0)`.
- Everything in the `terms` PyTree passed to `diffeqsolve(terms, ...)`.


!!! example

    A common example of computing gradients through `terms` is if using an [Equinox](https://github.com/patrick-kidger/equinox) module to represent a parameterised vector field. For example:

    ```python
    import equinox as eqx
    import diffrax

    class Func(eqx.Module):
        mlp: eqx.nn.MLP

        def __call__(self, t, y, args):
            return self.mlp(y)

    mlp = eqx.nn.MLP(...)
    func = Func(mlp)
    term = diffrax.ODETerm(func)
    diffrax.diffeqsolve(term, ...)
    ```

    In this case `diffrax.ODETerm`, `Func` and `eqx.nn.MLP` are all PyTrees, so all of the parameters inside `mlp` are visible to `diffeqsolve` and it can compute gradients with respect to them.

However if you were to do:

```python
model = ...

def func(t, y, args):
    return model(y)

term = diffrax.ODETerm(func)
diffrax.diffeqsolve(term, ..., adjoint=diffrax.BacksolveAdjoint())
```

then the parameters of `model` are not visible to `diffeqsolve` so gradients will not be computed with respect to them.
