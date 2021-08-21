# FAQ

### Is it possible to JIT a `diffeqsolve`?
In general, solving a differential equation involves a variable number of time steps. This kind of variable-step behaviour isn't something that JAX is capable of jitting.

(It should be possible to add support for this in certain special cases, but this hasn't happened yet.)

Fortunately, `diffeqsolve` can still JIT-compile individual steps, so good performance is still maintained. (This may be turned off via `diffeqsolve(..., jit=False)` if desired.)

### Why is `diffeqsolve` re-JIT-ing my forward pass each time I call it? (Possibly the only symptom is that things seem slow.)
One of two things is probably causing this. (Both are fixable.)

Possibility one: you're doing something like this:
```python
for _ in range(steps):  # e.g. a training loop
    fn = lambda t, y, args: -y
    diffeqsolve(euler(fn), ...)  # forward pass of model
```
in which the vector field (`fn`) keeps being redefined. This means that `jax.jit` has to recompile everything, as it can't tell that `fn` does the same thing each time.

The fix is to factor out `fn` and define it at the global scope. That was `jax.jit` knows its the same function each time. The `args` argument to `fn` can be used in place of any variables captured via closure.

Possibility two: you're using JAX version <= 0.2.18, which has a bug that re-triggers unnecessary recompilations. This can be fixed by switching every `jax.vmap(...)` for `jax.vmap(..., axis_name='')`; i.e. passing a dummy axis name.

### Why do the first few (typically <5) backpropagations through `diffeqsolve` take longer?
You're probably (a) using an adaptive step size controller and (b) are backpropagating via discretise-then-optimise.

In this case a variable number of steps have to be recorded during the forward pass. Certain parts of the internals need to be re-JIT'd when the number of steps varies. After the first few calls to the solver this stops happening, as all the possible step counts will have occured, and JAX can re-use its previously compiled computational graphs.

Diffrax actually optimises this a bit further -- we pick a number, e.g. 20, and pad the total number of steps to that value. This helps ensure that e.g. 16 steps and 17 steps will both use the same JIT'd function, without incurring too much overhead.

Finally, if you want, the adaptive step size controllers offer an `unvmap_dt` option. This will lock together the timesteps to be the same value for every batch element. This will avoid the need for these additional recompilations. It usually improves the speed of the solver as well. (The trade-off is that batch elements now have a (weak) effect on each other, which breaks the `vmap` abstraction slightly. This is often fine for a lot of applications, like neural ODE training -- e.g. [torchdiffeq](https://github.com/rtqichen/torchdiffeq) has always done this.)
