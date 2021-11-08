# FAQ

### Is it possible to JIT a `diffeqsolve`?
No: solving a differential equation can involve a variable number of time steps. This kind of variable-step behaviour is something that JAX isn't always capable of jitting.

Instead, Diffrax uses JIT internally, so that good performance is still maintained.

(It should be possible to add support for JIT in certain special cases, but this hasn't happened yet.)

### Why is `diffeqsolve` re-JIT-ing my forward pass each time I call it? (Possibly the only symptom is that things seem slow.)
Probably you're doing something like this:
```python
for _ in range(steps):  # e.g. a training loop
    fn = lambda t, y, args: -y
    diffeqsolve(euler(fn), ...)  # forward pass of model
```
in which the vector field (`fn`) keeps being redefined. This means that `jax.jit` has to recompile everything, as it can't tell that `fn` does the same thing each time.

The fix is to factor out `fn` -- e.g. define it at the global scope. That way `jax.jit` knows it is the same function each time. (The `args` argument to `fn` can be used in place of any variables previously captured via closure.)

### Why do the first few (typically <5) backpropagations through `diffeqsolve` take longer?
You're probably (a) using an adaptive step size controller and (b) are backpropagating via discretise-then-optimise.

In this case a variable number of steps have to be recorded during the forward pass. Certain parts of the internals need to be re-JIT'd when the number of steps varies. After the first few calls to the solver this stops happening, as all the possible step counts will have occured, and JAX can re-use its previously compiled computational graphs.

Diffrax actually optimises this a bit further -- we pick a number, e.g. 20, and pad the total number of steps to that value. This helps ensure that e.g. 16 steps and 17 steps will both use the same JIT'd function, without incurring too much overhead.
