# FAQ

**Why does jax.jit(diffeqsolve)(...) fail?**<br>
In general, solving a differential equation involves a variable number of time steps. This kind of variable-step behaviour isn't something that JAX is capable of jitting.

(More precisely: it does offer a limited way to do this via jax.lax.while_loop, but this doesn't support backpropagation, which is usually more important.)
