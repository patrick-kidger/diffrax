# JAX tricks

We use a few tricks that advanced users of JAX may find interesting. These can be found in `diffrax.jax_tricks`.

**autojit**<br>
This is `jax.jit` with two extra bells and whistles.

Its first (and actually less important) trick is that anything that can't be used as a traced argument -- i.e. any Python object that can't be converted into a `jax.numpy.array` -- is automatically marked as a static argument. This is in addition to any arguments marked as a `static_argnum`. Take care with this to make sure you don't accidentally recompile many many times (e.g. see the [`chex`](https://github.com/deepmind/chex) to help catch this error).

Its second trick is that it flattens PyTrees prior to passing them into `jax.jit`. Because of the automatic static-vs-traced argument detection (which occurs *after* flattening the tree), it means that `jax.jit` can understand PyTrees with a mix of static-able and trace-able components. (Contrast marking the whole PyTree as a `static_argnum` in typical `jax.jit`, which makes the whole PyTree static, potentially provoking lots of recompilations.)

**tree_dataclass**<br>
This is equivalent to the `dataclasses.dataclass` standard library decorator, except that it also marks the dataclass as being PyTree-able (i.e. can be flatten/unflattened to/from its arguments).

*The real magic is the way `tree_dataclass` and `autojit` combine with each other.*

Dataclasses offer a convenient way to express `parameterised functions`.

