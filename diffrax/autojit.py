from dataclasses import dataclass
import functools as ft
import jax
import jax.numpy as jnp
from typing import Any


@ft.lru_cache(maxsize=4096)
def _autojit_cache(f, args_treedef, static_argnums, **jitkwargs):
    @ft.partial(jax.jit, static_argnums=static_argnums, **jitkwargs)
    def f_wrapped(*args):
        args = jax.tree_unflatten(args_treedef, args)
        return f(*args)

    return f_wrapped


@dataclass(frozen=True)
class _UnPyTreeAble:
    value: Any


def autojit(f, static_argnums=None, static_argnames=None, donate_argnums=(), **jitkwargs):
    """JIT with some bells and whistles:
    - Automatically sets whether arguments are static or not: if they can be traced they will be, and will only be
      static if they have to be. The static_argnums can still be used to specify any extra static arguments, e.g.
      to dispatch on boolean conditionals.
    - If passed a PyTree of mixed static-able/non-static-able quantities, then these will be auto-detected as above.
      In particular this means that PyTree arguments with non-static-able quantities (e.g. functions) can still have
      tracing performed with respect to all their other quantities.
    """
    if isinstance(static_argnums, int):
        static_argnums = (static_argnums,)
    if static_argnames is not None:
        raise NotImplementedError
    if donate_argnums != ():
        raise NotImplementedError

    @ft.wraps(f)
    def f_wrapper(*args):
        if static_argnums is not None:
            args = list(args)
            for index in static_argnums:
                args[index] = _UnPyTreeAble(args[index])
        args_flat, args_treedef = jax.tree_flatten(args)
        new_static_argnums = []
        for i, arg in enumerate(args_flat):
            try:
                jnp.array(arg)
            except TypeError:
                new_static_argnums.append(i)
        new_static_argnums = tuple(new_static_argnums)
        if static_argnums is not None:
            args_flat = [arg.value if isinstance(arg, _UnPyTreeAble) else arg]
        f_jitted = _autojit_cache(f, args_treedef, new_static_argnums, **jitkwargs)
        return f_jitted(*args_flat)

    return f_wrapper
