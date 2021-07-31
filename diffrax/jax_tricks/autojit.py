from dataclasses import dataclass
import functools as ft
import jax
import jax.numpy as jnp
from typing import Any


@ft.lru_cache(maxsize=4096)
def _autojit_cache(f, args_treedef, **jitkwargs):
    @ft.partial(jax.jit, **jitkwargs)
    def f_wrapped(*args):
        args = jax.tree_unflatten(args_treedef, args)
        return f(*args)

    return f_wrapped


@dataclass(frozen=True)
class _UnPyTreeAble:
    value: Any


_marker_sentinel = object()


def autojit(f, static_argnums=None, static_argnames=None, donate_argnums=(), **jitkwargs):
    """
    A jax.jit that automatically sets whether arguments are static or not: if they can be traced they will be, and
    will only be static if they have to be. The static_argnums can still be used to specify any extra static arguments,
    e.g. to dispatch on boolean conditionals.

    The above applies recursively inside PyTrees, so that e.g. some parts of the PyTree will be static and some will be
    traced.
    """
    if isinstance(static_argnums, int):
        static_argnums = (static_argnums,)
    if static_argnames is not None:
        raise NotImplementedError
    if isinstance(donate_argnums, int):
        donate_argnums = (donate_argnums,)

    @ft.wraps(f)
    def f_wrapper(*args):
        if donate_argnums != ():
            marker_args = list(args)
            for i in donate_argnums:
                marker_args[i] = jax.tree_map(lambda _: _marker_sentinel, marker_args[i])
            marker_args_flat, _ = jax.tree_flatten(marker_args)
            new_donate_argnums = tuple(i for i, arg in enumerate(marker_args_flat) if arg is _marker_sentinel)
        else:
            new_donate_argnums = ()
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
        f_jitted = _autojit_cache(
            f, args_treedef, static_argnums=new_static_argnums, donate_argnums=new_donate_argnums, **jitkwargs
        )
        return f_jitted(*args_flat)

    return f_wrapper
