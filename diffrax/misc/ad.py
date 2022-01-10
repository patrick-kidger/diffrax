from typing import Any

import jax
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.xla as xla

from ..custom_types import PyTree


def is_perturbed(x: Any) -> bool:
    if isinstance(x, jax.ad.JVPTracer):
        return True
    elif isinstance(x, jax.core.Tracer):
        return any(is_perturbed(attr) for name, attr in x._contents())
    else:
        return False


def nondifferentiable_input(x: PyTree, name: str) -> None:
    if any(is_perturbed(xi) for xi in jax.tree_leaves(x)):
        raise ValueError(f"Cannot differentiate {name}.")


_nondifferentiable_output_p = jax.core.Primitive("nondifferentiable_output")


def _nondifferentiable_output_batch(x, batch_axes):
    (x,) = x
    (batch_axes,) = batch_axes
    return nondifferentiable_output(x), batch_axes


def _nondifferentiable_output_jvp(primals, tangents):
    (primals,) = primals
    (tangents,) = tangents
    return nondifferentiable_output(primals), nondifferentiable_output(tangents)


def _nondifferentiable_output_transpose(cts_in, _):
    if isinstance(cts_in, ad.Zero):
        return cts_in
    else:
        raise RuntimeError(
            "Attempted to backpropagate through a value for which this is invalid."
        )


_nondifferentiable_output_p.def_impl(lambda x: x)
_nondifferentiable_output_p.def_abstract_eval(lambda x: x)
batching.primitive_batchers[
    _nondifferentiable_output_p
] = _nondifferentiable_output_batch
xla.register_translation(
    _nondifferentiable_output_p,
    xla.lower_fun(lambda x: x, multiple_results=False, new_style=True),
)
ad.primitive_jvps[_nondifferentiable_output_p] = _nondifferentiable_output_jvp
ad.primitive_transposes[
    _nondifferentiable_output_p
] = _nondifferentiable_output_transpose


def nondifferentiable_output(x: PyTree) -> PyTree:
    return _nondifferentiable_output_p.bind(x)
