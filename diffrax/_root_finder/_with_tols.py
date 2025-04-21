import typing

import equinox as eqx
import jax.numpy as jnp
import optimistix as optx
import wadler_lindig as wl


class _UseStepSizeTol:
    def __repr__(self):
        return "<taken from `diffeqsolve(..., stepsize_controller=...)` argument>"


use_stepsize_tol = _UseStepSizeTol()


class _WithTolsDoc:
    def __init__(self, cls, args, kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def __pdoc__(self, **kwargs):
        first_call = wl.bracketed(
            begin=wl.TextDoc("diffrax.with_stepsize_controller_tols("),
            docs=[wl.pdoc(self.cls, **kwargs)],
            sep=wl.comma,
            end=wl.TextDoc(")"),
            indent=kwargs["indent"],
        )
        second_call = wl.bracketed(
            begin=first_call + wl.TextDoc("("),
            docs=[wl.pdoc(arg, **kwargs) for arg in self.args]
            + wl.named_objs(self.kwargs.items(), **kwargs),
            sep=wl.comma,
            end=wl.TextDoc(")"),
            indent=kwargs["indent"],
        )
        return second_call


def with_stepsize_controller_tols(cls: type[optx.AbstractRootFinder]):
    """Wraps a root finding class to indicate that it should use the same tolerances as
    were provided to an adaptive stepsize controller.

    **Arguments:**

    - `cls`: a subclass of `optimistix.AbstractRootFinder`.

    **Returns:**

    A wrapped version of `cls` that no longer accepts the `atol`, `rtol` or `norm`
    arguments, and will instead copy them from the adaptive step size controller.

    !!! Example

        ```python
        import diffrax as dfx
        import optimistix as optx

        root_finder = dfx.with_stepsize_controller_tols(optx.Chord)()
        solver = dfx.Kvaerno5(root_finder=root_finder)
        stepsize_controller = dfx.PIDController(rtol=1e-8, atol=1e-8)

        dfx.diffeqsolve(..., solver=solver, stepsize_controller=stepsize_controller)
        ```
    """

    def make(*args, **kwargs):
        if hasattr(typing, "GENERATING_DOCUMENTATION") and not typing.TYPE_CHECKING:
            return _WithTolsDoc(cls, args, kwargs)
        # Use `inf` as a dummy value to avoid triggering typechecking errors.
        # Pass `norm` explicitly to disallow passing it through `*args, **kwargs`.
        self = cls(*args, **kwargs, rtol=jnp.inf, atol=jnp.inf, norm=optx.max_norm)
        self = eqx.tree_at(
            lambda s: (s.rtol, s.atol, s.norm),
            self,
            (use_stepsize_tol, use_stepsize_tol, use_stepsize_tol),
        )
        return self

    return make
