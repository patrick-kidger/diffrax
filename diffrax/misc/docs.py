import typing


def in_public_docs(obj):
    if getattr(typing, "GENERATING_DOCUMENTATION", False):
        assert "_import_alias" not in obj.__dict__
        obj._import_alias = "diffrax." + obj.__name__
    return obj


def solver_wrapper(solver, *, ode=False, sde=False):
    doc = [f"Convenience wrapper for [{solver._import_alias}][].", ""]

    ode_doc = """
**Arguments:**

- `vector_field`: The vector field, which is a function `PyTree -> PyTree`. The input
    and output PyTrees must have identical structure, and each leaf must be a JAX array
    with identical shapes.
    """

    sde_doc = """
**Arguments:**

- `{vf}`: The drift, which is a function `PyTree -> PyTree`. The input and output
    PyTrees must have identical structure, and each leaf must be a JAX array with
    identical shapes.
- `diffusion`: The diffusion, which is a function between `PyTree -> PyTree`. The
    input and output PyTrees must have identical structure. If the Brownian motion
    has shape `(w1, ..., wk)`, then for each input leaf of shape `(d1, ..., dn)` the
    corresponding output leaf must have shape `(d1, ..., dn, w1, ..., wk)`. (And a
    'generalised' matrix-vector product is performed between diffusion and noise.)
- `bm`: A Brownian motion.
    """
    if ode:
        if sde:
            doc.append(
                f"""Can be used to solve either ODEs or SDEs.

            When solving ODEs:

            {ode_doc}

            When solving SDEs:

            {sde_doc.format(vf='vector_field')}
            """
            )
        else:
            doc.append(ode_doc)
    else:
        if sde:
            doc.append(sde_doc.format(vf="drift"))
        else:
            raise ValueError

    doc.append(
        f"""
**Returns:**

An instance of [{solver._import_alias}][].
"""
    )
    doc.append("")
    doc.append(f"See [{solver._import_alias}][] for more details on the solver itself.")

    doc = "".join(doc)

    def _solver_wrapper(fn):
        fn.__doc__ = doc
        return fn

    return _solver_wrapper
