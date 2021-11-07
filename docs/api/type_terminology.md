# Terminology for types

This API documentation uses a few convenient shorthands to refer some types.

- `Scalar` refers to either an `int`, `float`, or a JAX array with shape `()`.
- `PyTree` refers to any PyTree.
- `Array` refers to a JAX array.

---

In addition shapes and dtypes of `Array`s are annotated:

- `Array["dim1", "dim2"]` refers to a JAX array with shape `(dim1, dim2)`, and so on for other shapes.
    - If a dimension is named in this way, then it should match up and be of equal size to the equally-named dimensions of all other arrays passed at the same time.
    - `Array[()]` refers to an array with shape `()`.
    - `...` refers to an arbitrary number of dimensions, e.g. `Array["times", ...]`.
- `Array[bool]` refers to a JAX array with Boolean dtype. (And so on for other dtypes.)
- These are combined via e.g. `Array["dim1", "dim2", bool]`.
- The above syntax is essentially inspired by [torchtyping](https://github.com/patrick-kidger/torchtyping).

Similarly `PyTree["dims"]` is used to refer to a PyTree all of whose leaves are JAX arrays with shape `(dims,)`. (And so on.)
