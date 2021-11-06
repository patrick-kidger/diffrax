import typing
from typing import Any, Dict, Union


# Custom flag we set when generating documentation.
if getattr(typing, "GENERATING_DOCUMENTATION", False):

    class Array:
        def __class_getitem__(cls, item):
            return Array

    class PyTree:
        def __class_getitem__(cls, item):
            return PyTree

    # Dirty hack. When these appear inside generic types, e.g. Tuple[Array], then
    # they are stringified in a custom way (by pytkdocs and Python's typing).
    # However Python's typing has a particular exception when __module__ == "builtins",
    # which we exploit here.
    # https://github.com/python/cpython/blob/634984d7dbdd91e0a51a793eed4d870e139ae1e0/Lib/typing.py#L203  # noqa: E501
    Array.__module__ = "builtins"
    PyTree.__module__ = "builtins"

    Scalar = "Scalar"

else:

    class Array:
        def __class_getitem__(cls, item):
            return Any

    class PyTree:
        def __class_getitem__(cls, item):
            return Any

    Scalar = Union[int, float, Array[()]]


Bool = Union[bool, Array[(), bool]]

DenseInfo = Dict[str, PyTree]
