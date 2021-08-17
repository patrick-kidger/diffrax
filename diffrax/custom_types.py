from typing import Any, Dict, Union


class Array:
    def __class_getitem__(cls, item):
        return Any


class PyTree:
    def __class_getitem__(cls, item):
        return Any


DenseInfo = Dict[str, PyTree]

Scalar = Union[int, float, Array[()]]
