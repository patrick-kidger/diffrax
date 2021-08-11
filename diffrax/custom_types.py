from typing import Any, Dict, Union


class Array:
    def __class_getitem__(cls, item):
        return Any


PyTree = Any

DenseInfo = Dict[str, PyTree]

Scalar = Union[int, float, Array[()]]
