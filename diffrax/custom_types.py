from typing import Any, Optional, Tuple, Union


class Array:
    def __class_getitem__(cls, item):
        return Any


class PyTree:
    def __class_getitem__(cls, item):
        return Any


Scalar = Union[int, float, Array[()]]

SquashTreeDef = Tuple[Any, Optional[Tuple[int]], Optional[Tuple[int]]]
