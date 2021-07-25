from typing import Any, Optional, Tuple


class Array:
    def __class_getitem__(cls, item):
        return Any


class PyTree:
    def __class_getitem__(cls, item):
        return Any


Scalar = Array[()]


SquashTreeDef = Tuple[Any, Optional[Tuple[int]], Optional[Tuple[int]]]

