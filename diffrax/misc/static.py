from typing import Generic, TypeVar

import equinox as eqx


_T = TypeVar("_T")


class Static(eqx.Module, Generic[_T]):
    value: _T = eqx.static_field()
