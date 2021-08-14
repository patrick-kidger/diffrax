import numpy as np


class frozenndarray:
    def __init__(self, *, array, **kwargs):
        super().__init__(**kwargs)
        array.flags.writeable = False
        _hash = hash(array.data.tobytes())
        self._array = array
        self._hash = _hash

    def __repr__(self):
        return f"{type(self).__name__}(array={self._array})"

    def __array__(self):
        return self._array

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return (
            isinstance(other, frozenndarray)
            and (self._array.shape == other._array.shape)
            and (self._array.dtype == other._array.dtype)
            and (self._array == other._array).all()
        )


def frozenarray(*args, **kwargs):
    return frozenndarray(array=np.array(*args, **kwargs))
