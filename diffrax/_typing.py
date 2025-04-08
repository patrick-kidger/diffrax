import inspect
import types
from typing import (
    Annotated,
    Any,
    Generic,
    get_args,
    get_origin,
    Optional,
    Protocol,
    TypeVar,
    Union,
)
from typing_extensions import TypeAlias


# We don't actually care what people have subscripted with.
# In practice this should be thought of as TypeLike = Union[type, types.UnionType]. Plus
# maybe type(Literal) and so on?
TypeLike: TypeAlias = Any


_T = TypeVar("_T")


class _Foo(Generic[_T]):
    pass


_generic_alias_types = (types.GenericAlias, type(_Foo[int]))
_union_origins = (Union, types.UnionType)
del _Foo, _T


def better_isinstance(x, annotation) -> bool:
    """As `isinstance`, but supports a few other types that are useful to us."""
    origin = get_origin(annotation)
    if origin in _union_origins:
        return any(better_isinstance(x, arg) for arg in get_args(annotation))
    elif isinstance(annotation, _generic_alias_types):
        assert origin is not None
        return better_isinstance(x, origin)
    elif annotation is Any:
        return True
    elif isinstance(annotation, type):
        return isinstance(x, annotation)
    else:
        raise NotImplementedError(
            f"Do not know how to check whether `{x}` is an instance of `{annotation}`."
        )


def get_origin_no_specials(x, error_msg: str) -> Optional[type]:
    """As `typing.get_origin`, but ignores `Annotated` and throws a
    `NotImplementedError` if passed any other non-class: `Union`, `Literal`, etc. Serves
    as a guard against the full weirdness of the Python type system.

    **Arguments:**

    - `x`: the type to apply `get_origin` to.
    - `error_msg`: if a disallowed type is used, then this will appear in the error
        message.

    **Returns:**

    As `get_origin`, specifically either `None` or a class.
    """
    origin = get_origin(x)
    if origin in _union_origins:
        raise NotImplementedError(f"Cannot use unions in `{error_msg}`.")
    elif origin is Annotated:
        # We do allow Annotated, just because it's easy to handle.
        return get_origin_no_specials(get_args(x)[0], error_msg)
    elif origin is None or inspect.isclass(origin):
        return origin
    else:
        raise NotImplementedError(f"Cannot use {x} in `{error_msg}`.")


def get_args_of(base_cls: type, cls, error_msg: str) -> tuple[TypeLike, ...]:
    """Equivalent to `get_args(cls)`, except that it tracks through the type hierarchy
    finding the way in which `cls` subclasses `base_cls`, and returns the arguments that
    subscript that instead.

    For example,
    ```python
    class Foo(Generic[T]):
        pass

    class Bar(Generic[S]):
        pass

    class Qux(Foo[T], Bar[S]):
        pass

    get_args_of(Foo, Qux[int, str], "...")  # int
    ```

    In addition, any unfilled type variables are returned as `Any`.

    **Arguments:**

    - `base_cls`: the class to get parameters with respect to.
    - `cls`: the class or subscripted generic to get arguments with respect to.
    - `error_msg`: if anything goes wrong, mention this in the error message.

    **Returns:**

    A tuple of types indicating the arguments. In addition, any unfilled type variables
    are returned as `Any`.
    """

    if not inspect.isclass(base_cls):
        raise TypeError(f"{base_cls} should be a class")
    if not hasattr(base_cls, "__parameters__"):
        raise TypeError(f"{base_cls} should be an unsubscripted generic")

    origin = get_origin_no_specials(cls, error_msg)
    if inspect.isclass(cls):
        # Unsubscripted
        assert origin is None
        origin = cls
        params = [Any for _ in getattr(cls, "__parameters__", ())]
    else:
        # Subscripted
        assert origin is not None
        params: list[TypeLike] = []
        for param in get_args(cls):
            if isinstance(param, TypeVar):
                params.append(Any)
            else:
                params.append(param)
    if issubclass(origin, base_cls):
        out = _get_args_of_impl(base_cls, origin, tuple(params), error_msg)
        if out is None:
            # Dependency is purely inheritance without subscripting
            return tuple(Any for _ in base_cls.__parameters__)
        else:
            return out
    else:
        raise TypeError(f"{cls} is not a subclass of {base_cls}")


def _get_args_of_impl(
    base_cls: type, cls: type, params: tuple[TypeLike, ...], error_msg
) -> Optional[tuple[TypeLike, ...]]:
    if cls is base_cls:
        return params
    assert len(cls.__parameters__) == len(params)
    param_lookup = {k: v for k, v in zip(cls.__parameters__, params)}
    base_params: set[tuple[TypeLike, ...]] = set()
    # If we've gotten this far then `cls` is known to have been subscripted, so it
    # should have an `__orig_bases__` attribute. (Where as e.g. `class Foo: pass` does
    # not have one)
    for x in cls.__orig_bases__:
        x_origin = get_origin_no_specials(x, error_msg)
        if x_origin in (Generic, Protocol):
            continue
        if inspect.isclass(x):
            # Unsubscripted, ignore.
            assert x_origin is None
        else:
            # Subscripted, should pass in parameters
            assert x_origin is not None
            assert len(get_args(x)) > 0
            x_params = [
                param_lookup.get(arg, Any) if isinstance(arg, TypeVar) else arg
                for arg in get_args(x)
            ]
            if issubclass(x_origin, base_cls):
                base_params_i = _get_args_of_impl(
                    base_cls, x_origin, tuple(x_params), error_msg
                )
                if base_params_i is not None:
                    base_params.add(base_params_i)
            # Else ignore, we won't be able to recurse down to `base_cls` this way.
    if len(base_params) == 0:
        # `base_cls` is a superclass of `cls`, as we have earlier guards against this.
        assert issubclass(cls, base_cls)
        # However that dependency is purely normal inheritance, no subscripting.
        return None
    elif len(base_params) == 1:
        return base_params.pop()
    else:
        if len(params) == 0:
            error_cls = cls
        else:
            error_cls = cls[params]
        raise TypeError(
            f"{error_cls} inherits from {base_cls} in multiple incompatible ways."
        )
