from typing import Annotated, Any, Generic, Literal, TypeVar, Union

import diffrax as dfx
import pytest
from diffrax._custom_types import RealScalarLike
from diffrax._typing import get_args_of, get_origin_no_specials


T = TypeVar("T")
S = TypeVar("S")
U = TypeVar("U")


class Foo(Generic[T]):
    pass


class Bar(Generic[S]):
    pass


class Baz:
    pass


def test_get_origin_no_specials():
    assert get_origin_no_specials(int, "") is None
    assert get_origin_no_specials(tuple[int, ...], "") is tuple
    assert get_origin_no_specials(Foo[int], "") is Foo
    assert get_origin_no_specials(Annotated[tuple[int, ...], 1337], "") is tuple
    # Weird, but legal
    assert get_origin_no_specials(Generic[T], "") is Generic  # pyright: ignore

    with pytest.raises(NotImplementedError, match="qwerty"):
        get_origin_no_specials(Union[int, str], "qwerty")
    with pytest.raises(NotImplementedError, match="qwerty"):
        get_origin_no_specials(Literal[4], "qwerty")


def test_get_args_of_not_generic():
    with pytest.raises(TypeError, match="unsubscripted generic"):
        get_args_of(Baz, Foo, "")
    with pytest.raises(TypeError, match="unsubscripted generic"):
        get_args_of(Baz, Foo[int], "")
    with pytest.raises(TypeError, match="unsubscripted generic"):
        get_args_of(int, Foo, "")


def test_get_args_of_not_subclass():
    with pytest.raises(TypeError, match="is not a subclass"):
        get_args_of(Foo, Bar, "")
    with pytest.raises(TypeError, match="is not a subclass"):
        get_args_of(Foo, Baz, "")
    with pytest.raises(TypeError, match="is not a subclass"):
        get_args_of(Foo, int, "")


def test_get_args_of_single_inheritance():
    class Qux1(Foo):
        pass

    class Qux2(Foo[int]):
        pass

    class Qux3(Foo[T]):
        pass

    assert get_args_of(Foo, Qux1, "") == (Any,)
    assert get_args_of(Foo, Qux2, "") == (int,)
    assert get_args_of(Foo, Qux3, "") == (Any,)
    assert get_args_of(Foo, Qux3[str], "") == (str,)


def test_get_args_irrelevant_inheritance():
    class Qux1(Foo, str):
        pass

    class Qux2(Foo[int], str):
        pass

    class Qux3(Foo[T], str):
        pass

    assert get_args_of(Foo, Qux1, "") == (Any,)
    assert get_args_of(Foo, Qux2, "") == (int,)
    assert get_args_of(Foo, Qux3, "") == (Any,)
    assert get_args_of(Foo, Qux3[str], "") == (str,)


def test_get_args_double_inheritance():
    class Qux1(Foo, Bar):
        pass

    class Qux2(Foo[int], Bar):
        pass

    class Qux3(Foo[T], Bar):
        pass

    assert get_args_of(Foo, Qux1, "") == (Any,)
    assert get_args_of(Foo, Qux2, "") == (int,)
    assert get_args_of(Foo, Qux3, "") == (Any,)
    assert get_args_of(Foo, Qux3[bool], "") == (bool,)

    class Qux4(Foo, Bar[str]):
        pass

    class Qux5(Foo[int], Bar[str]):
        pass

    class Qux6(Foo[T], Bar[str]):
        pass

    assert get_args_of(Foo, Qux4, "") == (Any,)
    assert get_args_of(Foo, Qux5, "") == (int,)
    assert get_args_of(Foo, Qux6, "") == (Any,)
    assert get_args_of(Foo, Qux6[bool], "") == (bool,)

    class Qux7(Foo, Bar[S]):
        pass

    class Qux8(Foo[int], Bar[S]):
        pass

    class Qux9(Foo[T], Bar[S]):
        pass

    assert get_args_of(Foo, Qux7, "") == (Any,)
    assert get_args_of(Foo, Qux7[bool], "") == (Any,)
    assert get_args_of(Foo, Qux8, "") == (int,)
    assert get_args_of(Foo, Qux8[bool], "") == (int,)
    assert get_args_of(Foo, Qux9, "") == (Any,)
    assert get_args_of(Foo, Qux9[bool, str], "") == (bool,)

    class Qux10(Foo, Bar[T]):
        pass

    class Qux11(Foo[int], Bar[T]):
        pass

    class Qux12(Foo[T], Bar[T]):
        pass

    assert get_args_of(Foo, Qux10, "") == (Any,)
    assert get_args_of(Foo, Qux11, "") == (int,)
    assert get_args_of(Foo, Qux12, "") == (Any,)
    assert get_args_of(Foo, Qux12[bool], "") == (bool,)


def test_get_args_double_inheritance_reverse():
    class Qux1(Foo, Bar):
        pass

    class Qux2(Foo[int], Bar):
        pass

    class Qux3(Foo[T], Bar):
        pass

    assert get_args_of(Bar, Qux1, "") == (Any,)
    assert get_args_of(Bar, Qux2, "") == (Any,)
    assert get_args_of(Bar, Qux3, "") == (Any,)
    assert get_args_of(Bar, Qux3[bool], "") == (Any,)

    class Qux4(Foo, Bar[str]):
        pass

    class Qux5(Foo[int], Bar[str]):
        pass

    class Qux6(Foo[T], Bar[str]):
        pass

    assert get_args_of(Bar, Qux4, "") == (str,)
    assert get_args_of(Bar, Qux5, "") == (str,)
    assert get_args_of(Bar, Qux6, "") == (str,)
    assert get_args_of(Bar, Qux6[bool], "") == (str,)

    class Qux7(Foo, Bar[S]):
        pass

    class Qux8(Foo[int], Bar[S]):
        pass

    class Qux9(Foo[T], Bar[S]):
        pass

    assert get_args_of(Bar, Qux7, "") == (Any,)
    assert get_args_of(Bar, Qux7[bool], "") == (bool,)
    assert get_args_of(Bar, Qux8, "") == (Any,)
    assert get_args_of(Bar, Qux8[bool], "") == (bool,)
    assert get_args_of(Bar, Qux9, "") == (Any,)
    assert get_args_of(Bar, Qux9[bool, str], "") == (str,)

    class Qux10(Foo, Bar[T]):
        pass

    class Qux11(Foo[int], Bar[T]):
        pass

    class Qux12(Foo[T], Bar[T]):
        pass

    assert get_args_of(Bar, Qux10, "") == (Any,)
    assert get_args_of(Bar, Qux11, "") == (Any,)
    assert get_args_of(Bar, Qux12, "") == (Any,)
    assert get_args_of(Bar, Qux12[bool], "") == (bool,)


def test_get_args_of_complicated():
    class X1(Generic[T, S]):
        pass

    class X2(X1[T, T], Generic[T, S]):
        pass

    class X3(X2):
        pass

    class X4(X2[int, T]):
        pass

    class X5(str, X1[str, str]):
        pass

    class X6(X1[S, T], Generic[T, U, S]):
        pass

    # This one is invalid at static type-checking time.
    class X7(X6[int, str, bool], X2[int, str]):  # pyright: ignore
        pass

    class X8(X6[bool, T, bool], X2[bool, int]):
        pass

    class X9(X3, X2[int, str]):
        pass

    # Some of these are invalid at static type-checking time.
    assert get_args_of(X1, X1, "") == (Any, Any)
    assert get_args_of(X1, X1[int, S], "") == (int, Any)  # pyright: ignore
    assert get_args_of(X1, X1[int, str], "") == (int, str)

    assert get_args_of(X1, X2, "") == (Any, Any)
    assert get_args_of(X1, X2[T, str], "") == (Any, Any)  # pyright: ignore
    assert get_args_of(X1, X2[str, T], "") == (str, str)  # pyright: ignore
    assert get_args_of(X1, X2[int, str], "") == (int, int)

    assert get_args_of(X2, X3, "") == (Any, Any)

    assert get_args_of(X2, X4, "") == (int, Any)
    assert get_args_of(X2, X4[str], "") == (int, str)

    assert get_args_of(X1, X5, "") == (str, str)

    assert get_args_of(X1, X6, "") == (Any, Any)
    assert get_args_of(X1, X6[int, str, bool], "") == (bool, int)

    with pytest.raises(TypeError, match="multiple incompatible ways"):
        assert get_args_of(X1, X7, "") == (Any, Any)
    assert get_args_of(X6, X7, "") == (int, str, bool)
    assert get_args_of(X2, X7, "") == (int, str)

    assert get_args_of(X1, X8, "") == (bool, bool)
    assert get_args_of(X1, X8[float], "") == (bool, bool)
    assert get_args_of(X6, X8, "") == (bool, Any, bool)
    assert get_args_of(X6, X8[float], "") == (bool, float, bool)
    assert get_args_of(X2, X8, "") == (bool, int)
    assert get_args_of(X2, X8[float], "") == (bool, int)

    assert get_args_of(X3, X9, "") == ()
    assert get_args_of(X2, X9, "") == (int, str)
    assert get_args_of(X1, X9, "") == (int, int)


_abstract_args = lambda cls: get_args_of(dfx.AbstractTerm, cls, "")


def test_abstract_term():
    assert _abstract_args(dfx.AbstractTerm) == (Any, Any)
    assert _abstract_args(dfx.AbstractTerm[int, str]) == (int, str)


def test_ode_term():
    assert _abstract_args(dfx.ODETerm) == (Any, RealScalarLike)
    assert _abstract_args(dfx.ODETerm[int]) == (int, RealScalarLike)


def test_control_term():
    assert _abstract_args(dfx.ControlTerm) == (Any, Any)
    assert _abstract_args(dfx.ControlTerm[int, str]) == (int, str)


def test_weakly_diagonal_control_term():
    assert _abstract_args(dfx.WeaklyDiagonalControlTerm) == (Any, Any)
    assert _abstract_args(dfx.WeaklyDiagonalControlTerm[int, str]) == (int, str)
