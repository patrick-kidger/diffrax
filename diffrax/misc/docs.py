import functools as ft
import typing
from typing import Optional


_T = typing.TypeVar("_T")


# So this function is a bit of a wart. Basically it serves to tell an object its
# "import name", e.g. "diffrax.foo" rather than "diffrax.folder.file.foo". Such names
# being very common because we've publicly explosed that functionality by importing foo
# in __init__.py.
#
# This is needed when generating cross-references in documentation. We have an object
# and want to know how to refer to it in the documentation.
def in_public_docs(
    obj: Optional[_T] = None,
    *,
    module: str = "diffrax",
    submodule: Optional[str] = None
):
    if submodule is not None:
        module = module + "." + submodule

    if obj is None:
        return ft.partial(_in_public_docs, module=module)
    else:
        return _in_public_docs(obj, module)


def _in_public_docs(obj: _T, module: str) -> _T:
    if getattr(typing, "GENERATING_DOCUMENTATION", False):
        assert "_import_alias" not in obj.__dict__
        obj._import_alias = module + "." + obj.__name__
    return obj
