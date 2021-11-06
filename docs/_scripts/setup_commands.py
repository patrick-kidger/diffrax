import argparse
import importlib
import traceback
import typing
import warnings
from typing import Any

import pytkdocs


def _import(string: str) -> Any:
    pieces = string.split(".")
    if len(pieces) == 1:
        # Not a relative module.class lookup.
        # Must be a builtin.
        return None
    obj = importlib.import_module(pieces[0])
    for piece in pieces[1:]:
        obj = getattr(obj, piece)
    return obj


def _postprocess(obj, path, bases):
    # (f)
    if "bases" in obj:
        bases = []
        for base in obj["bases"]:
            base = _import(base)
            if base is None:
                continue
            # Only include those bases that are part of the public documentation.
            if "_import_alias" in base.__dict__:
                bases.append(base)
        obj["bases"] = [base._import_alias for base in bases]

    # (a)
    obj["path"] = path

    # (b)
    for prop in ("dataclass", "special"):
        try:
            obj["properties"].remove(prop)
        except ValueError:
            pass
    try:
        _obj = _import(path)
    except AttributeError:
        pass
        # Can happen when doing:
        #
        # @dataclass
        # class A:
        #       myfield: int
        #
        # and then trying to access `A.myfield`.
        # This doesn't actually exist on the class object -- only on instances.
        # Which means that the _import line will fail.
    else:
        if getattr(_obj, "__isabstractmethod__", False):
            obj["properties"].append("abstractmethod")

    # (c)
    if obj["docstring"] == "":
        docstring = ""
        obj_name = obj["name"]
        if "inherited" in obj["properties"]:
            for base in bases:
                if obj_name in base.__dict__:
                    base_alias = base.__dict__["_import_alias"] + "." + obj_name
                    docstring = f"Inherited from [`{base_alias}`][]."
                    break
            else:
                raise RuntimeError(
                    f"Inherited object {obj_name} not available on a public base class."
                )
        else:
            for base in bases:
                if obj_name in base.__dict__:
                    base_method = getattr(base, obj_name)
                    if getattr(base_method, "__isabstractmethod__", False):
                        base_alias = base.__dict__["_import_alias"] + "." + obj_name
                        docstring = f"Implements [`{base_alias}`][]."
                        break
        if docstring != "":
            obj["docstring_sections"] = [{"type": "markdown", "value": docstring}]

    # (g)
    if obj["name"] == "__init__":
        del obj["signature"]["return_annotation"]

    # Delete properties that are similar to those we're modifying. This is essentially
    # "assert they're not used downstream".
    # (As they may report the wrong result, given that we modify their siblings.)
    del obj["docstring"]
    del obj["parent_path"]

    # Recurse into methods of classes, etc.
    for child in obj["children"].values():
        child_path = path + "." + child["name"]
        _postprocess(child, child_path, bases)


def main():
    # This does a few things.

    # (a)
    # When doing the following the documentation markdown file:
    #
    # ::: diffrax.something
    #
    # Then this tweak will mean that `diffrax.something` is the name also used in
    # the documentation, rather than something potentially larger like
    # diffrax.folder.file.something.

    # (b)
    # A few properties are removed to avoid common visual noise. Some extras are added.

    # (c)
    # Some docstrings are automatically provided for inherited methods, or methods
    # implementing abstract methods

    # (d)
    # By default pytkdocs has some really weird behaviour in which the docstring for
    # inherited magic methods are removed. This change enhances that check to apply
    # to all methods, not just magic methods.

    # (e)
    # Set a flag to say we're generating documentation is set, which Diffrax uses to
    # customise how its types are displayed.

    # (f)
    # Only include those base classes that are part of the public documentation.

    # (g)
    # Skip the "-> None" return type annotation for __init__ methods.

    _process_config = pytkdocs.cli.process_config

    def process_config(config):
        paths = [c["path"] for c in config["objects"]]
        out = _process_config(config)
        for path, out_object in zip(paths, out["objects"]):
            # (a, b, c, f, g)
            try:
                _postprocess(out_object, path, bases=None)
            except Exception as e:
                tb = traceback.format_exc()
                warnings.warn(
                    f"When loading {out_object['name']}, an exception of type "
                    f"{type(e)} was raised, with message '{str(e)}' and "
                    f"traceback:\n{tb}."
                )
        return out

    pytkdocs.cli.process_config = process_config

    # (d)
    pytkdocs.loader.RE_SPECIAL = argparse.Namespace(match=lambda x: True)

    # (e)
    typing.GENERATING_DOCUMENTATION = True
