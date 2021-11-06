import argparse
import typing

import pytkdocs


def _postprocess_properties(x):
    for prop in ("dataclass", "special"):
        try:
            x["properties"].remove(prop)
        except ValueError:
            pass
    if "inherited" in x["properties"] and x["docstring"] == "":
        docstring = "Inherited; check the documentation for the parent class(es)."
        x["docstring"] = docstring
        x["docstring_sections"] = [{"type": "markdown", "value": docstring}]
    for child in x["children"].values():
        _postprocess_properties(child)


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
    # A few properties are removed to avoid common visual noise.

    # (c)
    # By default pytkdocs has some really weird behaviour in which the docstring for
    # inherited magic methods are removed. This change enhances that check to apply
    # to all methods, not just magic methods.

    # (d)
    # A flag to say we're generating documentation is set, which Diffrax uses to
    # customise how its types are displayed.

    _process_config = pytkdocs.cli.process_config

    def process_config(config):
        paths = [c["path"] for c in config["objects"]]
        out = _process_config(config)
        for path, out_object in zip(paths, out["objects"]):
            # (a)
            out_object["path"] = path
            # (b)
            _postprocess_properties(out_object)
        return out

    pytkdocs.cli.process_config = process_config

    # (c)
    pytkdocs.loader.RE_SPECIAL = argparse.Namespace(match=lambda x: True)

    # (d)
    typing.GENERATING_DOCUMENTATION = True
