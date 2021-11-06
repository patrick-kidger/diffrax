import typing


def in_public_docs(obj):
    if getattr(typing, "GENERATING_DOCUMENTATION", False):
        assert "_import_alias" not in obj.__dict__
        obj._import_alias = "diffrax." + obj.__name__
    return obj
