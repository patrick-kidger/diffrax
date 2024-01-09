import jax.tree_util as jtu
from jaxtyping import PyTree

from ._brownian import AbstractBrownianPath, UnsafeBrownianPath
from ._path import AbstractPath
from ._term import AbstractTerm


# Assumes that the SDE-ness is interpretable by finding AbstractBrownianPath.
# In principle a user could re-create terms, controls, etc. without going via this,
# though. So this is a bit imperfect.
#
# Fortunately, at time of writing this is used for two things:
# - solver.error_order
# - error checking
# The former can be overridden by `PIDController(error_order=...)` and the latter is
# really just to catch common errors.
# That is, for the power user who implements enough to bypass this check -- probably
# they know what they're doing and can handle both of these cases appropriately.
def _is_brownian(x):
    return isinstance(x, AbstractBrownianPath)


def _is_unsafe_brownian(x):
    return isinstance(x, UnsafeBrownianPath)


def _is_path(x):
    return isinstance(x, AbstractPath)


def is_sde(terms: PyTree[AbstractTerm]) -> bool:
    leaves, _ = jtu.tree_flatten(terms, is_leaf=_is_brownian)
    return any(_is_brownian(leaf) for leaf in leaves)


def is_unsafe_sde(terms: PyTree[AbstractTerm]) -> bool:
    leaves, _ = jtu.tree_flatten(terms, is_leaf=_is_unsafe_brownian)
    return any(_is_unsafe_brownian(leaf) for leaf in leaves)


def is_cde(terms: PyTree[AbstractTerm]) -> bool:
    leaves, _ = jtu.tree_flatten(terms, is_leaf=_is_path)
    return any(_is_path(leaf) and not _is_brownian(leaf) for leaf in leaves)
