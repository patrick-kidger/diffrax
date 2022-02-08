import jax

from .brownian import AbstractBrownianPath, UnsafeBrownianPath
from .custom_types import PyTree
from .term import AbstractTerm


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
def is_sde(terms: PyTree[AbstractTerm]) -> bool:
    is_brownian = lambda x: isinstance(x, AbstractBrownianPath)
    leaves, _ = jax.tree_flatten(terms, is_leaf=is_brownian)
    return any(is_brownian(leaf) for leaf in leaves)


def is_unsafe_sde(terms: PyTree[AbstractTerm]) -> bool:
    is_brownian = lambda x: isinstance(x, UnsafeBrownianPath)
    leaves, _ = jax.tree_flatten(terms, is_leaf=is_brownian)
    return any(is_brownian(leaf) for leaf in leaves)
