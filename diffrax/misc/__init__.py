from .bounded_while_loop import bounded_while_loop, HadInplaceUpdate
from .frozenarray import frozenarray, frozenndarray
from .misc import (
    branched_error_if,
    check_no_derivative,
    ContainerMeta,
    error_if,
    fill_forward,
    force_bitcast_convert_type,
    is_perturbed,
    left_broadcast_to,
    linear_rescale,
    nextafter,
    nextbefore,
    rms_norm,
)
from .ravel import ravel_pytree
from .sde_kl_divergence import sde_kl_divergence
from .unvmap import unvmap_all, unvmap_any, unvmap_max
