from collections.abc import Callable
from typing import Optional, TypeVar

from equinox import AbstractVar
from jaxtyping import PyTree

from .._custom_types import RealScalarLike
from .base import AbstractStepSizeController


_ControllerState = TypeVar("_ControllerState")
_Dt0 = TypeVar("_Dt0", None, RealScalarLike, Optional[RealScalarLike])


class AbstractAdaptiveStepSizeController(
    AbstractStepSizeController[_ControllerState, _Dt0]
):
    """Indicates an adaptive step size controller.

    Accepts tolerances `rtol` and `atol`. When used in conjunction with an implicit
    solver ([`diffrax.AbstractImplicitSolver`][]), then these tolerances will
    automatically be used as the tolerances for the nonlinear solver passed to the
    implicit solver, if they are not specified manually.
    """

    rtol: AbstractVar[RealScalarLike]
    atol: AbstractVar[RealScalarLike]
    norm: AbstractVar[Callable[[PyTree], RealScalarLike]]

    def __check_init__(self):
        if self.rtol is None or self.atol is None:
            raise ValueError(
                "The default values for `rtol` and `atol` were removed in Diffrax "
                "version 0.1.0. (As the choice of tolerance is nearly always "
                "something that you, as an end user, should make an explicit choice "
                "about.)\n"
                "If you want to match the previous defaults then specify "
                "`rtol=1e-3`, `atol=1e-6`. For example:\n"
                "```\n"
                "diffrax.PIDController(rtol=1e-3, atol=1e-6)\n"
                "```\n"
            )
