import importlib.metadata

from ._adjoint import (
    AbstractAdjoint as AbstractAdjoint,
    BacksolveAdjoint as BacksolveAdjoint,
    DirectAdjoint as DirectAdjoint,
    ImplicitAdjoint as ImplicitAdjoint,
    RecursiveCheckpointAdjoint as RecursiveCheckpointAdjoint,
)
from ._autocitation import citation as citation, citation_rules as citation_rules
from ._brownian import (
    AbstractBrownianPath as AbstractBrownianPath,
    UnsafeBrownianPath as UnsafeBrownianPath,
    VirtualBrownianTree as VirtualBrownianTree,
)
from ._custom_types import (
    AbstractBrownianIncrement as AbstractBrownianIncrement,
    AbstractSpaceTimeLevyArea as AbstractSpaceTimeLevyArea,
    AbstractSpaceTimeTimeLevyArea as AbstractSpaceTimeTimeLevyArea,
    BrownianIncrement as BrownianIncrement,
    SpaceTimeLevyArea as SpaceTimeLevyArea,
    SpaceTimeTimeLevyArea as SpaceTimeTimeLevyArea,
)
from ._event import (
    AbstractDiscreteTerminatingEvent as AbstractDiscreteTerminatingEvent,
    DiscreteTerminatingEvent as DiscreteTerminatingEvent,
    SteadyStateEvent as SteadyStateEvent,
)
from ._global_interpolation import (
    AbstractGlobalInterpolation as AbstractGlobalInterpolation,
    backward_hermite_coefficients as backward_hermite_coefficients,
    CubicInterpolation as CubicInterpolation,
    DenseInterpolation as DenseInterpolation,
    linear_interpolation as linear_interpolation,
    LinearInterpolation as LinearInterpolation,
    rectilinear_interpolation as rectilinear_interpolation,
)
from ._integrate import diffeqsolve as diffeqsolve
from ._local_interpolation import (
    AbstractLocalInterpolation as AbstractLocalInterpolation,
    FourthOrderPolynomialInterpolation as FourthOrderPolynomialInterpolation,
    LocalLinearInterpolation as LocalLinearInterpolation,
    ThirdOrderHermitePolynomialInterpolation as ThirdOrderHermitePolynomialInterpolation,  # noqa: E501
)
from ._misc import adjoint_rms_seminorm as adjoint_rms_seminorm
from ._path import AbstractPath as AbstractPath
from ._progress_meter import (
    AbstractProgressMeter as AbstractProgressMeter,
    NoProgressMeter as NoProgressMeter,
    TextProgressMeter as TextProgressMeter,
    TqdmProgressMeter as TqdmProgressMeter,
)
from ._root_finder import (
    VeryChord as VeryChord,
    with_stepsize_controller_tols as with_stepsize_controller_tols,
)
from ._saveat import SaveAt as SaveAt, SubSaveAt as SubSaveAt
from ._solution import (
    is_event as is_event,
    is_okay as is_okay,
    is_successful as is_successful,
    RESULTS as RESULTS,
    Solution as Solution,
)
from ._solver import (
    AbstractAdaptiveSolver as AbstractAdaptiveSolver,
    AbstractDIRK as AbstractDIRK,
    AbstractERK as AbstractERK,
    AbstractESDIRK as AbstractESDIRK,
    AbstractImplicitSolver as AbstractImplicitSolver,
    AbstractItoSolver as AbstractItoSolver,
    AbstractRungeKutta as AbstractRungeKutta,
    AbstractSDIRK as AbstractSDIRK,
    AbstractSolver as AbstractSolver,
    AbstractSRK as AbstractSRK,
    AbstractStratonovichSolver as AbstractStratonovichSolver,
    AbstractWrappedSolver as AbstractWrappedSolver,
    Bosh3 as Bosh3,
    ButcherTableau as ButcherTableau,
    CalculateJacobian as CalculateJacobian,
    Dopri5 as Dopri5,
    Dopri8 as Dopri8,
    Euler as Euler,
    EulerHeun as EulerHeun,
    GeneralShARK as GeneralShARK,
    HalfSolver as HalfSolver,
    Heun as Heun,
    ImplicitEuler as ImplicitEuler,
    ItoMilstein as ItoMilstein,
    KenCarp3 as KenCarp3,
    KenCarp4 as KenCarp4,
    KenCarp5 as KenCarp5,
    Kvaerno3 as Kvaerno3,
    Kvaerno4 as Kvaerno4,
    Kvaerno5 as Kvaerno5,
    LeapfrogMidpoint as LeapfrogMidpoint,
    Midpoint as Midpoint,
    MultiButcherTableau as MultiButcherTableau,
    Ralston as Ralston,
    ReversibleHeun as ReversibleHeun,
    SEA as SEA,
    SemiImplicitEuler as SemiImplicitEuler,
    ShARK as ShARK,
    Sil3 as Sil3,
    SlowRK as SlowRK,
    SPaRK as SPaRK,
    SRA1 as SRA1,
    StochasticButcherTableau as StochasticButcherTableau,
    StratonovichMilstein as StratonovichMilstein,
    Tsit5 as Tsit5,
)
from ._step_size_controller import (
    AbstractAdaptiveStepSizeController as AbstractAdaptiveStepSizeController,
    AbstractStepSizeController as AbstractStepSizeController,
    ConstantStepSize as ConstantStepSize,
    PIDController as PIDController,
    StepTo as StepTo,
)
from ._term import (
    AbstractTerm as AbstractTerm,
    ControlTerm as ControlTerm,
    MultiTerm as MultiTerm,
    ODETerm as ODETerm,
    WeaklyDiagonalControlTerm as WeaklyDiagonalControlTerm,
)


__version__ = importlib.metadata.version("diffrax")
