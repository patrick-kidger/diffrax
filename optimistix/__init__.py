import importlib.metadata

from . import compat as compat, internal as internal
from ._adjoint import (
    AbstractAdjoint as AbstractAdjoint,
    ImplicitAdjoint as ImplicitAdjoint,
    RecursiveCheckpointAdjoint as RecursiveCheckpointAdjoint,
)
from ._fixed_point import (
    AbstractFixedPointSolver as AbstractFixedPointSolver,
    fixed_point as fixed_point,
)
from ._iterate import (
    AbstractIterativeSolver as AbstractIterativeSolver,
)
from ._least_squares import (
    AbstractLeastSquaresSolver as AbstractLeastSquaresSolver,
    least_squares as least_squares,
)
from ._minimise import (
    AbstractMinimiser as AbstractMinimiser,
    minimise as minimise,
)
from ._misc import (
    max_norm as max_norm,
    rms_norm as rms_norm,
    two_norm as two_norm,
)
from ._root_find import (
    AbstractRootFinder as AbstractRootFinder,
    root_find as root_find,
)
from ._search import (
    AbstractDescent as AbstractDescent,
    AbstractSearch as AbstractSearch,
    FunctionInfo as FunctionInfo,
)
from ._solution import RESULTS as RESULTS, Solution as Solution
from ._solver import (
    AbstractBFGS as AbstractBFGS,
    AbstractGaussNewton as AbstractGaussNewton,
    AbstractGradientDescent as AbstractGradientDescent,
    BacktrackingArmijo as BacktrackingArmijo,
    BestSoFarFixedPoint as BestSoFarFixedPoint,
    BestSoFarLeastSquares as BestSoFarLeastSquares,
    BestSoFarMinimiser as BestSoFarMinimiser,
    BestSoFarRootFinder as BestSoFarRootFinder,
    BFGS as BFGS,
    Bisection as Bisection,
    Chord as Chord,
    ClassicalTrustRegion as ClassicalTrustRegion,
    dai_yuan as dai_yuan,
    DampedNewtonDescent as DampedNewtonDescent,
    Dogleg as Dogleg,
    DoglegDescent as DoglegDescent,
    FixedPointIteration as FixedPointIteration,
    fletcher_reeves as fletcher_reeves,
    GaussNewton as GaussNewton,
    GradientDescent as GradientDescent,
    hestenes_stiefel as hestenes_stiefel,
    IndirectDampedNewtonDescent as IndirectDampedNewtonDescent,
    IndirectLevenbergMarquardt as IndirectLevenbergMarquardt,
    LearningRate as LearningRate,
    LevenbergMarquardt as LevenbergMarquardt,
    LinearTrustRegion as LinearTrustRegion,
    NelderMead as NelderMead,
    Newton as Newton,
    NewtonDescent as NewtonDescent,
    NonlinearCG as NonlinearCG,
    NonlinearCGDescent as NonlinearCGDescent,
    OptaxMinimiser as OptaxMinimiser,
    polak_ribiere as polak_ribiere,
    SteepestDescent as SteepestDescent,
)


__version__ = importlib.metadata.version("optimistix")
