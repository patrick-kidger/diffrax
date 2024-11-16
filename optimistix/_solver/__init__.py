from .backtracking import BacktrackingArmijo as BacktrackingArmijo
from .best_so_far import (
    BestSoFarFixedPoint as BestSoFarFixedPoint,
    BestSoFarLeastSquares as BestSoFarLeastSquares,
    BestSoFarMinimiser as BestSoFarMinimiser,
    BestSoFarRootFinder as BestSoFarRootFinder,
)
from .bfgs import AbstractBFGS as AbstractBFGS, BFGS as BFGS
from .bisection import Bisection as Bisection
from .dogleg import Dogleg as Dogleg, DoglegDescent as DoglegDescent
from .fixed_point import FixedPointIteration as FixedPointIteration
from .gauss_newton import (
    AbstractGaussNewton as AbstractGaussNewton,
    GaussNewton as GaussNewton,
    NewtonDescent as NewtonDescent,
)
from .gradient_methods import (
    AbstractGradientDescent as AbstractGradientDescent,
    GradientDescent as GradientDescent,
    SteepestDescent as SteepestDescent,
)
from .learning_rate import LearningRate as LearningRate
from .levenberg_marquardt import (
    DampedNewtonDescent as DampedNewtonDescent,
    IndirectDampedNewtonDescent as IndirectDampedNewtonDescent,
    IndirectLevenbergMarquardt as IndirectLevenbergMarquardt,
    LevenbergMarquardt as LevenbergMarquardt,
)
from .nelder_mead import NelderMead as NelderMead
from .newton_chord import Chord as Chord, Newton as Newton
from .nonlinear_cg import (
    dai_yuan as dai_yuan,
    fletcher_reeves as fletcher_reeves,
    hestenes_stiefel as hestenes_stiefel,
    NonlinearCG as NonlinearCG,
    NonlinearCGDescent as NonlinearCGDescent,
    polak_ribiere as polak_ribiere,
)
from .optax import OptaxMinimiser as OptaxMinimiser
from .trust_region import (
    ClassicalTrustRegion as ClassicalTrustRegion,
    LinearTrustRegion as LinearTrustRegion,
)
