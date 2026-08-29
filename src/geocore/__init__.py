"""geocore — geometric computation core.

Clifford/Pauli algebra, geometric objects (Pauli, Rotation), geometric
operator dispatch, and machine-precision verification.  The derivation
engine is the geometry theory; the presentation is standard mathematics
verified to machine precision (engine-presentation separation).
"""

__version__ = "0.1.0"

from . import circuit, clifford, derivatives, geostats, hyperbolic, invariants, manifolds, objects, ops, optim, qec, rotations, shortcuts, spectral, sphere, verify, viz  # noqa: F401
from .hyperbolic import HyperbolicPlane  # noqa: F401
from .manifolds import EuclideanSpace, PolarPlane  # noqa: F401
from .circuit import Circuit  # noqa: F401
from .spectral import Circle  # noqa: F401
from .sphere import Sphere  # noqa: F401
from .objects import Clifford, Pauli, Rotation  # noqa: F401
from .ops import dispatch, get_op, op  # noqa: F401
from .optim import RiemannianAdam, RiemannianSGD, BatchOptimizationResult, OptimizationResult, minimize, minimize_batch  # noqa: F401
from .geostats import frechet_mean, frechet_variance, principal_directions, tangent_covariance  # noqa: F401
from .rotations import optimize_pauli_rotations  # noqa: F401

__all__ = [
    "clifford",
    "circuit",
    "derivatives",
    "geostats",
    "hyperbolic",
    "invariants",
    "manifolds",
    "objects",
    "ops",
    "optim",
    "qec",
    "rotations",
    "shortcuts",
    "spectral",
    "sphere",
    "verify",
    "viz",
    "Pauli",
    "Circuit",
    "PolarPlane",
    "EuclideanSpace",
    "Sphere",
    "HyperbolicPlane",
    "Circle",
    "Rotation",
    "RiemannianSGD",
    "RiemannianAdam",
    "OptimizationResult",
    "BatchOptimizationResult",
    "minimize",
    "minimize_batch",
    "frechet_mean",
    "frechet_variance",
    "principal_directions",
    "tangent_covariance",
    "dispatch",
    "get_op",
    "op",
    "optimize_pauli_rotations",
    "__version__",
]
