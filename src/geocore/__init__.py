"""geocore — geometric computation core.

Clifford/Pauli algebra, geometric objects (Pauli, Rotation), geometric
operator dispatch, and machine-precision verification.  The derivation
engine is the geometry theory; the presentation is standard mathematics
verified to machine precision (engine-presentation separation).
"""

__version__ = "0.1.0"

from . import clifford, invariants, objects, ops, rotations, shortcuts, verify  # noqa: F401
from .objects import Pauli, Rotation  # noqa: F401
from .ops import dispatch, get_op, op  # noqa: F401
from .rotations import optimize_pauli_rotations  # noqa: F401

__all__ = [
    "clifford",
    "invariants",
    "objects",
    "ops",
    "rotations",
    "shortcuts",
    "verify",
    "Pauli",
    "Rotation",
    "dispatch",
    "get_op",
    "op",
    "optimize_pauli_rotations",
    "__version__",
]
