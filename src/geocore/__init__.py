"""geocore — geometric computation core.

Clifford/Pauli algebra, geometric rotation objects, and machine-precision
verification.  The derivation engine is the geometry theory; the
presentation is standard mathematics verified to machine precision.
"""

__version__ = "0.1.0"

from . import clifford, rotations, verify  # noqa: F401
from .rotations import PauliRotation, optimize_pauli_rotations  # noqa: F401

__all__ = ["clifford", "rotations", "verify", "PauliRotation", "optimize_pauli_rotations", "__version__"]
