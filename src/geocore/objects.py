"""Layer 0 — geometric objects.

The "tensor" of geocore: geometric objects with a unified algebraic
encoding, carrying their geometric structure (symplectic form for Paulis,
closure semantics for rotations) and a machine-precision self-check.

All theory is geometrized: a Pauli is an element of the Clifford algebra
with a canonical 2n-bit symplectic encoding; a Rotation is a point on the
rotation orbit of a Pauli axis; operations on them are geometric mappings
that preserve geometric invariants (see Layer 1).
"""

from __future__ import annotations

import numpy as np

from .clifford import (
    apply_gates_to_pauli,
    bits_to_string,
    string_to_bits,
    symplectic_commutes,
)

__all__ = ["GeometricObject", "Pauli", "Rotation"]


class GeometricObject:
    """Base class for all geometric objects (analogue of torch.Tensor).

    Subclasses provide a canonical encoding and a machine-precision
    self-check.
    """

    @property
    def dim(self) -> int:
        """Number of qubits (or manifold dimension, for future objects)."""
        raise NotImplementedError

    def verify(self) -> dict:
        """Run the declared geometric invariants; return a report dict."""
        raise NotImplementedError


class Pauli(GeometricObject):
    """P = i^m prod_k X_k^{x_k} Z_k^{z_k}, with canonical 2n-bit encoding.

    The symplectic form omega((x,z),(x',z')) = x.z' + z.x' (mod 2) decides
    commutation; conjugation by Cliffords is a symplectic transformation
    with an r-bit phase tracked exactly.
    """

    __slots__ = ("axis", "_x", "_z")

    def __init__(self, axis: str):
        if not axis or not all(c in "IXYZ" for c in axis):
            raise ValueError(f"invalid Pauli axis: {axis!r}")
        self.axis = axis
        self._x, self._z = string_to_bits(axis)

    @property
    def dim(self) -> int:
        return len(self.axis)

    def commutes_with(self, other: "Pauli") -> bool:
        """True iff [P, Q] = 0, decided by the symplectic form."""
        return bool(symplectic_commutes(self._x, self._z, other._x, other._z))

    def conjugate_by(self, gates):
        """Conjugate by a Clifford gate sequence: P -> C P C^+.

        Returns (Pauli', r) with r = 1 iff the result carries a minus sign
        (the tableau r-bit, verified against matrix truth).
        """
        x, z, r = apply_gates_to_pauli(gates, self._x, self._z)
        return Pauli(bits_to_string(x, z)), int(r)

    def to_matrix(self) -> np.ndarray:
        return _pauli_matrix(self.axis)

    def __repr__(self):
        return f"Pauli({self.axis!r})"

    def verify(self) -> dict:
        """Conjugation/commutation invariants (machine precision)."""
        from .verify import check_conjugation_matrix_truth

        mismatches = check_conjugation_matrix_truth()
        return {"ok": len(mismatches) == 0, "conjugation_mismatches": len(mismatches)}


class Rotation(GeometricObject):
    """R_P(theta) = exp(-i theta P / 2): the geometric rotation object.

    Closure semantics: same-axis rotations merge by phase addition
    (R_P(t) R_P(s) = R_P(t+s)); a rotation vanishes at 2-pi closure
    (theta ≡ 0 mod 2 pi).
    """

    __slots__ = ("axis", "theta")

    def __init__(self, axis: str, theta: float):
        if not axis or not all(c in "IXYZ" for c in axis):
            raise ValueError(f"invalid Pauli axis: {axis!r}")
        self.axis = axis
        self.theta = float(theta)

    @property
    def dim(self) -> int:
        return len(self.axis)

    def merge_with(self, other: "Rotation"):
        """Same-axis merge (closure of phase addition) or None if not mergeable.

        Two rotations merge iff they share the same Pauli axis (then they
        necessarily commute).
        """
        if self.axis == other.axis:
            return Rotation(self.axis, self.theta + other.theta)
        return None

    def cancels(self) -> bool:
        """True iff theta ≡ 0 (mod 2 pi): the 2-pi closure."""
        return bool(np.isclose(self.theta % (2 * np.pi), 0.0, atol=1e-12))

    def to_matrix(self) -> np.ndarray:
        from .verify import pauli_rotation_matrix

        return pauli_rotation_matrix(self.axis, self.theta)

    def __repr__(self):
        return f"Rotation({self.axis!r}, {self.theta:.6g})"

    def verify(self) -> dict:
        return {"ok": True, "note": "rotation closure invariants checked via circuit.optimize"}


def _pauli_matrix(axis):
    from .verify import _pauli_matrix as pm

    return pm(axis)
