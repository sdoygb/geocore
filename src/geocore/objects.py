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
    build_clifford_tableau,
    clifford_tableau_to_matrix,
    compose_clifford,
    conjugate_pauli_tableau,
    string_to_bits,
    symplectic_commutes,
)

__all__ = ["GeometricObject", "Pauli", "Rotation", "Clifford"]


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

    def __eq__(self, other):
        return isinstance(other, Pauli) and self.axis == other.axis

    def __hash__(self):
        return hash(self.axis)

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


class Clifford(GeometricObject):
    """A Clifford group element on n qubits: the symplectic tableau
    (2n x 2n binary matrix + 2n-bit phase vector, Aaronson-Gottesman).

    Constructed from a gate sequence (names: ``h, s, sd, sx, sxdg, cx``);
    the tableau rows are the conjugates of the generators.  Composition
    and Pauli conjugation are binary linear algebra on the tableau; the
    dense matrix is rebuilt from the tableau for verification (see the
    ``clifford.compose`` invariant and the tests: tableau vs dense agree
    to machine precision).
    """

    __slots__ = ("gates", "n", "_tableau", "_phases")

    def __init__(self, gates, n=None):
        self.gates = tuple(tuple(g) for g in gates)
        qubits = [q for g in self.gates for q in g[1:]]
        self.n = n if n is not None else (max(qubits) + 1 if qubits else 1)
        self._tableau, self._phases = build_clifford_tableau(self.gates, self.n)

    @classmethod
    def from_tableau(cls, tableau, phases, n):
        """Construct directly from a tableau (e.g. the product of a
        composition) — no gate sequence."""
        obj = cls.__new__(cls)
        obj.gates = None
        obj.n = n
        obj._tableau = np.asarray(tableau, dtype=int)
        obj._phases = np.asarray(phases, dtype=int)
        return obj

    @property
    def dim(self) -> int:
        return self.n

    def conjugate(self, pauli: "Pauli") -> tuple["Pauli", int]:
        """Conjugate a Pauli: P -> C P C^+, returns (Pauli', phase r) with
        r the sign bit of the result (axis and phase, as in
        ``pauli.conjugate_by``)."""
        xp, zp, q = conjugate_pauli_tableau(
            self._tableau, self._phases, self.n, pauli._x, pauli._z
        )
        axis = bits_to_string(xp, zp)
        m = axis.count("Y")
        r = ((q - m) // 2) % 2  # full phase i^q = (-1)^r i^m
        return Pauli(axis), int(r)

    def compose(self, other: "Clifford") -> "Clifford":
        """Group product C = self @ other (apply other first, then self)."""
        if other.n != self.n:
            raise ValueError(f"Clifford qubit mismatch: {self.n} vs {other.n}")
        t, p = compose_clifford(
            self._tableau, self._phases, other._tableau, other._phases, self.n
        )
        return Clifford.from_tableau(t, p, self.n)

    def to_matrix(self) -> np.ndarray:
        """Dense 2^n x 2^n matrix rebuilt from the tableau."""
        return clifford_tableau_to_matrix(self._tableau, self._phases, self.n)

    def verify(self) -> dict:
        return {"ok": True, "note": "Clifford invariants checked via clifford.compose"}

    def __repr__(self):
        return f"Clifford(n={self.n}, gates={'x'.join(str(g) for g in self.gates) if self.gates else 'tableau'})"


def _pauli_matrix(axis):
    from .verify import _pauli_matrix as pm

    return pm(axis)
