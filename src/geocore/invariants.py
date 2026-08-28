"""Layer 2 — automatic verification (the analogue of autograd).

PyTorch's core automatically *differentiates*; geocore's core automatically
*verifies*.  Operators declare the geometric invariants they preserve; the
verification layer checks them to machine precision on every call (unless
``no_verify()`` is active, the analogue of ``torch.no_grad()``).

All theory is geometrized: the invariants below check exactly the geometric
theorems the operators claim — the symplectic form deciding commutation,
matrix-truth of Clifford conjugation (axis and phase), closure of phase
addition on a Pauli axis, the 2-pi closure, and unitary equivalence of
circuit optimization.
"""

from __future__ import annotations

import contextlib
import dataclasses

import numpy as np

__all__ = [
    "VerificationReport",
    "VerificationError",
    "Invariant",
    "SymplecticForm",
    "ConjugationMatrixTruth",
    "RotationActionClosure",
    "MergeClosure",
    "CancellationClosure",
    "UnitaryEquivalence",
    "VerificationContext",
    "no_verify",
    "verify_invariants",
]


@dataclasses.dataclass
class VerificationReport:
    """Result of one invariant check (machine precision)."""

    ok: bool
    max_error: float = 0.0
    details: str = ""


class VerificationError(RuntimeError):
    """Raised when an operator's invariant fails (strict by default)."""


class Invariant:
    """A geometric invariant preserved by an operator, checked to machine
    precision: ``check(result, *args)``."""

    name: str = "invariant"
    atol: float = 1e-9

    def check(self, result, *args, **kwargs) -> VerificationReport:
        raise NotImplementedError


class SymplecticForm(Invariant):
    """pauli.commutes: result must equal omega(a, b) == 0 computed directly."""

    name = "symplectic_form"

    def check(self, result, a, b, **kwargs) -> VerificationReport:
        from .clifford import symplectic_commutes

        expected = bool(symplectic_commutes(a._x, a._z, b._x, b._z))
        return VerificationReport(
            ok=(result == expected),
            max_error=0.0 if result == expected else 1.0,
            details=(
                f"commutes({a.axis},{b.axis})={result} but omega-form gives {expected}"
                if result != expected
                else ""
            ),
        )


class ConjugationMatrixTruth(Invariant):
    """pauli.conjugate_by: conjugated Pauli (axis and phase) must equal
    C P C^+ computed as explicit matrices."""

    name = "conjugation_matrix_truth"

    def check(self, result, a, gates, **kwargs) -> VerificationReport:
        from .verify import _gate_matrix, _pauli_matrix

        conj, r = result
        n = a.dim
        C = np.eye(2**n, dtype=complex)
        for gate, args in gates:
            C = _gate_matrix(gate, args, n) @ C
        truth = C @ a.to_matrix() @ C.conj().T
        got = conj.to_matrix() * ((-1) ** r)
        err = float(np.abs(truth - got).max())
        return VerificationReport(
            ok=err < self.atol,
            max_error=err,
            details=f"conjugation matrix-truth max error {err:.2e}",
        )


class MergeClosure(Invariant):
    """rotation.merge: a merge result must equal R_P(t) R_P(s); a None
    result requires distinct axes (no same-axis orbit closure)."""

    name = "merge_closure"

    def check(self, result, a, b, **kwargs) -> VerificationReport:
        if result is None:
            ok = a.axis != b.axis
            return VerificationReport(
                ok=ok,
                details=("merged same-axis rotations but got None" if not ok else ""),
            )
        err = float(np.abs(result.to_matrix() - a.to_matrix() @ b.to_matrix()).max())
        return VerificationReport(
            ok=err < self.atol,
            max_error=err,
            details=f"merge closure max error {err:.2e}",
        )


class CancellationClosure(Invariant):
    """rotation.cancels: result must equal (theta ≡ 0 mod 2 pi)."""

    name = "cancellation_closure"

    def check(self, result, a, **kwargs) -> VerificationReport:
        expected = bool(np.isclose(a.theta % (2 * np.pi), 0.0, atol=1e-12))
        return VerificationReport(
            ok=(result == expected),
            details=f"cancels({a})={result} but 2-pi closure gives {expected}"
            if result != expected
            else "",
        )


class RotationActionClosure(Invariant):
    """rotation.apply_to_state: result must equal the closed-form action
    cos(theta/2)|psi> - i sin(theta/2) P|psi> (exists because P^2 = I)."""

    name = "rotation_action_closure"

    def check(self, result, rotation, state, **kwargs) -> VerificationReport:
        from .clifford import rotation_action_closed_form

        expected = rotation_action_closed_form(rotation.axis, rotation.theta, state)
        err = float(np.abs(np.asarray(result) - expected).max())
        return VerificationReport(
            ok=err < self.atol,
            max_error=err,
            details=f"rotation action closure max error {err:.2e}",
        )


class UnitaryEquivalence(Invariant):
    """circuit.optimize: U(input) == U(clifford_forward) @ U(output)."""

    name = "unitary_equivalence"

    def check(self, result, rotations, **kwargs) -> VerificationReport:
        from .verify import check_unitary_equivalence

        opt, cl = result
        ok = check_unitary_equivalence(rotations, opt, cl, atol=self.atol)
        return VerificationReport(
            ok=bool(ok),
            details=(
                f"circuit.optimize({len(rotations)} -> {len(opt)} rotations) "
                "failed unitary equivalence"
                if not ok
                else ""
            ),
        )


# ---------------------------------------------------------------------------
# Verification context (analogue of torch.no_grad())
# ---------------------------------------------------------------------------


class VerificationContext:
    """Global switch controlling automatic verification.

    Strict by default: a failing invariant raises VerificationError.  Use
    ``no_verify()`` to disable automatic verification for a block (e.g. for
    measured-speed paths where the invariant is checked once up front).
    """

    _enabled = True

    @classmethod
    def is_enabled(cls) -> bool:
        return cls._enabled


@contextlib.contextmanager
def no_verify():
    """Disable automatic verification in the enclosing block."""
    prev = VerificationContext._enabled
    VerificationContext._enabled = False
    try:
        yield
    finally:
        VerificationContext._enabled = prev


def verify_invariants(operator, result, *args, **kwargs) -> list[VerificationReport]:
    """Run all invariants of an operator against a call; return reports."""
    reports = []
    for inv in operator.invariants:
        reports.append(inv.check(result, *args, **kwargs))
    return reports
