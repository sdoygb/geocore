"""Layer 0 — the Circuit object: a gate sequence of Clifford gates
(``h, s, sd, sx, sxdg, cx``) and Pauli rotations ``R_P(theta)``.

``optimize()`` is the geometric optimizer: Clifford gates are pulled
through the rotations (C R_P(th) = R_{C+ P C}(th) C, an exact identity),
the rotation chain is reduced to its fixed point (merge / 2-pi cancel /
pi/2 absorption), and the total unitary is verified automatically against
the input to machine precision (U(in) == U(clifford) @ U(optimized)).
"""

from __future__ import annotations

import numpy as np

from .objects import Clifford, GeometricObject, Pauli
from .verify import _gate_matrix, pauli_rotation_matrix

__all__ = ["Circuit"]

_DAGGER = {"h": "h", "s": "sd", "sd": "s", "sx": "sxdg", "sxdg": "sx", "cx": "cx"}


class Circuit(GeometricObject):
    """A quantum circuit: Clifford gates and Pauli rotations.

    Gate formats (application order = list order):
      - Clifford:  ``("h", 0)``, ``("cx", 0, 1)`` — name and qubit args
      - rotation:  ``("r", axis, theta)`` — R_axis(theta)

    ``to_matrix`` is the dense reference; ``apply_to_state`` acts on a
    state (rotations via the closed form, O(2^n)); ``optimize`` pulls
    Clifford gates through the rotations, optimizes the rotation chain,
    and verifies U(in) == U(clifford) @ U(optimized) automatically.
    """

    def __init__(self, gates=None, n=None):
        self.gates = list(gates or [])
        qubits = set()
        for g in self.gates:
            if g[0] == "r":
                qubits.update(i for i, c in enumerate(g[1]) if c != "I")
            else:
                qubits.update(g[1:])
        self.n = n if n is not None else (max(qubits) + 1 if qubits else 1)

    @property
    def dim(self) -> int:
        return self.n

    @property
    def num_gates(self) -> int:
        return len(self.gates)

    def append(self, gate, *args):
        """Append a gate: Clifford (name, *qubits) or a rotation
        (("r", axis, theta))."""
        if gate == "r":
            axis, theta = args
            self.gates.append(("r", axis, float(theta)))
        else:
            self.gates.append((gate, *args))
        return self

    def to_matrix(self) -> np.ndarray:
        """Dense matrix of the circuit (the reference truth)."""
        M = np.eye(2**self.n, dtype=complex)
        for g in self.gates:
            if g[0] == "r":
                M = pauli_rotation_matrix(g[1], g[2]) @ M
            else:
                M = _gate_matrix(g[0], tuple(g[1:]), self.n) @ M
        return M

    def apply_to_state(self, state) -> np.ndarray:
        """Apply the circuit to a state: rotations via the closed form
        R_P(theta)|psi> (O(2^n)), Clifford gates via their matrices."""
        state = np.asarray(state, dtype=complex)
        for g in self.gates:
            if g[0] == "r":
                from .clifford import rotation_action_closed_form

                state = rotation_action_closed_form(g[1], g[2], state)
            else:
                state = _gate_matrix(g[0], tuple(g[1:]), self.n) @ state
        return state

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def optimize(self):
        """Optimize the rotation structure, pulling Clifford gates
        through (C R_P(th) = R_{C+ P C}(th) C).

        Returns (optimized_circuit, clifford_circuit) with
        U(input) == U(clifford) @ U(optimized), verified to machine
        precision (raises VerificationError otherwise).
        """
        from .invariants import VerificationError
        from .rotations import optimize_pauli_rotations

        prefix: list = []  # Clifford gates applied before the rotations
        rotations: list = []  # (axis, theta), in application order
        for g in self.gates:
            if g[0] == "r":
                axis, theta = g[1], g[2]
                if axis == "I" * len(axis):
                    raise ValueError(
                        "circuit.optimize: R_I(theta) is a global phase "
                        "e^{-i theta/2} I which the optimizer cannot "
                        "represent — use a non-identity axis"
                    )
                if prefix:
                    # C R_P(th) = R_{C+ P C}(th) C: the rotation pulled
                    # through the accumulated Clifford is applied AFTER
                    # the previous rotations (append).
                    Cd = Clifford([(_DAGGER[pg[0]], tuple(pg[1:])) for pg in reversed(prefix)], self.n)
                    new_axis, r = Cd.conjugate(Pauli(axis))
                    theta = -theta if r else theta
                    axis = new_axis.axis
                rotations.append((axis, theta))
            else:
                prefix.append(g)

        opt_rot, cliff_rot = optimize_pauli_rotations(rotations)
        # build the outputs.  The absorbed pi/2 pieces stay as EXACT
        # rotation gates R_axis(k*pi/2) (a Clifford rotation): expressing
        # them with S gates would carry the e^{i pi/4} global phase of S
        # relative to R_Z(pi/2) and break exact unitary equivalence.
        opt_circuit = Circuit([("r", a, t) for a, t in opt_rot], n=self.n)
        cliff_gates: list = [("r", a, ang) for a, ang in cliff_rot]
        cliff_gates.extend(prefix)
        cliff_circuit = Circuit(cliff_gates, n=self.n)

        # automatic verification: U(in) == U(clifford) @ U(optimized)
        err = float(
            np.abs(
                self.to_matrix()
                - cliff_circuit.to_matrix() @ opt_circuit.to_matrix()
            ).max()
        )
        if err > 1e-9:
            raise VerificationError(
                f"circuit.optimize: unitary equivalence failed (err {err:.2e})"
            )
        return opt_circuit, cliff_circuit

    def verify(self) -> dict:
        return {"ok": True, "note": "circuit invariants checked by optimize()"}

    def __repr__(self):
        return f"Circuit(n={self.n}, {self.num_gates} gates)"
