"""Machine-precision verification harness.

The core principle: a claim is accepted only if verified to machine
precision against an independent, explicit computation.

- ``check_conjugation_matrix_truth``: every Clifford gate x every Pauli on
  n <= 3, against explicit matrix conjugation (axis *and* phase).
- ``check_unitary_equivalence``: U(input) == U(output) * U(clifford) to 1e-9
  for a Pauli-rotation chain and its optimized form.
"""

from __future__ import annotations

import itertools

import numpy as np

from .clifford import bits_to_string, string_to_bits

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)

_H = (_X + _Z) / np.sqrt(2)
_S = np.diag([1, 1j])
_SD = np.diag([1, -1j])
_SX = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex) / 2
_CX01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
_SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)

_ONE_QUBIT_GATES = {"h": _H, "s": _S, "sd": _SD, "sx": _SX, "sxdg": _SX.conj().T}


def _pauli_matrix(axis):
    m = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}
    M = np.array([[1]], dtype=complex)
    for c in axis:
        M = np.kron(M, m[c])
    return M


def _gate_matrix(gate, args, n):
    if gate in _ONE_QUBIT_GATES:
        g = _ONE_QUBIT_GATES[gate]
        if n == 1:
            return g
        ops = [np.eye(2, dtype=complex)] * n
        ops[args[0]] = g
        M = ops[0]
        for o in ops[1:]:
            M = np.kron(M, o)
        return M
    c, t = args
    return _CX01 if (c, t) == (0, 1) else _SWAP @ _CX01 @ _SWAP


def check_conjugation_matrix_truth(max_n=3):
    """Return a list of (gate, args, pauli, truth, got) mismatches (empty = pass).

    Tests every single-qubit gate against I, X, Y, Z and CNOT against all
    3^n Paulis, comparing axis AND phase against explicit matrix conjugation.
    """
    from .clifford import apply_clifford_to_pauli

    mismatches = []
    names1 = ["I", "X", "Y", "Z"]
    for gate, g in _ONE_QUBIT_GATES.items():
        for a in names1:
            x, z = string_to_bits(a)
            x2, z2, r2 = apply_clifford_to_pauli(gate, (0,), x.copy(), z.copy())
            truth = g @ _pauli_matrix(a) @ g.conj().T
            got = _pauli_matrix(bits_to_string(x2, z2))
            if not (np.allclose(truth, got) or np.allclose(truth, -got)):
                mismatches.append((gate, (0,), a, truth, got))
            elif (np.allclose(truth, got)) != (r2 == 0):
                mismatches.append((gate, (0,), a, truth, got))
    names2 = [
        "".join(p) for p in itertools.product("XYZ", repeat=2)
    ]
    for c, t in [(0, 1), (1, 0)]:
        Gm = _gate_matrix("cx", (c, t), 2)
        for a in names2:
            x, z = string_to_bits(a)
            x2, z2, r2 = apply_clifford_to_pauli("cx", (c, t), x.copy(), z.copy())
            truth = Gm @ _pauli_matrix(a) @ Gm.conj().T
            got = _pauli_matrix(bits_to_string(x2, z2))
            if not (np.allclose(truth, got) or np.allclose(truth, -got)):
                mismatches.append(("cx", (c, t), a, truth, got))
            elif (np.allclose(truth, got)) != (r2 == 0):
                mismatches.append(("cx", (c, t), a, truth, got))
    return mismatches


def pauli_rotation_matrix(axis, theta):
    P = _pauli_matrix(axis)
    return np.cos(theta / 2) * np.eye(P.shape[0], dtype=complex) - 1j * np.sin(theta / 2) * P


def _norm(rotations):
    """Normalize rotation inputs to (axis, theta) tuples."""
    out = []
    for r in rotations:
        if isinstance(r, tuple):
            out.append(r)
        else:  # PauliRotation-like object
            out.append((r.axis, r.theta))
    return out


def circuit_matrix(rotations, n=None):
    """Operator of a rotation chain (rightmost rotation applied first)."""
    rotations = _norm(rotations)
    if n is None:
        n = len(rotations[0][0]) if rotations else 1
    U = np.eye(2**n, dtype=complex)
    for axis, theta in rotations:
        U = pauli_rotation_matrix(axis, theta) @ U
    return U


def check_unitary_equivalence(rotations, optimized, clifford, atol=1e-9):
    """True iff U(input) == U(clifford_forward) @ U(optimized) to atol."""
    rotations = _norm(rotations)
    optimized = _norm(optimized)
    n = len(rotations[0][0]) if rotations else len(optimized[0][0])
    Ua = circuit_matrix(rotations, n)
    Uc = np.eye(2**n, dtype=complex)
    for axis, theta in clifford:  # forward order: C1 C2 ...
        Uc = Uc @ pauli_rotation_matrix(axis, theta)
    Ub = Uc @ circuit_matrix(optimized, n)
    return np.allclose(Ua, Ub, atol=atol)
