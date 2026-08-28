"""Clifford (Pauli) algebra core.

A Pauli on n qubits is encoded as a 2n-bit symplectic vector (x, z):
P = i^m * prod_k X_k^{x_k} Z_k^{z_k}.  Commutation is decided by the
symplectic form omega((x,z),(x',z')) = x.z' + z.x' (mod 2).

Conjugation by a Clifford gate is a symplectic transformation; the +/- phase
is tracked with a tableau-style r-bit.  All phase rules use *old* bit values
and are verified against explicit matrix conjugation for every gate x every
Pauli (see geocore.verify; 0 failures).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "symplectic_commutes",
    "string_to_bits",
    "bits_to_string",
    "apply_gates_to_pauli",
]

PAULI_CHARS = "IXYZ"


def symplectic_commutes(x1, z1, x2, z2) -> bool:
    """True iff [P1, P2] = 0, where Pi = Pauli(x_i, z_i)."""
    return (np.dot(x1, z2) + np.dot(z1, x2)) % 2 == 0


def string_to_bits(axis: str) -> tuple[np.ndarray, np.ndarray]:
    """'XZIY' -> (x-bits, z-bits); X/Y set x, Z/Y set z."""
    n = len(axis)
    x = np.zeros(n, dtype=np.uint8)
    z = np.zeros(n, dtype=np.uint8)
    for i, c in enumerate(axis):
        if c in ("X", "Y"):
            x[i] = 1
        if c in ("Z", "Y"):
            z[i] = 1
    return x, z


def bits_to_string(x, z) -> str:
    out = []
    for xi, zi in zip(x, z):
        out.append(
            "I" if (xi, zi) == (0, 0) else "Y" if (xi, zi) == (1, 1) else "X" if xi else "Z"
        )
    return "".join(out)


def _apply_h(x, z, r, q):
    r ^= x[q] & z[q]
    x[q], z[q] = z[q], x[q]
    return x, z, r


def _apply_s(x, z, r, q):
    r ^= x[q] & z[q]
    z[q] ^= x[q]
    return x, z, r


def _apply_sd(x, z, r, q):
    r ^= x[q] & (z[q] ^ 1)
    z[q] ^= x[q]
    return x, z, r


def _apply_sqrt_x(x, z, r, q):
    r ^= z[q] & (x[q] ^ 1)
    x[q] ^= z[q]
    return x, z, r


def _apply_sqrt_xd(x, z, r, q):
    r ^= z[q] & x[q]
    x[q] ^= z[q]
    return x, z, r


def _apply_cnot(x, z, r, control, target):
    r ^= x[control] & z[target] & (x[target] ^ z[control] ^ 1)
    x[target] ^= x[control]
    z[control] ^= z[target]
    return x, z, r


_GATE_APPLIERS = {
    "h": _apply_h,
    "s": _apply_s,
    "sd": _apply_sd,
    "sx": _apply_sqrt_x,
    "sxdg": _apply_sqrt_xd,
    "cx": _apply_cnot,
}


def apply_clifford_to_pauli(gate: str, args, x, z, r=0):
    """Conjugate Pauli(x, z) by one Clifford gate; returns (x', z', r').

    Gate names: h, s, sd, sx, sxdg (single qubit), cx (two qubits).
    r=1 means the resulting Pauli carries a minus sign.
    """
    fn = _GATE_APPLIERS[gate]
    if gate == "cx":
        return fn(x, z, r, args[0], args[1])
    return fn(x, z, r, args[0])


def apply_gates_to_pauli(gates, x, z):
    """Conjugate Pauli(x, z) by a gate sequence; returns (x', z', r)."""
    r = 0
    for gate, args in gates:
        x, z, r = apply_clifford_to_pauli(gate, args, x, z, r)
    return x, z, r
