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


def pauli_action_on_state(axis: str, state):
    """Action of the Pauli P on a state vector (O(2^n)).

    Decompose P = i^m X^x Z^z (Y = i X Z, so m = number of Y factors).
    Z acts first (signs by original bits), then X (a bitwise permutation),
    then the global phase i^m.  Never builds the dense 2^n x 2^n matrix.
    """
    n = len(axis)
    xbits = [q for q, c in enumerate(axis) if c in ("X", "Y")]
    zbits = [q for q, c in enumerate(axis) if c in ("Z", "Y")]
    n_y = axis.count("Y")

    idx = np.arange(state.size)
    out = state.copy()
    # Z^z: sign flips by the original bit values
    for q in zbits:
        bit = 1 << (n - 1 - q)
        out = np.where((idx >> (n - 1 - q)) & 1, -out, out)
    # X^x: accumulate the bitwise permutation, apply once
    for q in xbits:
        bit = 1 << (n - 1 - q)
        idx = idx ^ bit
    out = out[idx]
    # global phase i^m
    return (1j ** n_y) * out


def rotation_action_closed_form(axis: str, theta: float, state):
    """R_P(theta)|psi> = cos(theta/2)|psi> - i sin(theta/2) P|psi>.

    Closed form because P^2 = I (the rotation orbit of the Pauli axis
    closes in two steps); O(2^n) instead of O(8^n) for a dense expm.
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return c * state - 1j * s * pauli_action_on_state(axis, state)


# ---------------------------------------------------------------------------
# Clifford tableau representation (Aaronson-Gottesman).
#
# A Clifford element on n qubits is the 2n x 2n binary symplectic tableau
# whose rows are the conjugates of the generators (X_i in rows 0..n-1,
# Z_i in rows n..2n-1), plus a 2n-bit phase vector.  Composition and
# conjugation are binary linear algebra; the dense matrix is rebuilt from
# the tableau for verification.
# ---------------------------------------------------------------------------


def build_clifford_tableau(gates, n):
    """Tableau (2n x 2n) and phases (2n,) from a gate sequence: row i is
    the conjugation of generator i (X_i for i < n, Z_{i-n} otherwise).

    The phase is the full 2-bit phase q (i^q, q in 0..3): the conjugated
    generator is i^q X^x Z^z.  (Y = i X Z, so a Y factor carries an i that
    the single sign bit of ``apply_gates_to_pauli`` does not track — it
    must enter the multiplication phase, hence q = (2r + m) mod 4 with m
    the number of Y factors in the row.)
    """
    tableau = np.zeros((2 * n, 2 * n), dtype=int)
    phases = np.zeros(2 * n, dtype=int)
    for i in range(2 * n):
        x = np.zeros(n, dtype=int)
        z = np.zeros(n, dtype=int)
        if i < n:
            x[i] = 1
        else:
            z[i - n] = 1
        xp, zp, r = apply_gates_to_pauli(gates, x, z)
        tableau[i, :n] = xp
        tableau[i, n:] = zp
        m = int(bits_to_string(xp, zp).count("Y"))
        phases[i] = (2 * r + m) % 4
    return tableau, phases


def _row_phase(x_row, z_row, q_row, x_acc, z_acc, q_acc):
    """Phase of (accumulated Pauli) x (row Pauli): the row carries i^{q_row}
    X^x Z^z, the accumulated i^{q_acc} X^{xa} Z^{za}; the product gains the
    Z-crossing-X interaction (-1)^{za . x} = i^{2 (za . x)}."""
    inter = int(np.dot(z_acc, x_row)) % 2
    return (q_acc + q_row + 2 * inter) % 4


def conjugate_pauli_tableau(tableau, phases, n, x, z, q0=None):
    """Conjugate X^x Z^z by the Clifford (tableau, phases).

    The conjugate is the ordered product of the conjugated generators,
    each i^{q} X^x Z^z; phases and the Z-crossing-X interaction accumulate
    mod 4.  Returns (x', z', q) with the full phase i^q.

    ``q0`` is the initial phase of the input: by default the input is a
    Pauli in Y-semantics, i^{m_in} X^x Z^z with m_in overlapping bits
    (a Y factor); pass ``q0=0`` when the input is a raw XZ pair whose
    phase is tracked separately (as in composition).
    """
    x = np.asarray(x, dtype=int)
    z = np.asarray(z, dtype=int)
    xp = np.zeros(n, dtype=int)
    zp = np.zeros(n, dtype=int)
    if q0 is None:
        q0 = int(np.sum(x & z)) % 4  # Y factors of the input carry i
    q = q0
    for i in range(n):
        if x[i]:
            q = _row_phase(tableau[i, :n], tableau[i, n:], phases[i], xp, zp, q)
            xp ^= tableau[i, :n]
            zp ^= tableau[i, n:]
        if z[i]:
            q = _row_phase(tableau[n + i, :n], tableau[n + i, n:], phases[n + i], xp, zp, q)
            xp ^= tableau[n + i, :n]
            zp ^= tableau[n + i, n:]
    return xp, zp, q


def compose_clifford(t1, p1, t2, p2, n):
    """C = C1 @ C2: row i of C is C1 conjugating (C2 G_i C2^+), whose
    Pauli is tableau2 row i with phase p2[i] (i^{p2[i]})."""
    t = np.zeros_like(t1)
    p = np.zeros_like(p1)
    for i in range(2 * n):
        xp, zp, q = conjugate_pauli_tableau(
            t1, p1, n, t2[i, :n], t2[i, n:], q0=0
        )
        p[i] = (q + p2[i]) % 4
        t[i, :n] = xp
        t[i, n:] = zp
    return t, p


def _state_apply_row(x_row, z_row, q_row, psi):
    """Apply the conjugated generator i^{q} X^x Z^z to a state: the XZ
    action is pauli_action_on_state (which includes the Y = i X Z factors),
    and the remaining i-power is i^{q - m} with m the row's Y count."""
    axis = bits_to_string(x_row, z_row)
    m = axis.count("Y")
    return (1j ** ((q_row - m) % 4)) * pauli_action_on_state(axis, psi)


def clifford_tableau_to_matrix(tableau, phases, n):
    """Dense 2^n x 2^n matrix from the tableau, exact up to a global phase.

    A tableau determines the Clifford only projectively (C and e^{iθ}C
    share a tableau), so the reconstruction pins the global phase by the
    convention that the projection of |0> onto C|0> is real positive.
    Columns: C|j> = (C X^j C^+) C|0>, where the conjugated X^j has an
    exact phase from the tableau — hence all columns share the same
    global-phase factor.  Use ``matrices_equal_up_to_phase`` to compare
    against a gate-sequence matrix.
    """
    d = 2**n
    M = np.zeros((d, d), dtype=complex)
    # column 0: C|0> has stabilizers {C Z_k C+} (all +1).  Project a
    # computational basis state onto the stabilizer space; if the overlap
    # vanishes (e.g. C|0> has no |0> component), try the next basis state.
    psi = None
    for a in range(d):
        ea = np.zeros(d, dtype=complex)
        ea[a] = 1.0
        cand = ea.copy()
        for k in range(n):
            ppsi = _state_apply_row(
                tableau[n + k, :n], tableau[n + k, n:], phases[n + k], cand
            )
            cand = (cand + ppsi) / 2.0
        if np.linalg.norm(cand) > 1e-12:
            psi = cand / np.linalg.norm(cand)
            break
    if psi is None:
        raise ValueError("clifford_tableau_to_matrix: no computational basis seed")
    M[:, 0] = psi
    # columns j: C|j> = (C X^j C+) C|0>
    for j in range(1, d):
        x = np.array([(j >> (n - 1 - q)) & 1 for q in range(n)], dtype=int)
        xp, zp, q = conjugate_pauli_tableau(
            tableau, phases, n, x, np.zeros(n, dtype=int)
        )
        M[:, j] = _state_apply_row(xp, zp, q, psi)
    return M


def matrices_equal_up_to_phase(M, N, atol=1e-9) -> tuple[bool, float]:
    """Compare two Clifford matrices up to a global phase (the tableau
    determines the Clifford only projectively).  The global phase of a
    Clifford circuit is an 8th root of unity (e.g. sqrt-X carries
    e^{i pi/4}), so the eight candidates suffice."""
    best = min(
        float(np.abs(M - lam * N).max())
        for lam in np.exp(1j * np.arange(8) * np.pi / 4)
    )
    return best < atol, best


def clifford_gates_to_matrix(gates, n):
    """Dense matrix from the gate sequence directly (the independent
    reference truth for the tableau machinery)."""
    from .verify import _gate_matrix

    M = np.eye(2**n, dtype=complex)
    for gate, args in gates:
        M = _gate_matrix(gate, args, n) @ M
    return M
