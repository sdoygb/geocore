"""Geometric rotation objects and Pauli-rotation optimization.

R_P(theta) = exp(-i theta P / 2) is a first-class geometric object.  The
optimization machinery is organized by geometric closure principles:

- merge:  R_P(t) R_P(s) = R_P(t+s)   (closure of phase addition, same axis)
- cancel: R_P(theta) = 1  iff theta ≡ 0 (mod 2 pi)   (2-pi closure)
- pull-through: [C, R_P(th)] -> [R_{C^+ P C}(th), C]   (conjugation by C^+,
  the dagger piece), with the pi/2 part of any merged angle absorbed into a
  Clifford accumulated at the end of the circuit
- termination: iterate to the fixed point where no merge is possible
  (completeness / 圆满)

The output is (rotations, accumulated Clifford list [C1, C2, ...]) with
operator semantics U_out = C1 C2 ... (rotations operator).
"""

from __future__ import annotations

import numpy as np

from .clifford import apply_gates_to_pauli, bits_to_string, string_to_bits

__all__ = ["PauliRotation", "optimize_pauli_rotations"]


class PauliRotation:
    """A geometric rotation R_P(theta) about the Pauli axis P."""

    __slots__ = ("axis", "theta")

    def __init__(self, axis: str, theta: float):
        if not all(c in "IXYZ" for c in axis):
            raise ValueError(f"invalid Pauli axis: {axis!r}")
        self.axis = axis
        self.theta = theta

    def __repr__(self):
        return f"R_{self.axis}({self.theta:.6g})"


class _Bucket:
    """Bucket of rotations keyed by axis (symplectic representation)."""

    def __init__(self, n: int):
        self.n = n
        self.xs: list[np.ndarray] = []
        self.zs: list[np.ndarray] = []
        self.angles: list[float] = []

    def insert(self, x, z):
        self.xs.append(x)
        self.zs.append(z)
        self.angles.append(0.0)
        return len(self.xs) - 1

    def pop_last(self):
        self.xs.pop()
        self.zs.pop()
        self.angles.pop()

    def commutes(self, i: int, j: int) -> bool:
        from .clifford import symplectic_commutes

        return symplectic_commutes(self.xs[i], self.zs[i], self.xs[j], self.zs[j])

    def same_axis(self, i: int, j: int) -> bool:
        return np.array_equal(self.xs[i], self.xs[j]) and np.array_equal(
            self.zs[i], self.zs[j]
        )


def _simplify_angle(theta: float) -> tuple[float, int]:
    """Split theta = theta' + k*pi/2 (k = round(theta / (pi/2)))."""
    k = round(theta / (np.pi / 2))
    return theta - k * (np.pi / 2), k


def _pi2_piece(axis: str, k: int):
    """Clifford circuit implementing R_axis(k*pi/2).

    R_P(pi/2) = (prod C_q)^+ . fold . S_target . fold^+ . (prod C_q),
    where C_q maps the per-qubit factor to Z (X -> H, Y -> SX), the fold
    C = CX(q_k,0)...CX(q_1,0) satisfies C . S_target . C^+ = R_{prod Z_q}
    (the target's Z picks up every control), and each dagger is the
    reversed gate sequence (CX is self-inverse, but order matters).
    """
    support = [i for i, c in enumerate(axis) if c != "I"]
    if not support:
        return []
    gates = []
    for q in support:
        c = axis[q]
        if c == "X":
            gates.append(("h", (q,)))
        elif c == "Y":
            gates.append(("sx", (q,)))
    target = support[0]
    others = support[1:]
    for q in reversed(others):  # fold^+ : reversed C
        gates.append(("cx", (q, target)))
    for _ in range(k % 4):
        gates.append(("s", (target,)))
    for q in others:  # fold : C
        gates.append(("cx", (q, target)))
    for q in reversed(support):  # (prod C_q)^+ : reversed, daggers
        c = axis[q]
        if c == "X":
            gates.append(("h", (q,)))
        elif c == "Y":
            gates.append(("sxdg", (q,)))
    return gates


def _pi2_piece_dagger(axis: str, k: int):
    """Dagger of the pi/2 piece: implements R_axis(-k*pi/2) = C^+.

    The circuit transformation [C, R_P(th)] -> [R_{C^+ P C}(th), C] requires
    conjugation by C^+, hence the dagger piece (verified against matrix
    truth; the non-dagger direction flips the angle sign).
    """
    inv = {"h": "h", "s": "sd", "sd": "s", "sx": "sxdg", "sxdg": "sx", "cx": "cx"}
    return [(inv[g], args) for g, args in reversed(_pi2_piece(axis, k))]


def _merge_pass(rotations: list[tuple[str, float]]):
    """One merge/cancel pass; returns (rotations, accumulated Clifford list)."""
    if not rotations:
        return [], []
    n = len(rotations[0][0])
    bucket = _Bucket(n)
    rest = [
        (string_to_bits(axis)[0], string_to_bits(axis)[1], theta)
        for axis, theta in rotations
    ]
    rot_index = 0
    clifford: list[tuple[str, float]] = []
    while rest:
        x, z, theta = rest.pop(0)
        idx = bucket.insert(x, z)
        merged = False
        for i in range(idx - 1, -1, -1):
            if not bucket.commutes(idx, i):
                break
            if bucket.same_axis(i, idx):
                bucket.angles[i] += theta
                merged = True
                new_angle, k = _simplify_angle(bucket.angles[i])
                if k % 4 != 0:
                    piece = _pi2_piece_dagger(bits_to_string(x, z), k)
                    new_rest = []
                    for xr, zr, th in rest:
                        xr2, zr2, r2 = apply_gates_to_pauli(piece, xr, zr)
                        new_rest.append((xr2, zr2, (-1 if r2 else 1) * th))
                    rest = new_rest
                    clifford.append((bits_to_string(x, z), k * (np.pi / 2)))
                bucket.angles[i] = new_angle
                if np.isclose(new_angle % (2 * np.pi), 0.0, atol=1e-12):
                    bucket.xs[i][:] = 0  # 2-pi closure: rotation vanishes
                    bucket.zs[i][:] = 0
                    bucket.angles[i] = 0.0
                break
        if merged:
            bucket.pop_last()
        else:
            bucket.angles[idx] = theta
        rot_index += 1
    out = []
    for i in range(len(bucket.xs)):
        axis = bits_to_string(bucket.xs[i], bucket.zs[i])
        if axis != "I" * n and abs(bucket.angles[i]) > 1e-12:
            out.append((axis, bucket.angles[i]))
    return out, clifford


def optimize_pauli_rotations(rotations):
    """Optimize a chain of Pauli rotations to the fixed point.

    Parameters
    ----------
    rotations : iterable of (axis, theta) or PauliRotation

    Returns
    -------
    (optimized_rotations, clifford) : tuple of lists
        optimized_rotations: list of (axis, theta).
        clifford: accumulated Clifford rotations [C1, C2, ...] applied after
        the rotations (operator semantics U = C1 C2 ... (rotations op)).
    """
    current = [(r.axis, r.theta) if isinstance(r, PauliRotation) else (r[0], r[1]) for r in rotations]
    clifford: list[tuple[str, float]] = []
    while True:
        new, cl = _merge_pass(current)
        clifford.extend(cl)
        if len(new) >= len(current):
            break
        current = new
    return current, clifford
