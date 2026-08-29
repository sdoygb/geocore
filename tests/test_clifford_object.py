"""Tests for the Clifford group element object (L0 extension): the
symplectic tableau machinery (composition, conjugation, dense
reconstruction) verified against dense matrix truth."""

import numpy as np
import pytest

from geocore import Clifford, Pauli
from geocore.clifford import (
    clifford_gates_to_matrix,
    matrices_equal_up_to_phase,
)
from geocore.ops import clifford_compose, clifford_conjugate

rng = np.random.default_rng(31)

_SINGLE = ["h", "s", "sd", "sx", "sxdg"]


def rand_gates(n, n_gates):
    gates = []
    for _ in range(n_gates):
        g = rng.choice(_SINGLE + (["cx"] if n >= 2 else []))
        if g == "cx":
            c, t = rng.choice(n, 2, replace=False)
            gates.append((g, (int(c), int(t))))
        else:
            gates.append((g, (int(rng.integers(n)),)))
    return gates


def _rand_pauli(n):
    return Pauli("".join(rng.choice(list("XYZI"), n)))


def test_tableau_matches_gates_dense():
    """The tableau reconstruction equals the gate-sequence dense matrix up
    to a global phase (the tableau determines the Clifford projectively),
    to machine precision."""
    for n in [1, 2, 3]:
        for _ in range(10):
            gates = rand_gates(n, 6)
            C = Clifford(gates, n)
            ok, err = matrices_equal_up_to_phase(
                C.to_matrix(), clifford_gates_to_matrix(gates, n)
            )
            assert ok, (n, gates)
            assert err < 1e-9


def test_clifford_is_unitary():
    for n in [1, 2, 3]:
        C = Clifford(rand_gates(n, 8), n)
        err = np.abs(C.to_matrix() @ C.to_matrix().conj().T - np.eye(2**n)).max()
        assert err < 1e-9


def test_conjugate_verified_against_dense():
    """clifford.conjugate (axis and phase) equals C P C+ as explicit
    matrices — the op invariant runs automatically."""
    for n in [1, 2, 3]:
        for _ in range(5):
            C = Clifford(rand_gates(n, 5), n)
            P = _rand_pauli(n)
            conj, r = clifford_conjugate(C, P)  # raises if mismatch
            assert isinstance(conj, Pauli)


def test_compose_verified_against_dense():
    """clifford.compose equals the dense product C1 @ C2 up to a global
    phase (the op invariant runs automatically)."""
    for n in [1, 2, 3]:
        for _ in range(5):
            C1 = Clifford(rand_gates(n, 5), n)
            C2 = Clifford(rand_gates(n, 5), n)
            C = clifford_compose(C1, C2)  # raises if mismatch
            assert C.n == n


def test_compose_associative():
    """C1 (C2 C3) == (C1 C2) C3 (tableau composition is associative)."""
    C1, C2, C3 = (Clifford(rand_gates(2, 4), 2) for _ in range(3))
    ok, err = matrices_equal_up_to_phase(
        C1.compose(C2.compose(C3)).to_matrix(),
        (C1.compose(C2)).compose(C3).to_matrix(),
    )
    assert ok
    assert err < 1e-9


def test_composition_conjugation_consistency():
    """(C1 C2) P (C2+ C1+) == C1 (C2 P C2+) C1+, phases included: the
    phase of C1 applied to (-1)^{r2} axis is r1 XOR r2."""
    for _ in range(50):
        C1, C2 = Clifford(rand_gates(2, 4), 2), Clifford(rand_gates(2, 4), 2)
        P = _rand_pauli(2)
        axis2, r2 = C2.conjugate(P)
        axis1, r1 = C1.conjugate(Pauli(axis2.axis))
        lhs = C1.compose(C2).conjugate(P)
        assert lhs[0].axis == axis1.axis
        assert lhs[1] == (r1 + r2) % 2


def test_classic_gate_rules():
    """The textbook Clifford conjugation rules, exact."""
    C = Clifford([("cx", (0, 1))], 2)
    assert C.conjugate(Pauli("XI")) == (Pauli("XX"), 0)
    assert C.conjugate(Pauli("IZ")) == (Pauli("ZZ"), 0)
    assert C.conjugate(Pauli("YI")) == (Pauli("YX"), 0)
    assert Clifford([("s", (0,))], 1).conjugate(Pauli("Y")) == (Pauli("X"), 1)


def test_identities_up_to_phase():
    """H^2 = I and S^2 = Z (the reconstruction is exact up to the global
    phase that the tableau cannot determine)."""
    H2 = Clifford([("h", (0,)), ("h", (0,))], 1).to_matrix()
    S2 = Clifford([("s", (0,)), ("s", (0,))], 1).to_matrix()
    assert matrices_equal_up_to_phase(H2, np.eye(2))[0]
    assert matrices_equal_up_to_phase(S2, np.diag([1, -1]))[0]


def test_single_qubit_clifford_x_orbit():
    """The single-qubit Clifford group maps X to any of ±X, ±Y, ±Z — three
    axes."""
    axes = set()
    seqs = [
        ("h",), ("s",), ("sd",), ("sx",), ("sxdg",),
        ("h", "s"), ("s", "h"), ("h", "s", "h"), ("s", "h", "s"),
    ]
    for seq in seqs:
        C = Clifford([(g, (0,)) for g in seq], 1)
        axes.add(C.conjugate(Pauli("X"))[0].axis)
    assert axes == {"X", "Y", "Z"}


def test_compose_qubit_mismatch_raises():
    with pytest.raises(ValueError):
        Clifford([("h", (0,))], 1).compose(Clifford([("cx", (0, 1))], 2))
