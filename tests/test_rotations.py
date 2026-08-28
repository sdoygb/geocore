"""Tests: Pauli-rotation optimization — unitary equivalence and known cases."""

import numpy as np

from geocore import verify
from geocore.rotations import PauliRotation, optimize_pauli_rotations


def test_known_merge():
    rots = [("XX", 0.3), ("XX", 0.4)]
    opt, cl = optimize_pauli_rotations(rots)
    assert opt == [("XX", 0.7)] and cl == []
    assert verify.check_unitary_equivalence(rots, opt, cl)


def test_known_cancel():
    rots = [("XX", np.pi / 4), ("XX", -np.pi / 4)]
    opt, cl = optimize_pauli_rotations(rots)
    assert opt == []
    assert verify.check_unitary_equivalence(rots, opt, cl)


def test_issue_example():
    """The 5-rotation example from the Qiskit issue discussion: 5 -> 3."""
    rots = [
        ("XXII", -np.pi / 4),
        ("ZZIY", np.pi / 4),
        ("ZZYI", np.pi / 4),
        ("YYXX", np.pi / 4),
        ("XXII", np.pi / 4),
    ]
    opt, cl = optimize_pauli_rotations(rots)
    assert len(opt) == 3
    assert verify.check_unitary_equivalence(rots, opt, cl)


def test_pauli_rotation_object_input():
    rots = [PauliRotation("XX", 0.3), PauliRotation("XX", 0.4)]
    opt, cl = optimize_pauli_rotations(rots)
    assert opt == [("XX", 0.7)]
    assert verify.check_unitary_equivalence(rots, opt, cl)


def test_random_unitary_equivalence():
    rng = np.random.default_rng(7)
    for n in [1, 2, 3]:
        for _ in range(60):
            m = int(rng.integers(2, 7))
            rots = [
                ("".join(rng.choice(["X", "Y", "Z"], n)), float(rng.uniform(0.3, 1.2)))
                for _ in range(m)
            ]
            opt, cl = optimize_pauli_rotations(rots)
            assert len(opt) <= len(rots)
            assert verify.check_unitary_equivalence(rots, opt, cl)


def test_reduces_non_clifford_count_on_pi4_chains():
    """On pi/4 chains the pass should genuinely reduce rotation count."""
    rng = np.random.default_rng(3)
    reduced = 0
    for _ in range(40):
        m = int(rng.integers(4, 9))
        rots = [
            ("".join(rng.choice(["X", "Y", "Z"], 2)), float(rng.choice([np.pi / 4, -np.pi / 4])))
            for _ in range(m)
        ]
        opt, cl = optimize_pauli_rotations(rots)
        assert verify.check_unitary_equivalence(rots, opt, cl)
        if len(opt) < len(rots):
            reduced += 1
    assert reduced > 0, "expected at least one pi/4 chain to reduce"
