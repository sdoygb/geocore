"""Tests for the Circuit object: gate sequences of Clifford gates and
Pauli rotations, with the geometric optimizer (Clifford pull-through +
rotation-chain reduction) verified to preserve the unitary exactly."""

import numpy as np
import pytest

from geocore import Circuit
from geocore.invariants import VerificationError

rng = np.random.default_rng(41)


def rand_circuit(n, depth=10):
    gates = []
    for _ in range(depth):
        g = rng.choice(["h", "s", "sx", "cx", "r"] if n >= 2 else ["h", "s", "sx", "r"])
        if g == "r":
            axis = "".join(rng.choice(list("IXYZ"), n))
            if axis == "I" * n:
                axis = "I" * (n - 1) + "X"
            gates.append(("r", axis, rng.uniform(-1.0, 1.0)))
        elif g == "cx":
            c, t = rng.choice(n, 2, replace=False)
            gates.append(("cx", int(c), int(t)))
        else:
            gates.append((g, int(rng.integers(n))))
    return Circuit(gates, n=n)


def test_apply_to_state_matches_matrix():
    """apply_to_state (closed-form rotations) == to_matrix @ state."""
    for _ in range(10):
        circ = rand_circuit(int(rng.integers(1, 4)), depth=6)
        state = rng.standard_normal(2 ** circ.n) + 1j * rng.standard_normal(2 ** circ.n)
        assert np.abs(circ.apply_to_state(state) - circ.to_matrix() @ state).max() < 1e-9


def test_optimize_preserves_unitary_random():
    """U(input) == U(clifford) @ U(optimized) exactly, for random mixed
    circuits (Clifford pull-through + rotation merge/absorb)."""
    for n in [2, 3]:
        for _ in range(15):
            circ = rand_circuit(n)
            opt, cliff = circ.optimize()  # raises VerificationError if not equivalent
            err = np.abs(circ.to_matrix() - cliff.to_matrix() @ opt.to_matrix()).max()
            assert err < 1e-9
            assert opt.num_gates <= circ.num_gates


def test_optimize_never_increases_gates_and_reduces_known_case():
    """The textbook 5 -> 3 rotation reduction, exact."""
    rot = Circuit([
        ("r", "XXII", -np.pi / 4), ("r", "ZZIY", np.pi / 4),
        ("r", "ZZYI", np.pi / 4), ("r", "YYXX", np.pi / 4),
        ("r", "XXII", np.pi / 4),
    ], n=4)
    opt, cliff = rot.optimize()
    assert opt.num_gates == 3
    assert np.abs(rot.to_matrix() - cliff.to_matrix() @ opt.to_matrix()).max() < 1e-9


def test_optimize_merges_and_absorbs_pi_over_2():
    """Same-axis rotations merge; the pi/2 part becomes an exact Clifford
    rotation in the clifford output."""
    c = Circuit([("r", "YZ", 0.604), ("r", "YZ", 0.962)], n=2)
    opt, cliff = c.optimize()
    assert opt.num_gates == 1
    assert any(g[0] == "r" and abs(g[2]) == pytest.approx(np.pi / 2) for g in cliff.gates)
    assert np.abs(c.to_matrix() - cliff.to_matrix() @ opt.to_matrix()).max() < 1e-9


def test_pull_through_identity():
    """C R_P(th) = R_{C P C+}(th) C at the matrix level (H R_X = R_Z H)."""
    lhs = Circuit([("r", "X", 0.5), ("h", 0)], n=1).to_matrix()
    rhs = Circuit([("h", 0), ("r", "Z", 0.5)], n=1).to_matrix()
    assert np.abs(lhs - rhs).max() < 1e-12


def test_optimize_clifford_only_circuit():
    """A Clifford-only circuit optimizes to the same unitary (all gates
    in the clifford output)."""
    c = Circuit([("h", 0), ("cx", 0, 1), ("s", 1), ("h", 1)], n=2)
    opt, cliff = c.optimize()
    assert opt.num_gates == 0
    assert np.abs(c.to_matrix() - cliff.to_matrix()).max() < 1e-9


def test_identity_rotation_raises():
    """R_I(theta) is a global phase the optimizer cannot represent; it
    raises a clear error."""
    with pytest.raises(ValueError, match="global phase"):
        Circuit([("r", "II", 0.5)], n=2).optimize()


def test_empty_circuit():
    opt, cliff = Circuit([], n=2).optimize()
    assert np.abs(np.eye(4) - cliff.to_matrix() @ opt.to_matrix()).max() < 1e-12


def test_append_api():
    c = Circuit(n=2)
    c.append("h", 0).append("cx", 0, 1).append("r", "XX", 0.3)
    assert c.num_gates == 3
    assert c.gates == [("h", 0), ("cx", 0, 1), ("r", "XX", 0.3)]


def test_wrong_dimension_state_raises():
    c = Circuit([("h", 0)], n=1)
    with pytest.raises(ValueError):
        c.apply_to_state(np.zeros(4, dtype=complex))
