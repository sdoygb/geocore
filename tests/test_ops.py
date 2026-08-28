"""Tests: Layer 1 geometric operator dispatch."""

import numpy as np

import geocore as gc
from geocore import Pauli, Rotation, get_op, op


def test_dispatch_routes_by_geometric_type():
    assert get_op("pauli.commutes")(Pauli("X"), Pauli("Z")) is False
    assert get_op("pauli.commutes")(Pauli("X"), Pauli("X")) is True
    assert get_op("rotation.cancels")(Rotation("X", 2 * np.pi)) is True
    assert get_op("rotation.cancels")(Rotation("X", 0.5)) is False


def test_pauli_conjugate_operator():
    gates = [("h", (0,))]
    conj, r = get_op("pauli.conjugate_by")(Pauli("X"), gates)
    assert conj.axis == "Z" and r == 0
    conj2, r2 = get_op("pauli.conjugate_by")(Pauli("Y"), [("s", (0,))])
    assert conj2.axis == "X" and r2 == 1  # S.Y.S^+ = -X (phase matters)


def test_rotation_merge_operator():
    m = get_op("rotation.merge")(Rotation("XX", 0.3), Rotation("XX", 0.4))
    assert m.axis == "XX" and np.isclose(m.theta, 0.7)
    assert get_op("rotation.merge")(Rotation("XX", 0.3), Rotation("YY", 0.4)) is None


def test_circuit_optimize_operator():
    rots = [("XXII", -np.pi / 4), ("ZZIY", np.pi / 4), ("ZZYI", np.pi / 4),
            ("YYXX", np.pi / 4), ("XXII", np.pi / 4)]
    opt, cl = get_op("circuit.optimize")(rots)
    assert len(opt) == 3
    from geocore import verify

    assert verify.check_unitary_equivalence(rots, opt, cl)


def test_operator_registry_is_geometric():
    """Every v0.1 operator documents its geometric theorem."""
    for name in ["pauli.commutes", "pauli.conjugate_by", "rotation.merge",
                 "rotation.cancels", "circuit.optimize"]:
        o = get_op(name)
        assert o.theorem, f"operator {name} missing geometric theorem"


def test_unknown_types_raise():
    import pytest

    with pytest.raises(NotImplementedError):
        get_op("rotation.merge")(1, 2)


def test_objects_verify():
    assert Pauli("XYZ").verify()["ok"]
    assert Rotation("X", 0.3).verify()["ok"]
