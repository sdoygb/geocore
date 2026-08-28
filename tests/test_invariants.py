"""Tests: Layer 2 automatic verification."""

import numpy as np
import pytest

import geocore as gc
from geocore import Pauli, Rotation, get_op
from geocore.invariants import VerificationError, no_verify


def test_verification_runs_automatically_and_passes():
    # every op call self-checks; correct calls must pass silently
    assert get_op("pauli.commutes")(Pauli("X"), Pauli("Z")) is False
    assert get_op("rotation.merge")(Rotation("XX", 0.3), Rotation("XX", 0.4)).theta == 0.7
    opt, cl = get_op("circuit.optimize")(
        [("XXII", -np.pi / 4), ("ZZIY", np.pi / 4), ("ZZYI", np.pi / 4),
         ("YYXX", np.pi / 4), ("XXII", np.pi / 4)]
    )
    assert len(opt) == 3


def test_failing_invariant_raises():
    """A deliberately wrong implementation must be caught by the invariant."""

    # sabotage: register a wrong impl for pauli.commutes
    op_reg = get_op("pauli.commutes")

    @op_reg.register(Pauli, Pauli)
    def _wrong(a, b):
        return not a.commutes_with(b)

    try:
        with pytest.raises(VerificationError):
            op_reg(Pauli("X"), Pauli("Z"))  # X, Z anticommute -> impl says True -> fail
    finally:
        # restore
        del op_reg._implementations[(Pauli, Pauli)]
        # re-register correct impl
        from geocore.ops import pauli_commutes

        @op_reg.register(Pauli, Pauli)
        def _correct(a, b):
            return a.commutes_with(b)


def test_no_verify_disables_automatic_verification():
    op_reg = get_op("pauli.commutes")

    @op_reg.register(Pauli, Pauli)
    def _wrong(a, b):
        return not a.commutes_with(b)

    try:
        # without no_verify: raises
        with pytest.raises(VerificationError):
            op_reg(Pauli("X"), Pauli("Z"))
        # with no_verify: passes silently (verification disabled)
        with no_verify():
            assert op_reg(Pauli("X"), Pauli("Z")) is True
    finally:
        del op_reg._implementations[(Pauli, Pauli)]
        from geocore.ops import pauli_commutes

        @op_reg.register(Pauli, Pauli)
        def _correct(a, b):
            return a.commutes_with(b)


def test_all_v01_ops_have_invariants():
    for name in ["pauli.commutes", "pauli.conjugate_by", "rotation.merge",
                 "rotation.cancels", "circuit.optimize"]:
        assert get_op(name).invariants, f"operator {name} has no invariants"


def test_conjugation_matrix_truth_runs():
    conj, r = get_op("pauli.conjugate_by")(Pauli("Y"), [("s", (0,))])
    assert conj.axis == "X" and r == 1
