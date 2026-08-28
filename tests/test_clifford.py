"""Tests: conjugation primitives against matrix truth."""

import numpy as np

from geocore import verify
from geocore.clifford import string_to_bits


def test_conjugation_matrix_truth_zero_failures():
    mismatches = verify.check_conjugation_matrix_truth()
    assert mismatches == [], f"{len(mismatches)} conjugation mismatches: {mismatches[:3]}"


def test_specific_phase_cases():
    """Spot checks of the subtle phase cases that bit us during development."""
    from geocore.clifford import apply_gates_to_pauli, bits_to_string

    # S^+ . Y . S = -X  (phase matters)
    x, z = string_to_bits("Y")
    x2, z2, r2 = apply_gates_to_pauli([("sd", (0,))], x, z)
    assert bits_to_string(x2, z2) == "X" and r2 == 0

    # sqrt(X) . Z . sqrt(X)^+ = -Y
    x, z = string_to_bits("Z")
    x2, z2, r2 = apply_gates_to_pauli([("sx", (0,))], x, z)
    assert bits_to_string(x2, z2) == "Y" and r2 == 1

    # CNOT(1,0) . (Z X) . CNOT(1,0) = -Y Y  (component reordering phase)
    x, z = string_to_bits("ZX")
    x2, z2, r2 = apply_gates_to_pauli([("cx", (1, 0))], x, z)
    assert bits_to_string(x2, z2) == "YY" and r2 == 1
