"""Tests for the N-sector (particle-number) reduction of the discrete-
evolution solver (examples/vqe_sector_reduction.py): the sector
Hamiltonian's ground state equals the full-space one, the sector
evolution matches (or beats) the full-space one, and the cc-pVDZ
dimension reduction is the promised 3.7e6x.

Requires openfermion + openfermionpyscf (for LiH/H2O Hamiltonians).
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from vqe_sector_reduction import (  # noqa: E402
    sector_evolve,
    sector_hamiltonian,
    sector_states,
    pauli_action_int,
)
from vqe_lih_evolution import lih_hamiltonian  # noqa: E402

pytest.importorskip("openfermion")
pytest.importorskip("openfermionpyscf")

LIH = [["Li", [0, 0, 0]], ["H", [0, 0, 1.6]]]


def test_sector_ground_state_equals_full():
    """The N-sector Hamiltonian's GS equals the full-space GS (the
    particle number is a conserved quantity)."""
    n, diag, off, gs, E0, fci = lih_hamiltonian(geometry=LIH)
    hd, H_off = sector_hamiltonian(n, 4, diag, off)
    ev = np.linalg.eigvalsh(np.diag(hd) + H_off.toarray())
    assert abs(ev[0] - E0) < 1e-8


def test_sector_evolution_matches_full():
    """Zero-gradient evolution inside the N sector reaches the sector
    ground state (fidelity ~1)."""
    n, diag, off, gs, E0, fci = lih_hamiltonian(geometry=LIH)
    hd, H_off = sector_hamiltonian(n, 4, diag, off)
    _, v = np.linalg.eigh(np.diag(hd) + H_off.toarray())
    psi, _, _ = sector_evolve(n, 4, diag, off, 100, 40)
    assert abs(np.vdot(v[:, 0], psi)) ** 2 > 0.99


def test_sector_dimension_reduction():
    """The sector is much smaller than the full space."""
    n, diag, off, gs, E0, fci = lih_hamiltonian(geometry=LIH)
    states = sector_states(n, 4)
    assert len(states) == 495            # C(12, 4)
    assert 2**n / len(states) > 8        # >= 8x reduction


def test_cc_pvdz_reduction_is_3_7_million_x():
    """LiH cc-pVDZ: C(38,4) = 73815 vs 2^38 (3.7e6x) — the computable
    sector where the full space was impossible."""
    from math import comb
    assert comb(38, 4) == 73815
    assert 2**38 / comb(38, 4) > 3.7e6


def test_pauli_action_int_phase():
    """P|z> phase: Y = i X Z, so Y on an occupied bit gives -i (the Z
    sign -1 times i), Z gives (-1)^bit."""
    t, ph = pauli_action_int(0b100, "YI", 3)   # |100>: qubit0=1
    assert t == 0b000
    assert abs(ph - (-1j)) < 1e-12             # (-1) * i
    t, ph = pauli_action_int(0b100, "ZI", 3)
    assert t == 0b100
    assert abs(ph - (-1.0)) < 1e-12
