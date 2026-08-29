"""Tests for the LiH molecule with the discrete-evolution solver
(examples/vqe_lih_evolution.py): the openfermion JW Hamiltonian
reproduces FCI to machine precision, the diagonal-part ground state is
the HF state, and the zero-gradient discrete adiabatic evolution
converges inside chemical accuracy (1.6e-3 Ha) at 12 qubits.

Note: requires openfermion + openfermionpyscf (pip install), and the
LiH FCI/SCF data generation takes a few seconds.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from vqe_lih_evolution import (  # noqa: E402
    _energy,
    evolve_lih,
    lih_hamiltonian,
)

pytest.importorskip("openfermion")
pytest.importorskip("openfermionpyscf")


@pytest.fixture(scope="module")
def lih():
    return lih_hamiltonian()


def test_jw_reproduces_fci(lih):
    """The openfermion JW ground state matches FCI to machine
    precision, and the reconstructed energy is exact."""
    n, diag, off, gs, E0, fci = lih
    assert n == 12
    assert abs(E0 - fci) < 1e-8
    assert abs(_energy(gs, diag, off) - E0) < 1e-8


def test_diagonal_ground_state_is_hf(lih):
    """The H_diag ground state (the evolution start) is the HF state:
    its energy equals the SCF electronic energy."""
    n, diag, off, gs, E0, fci = lih
    idx0 = int(np.argmin(diag.real))
    assert bin(idx0).count("1") == 4          # 4 electrons (LiH)
    e_hf = diag.real.min()
    assert abs(e_hf - (-7.8619)) < 0.01       # HF energy


def test_lih_converges_inside_chemical_accuracy(lih):
    """Zero-gradient discrete evolution reaches chemical accuracy
    (1.6e-3 Ha) at 12 qubits."""
    n, diag, off, gs, E0, fci = lih
    psi = evolve_lih(n, diag, off, 100, 40)
    fid = abs(np.vdot(gs, psi)) ** 2
    E = _energy(psi, diag, off)
    assert fid > 0.99
    assert abs(E - E0) < 1.6e-3


def test_lih_more_steps_more_accurate(lih):
    n, diag, off, gs, E0, fci = lih
    e1 = _energy(evolve_lih(n, diag, off, 50, 20), diag, off)
    e2 = _energy(evolve_lih(n, diag, off, 200, 80), diag, off)
    assert abs(e2 - E0) < abs(e1 - E0)
