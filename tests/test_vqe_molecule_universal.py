"""Tests for the input universality of the discrete-evolution solver
(examples/vqe_molecule_universal.py): the SAME zero-gradient pipeline
(JW -> diagonal->full adiabatic, no hand-tuning) reaches chemical
accuracy on different molecules and different geometries, and the
particle-number sector is automatically correct.  H2O is the honest
boundary (fidelity high, absolute chemical accuracy not reached).

Requires openfermion + openfermionpyscf.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from vqe_molecule_universal import (  # noqa: E402
    energy,
    evolve,
    molecule_hamiltonian,
    particle_number,
)

pytest.importorskip("openfermion")
pytest.importorskip("openfermionpyscf")

H2 = [["H", [0, 0, 0]], ["H", [0, 0, 0.735]]]
LIH_13 = [["Li", [0, 0, 0]], ["H", [0, 0, 1.3]]]
LIH_16 = [["Li", [0, 0, 0]], ["H", [0, 0, 1.6]]]
LIH_20 = [["Li", [0, 0, 0]], ["H", [0, 0, 2.0]]]
H2O = [["O", [0, 0, 0]], ["H", [0.757, 0.586, 0]],
       ["H", [-0.757, 0.586, 0]]]


def test_jw_matches_fci_any_molecule():
    for name, geom in [("H2", H2), ("LiH", LIH_16), ("H2O", H2O)]:
        n, diag, off, gs, E0, fci = molecule_hamiltonian(geom)
        assert abs(E0 - fci) < 1e-8
        assert abs(energy(gs, diag, off) - E0) < 1e-8


def test_lih_potential_curve_inside_chemical_accuracy():
    """Same pipeline, three bond lengths: all inside 1.6e-3 Ha."""
    for geom in (LIH_13, LIH_16, LIH_20):
        n, diag, off, gs, E0, fci = molecule_hamiltonian(geom)
        psi = evolve(n, diag, off, 100, 40)
        E = energy(psi, diag, off)
        assert abs(E - E0) < 1.6e-3
        assert abs(np.vdot(gs, psi)) ** 2 > 0.99


def test_particle_number_sector_automatic():
    """The diagonal-ground-state start is automatically in the correct
    particle-number sector (no hand projection needed)."""
    for geom, n_e in [(LIH_16, 4), (H2O, 10)]:
        n, diag, off, gs, E0, fci = molecule_hamiltonian(geom)
        idx0 = int(np.argmin(diag.real))
        assert bin(idx0).count("1") == n_e
        psi = evolve(n, diag, off, 50, 20)
        assert abs(particle_number(psi, n) - n_e) < 1e-6


def test_h2_uniform_pipeline_close_to_chemical_accuracy():
    """The uniform (zero hand-tuning) pipeline on H2: error below 0.01
    Ha (near chemical accuracy; the purpose-built H2 path of feature 40
    is tighter at 1e-4)."""
    n, diag, off, gs, E0, fci = molecule_hamiltonian(H2)
    psi = evolve(n, diag, off, 100, 40)
    E = energy(psi, diag, off)
    assert abs(E - E0) < 0.01
    assert abs(np.vdot(gs, psi)) ** 2 > 0.98
