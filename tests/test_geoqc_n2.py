"""Tests for the N2 bond-dissociation benchmark (examples/geoqc_n2.py):
the textbook multireference problem solved by the geometrised
pipeline.  Verified: the exterior-sector (FCI-level) ground state
equals openfermion FCI at equilibrium and in the dissociation region,
and CCSD — the single-reference reference — fails badly at large R
(the point of the benchmark).

Requires openfermion, openfermionpyscf and pyscf.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from geoqc.integrals import ao_integrals, mo_transform, spin_orbital_integrals  # noqa: E402
from geoqc.scf import grassmann_scf, fock_matrix  # noqa: E402
from geoqc import exterior  # noqa: E402
from scipy.linalg import sqrtm  # noqa: E402
from scipy import sparse  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402

pytest.importorskip("openfermion")
pytest.importorskip("openfermionpyscf")
pytest.importorskip("pyscf")


def _fci_at(R, eps=1e-4):
    """FCI-level N2 energy at R via the geometrised pipeline."""
    geom = [["N", [0, 0, 0]], ["N", [0, 0, R]]]
    n, h, eri, S, nuc = ao_integrals(geom, "sto-3g")
    E, P, C, C_o, _, _ = grassmann_scf(h, eri, S, 7)
    X = np.asarray(sqrtm(np.linalg.inv(S)).real)
    h_o = X.T @ h @ X
    eri_o = mo_transform(X, eri)
    F = fock_matrix(h_o, eri_o, 2.0 * C_o @ C_o.T)
    _, C_all = np.linalg.eigh(F)
    o = C_all.T @ h_o @ C_all
    t = mo_transform(C_all, eri_o)
    o_s, t_s = spin_orbital_integrals(o, t)
    hd, H_off = exterior.exterior_hamiltonian(20, 14, o_s, t_s,
                                              float(nuc), eps)
    H = sparse.diags(hd) + H_off
    w, _ = spla.eigsh(H, k=1, which="SA")
    return w[0]


def test_n2_fci_equals_reference_equilibrium():
    """At equilibrium (R=1.1 A) the exterior-sector ground state
    equals openfermion FCI."""
    from openfermion import MolecularData
    from openfermionpyscf import run_pyscf
    R = 1.1
    m = run_pyscf(MolecularData(geometry=[["N", [0, 0, 0]],
                                          ["N", [0, 0, R]]],
                                basis="sto-3g", multiplicity=1),
                  run_fci=True)
    assert abs(_fci_at(R) - m.fci_energy) < 1e-6


def test_n2_fci_equals_reference_dissociation():
    """In the dissociation region (R=2.6 A) the exterior-sector ground
    state still equals openfermion FCI (~6e-5, solver numerics on the
    near-degenerate state)."""
    from openfermion import MolecularData
    from openfermionpyscf import run_pyscf
    R = 2.6
    m = run_pyscf(MolecularData(geometry=[["N", [0, 0, 0]],
                                          ["N", [0, 0, R]]],
                                basis="sto-3g", multiplicity=1),
                  run_fci=True)
    assert abs(_fci_at(R) - m.fci_energy) < 1e-4


def test_n2_ccsd_fails_in_dissociation():
    """The textbook point: CCSD fails at large R (single reference)
    while FCI is exact — |CCSD - FCI| grows to ~0.4 Ha at R=2.6."""
    from openfermion import MolecularData
    from openfermionpyscf import run_pyscf
    for R, lo in [(1.1, 0.05), (2.6, 0.3)]:
        m = run_pyscf(MolecularData(geometry=[["N", [0, 0, 0]],
                                              ["N", [0, 0, R]]],
                                    basis="sto-3g", multiplicity=1),
                      run_fci=True, run_ccsd=True)
        d = abs(m.ccsd_energy - m.fci_energy)
        if R == 1.1:
            assert d < lo       # fine near equilibrium
        else:
            assert d > lo       # fails in the dissociation region
