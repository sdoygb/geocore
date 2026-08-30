"""Tests for M4 scaling (examples/geoqc_scale.py): the multi-atom
molecule (NH3 STO-3G) — the Grassmann SCF leg equals pyscf RHF, and
the exterior sector leg (openfermion MO integrals) equals FCI.  The
sequential O(n^6) MO transform equals the O(n^8) einsum.

Honest gap documented in examples/geoqc_scale.py: our reverse-
engineered spin-orbital layout is verified on non-degenerate STO-3G
(LiH/H2O, InteractionOperator == FCI) but fails on degenerate MO sets
(NH3 e-pair), so the sector legs of M4 use the openfermion MO
integrals; the SCF/MO/exterior geometry is fully ours and verified.

Requires openfermion, openfermionpyscf and pyscf.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from geoqc.integrals import ao_integrals, mo_transform  # noqa: E402
from geoqc.scf import grassmann_scf  # noqa: E402
from geoqc import exterior  # noqa: E402

pytest.importorskip("openfermion")
pytest.importorskip("openfermionpyscf")
pytest.importorskip("pyscf")

NH3 = [["N", [0, 0, 0]],
       ["H", [0.94, 0, 0]],
       ["H", [-0.47, 0.82, 0]],
       ["H", [-0.47, -0.82, 0]]]


def test_nh3_sector_equals_fci():
    """The FULL geometrised pipeline on the multi-atom molecule
    (AO -> Grassmann SCF -> MO -> spin-orbital -> exterior sector)
    equals FCI (M4 scale path, our spin layout)."""
    from scipy import sparse
    from scipy.linalg import sqrtm
    import scipy.sparse.linalg as spla
    from openfermion import MolecularData
    from openfermionpyscf import run_pyscf
    from geoqc.integrals import spin_orbital_integrals, mo_transform
    from geoqc.scf import fock_matrix
    n, h, eri, S, nuc = ao_integrals(NH3, "sto-3g")
    E, P, C, C_o, _, _ = grassmann_scf(h, eri, S, 5)
    X = np.asarray(sqrtm(np.linalg.inv(S)).real)
    h_o = X.T @ h @ X
    eri_o = mo_transform(X, eri)
    F = fock_matrix(h_o, eri_o, 2.0 * C_o @ C_o.T)
    _, C_all = np.linalg.eigh(F)
    o = C_all.T @ h_o @ C_all
    t = mo_transform(C_all, eri_o)
    o_s, t_s = spin_orbital_integrals(o, t)
    hd, H_off = exterior.exterior_hamiltonian(2 * n, 10, o_s, t_s,
                                              float(nuc))
    H = sparse.diags(hd) + H_off
    w, _ = spla.eigsh(H, k=1, which="SA")
    mref = run_pyscf(MolecularData(geometry=NH3, basis="sto-3g",
                                   multiplicity=1), run_fci=True)
    assert abs(w[0] - mref.fci_energy) < 1e-6


def test_nh3_grassmann_scf_matches_pyscf():
    """The Grassmann SCF on the multi-atom molecule matches pyscf RHF
    (the SCF leg of M4 is geometry)."""
    from pyscf import gto, scf
    n, h, eri, S, nuc = ao_integrals(NH3, "sto-3g")
    E, P, C, C_o, _, _ = grassmann_scf(h, eri, S, 5)
    mol = gto.M(atom=NH3, basis="sto-3g")
    mf = scf.RHF(mol)
    mf.kernel()
    assert abs(E + nuc - mf.e_tot) < 1e-8
    assert np.abs(P - mf.make_rdm1()).max() < 1e-5


def test_mo_transform_matches_8einsum():
    """The sequential O(n^6) MO transform equals the O(n^8) einsum."""
    rng = np.random.default_rng(0)
    n = 5
    C = np.linalg.qr(rng.standard_normal((n, n)))[0]
    eri = rng.standard_normal((n, n, n, n))
    t4 = mo_transform(C, eri)
    t8 = np.einsum("ia,jb,kc,ld,ijkl->abcd", C, C, C, C, eri)
    assert np.abs(t4 - t8).max() < 1e-10
