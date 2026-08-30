"""Tests for the Grassmann-manifold SCF and the full geometrised
pipeline (M2/M3 of geoqc, examples/geoqc_scf.py): Hartree-Fock as a
fixed point on Gr(N,n), machine-verified equal to pyscf RHF; the
pipeline AO integrals -> Grassmann SCF -> MO -> spin-orbital ->
exterior N-sector ground state equals FCI.

Requires openfermion, openfermionpyscf and pyscf.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from geoqc.integrals import ao_integrals, spin_orbital_integrals  # noqa: E402
from geoqc.scf import fock_matrix, grassmann_scf, rhf_energy  # noqa: E402
from geoqc import exterior  # noqa: E402

pytest.importorskip("openfermion")
pytest.importorskip("openfermionpyscf")
pytest.importorskip("pyscf")

LIH = [["Li", [0, 0, 0]], ["H", [0, 0, 1.6]]]


def test_grassmann_scf_matches_pyscf_lih():
    """The Grassmann-manifold fixed point reproduces pyscf RHF
    (energy to 1e-10, density to 1e-5, electron count exact)."""
    from pyscf import gto, scf
    n, h, eri, S, nuc = ao_integrals(LIH, "sto-3g")
    E, P, C, grads, dists = grassmann_scf(h, eri, S, 2)
    mol = gto.M(atom=LIH, basis="sto-3g")
    mf = scf.RHF(mol)
    mf.kernel()
    assert abs(E + nuc - mf.e_tot) < 1e-10
    assert np.abs(P - mf.make_rdm1()).max() < 1e-5
    assert abs(np.trace(S @ P) - 4.0) < 1e-10
    assert grads[-1] < 1e-9


def test_grassmann_scf_gradient_decays():
    """The Grassmann gradient norm (the occupied-virtual block)
    decays to the fixed point [F, C] = 0 (F diagonal in the occupied
    subspace)."""
    n, h, eri, S, nuc = ao_integrals(LIH, "sto-3g")
    E, P, C, grads, dists = grassmann_scf(h, eri, S, 2)
    assert grads[0] > grads[-1]
    assert grads[-1] < 1e-9   # the fixed point (F C subset span(C))


def test_spin_orbital_integrals_are_physical():
    """The spin-orbital integrals (openfermion layout) give the FCI
    ground state via InteractionOperator (independent validation of
    the reverse-engineered layout)."""
    from pyscf import gto, scf
    import scipy.sparse.linalg as spla
    from openfermion import MolecularData, get_sparse_operator
    from openfermion.ops import InteractionOperator
    from openfermion.transforms import get_fermion_operator
    from openfermionpyscf import run_pyscf
    n, h, eri, S, nuc = ao_integrals(LIH, "sto-3g")
    mol = gto.M(atom=LIH, basis="sto-3g")
    mf = scf.RHF(mol)
    mf.kernel()
    C = mf.mo_coeff
    o = C.T @ h @ C
    t = np.einsum("ia,jb,kc,ld,ijkl->abcd", C, C, C, C, eri)
    o_s, t_s = spin_orbital_integrals(o, t)
    op = InteractionOperator(float(nuc), o_s, t_s)
    Hm = get_sparse_operator(get_fermion_operator(op))
    w, _ = spla.eigsh(Hm, k=1, which="SA")
    mref = run_pyscf(MolecularData(geometry=LIH, basis="sto-3g",
                                   multiplicity=1), run_fci=True)
    assert abs(w[0] - mref.fci_energy) < 1e-8


def test_full_pipeline_equals_fci():
    """AO integrals -> Grassmann SCF -> MO -> spin-orbital -> exterior
    N-sector ground state == FCI (the geometrised path end to end)."""
    from pyscf import gto, scf
    from scipy import sparse
    from scipy.linalg import sqrtm
    import scipy.sparse.linalg as spla
    from openfermion import MolecularData
    from openfermionpyscf import run_pyscf
    n, h, eri, S, nuc = ao_integrals(LIH, "sto-3g")
    E, P, C, _, _ = grassmann_scf(h, eri, S, 2)
    X = np.asarray(sqrtm(np.linalg.inv(S)).real)
    h_o = X.T @ h @ X
    eri_o = np.einsum("ia,jb,kc,ld,ijkl->abcd", X, X, X, X, eri)
    F = fock_matrix(h_o, eri_o, P)
    _, C_all = np.linalg.eigh(F)
    o = C_all.T @ h_o @ C_all
    t = np.einsum("ia,jb,kc,ld,ijkl->abcd", C_all, C_all, C_all, C_all,
                  eri_o)
    o_s, t_s = spin_orbital_integrals(o, t)
    hd, H_off = exterior.exterior_hamiltonian(12, 4, o_s, t_s, float(nuc))
    H = sparse.diags(hd) + H_off
    w, _ = spla.eigsh(H, k=1, which="SA")
    mref = run_pyscf(MolecularData(geometry=LIH, basis="sto-3g",
                                   multiplicity=1), run_fci=True)
    assert abs(w[0] - mref.fci_energy) < 1e-8
