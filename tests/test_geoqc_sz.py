"""Tests for the S_z (spin) sector projection (examples/geoqc_sz.py,
geoqc.exterior.exterior_hamiltonian_sz): the S_z = 0 sector contains
the singlet ground state, the S_z-sector build equals the full-N
sector to machine precision, and the sector dimensions follow
C(n/2, n_alpha) * C(n/2, n_beta).

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

LIH = [["Li", [0, 0, 0]], ["H", [0, 0, 1.6]]]


def _build(geom, basis, N, sz=None, eps=1e-4):
    n, h, eri, S, nuc = ao_integrals(geom, basis)
    E, P, C, C_o, _, _ = grassmann_scf(h, eri, S, N // 2)
    X = np.asarray(sqrtm(np.linalg.inv(S)).real)
    h_o = X.T @ h @ X
    eri_o = mo_transform(X, eri)
    F = fock_matrix(h_o, eri_o, 2.0 * C_o @ C_o.T)
    _, C_all = np.linalg.eigh(F)
    o = C_all.T @ h_o @ C_all
    t = mo_transform(C_all, eri_o)
    o_s, t_s = spin_orbital_integrals(o, t)
    if sz is None:
        return exterior.exterior_hamiltonian(2 * n, N, o_s, t_s,
                                             float(nuc), eps)
    return exterior.exterior_hamiltonian_sz(2 * n, N, sz, o_s, t_s,
                                            float(nuc), eps)


def test_sz_sector_dimension():
    """The S_z = 0 sector dimension is C(n/2, N/2)^2."""
    n, h, eri, S, nuc = ao_integrals(LIH, "sto-3g")
    hd, H = _build(LIH, "sto-3g", 4, sz=0)
    assert H.shape[0] == 225            # C(6,2)^2
    assert len(exterior.sector_states_sz(12, 4, 0)) == 225


def test_sz_sector_equals_full_lih():
    """The S_z = 0 sector ground state equals the full-N sector GS."""
    hd1, H1 = _build(LIH, "sto-3g", 4, sz=None)
    w1, _ = spla.eigsh(sparse.diags(hd1) + H1, k=1, which="SA")
    hd2, H2 = _build(LIH, "sto-3g", 4, sz=0)
    w2, _ = spla.eigsh(sparse.diags(hd2) + H2, k=1, which="SA")
    assert abs(w1[0] - w2[0]) < 1e-8


def test_sz_sector_equals_full_n2():
    """The S_z = 0 sector ground state equals the full-N sector GS for
    N2 (a larger N=14 sector)."""
    N2 = [["N", [0, 0, 0]], ["N", [0, 0, 1.1]]]
    hd1, H1 = _build(N2, "sto-3g", 14, sz=None, eps=1e-5)
    w1, _ = spla.eigsh(sparse.diags(hd1) + H1, k=1, which="SA")
    hd2, H2 = _build(N2, "sto-3g", 14, sz=0, eps=1e-5)
    w2, _ = spla.eigsh(sparse.diags(hd2) + H2, k=1, which="SA")
    assert H2.shape[0] == 14400         # C(10,7)^2
    assert abs(w1[0] - w2[0]) < 1e-8
