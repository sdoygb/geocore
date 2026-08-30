"""Tests for the matrix-free exterior evolution (feature 49,
examples/vqe_matfree.py): H|v> by the exterior action as a
LinearOperator — no sector matrix built.  Machine-verified: the
matrix-free matvec equals the sparse-matrix matvec, eigsh with the
LinearOperator reproduces the sparse ground state, and the Krylov
(matrix-free) evolution equals the sparse exponential evolution.

Requires openfermion + openfermionpyscf.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from vqe_matfree import exterior_action, krylov_expm, matfree_evolve  # noqa: E402
from vqe_exterior_algebra import exterior_hamiltonian, integrals_from_openfermion  # noqa: E402

pytest.importorskip("openfermion")
pytest.importorskip("openfermionpyscf")

LIH = [["Li", [0, 0, 0]], ["H", [0, 0, 1.6]]]


@pytest.fixture(scope="module")
def systems():
    out = {}
    for name, geom, ne in [("LiH", LIH, 4)]:
        n, o, t, const, fci = integrals_from_openfermion(geom, "sto-3g",
                                                         run_fci=True)
        L = exterior_action(n, ne, o, t, const)
        hd, H_sp = exterior_hamiltonian(n, ne, o, t, const)
        out[name] = {"L": L, "hd": hd, "H": H_sp, "dim": L.shape[0]}
    return out


def test_matvec_equals_sparse(systems):
    """The matrix-free exterior action equals the sparse matvec on
    random vectors (machine precision)."""
    from scipy import sparse
    s = systems["LiH"]
    H = sparse.diags(s["hd"]) + s["H"]
    rng = np.random.default_rng(0)
    for _ in range(3):
        v = rng.standard_normal((s["dim"], 2)).view(complex)[:, 0]
        v /= np.linalg.norm(v)
        assert np.linalg.norm(s["L"] @ v - H @ v) < 1e-10


def test_eigsh_matches_sparse(systems):
    """eigsh with the LinearOperator == eigsh with the sparse matrix."""
    from scipy import sparse
    import scipy.sparse.linalg as spla
    s = systems["LiH"]
    H = sparse.diags(s["hd"]) + s["H"]
    w1, _ = spla.eigsh(s["L"], k=1, which="SA")
    w2, _ = spla.eigsh(H, k=1, which="SA")
    assert abs(w1[0] - w2[0]) < 1e-10


def test_krylov_expm_precision(systems):
    """The Krylov exponential (discrete descent) matches scipy's
    expm_multiply at m = 20 (machine precision)."""
    from scipy import sparse
    from scipy.sparse.linalg import expm_multiply
    s = systems["LiH"]
    H = sparse.diags(s["hd"]) + s["H"]
    rng = np.random.default_rng(1)
    v = rng.standard_normal((s["dim"], 2)).view(complex)[:, 0]
    v /= np.linalg.norm(v)
    ref = expm_multiply(-1j * 0.667 * H, v)
    got = krylov_expm(lambda x: H @ x, v, 0.667, m=20)
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-10


def test_matfree_evolution_matches_sparse(systems):
    """The full matrix-free adiabatic evolution (Krylov per step)
    equals the sparse expm_multiply evolution."""
    from scipy import sparse
    from vqe_path_geometry import adiabatic_path
    s = systems["LiH"]
    H = sparse.diags(s["hd"]) + s["H"]
    p, T = 30, 40
    psi_mf = matfree_evolve(s["L"], s["hd"], p, T)
    path = adiabatic_path(s["H"], s["hd"], p, T)
    assert abs(np.vdot(psi_mf, path[-1])) ** 2 > 1 - 1e-9
