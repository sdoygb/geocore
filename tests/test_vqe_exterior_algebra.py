"""Tests for the exterior-algebra (Clifford) sector construction
(examples/vqe_exterior_algebra.py, feature 47): the N-electron sector
is the degree-N piece of the exterior algebra Lambda(C^n), creation is
the wedge product and annihilation the contraction, and the fermion
signs ARE the exterior grading.  The exterior build must equal the
JW-Pauli sector build to machine precision, and its ground state must
equal the full-space FCI energy.

Requires openfermion + openfermionpyscf.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from vqe_exterior_algebra import (  # noqa: E402
    exterior_hamiltonian,
    exterior_sign,
    integrals_from_openfermion,
)
from vqe_sector_reduction import sector_hamiltonian, sector_states  # noqa: E402
from vqe_lih_evolution import lih_hamiltonian  # noqa: E402

pytest.importorskip("openfermion")
pytest.importorskip("openfermionpyscf")

LIH = [["Li", [0, 0, 0]], ["H", [0, 0, 1.6]]]


def test_exterior_sign_is_grading():
    """exterior_sign(z, p) = (-1)^{# occupied orbitals below p}: the
    wedge/contraction sign is the exterior-algebra grading."""
    # z = |0110> on 4 orbitals (orbitals 1,2 occupied), big-endian
    n = 4
    z = 0b0110
    assert exterior_sign(z, 0, n) == 1.0     # nothing below 0
    assert exterior_sign(z, 1, n) == 1.0     # nothing occupied < 1
    assert exterior_sign(z, 2, n) == -1.0    # orbital 1 occupied
    assert exterior_sign(z, 3, n) == 1.0     # two occupied below -> +1
    assert exterior_sign(z, 4, n) == 1.0    # two occupied below -> +1


def test_exterior_matches_jw_pauli_lih():
    """Exterior build == JW-Pauli sector build (machine precision)."""
    from scipy import sparse
    n, o, t, const, fci = integrals_from_openfermion(LIH, "sto-3g",
                                                     run_fci=True)
    hd1, H1 = exterior_hamiltonian(n, 4, o, t, const)
    M1 = (sparse.diags(hd1) + H1).tocsr()
    M1.eliminate_zeros()
    n2, diag, off, gs, E0, fci2 = lih_hamiltonian(geometry=LIH)
    hd2, H2 = sector_hamiltonian(n2, 4, diag, off)
    M2 = (sparse.diags(hd2) + H2).tocsr()
    M2.eliminate_zeros()
    assert M1.nnz == M2.nnz
    assert np.allclose(M1.toarray(), M2.toarray(), atol=1e-10)


def test_exterior_sector_gs_equals_fci():
    """The exterior N-sector ground state equals the full-space FCI
    energy (the sector is exact)."""
    n, o, t, const, fci = integrals_from_openfermion(LIH, "sto-3g",
                                                     run_fci=True)
    hd, H_off = exterior_hamiltonian(n, 4, o, t, const)
    ev = np.linalg.eigvalsh(np.diag(hd) + H_off.toarray())
    n2, diag, off, gs, E0, fci2 = lih_hamiltonian(geometry=LIH)
    assert abs(ev[0] - E0) < 1e-8


def test_overlap_terms_are_kept():
    """Fermionic terms with {p,q} ∩ {r,s} non-empty (an orbital
    annihilated then re-created) contribute; excluding them breaks the
    equality with the JW-Pauli build (the diagonal comes out wrong)."""
    from scipy import sparse
    n, o, t, const, fci = integrals_from_openfermion(LIH, "sto-3g",
                                                     run_fci=True)
    hd, H = exterior_hamiltonian(n, 4, o, t, const)
    n2, diag, off, gs, E0, fci2 = lih_hamiltonian(geometry=LIH)
    hd2, H2 = sector_hamiltonian(n2, 4, diag, off)
    full_ext = hd + H.diagonal()
    full_pauli = hd2 + H2.diagonal()
    # the two-body diagonal (a+_p a+_q a_q a_p) lives in H's diagonal
    # for the exterior build and in hd for the Pauli build — both must
    # agree on the full diagonal
    assert np.allclose(full_ext, full_pauli, atol=1e-10)


def test_exterior_keeps_particle_number():
    """The exterior action never leaves the N sector (wedge/contraction
    move exactly one or two particles)."""
    n, o, t, const, fci = integrals_from_openfermion(LIH, "sto-3g",
                                                     run_fci=True)
    hd, H_off = exterior_hamiltonian(n, 4, o, t, const)
    states = sector_states(n, 4)
    assert hd.size == len(states) == 495
    coo = H_off.tocoo()
    assert coo.row.max() < 495 and coo.col.max() < 495
