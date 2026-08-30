"""Tests for the term-reduction layer of the N-sector build
(examples/vqe_term_reduction.py, feature 46): the combinatorial build
and the commuting-term merge both reproduce the naive per-state x
per-term build to machine precision, the flip-weight structure of a
molecular JW Hamiltonian is 2/4 (so every off term is N-conserving for
N >= 2), and spectral truncation's error is calibrated below chemical
accuracy.

Requires openfermion + openfermionpyscf.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from vqe_sector_reduction import sector_hamiltonian  # noqa: E402
from vqe_term_reduction import (  # noqa: E402
    group_by_flip_set,
    n_conserving_terms,
    sector_hamiltonian_fast,
    sector_hamiltonian_merged,
    term_budget,
)
from vqe_lih_evolution import lih_hamiltonian  # noqa: E402

pytest.importorskip("openfermion")
pytest.importorskip("openfermionpyscf")

LIH = [["Li", [0, 0, 0]], ["H", [0, 0, 1.6]]]


def _builds(off, n, N, diag, eps=0.0):
    hd1, H1 = sector_hamiltonian(n, N, diag, off)          # naive
    hd2, H2 = sector_hamiltonian_fast(n, N, diag, off, eps)
    hd3, H3 = sector_hamiltonian_merged(n, N, diag, off, eps)
    return hd1, H1, hd2, H2, hd3, H3


def test_combinatorial_build_matches_naive():
    """The combinatorial build (enumerate matrix elements per term)
    equals the naive per-state x per-term build to machine precision."""
    n, diag, off, gs, E0, fci = lih_hamiltonian(geometry=LIH)
    hd1, H1, hd2, H2, _, _ = _builds(off, n, 4, diag)
    assert np.allclose(hd1, hd2)
    assert np.allclose(H1.toarray(), H2.toarray(), atol=1e-12)


def test_commuting_merge_matches_naive():
    """Terms sharing the same flip set form a commuting family; the
    merged build equals the naive build to machine precision."""
    n, diag, off, gs, E0, fci = lih_hamiltonian(geometry=LIH)
    hd1, H1, _, _, hd3, H3 = _builds(off, n, 4, diag)
    assert np.allclose(hd1, hd3)
    assert np.allclose(H1.toarray(), H3.toarray(), atol=1e-12)


def test_merge_group_structure_is_real():
    """Merging actually groups terms: fewer flip-set groups than terms
    (LiH STO-3G: 552 terms -> 83 groups)."""
    n, diag, off, gs, E0, fci = lih_hamiltonian(geometry=LIH)
    groups = group_by_flip_set(n, 4, off)
    assert len(groups) < len(off)
    # every term's flip weight is 2 or 4 (1/2-body fermion JW), so all
    # off terms are N-conserving for N >= 2
    kept = n_conserving_terms(n, 4, off)
    assert len(kept) == len(off)
    for c, F, Zph, ny in kept:
        assert len(F) in (2, 4)


def test_budget_matches_built_size():
    """The a-priori element budget C(w,w/2)*C(n-w,N-w/2) per term is
    the per-term operation count; the built sparse matrix holds the
    UNIQUE elements (the commuting merge removes duplicate (row,col)
    entries), so built nnz <= budget, and the fast and merged builds
    agree on the unique set."""
    n, diag, off, gs, E0, fci = lih_hamiltonian(geometry=LIH)
    kept, nnz, _ = term_budget(n, 4, off)
    _, H2 = sector_hamiltonian_fast(n, 4, diag, off)
    _, H3 = sector_hamiltonian_merged(n, 4, diag, off)
    assert H2.nnz == H3.nnz          # same unique matrix elements
    assert H2.nnz <= nnz             # budget counts duplicates
    assert H2.nnz > 0


def test_truncation_error_within_chemical_accuracy():
    """eps = 1e-3 truncation shifts the sector ground state by less
    than 1.6e-3 Ha (1 kcal/mol) on LiH STO-3G — the calibration that
    justifies the cc-pVDZ truncation."""
    from scipy import sparse
    import scipy.sparse.linalg as spla
    n, diag, off, gs, E0, fci = lih_hamiltonian(geometry=LIH)
    hd_r, H_r = sector_hamiltonian_fast(n, 4, diag, off)
    hd_t, H_t = sector_hamiltonian_fast(n, 4, diag, off, 1e-3)
    Hr = sparse.diags(hd_r) + H_r
    Ht = sparse.diags(hd_t) + H_t
    wr = spla.eigsh(Hr, k=1, which="SA", return_eigenvectors=False)[0]
    wt = spla.eigsh(Ht, k=1, which="SA", return_eigenvectors=False)[0]
    assert abs(wt - wr) < 1.6e-3
