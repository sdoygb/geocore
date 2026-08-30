"""Tests for the Schubert-cell structure of the N-sector states
(feature 50, examples/vqe_schubert.py): the sector basis is the
Grassmannian Gr(N,n) with its Schubert cell decomposition.  Verified:
states <-> box partitions bijectively, every excitation has even
Bruhat distance (S_z conservation -> bipartite lattice), and the
diagonal energy trends with the cell weight.

Requires openfermion + openfermionpyscf.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from vqe_schubert import (  # noqa: E402
    bruhat_le,
    excitation_bruhat_distance,
    partition_of_state,
    partition_weight,
    state_of_partition,
)
from vqe_exterior_algebra import exterior_hamiltonian, integrals_from_openfermion  # noqa: E402
from vqe_sector_reduction import sector_states  # noqa: E402

pytest.importorskip("openfermion")
pytest.importorskip("openfermionpyscf")

LIH = [["Li", [0, 0, 0]], ["H", [0, 0, 1.6]]]


@pytest.fixture(scope="module")
def lih():
    n, o, t, const, fci = integrals_from_openfermion(LIH, "sto-3g",
                                                     run_fci=True)
    return n, o, t, const


def test_bijection_states_partitions(lih):
    """Sector states <-> legal partitions in the N x (n-N) box,
    bijectively (the Schubert cell decomposition of Gr(N,n))."""
    n, o, t, const = lih
    N = 4
    states = sector_states(n, N)
    parts = [partition_of_state(z, n, N) for z in states]
    # legality: decreasing, non-negative, inside the box
    for p in parts:
        assert all(p[j] >= p[j + 1] for j in range(N - 1))
        assert p[-1] >= 0 and p[0] <= n - N
    assert len(set(parts)) == len(states)         # injective
    back = [state_of_partition(p, n) for p in parts]
    assert all(b == z for b, z in zip(back, states))  # inverse


def test_bruhat_order_is_partial_order(lih):
    """The Bruhat (dominance) order is reflexive and transitive on the
    partitions of the sector (transitivity sampled)."""
    rng = np.random.default_rng(0)
    n, o, t, const = lih
    N = 4
    states = sector_states(n, N)
    parts = [partition_of_state(z, n, N) for z in states]
    for p in parts:
        assert bruhat_le(p, p)                     # reflexive
    for _ in range(300):                           # sampled transitivity
        a, b, c = [parts[i] for i in rng.integers(0, len(parts), 3)]
        if bruhat_le(a, b) and bruhat_le(b, c):
            assert bruhat_le(a, c)


def test_excitations_have_even_bruhat_distance(lih):
    """S_z conservation + interleaved spin order -> every H matrix
    element connects states of equal |lambda| parity (bipartite)."""
    n, o, t, const = lih
    hd, H_off = exterior_hamiltonian(n, 4, o, t, const)
    states = sector_states(n, 4)
    coo = H_off.tocoo()
    for i, j, v in zip(coo.row, coo.col, coo.data):
        if i != j and abs(v) > 1e-9:
            d = excitation_bruhat_distance(states[j], states[i], n, 4)
            assert d % 2 == 0


def test_diagonal_energy_trends_with_cell_weight(lih):
    """The diagonal energy trends with the Schubert cell weight
    |lambda| (Spearman rho > 0.5, shell means increasing)."""
    n, o, t, const = lih
    hd, H_off = exterior_hamiltonian(n, 4, o, t, const)
    states = sector_states(n, 4)
    wts = np.array([partition_weight(partition_of_state(z, n, 4))
                    for z in states])
    e_diag = (hd + H_off.diagonal()).real
    rho = np.corrcoef(wts, e_diag)[0, 1]
    assert rho > 0.5
    shells = {}
    for w, e in zip(wts, e_diag):
        shells.setdefault(w, []).append(e)
    means = [np.mean(shells[w]) for w in sorted(shells)]
    assert means[-1] > means[0]
