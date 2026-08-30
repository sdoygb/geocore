"""Migration-equivalence tests: the geoqc package must reproduce the
example-script implementations (features 45-50) to machine precision,
so the refactor into the package is verified rather than assumed.

Requires openfermion + openfermionpyscf (integrals from the standard
library, honestly labelled as physics input).
"""

import sys
import os

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "examples")))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

import geoqc  # noqa: E402
from geoqc import sector as gs  # noqa: E402
from geoqc import exterior as ge  # noqa: E402
from geoqc import schubert as gsc  # noqa: E402
from geoqc import manifold as gm  # noqa: E402
from geoqc import descent as gd  # noqa: E402
from geoqc import integrals as gi  # noqa: E402

from vqe_sector_reduction import (  # noqa: E402
    sector_hamiltonian as es_sector_hamiltonian,
    sector_states as es_sector_states,
)
from vqe_term_reduction import (  # noqa: E402
    sector_hamiltonian_fast as et_fast,
    sector_hamiltonian_merged as et_merged,
    term_budget as et_budget,
    sector_diagonal as et_diagonal,
)
from vqe_exterior_algebra import (  # noqa: E402
    exterior_hamiltonian as ee_hamiltonian,
    integrals_from_openfermion as ee_integrals,
)
from vqe_matfree import exterior_action as em_action, krylov_expm as em_krylov  # noqa: E402
from vqe_path_geometry import (  # noqa: E402
    fs_distance as ep_fs,
    path_geometry as ep_path_geometry,
)
from vqe_schubert import partition_of_state as es_partition  # noqa: E402

pytest.importorskip("openfermion")
pytest.importorskip("openfermionpyscf")

LIH = [["Li", [0, 0, 0]], ["H", [0, 0, 1.6]]]
H2O = [["O", [0, 0, 0]], ["H", [0.757, 0.586, 0]], ["H", [-0.757, 0.586, 0]]]


def test_integrals_match():
    """The integral interface reproduces the example implementation
    (two independent SCF/FCI runs agree to 1e-8)."""
    a = gi.integrals_from_openfermion(LIH, "sto-3g", run_fci=True)
    b = ee_integrals(LIH, "sto-3g", run_fci=True)
    for x, y in zip(a, b):
        if isinstance(x, np.ndarray):
            assert np.allclose(x, y)
        elif x is None:
            assert y is None
        else:
            assert abs(x - y) < 1e-8


def test_sector_states_and_builds_match():
    """Sector enumeration, naive/combinatorial/merged builds, budget,
    and diagonal all reproduce the example implementations."""
    n, o, t, const, _ = gi.integrals_from_openfermion(LIH, "sto-3g",
                                                      run_fci=True)
    nn, cnst, z_terms, off = gi.molecule_terms(LIH, "sto-3g")
    assert nn == n
    assert gs.sector_states(n, 4) == es_sector_states(n, 4)
    # naive vs example naive
    hd1, H1 = gs.sector_hamiltonian(n, 4, np.full(2**n, 0, dtype=complex), off)
    hd1b, H1b = es_sector_hamiltonian(n, 4, np.full(2**n, 0, dtype=complex), off)
    assert np.allclose(hd1, hd1b) and np.allclose(H1.toarray(), H1b.toarray())
    # fast vs example fast (needs a real diagonal)
    hd_d = gs.sector_diagonal(n, 4, cnst, z_terms)
    hd_d2 = et_diagonal(n, 4, cnst, z_terms)
    assert np.allclose(hd_d, hd_d2)
    hd2, H2 = gs.sector_hamiltonian_fast(n, 4, hd_d, off)
    hd2b, H2b = et_fast(n, 4, hd_d2, off)
    assert np.allclose(H2.toarray(), H2b.toarray(), atol=1e-12)
    # merged
    hd3, H3 = gs.sector_hamiltonian_merged(n, 4, hd_d, off)
    hd3b, H3b = et_merged(n, 4, hd_d2, off)
    assert np.allclose(H3.toarray(), H3b.toarray(), atol=1e-12)
    # budget
    assert gs.term_budget(n, 4, off) == et_budget(n, 4, off)


def test_exterior_hamiltonian_matches():
    """The exterior sector Hamiltonian reproduces the example build."""
    for geom, ne in [(LIH, 4), (H2O, 10)]:
        n, o, t, const, _ = gi.integrals_from_openfermion(geom, "sto-3g",
                                                          run_fci=True)
        hd1, H1 = ge.exterior_hamiltonian(n, ne, o, t, const)
        hd2, H2 = ee_hamiltonian(n, ne, o, t, const)
        assert np.allclose(hd1, hd2)
        assert np.allclose(H1.toarray(), H2.toarray(), atol=1e-12)


def test_exterior_action_matches():
    """The matrix-free exterior action reproduces the example matvec."""
    n, o, t, const, _ = gi.integrals_from_openfermion(LIH, "sto-3g",
                                                      run_fci=True)
    L1 = ge.exterior_action(n, 4, o, t, const)
    L2 = em_action(n, 4, o, t, const)
    rng = np.random.default_rng(0)
    v = rng.standard_normal((L1.shape[0], 2)).view(complex)[:, 0]
    v /= np.linalg.norm(v)
    assert np.linalg.norm(L1 @ v - L2 @ v) < 1e-12


def test_krylov_matches():
    """The Krylov exponential reproduces the example implementation."""
    n, o, t, const, _ = gi.integrals_from_openfermion(LIH, "sto-3g",
                                                      run_fci=True)
    from scipy import sparse
    hd, H_off = ge.exterior_hamiltonian(n, 4, o, t, const)
    H = sparse.diags(hd) + H_off
    rng = np.random.default_rng(1)
    v = rng.standard_normal((495, 2)).view(complex)[:, 0]
    v /= np.linalg.norm(v)
    r1 = gd.krylov_expm(lambda x: H @ x, v, 0.5, m=20)
    r2 = em_krylov(lambda x: H @ x, v, 0.5, m=20)
    assert np.linalg.norm(r1 - r2) < 1e-12


def test_schubert_and_manifold_match():
    """The Schubert bijection and the Fubini-Study geometry reproduce
    the example implementations."""
    n, o, t, const, _ = gi.integrals_from_openfermion(LIH, "sto-3g",
                                                      run_fci=True)
    states = gs.sector_states(n, 4)
    for z in states[:50]:
        assert (gsc.partition_of_state(z, n, 4)
                == es_partition(z, n, 4))
    rng = np.random.default_rng(2)
    a = rng.standard_normal((495, 2)).view(complex)[:, 0]
    b = rng.standard_normal((495, 2)).view(complex)[:, 0]
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    assert abs(gm.fs_distance(a, b) - ep_fs(a, b)) < 1e-12
