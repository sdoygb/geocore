"""Tests for the geometry of the adiabatic path (feature 48,
examples/vqe_path_geometry.py): the Fubini-Study metric on the
exterior-algebra N-sector state space.  Machine-verified: the path
length obeys the geodesic inequality, the final fidelity equals
cos^2 of the Fubini-Study distance, the instantaneous-GS path length
is T-independent, and the path-manifold deviation (the geometric
measure of adiabatic quality) decreases monotonically with T.

Requires openfermion + openfermionpyscf.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from vqe_path_geometry import (  # noqa: E402
    adiabatic_path,
    exterior_sector,
    fs_distance,
    instantaneous_path,
    manifold_deviation,
    path_geometry,
)
from vqe_exterior_algebra import integrals_from_openfermion  # noqa: E402

pytest.importorskip("openfermion")
pytest.importorskip("openfermionpyscf")

LIH = [["Li", [0, 0, 0]], ["H", [0, 0, 1.6]]]


@pytest.fixture(scope="module")
def sector():
    n, o, t, const, fci = integrals_from_openfermion(LIH, "sto-3g",
                                                     run_fci=True)
    hd, H_off, H = exterior_sector(n, 4, o, t, const)
    import scipy.sparse.linalg as spla
    w0, v0 = spla.eigsh(H, k=1, which="SA")
    return {"hd": hd, "H_off": H_off, "GS": v0[:, 0], "E0": w0[0]}


def test_geodesic_inequality(sector):
    """The path length L >= d_FS(psi0, psif) (triangle inequality on
    the polygonal arc)."""
    path = adiabatic_path(sector["H_off"], sector["hd"], 200, 40)
    L, _, _, d, _ = path_geometry(path)
    assert L >= d - 1e-12


def test_fidelity_equals_cos2_fs_distance(sector):
    """fidelity(psif, GS) == cos^2(d_FS(psif, GS)) exactly."""
    path = adiabatic_path(sector["H_off"], sector["hd"], 200, 40)
    fid = abs(np.vdot(path[-1], sector["GS"])) ** 2
    d = fs_distance(path[-1], sector["GS"])
    assert abs(fid - np.cos(d) ** 2) < 1e-12


def test_instantaneous_path_length_is_t_independent(sector):
    """The instantaneous-GS curve length is a property of the path,
    not of T (and converges with resolution)."""
    L1, _, _, _, _ = path_geometry(instantaneous_path(
        sector["H_off"], sector["hd"], 100))
    L2, _, _, _, _ = path_geometry(instantaneous_path(
        sector["H_off"], sector["hd"], 200))
    assert abs(L1 - L2) < 1e-3


def test_manifold_deviation_decreases_with_T(sector):
    """The path-manifold deviation d_FS(psi_k, GS(s_k)) — the geometric
    adiabatic quality — decreases monotonically with the adiabatic
    time T."""
    maxs = []
    for T in (10, 20, 40):
        path = adiabatic_path(sector["H_off"], sector["hd"], 200, T)
        dev = manifold_deviation(path, sector["H_off"], sector["hd"], 200)
        maxs.append(dev.max())
    assert maxs[0] > maxs[1] > maxs[2]


def test_excess_is_positive(sector):
    """The evolution path is longer than the instantaneous-GS curve
    (the non-adiabatic geometric cost is positive)."""
    path = adiabatic_path(sector["H_off"], sector["hd"], 200, 40)
    L, _, _, _, _ = path_geometry(path)
    Li, _, _, _, _ = path_geometry(instantaneous_path(
        sector["H_off"], sector["hd"], 200))
    assert L > Li
