"""Tests for the discrete-evolution scaling law and molecular
extension (examples/vqe_evolution_scaling.py): the Ising gap decays
polynomially (Delta ~ 3/n, n*Delta -> 3.0), the adiabatic time follows
T ~ const/Delta^2 (O(n^2), not exponential), and the H2 molecule
converges inside chemical accuracy with zero gradients.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from vqe_evolution_scaling import (  # noqa: E402
    adiabatic_time_req,
    boundary_correlation,
    h2_adiabatic,
    ising_gap,
    ising_gs_parity,
    pauli_matrix,
    sector_adiabatic,
    sector_alternating_init,
    sector_pure_evolution,
    H2_HAMILTONIAN,
)
from vqe_barren_prewarm import ising_ground_state  # noqa: E402
from vqe_discrete_evolution import diag_values  # noqa: E402
from vqe_barren_plateaus import ising_hamiltonian  # noqa: E402


def test_ising_gap_scales_as_three_over_n():
    """n*Delta -> 3.0 (polynomial gap, not exponential)."""
    for n in (6, 8, 10, 12):
        d = ising_gap(n)
        assert 2.0 < n * d < 4.0


def test_adiabatic_time_polynomial():
    """T_req * Delta^2 ~ const (T ~ 1/Delta^2 ~ O(n^2)), and the time
    grows far slower than exponential (2^n)."""
    times = []
    for n in (6, 8, 10, 12):
        d = ising_gap(n)
        T = adiabatic_time_req(n)
        assert T is not None
        times.append(T)
        assert T * d * d < 100        # ~const, 13-25 measured
    # growth from n=6 to n=12 is ~2x (O(n^2)), not 2^6 = 64x
    assert times[-1] / times[0] < 8


def test_h2_inside_chemical_accuracy():
    """H2 discrete adiabatic: energy error < 1.6e-3 Ha (chemical
    accuracy), zero gradients."""
    H = np.zeros((4, 4), dtype=complex)
    for c, pa in H2_HAMILTONIAN:
        H = H + c * pauli_matrix(pa)
    E0 = np.linalg.eigvalsh(H)[0].real
    psi = h2_adiabatic(200, 20)
    E = float(np.real(np.vdot(psi, H @ psi)))
    assert abs(E - E0) < 1.6e-3
    assert abs(np.vdot(psi, psi) - 1.0) < 1e-12


def test_h2_high_fidelity():
    H = np.zeros((4, 4), dtype=complex)
    for c, pa in H2_HAMILTONIAN:
        H = H + c * pauli_matrix(pa)
    _, gsv = np.linalg.eigh(H)
    psi = h2_adiabatic(1000, 100)
    assert abs(np.vdot(gsv[:, 0], psi)) ** 2 > 0.9


def test_gs_parity_alternates_with_n():
    """The Ising ground state sits in the Z2-even sector for even n and
    the Z2-odd sector for odd n (the spatial odd/even property)."""
    for n in (4, 6, 8):
        assert ising_gs_parity(n) == +1
    for n in (5, 7, 9):
        assert ising_gs_parity(n) == -1


def test_boundary_frustration_on_odd_n():
    """<Z0 Z_{n-1}> > 0 on odd chains (frustrated anti-ferro boundary),
    < 0 on even chains (matched) — the discrete spatial property."""
    for n in (4, 6, 8):
        assert boundary_correlation(n) < 0
    for n in (5, 7, 9):
        assert boundary_correlation(n) > 0


def test_symmetry_reduced_path_fixes_odd_n():
    """The plain |+>-based path is symmetry-forbidden on odd n (fid 0
    exactly); the symmetry-reduced (sector) path converges to ~1."""
    for n in (5, 7):
        par = ising_gs_parity(n)
        _, gs = ising_ground_state(n)
        psi = sector_adiabatic(n, par, 400, 40)
        assert abs(np.vdot(gs, psi)) ** 2 > 0.99


def test_even_n_plain_path_still_works():
    """Even n: the plain path (|+>, even sector) reaches the even-
    sector ground state."""
    for n in (4, 6):
        par = ising_gs_parity(n)
        _, gs = ising_ground_state(n)
        psi = sector_adiabatic(n, par, 400, 40)
        assert abs(np.vdot(gs, psi)) ** 2 > 0.99


def test_unified_sector_pure_path_odd_and_even():
    """The sector-pure Pauli-Trotter path (Sigma-ZZ -> Ising) works for
    BOTH odd and even n at scale — the fix of the symmetry-forbidden
    case without matrix projection."""
    for n in (5, 6, 7, 10):
        _, gs = ising_ground_state(n)
        C = diag_values(n, ising_hamiltonian(n))
        par = ising_gs_parity(n)
        init = sector_alternating_init(n, par)
        psi = sector_pure_evolution(n, 1000, 100, C, init)
        assert abs(np.vdot(gs, psi)) ** 2 > 0.90


def test_unified_path_scales_to_n14():
    """The unified path keeps fidelity > 0.9 at n=12 and n=14."""
    for n in (12, 14):
        _, gs = ising_ground_state(n)
        C = diag_values(n, ising_hamiltonian(n))
        par = ising_gs_parity(n)
        init = sector_alternating_init(n, par)
        psi = sector_pure_evolution(n, 1000, 100, C, init)
        assert abs(np.vdot(gs, psi)) ** 2 > 0.90


def test_unified_adiabatic_time_polynomial():
    """The unified-path T_req grows polynomially (3.2 at n=4 to 25.6
    at n=12 — ~n^2, far below 2^n), for odd and even n alike."""
    T4 = adiabatic_time_req(4)
    T12 = adiabatic_time_req(12)
    assert T4 is not None and T12 is not None
    assert T12 / T4 < 32          # ~8x for 3x the size (not 2^8)
    assert adiabatic_time_req(5) is not None   # odd n now works too
