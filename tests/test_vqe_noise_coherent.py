"""Tests for the coherent (unitary) rotation noise geometry
(examples/vqe_noise_coherent.py): coherent noise keeps states pure, the
energy track has the exact closed form A cos^2 + B sin^2 + C sin, the
FS-QFI is preserved exactly (zero metric contraction — the fourth
fingerprint), the natural gradient is NOT immune, and linear ZNE in the
eps = sin^2(th/2) space is exact when C = 0 (rotation axis commuting
at the state) and has a residual otherwise.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from vqe_noise_coherent import (  # noqa: E402
    deriv_states,
    energy_track,
    fs_qfi,
    pauli_mat,
    purity_max_eig,
    rot_U,
    zne_eps_extrap,
)
from vqe_barren_plateaus import hea_gates, _base_state  # noqa: E402
from vqe_noise_geometry import ising_matrix  # noqa: E402
from geocore.clifford import rotation_action_closed_form  # noqa: E402


def test_coherent_noise_keeps_state_pure():
    """Unitary rotation noise: rho stays rank-1 to machine precision."""
    n = 2
    gates = hea_gates(n, 1)
    base = _base_state(n)
    rng = np.random.default_rng(0)
    th0 = rng.uniform(-np.pi, np.pi, len(gates))
    psi = base.copy()
    for axis, idx in gates:
        psi = rotation_action_closed_form(axis, th0[idx], psi)
    for ax in ("XI", "IZ", "ZZ"):
        P = pauli_mat(ax, n)
        for th in (0.1, 0.9):
            assert abs(purity_max_eig(rot_U(P, th) @ psi) - 1.0) < 1e-12


def test_energy_track_closed_form():
    """E(th) = A cos^2(th/2) + B sin^2(th/2) + C sin(th) is exact."""
    n = 2
    gates = hea_gates(n, 1)
    base = _base_state(n)
    rng = np.random.default_rng(0)
    th0 = rng.uniform(-np.pi, np.pi, len(gates))
    psi = base.copy()
    for axis, idx in gates:
        psi = rotation_action_closed_form(axis, th0[idx], psi)
    H = ising_matrix(n)
    ths = np.array([0.05, 0.2, 0.5, 0.9, 1.3])
    for ax in ("XI", "IZ", "ZZ"):
        P = pauli_mat(ax, n)
        _, _, _, worst = energy_track(psi, H, P, ths)
        assert worst < 1e-10


def test_fs_qfi_preserved_by_coherent_noise():
    """A fixed unitary preserves the FS-QFI exactly (zero metric
    contraction) — the fourth fingerprint."""
    n = 2
    gates = hea_gates(n, 1)
    base = _base_state(n)
    rng = np.random.default_rng(0)
    th0 = rng.uniform(-np.pi, np.pi, len(gates))
    psi, D = deriv_states(th0, gates, base)
    F0 = fs_qfi(D, psi)
    for ax in ("XI", "IZ", "ZZ"):
        P = pauli_mat(ax, n)
        for th in (0.2, 0.7):
            Un = rot_U(P, th)
            F1 = fs_qfi([Un @ dd for dd in D], Un @ psi)
            assert np.max(np.abs(F1 - F0)) < 1e-10


def test_natural_gradient_not_immune():
    """Metric unchanged but gradient rotated: |g_nat| changes — the
    opposite of depolarizing noise."""
    n = 2
    gates = hea_gates(n, 1)
    base = _base_state(n)
    rng = np.random.default_rng(0)
    th0 = rng.uniform(-np.pi, np.pi, len(gates))
    psi, D = deriv_states(th0, gates, base)
    H = ising_matrix(n)
    F0 = fs_qfi(D, psi)
    g0 = 2.0 * np.real(np.array([np.vdot(dd, H @ psi) for dd in D]))
    P = pauli_mat("XI", n)
    Un = rot_U(P, 0.4)
    psip = Un @ psi
    gN = 2.0 * np.real(np.array([np.vdot(Un @ dd, H @ psip) for dd in D]))
    Fn = fs_qfi([Un @ dd for dd in D], psip)
    reg = 1e-10 * np.eye(len(gates))
    nat0 = np.linalg.solve(F0 + reg, g0)
    natN = np.linalg.solve(Fn + reg, gN)
    assert abs(np.linalg.norm(natN) - np.linalg.norm(nat0)) > 1e-3


def test_zne_eps_space_exact_when_c_zero():
    """Single qubit, H = Z, |psi> = |0>, noise axis X: C = <Y> = 0,
    E(th) = cos(th) = 1 - 2 sin^2(th/2) — linear in eps = sin^2(th/2),
    so linear ZNE in eps space is exact."""
    psi = np.array([1.0, 0.0], dtype=complex)
    H = np.diag([1.0, -1.0]).astype(complex)
    P = pauli_mat("X", 1)
    e_eps, _ = zne_eps_extrap(psi, H, P, np.array([0.3, 0.5, 0.7]))
    assert e_eps < 1e-10


def test_zne_eps_space_has_residual_when_c_nonzero():
    """2-qubit random state, Ising H, XI axis: C != 0, eps-space linear
    ZNE has an O(sqrt(eps(1-eps))) residual."""
    n = 2
    gates = hea_gates(n, 1)
    base = _base_state(n)
    rng = np.random.default_rng(0)
    th0 = rng.uniform(-np.pi, np.pi, len(gates))
    psi = base.copy()
    for axis, idx in gates:
        psi = rotation_action_closed_form(axis, th0[idx], psi)
    H = ising_matrix(n)
    P = pauli_mat("XI", n)
    e_eps, _ = zne_eps_extrap(psi, H, P, np.array([0.3, 0.5, 0.7]))
    assert e_eps > 1e-3


def test_variational_bound_trivial_for_pure_states():
    n = 2
    gates = hea_gates(n, 1)
    base = _base_state(n)
    rng = np.random.default_rng(0)
    th0 = rng.uniform(-np.pi, np.pi, len(gates))
    psi = base.copy()
    for axis, idx in gates:
        psi = rotation_action_closed_form(axis, th0[idx], psi)
    H = ising_matrix(n)
    E_gs = np.linalg.eigvalsh(H)[0].real
    for ax in ("XI", "IZ", "ZZ"):
        P = pauli_mat(ax, n)
        for th in (0.2, 0.7):
            psip = rot_U(P, th) @ psi
            assert float(np.real(np.vdot(psip, H @ psip))) >= E_gs - 1e-9
