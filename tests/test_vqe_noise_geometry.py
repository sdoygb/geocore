"""Tests for the noise-aware VQE geometry (examples/
vqe_noise_geometry.py): energy is exactly affine in the depolarizing
strength, the SLD-QFI contracts by the closed form c(lambda) to machine
precision, the natural gradient is (nearly) immune, the variational
bound survives while the SGD optimum is pulled (Adam masks the pull),
and ZNE's exactness order equals the number of noise points.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from vqe_barren_plateaus import hea_gates, _base_state  # noqa: E402
from vqe_noise_geometry import (  # noqa: E402
    apply_theta,
    contraction_factor,
    depolarize,
    depolarizing_circuit_energy,
    e_direct,
    e_lin,
    fs_qfi,
    ising_matrix,
    natural_gradient_contraction,
    noise_pull,
    sld_qfi,
)


def _deriv_states(theta, gates, base):
    from geocore.clifford import rotation_action_closed_form
    from geocore.derivatives import rotation_derivative
    F = [base.copy()]
    for axis, idx in gates:
        F.append(rotation_action_closed_form(axis, theta[idx], F[-1]))
    D = []
    for j in range(len(gates)):
        axis, idx = gates[j]
        dd = rotation_derivative(axis, theta[idx], F[j])
        for k in range(j + 1, len(gates)):
            dd = rotation_action_closed_form(gates[k][0], theta[gates[k][1]], dd)
        D.append(dd)
    return F[-1], D


def test_energy_is_affine_in_depolarizing_strength():
    """E(lambda) = (1-lambda) E_pure + lambda Tr(H)/d is EXACT."""
    n = 2
    d = 2**n
    gates = hea_gates(n, 1)
    base = _base_state(n)
    rng = np.random.default_rng(0)
    th = rng.uniform(-np.pi, np.pi, len(gates))
    psi = apply_theta(th, gates, base)
    H = ising_matrix(n)
    E_pure = float(np.real(np.vdot(psi, H @ psi)))
    TrH = float(np.real(np.trace(H)))
    for lam in (0.0, 0.2, 0.5, 0.8):
        rho = depolarize(psi, lam, d)
        assert abs(e_lin(lam, E_pure, TrH, d) - e_direct(rho, H)) < 1e-12


def test_sld_qfi_contracts_by_closed_form():
    """F_noisy = c(lambda) F_pure with c = (1-l)^2/(1-l+2l/d), verified
    against the numerical SLD to machine precision."""
    for n in (3, 4):
        d = 2**n
        gates = hea_gates(n, 1)
        base = _base_state(n)
        rng = np.random.default_rng(0)
        th = rng.uniform(-np.pi, np.pi, len(gates))
        psi, D = _deriv_states(th, gates, base)
        Fp = fs_qfi(D, psi)
        for lam in (0.05, 0.3, 0.7):
            c = contraction_factor(lam, d)
            Fn = sld_qfi(psi, D, lam)
            rel = np.max(np.abs(Fn - c * Fp)) / np.max(np.abs(Fp))
            assert rel < 1e-10


def test_natural_gradient_nearly_immune():
    """Euclidean gradient contracts by (1-lambda); natural gradient by
    (1-l+2l/d)/(1-l) ~= 1 for large d."""
    d = 2**10
    for lam in (0.1, 0.3):
        f_nat = natural_gradient_contraction(lam, d)
        assert abs(f_nat - 1.0) < 1e-3  # immune to O(1/d)
        assert abs(1.0 - (1 - lam)) > 0.05  # euclidean clearly contracts


def test_variational_bound_survives_noise():
    """Tr(rho H) >= E_gs for any physical mixed state (spectral
    theorem) — noise never pushes the energy below the ground state."""
    H = ising_matrix(6)
    E_gs = np.linalg.eigvalsh(H)[0].real
    rng = np.random.default_rng(1)
    for _ in range(30):
        ps = rng.normal(size=64) + 1j * rng.normal(size=64)
        ps = ps / np.linalg.norm(ps)
        for lam in (0.2, 0.7):
            rho = depolarize(ps, lam, 64)
            assert e_direct(rho, H) >= E_gs - 1e-9


def test_sgd_optimum_pulled_by_noise():
    """Fixed-step SGD: larger noise strength means a shallower optimum
    (E_pure(theta*) pulled higher above E_gs)."""
    pulls = []
    for lam in (0.0, 0.3, 0.6):
        _, _, pull = noise_pull(6, lam, steps=200, lr=0.05)
        pulls.append(pull)
    assert pulls[1] > pulls[0]
    assert pulls[2] > pulls[1]


def test_adam_masks_the_pull():
    """Adam normalizes the gradient, so a scalar (1-lambda) contraction
    leaves the trajectory unchanged — honest contrast."""
    _, _, pull0 = noise_pull(6, 0.0, steps=200, lr=0.2, adam=True)
    _, _, pull1 = noise_pull(6, 0.6, steps=200, lr=0.2, adam=True)
    assert abs(pull1 - pull0) < 1e-6


def test_zne_exactness_order_equals_noise_points():
    """L=1 noise point: linear extrapolation is exact.  L=2: linear has
    O(lambda^2) error, degree-2 polynomial extrapolation is exact."""
    H = ising_matrix(2)
    lam_pts = np.array([0.05, 0.15, 0.25])
    for L in (1, 2):
        Es = np.array([depolarizing_circuit_energy(2, L, H, None, lam)
                       for lam in lam_pts])
        E0 = depolarizing_circuit_energy(2, L, H, None, 0.0)
        E_lin = np.polyval(np.polyfit(lam_pts, Es, 1), 0.0)
        E_poly = np.polyval(np.polyfit(lam_pts, Es, L), 0.0)
        if L == 1:
            assert abs(E_lin - E0) < 1e-10
        else:
            assert abs(E_lin - E0) > 1e-4      # linear is wrong for 2 pts
            assert abs(E_poly - E0) < 1e-10    # degree-2 is exact
