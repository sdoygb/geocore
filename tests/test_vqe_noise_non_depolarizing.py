"""Tests for the non-depolarizing noise geometry (examples/
vqe_noise_non_depolarizing.py): amplitude damping has a scalar QFI
contraction (1-g) but a basis-dependent energy track (coherent linear,
population quadratic), phase damping has an anisotropic QFI
contraction, depolarizing is scalar (closed form), Pauli twirl shrinks
(not removes) the population residual, and the variational bound
survives any CPTP noise.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from vqe_noise_non_depolarizing import (  # noqa: E402
    ad1_kraus,
    channel2,
    channel_qfi_fingerprint,
    depol2,
    pauli_twirl2,
    pd1_kraus,
)

BELL = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2)
RHO_BELL = np.outer(BELL, BELL.conj())
X = np.array([[0, 1], [1, 0]], dtype=complex)
H_XX = np.kron(X, X)
H_ZZ = np.diag([1.0, -1.0, -1.0, 1.0]).astype(complex)
GAMS = np.array([0.05, 0.15, 0.25])


def _linear_zne_err(ch_fun, H, gams=GAMS):
    E0 = float(np.real(np.trace(RHO_BELL @ H)))
    Es = np.array([float(np.real(np.trace(ch_fun(RHO_BELL, g) @ H)))
                   for g in gams])
    return abs(np.polyval(np.polyfit(gams, Es, 1), 0.0) - E0)


def test_ad_energy_coherent_linear_population_quadratic():
    """Amplitude damping: H_XX (coherent) energy is exactly linear in g
    (linear ZNE exact); H_ZZ (population) is quadratic (O(g^2) error)."""
    ad = lambda rho, g: channel2(rho, g, ad1_kraus)  # noqa: E731
    assert _linear_zne_err(ad, H_XX) < 1e-10
    assert _linear_zne_err(ad, H_ZZ) > 1e-3


def test_ad_qfi_contraction_is_scalar():
    """Amplitude damping contracts the SLD-QFI by the scalar (1-g) to
    machine precision (even on an anisotropic circuit)."""
    ratios, _ = channel_qfi_fingerprint(
        0.3, lambda r, g: channel2(r, g, ad1_kraus))
    assert abs(ratios.mean() - 0.7) < 1e-6
    assert ratios.max() - ratios.min() < 1e-6


def test_depolarizing_qfi_contraction_is_scalar_closed_form():
    """Depolarizing: scalar c(0.3, d=4) = 0.7^2/(0.7+0.15) = 0.57647."""
    ratios, _ = channel_qfi_fingerprint(0.3, depol2, is_depol=True)
    c = 0.7 ** 2 / (0.7 + 2 * 0.3 / 4)
    assert abs(ratios.mean() - c) < 1e-6
    assert ratios.max() - ratios.min() < 1e-6


def test_phase_damping_qfi_contraction_is_anisotropic():
    """Phase damping: QFI contraction is anisotropic (population
    directions unchanged, coherence directions x (1-g))."""
    ratios, _ = channel_qfi_fingerprint(
        0.3, lambda r, g: channel2(r, g, pd1_kraus))
    assert ratios.max() - ratios.min() > 0.1


def test_pauli_twirl_shrinks_population_residual():
    """Twirling AD recovers the coherent part and shrinks (does not
    remove) the O(g^2) population residual."""
    ad = lambda rho, g: channel2(rho, g, ad1_kraus)  # noqa: E731
    e_raw = _linear_zne_err(ad, H_ZZ)
    e_tw = _linear_zne_err(lambda rho, g: pauli_twirl2(rho, g, ad1_kraus),
                           H_ZZ)
    assert e_tw < e_raw
    assert e_tw > 1e-3   # residual remains: twirl is not a cure-all


def test_variational_bound_survives_any_cptp():
    """Tr(rho H) >= E_gs holds for AD and PD noise (spectral theorem)."""
    from vqe_noise_geometry import ising_matrix
    H2 = ising_matrix(2)
    E_gs = np.linalg.eigvalsh(H2)[0].real
    rng = np.random.default_rng(1)
    for _ in range(30):
        ps = rng.normal(size=4) + 1j * rng.normal(size=4)
        ps = ps / np.linalg.norm(ps)
        rho = np.outer(ps, ps.conj())
        for g in (0.2, 0.7):
            for ch in (ad1_kraus, pd1_kraus):
                rho_n = channel2(rho, g, ch)
                assert float(np.real(np.trace(rho_n @ H2))) >= E_gs - 1e-9
