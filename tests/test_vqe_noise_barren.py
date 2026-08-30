"""Tests for noise-induced barren plateaus (NIBP) — the exact gradient
mechanism (examples/vqe_noise_barren.py): the chain-rule noisy gradient
equals the pure gradient at lambda=0 to machine precision, the gradient
contracts with noise ((1-lambda)^{L_eff}, exponential in depth), and
the trainability collapses monotonically with noise under fixed-step
SGD (Adam masks it — the feature-32 effect, reported).
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from vqe_barren_plateaus import hea_gates, _base_state, ising_hamiltonian  # noqa: E402
from geocore.clifford import rotation_action_closed_form, pauli_action_on_state  # noqa: E402
from geocore.derivatives import rotation_derivative  # noqa: E402
from vqe_noise_barren import (  # noqa: E402
    _grad_slot,
    _rho_forward,
    noisy_gradient_rms,
    noisy_vqe,
)

N, L = 6, 2


@pytest.fixture(scope="module")
def setup():
    gates = hea_gates(N, L)
    base = _base_state(N)
    H = ising_hamiltonian(N)
    return dict(gates=gates, base=base, H=H, d=2**N, block=len(gates) // L)


def test_noisy_gradient_matches_pure_at_zero_lambda(setup):
    """The exact chain-rule gradient equals the pure gradient at
    lambda=0 to machine precision."""
    s = setup
    rng = np.random.default_rng(1)
    th = rng.uniform(-np.pi, np.pi, len(s["gates"]))
    psi = s["base"].copy()
    for m in range(len(s["gates"])):
        ax2, idx2 = s["gates"][m]
        psi = rotation_action_closed_form(ax2, th[idx2], psi)
    Hpsi = np.zeros_like(psi)
    for c, ax in s["H"]:
        Hpsi += c * pauli_action_on_state(ax, psi)
    g_ref = np.zeros(len(s["gates"]))
    for j in range(len(s["gates"])):
        phi = s["base"].copy()
        for m in range(j):
            ax2, idx2 = s["gates"][m]
            phi = rotation_action_closed_form(ax2, th[idx2], phi)
        dj = rotation_derivative(s["gates"][j][0], th[s["gates"][j][1]], phi)
        for m in range(j + 1, len(s["gates"])):
            ax2, idx2 = s["gates"][m]
            dj = rotation_action_closed_form(ax2, th[idx2], dj)
        g_ref[j] = 2 * np.real(np.vdot(dj, Hpsi))
    rhos = _rho_forward(N, L, 0.0, s["gates"], th, s["base"])
    g_noisy = np.array([_grad_slot(j, s["gates"], th, rhos, L, 0.0,
                                   s["H"], s["d"], s["block"])
                        for j in range(len(s["gates"]))])
    assert np.max(np.abs(g_noisy - g_ref)) < 1e-12


def test_gradient_contracts_with_noise(setup):
    """Gradient RMS decreases monotonically with the depolarizing
    strength; at lambda=0.8 it is ~0.08x the noiseless value
    ((1-0.8)^L_eff with L_eff in (1, L))."""
    s = setup
    g0 = noisy_gradient_rms(N, L, 0.0, s["gates"], s["base"], s["H"])
    g2 = noisy_gradient_rms(N, L, 0.5, s["gates"], s["base"], s["H"])
    g8 = noisy_gradient_rms(N, L, 0.8, s["gates"], s["base"], s["H"])
    assert g2 < g0
    assert g8 < g2
    # contraction ~ (1-lambda)^{L_eff}, L_eff in (1, L): bounds
    assert (0.2 ** L) < g8 / g0 < (0.2 ** 1)
    assert (0.5 ** L) < g2 / g0 < (0.5 ** 1)


def test_trainability_collapses_with_noise(setup):
    """Fixed-step SGD: the pure-state energy of the final theta gets
    monotonically worse with noise (Adam would mask the contraction)."""
    s = setup
    _, e0 = noisy_vqe(N, L, 0.0, s["gates"], s["base"], s["H"],
                     steps=300)
    _, e6 = noisy_vqe(N, L, 0.6, s["gates"], s["base"], s["H"],
                     steps=300)
    assert e6 > e0  # worse (less negative) under noise
