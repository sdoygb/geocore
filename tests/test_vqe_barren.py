"""Tests for the barren-plateau diagnostics (examples/vqe_barren_plateaus.py):
the analytic gradient is machine-verified, the width scan reproduces the
cost-function dependence of VQE trainability (global n-local cost has a
steeper per-qubit gradient falloff than the 2-local Ising energy), the
small 2-qubit system shows no plateau, and the shallow local-cost VQE
still trains at n=8.

All statistics are reproducible (fixed RNG seeds) and use the exact
analytic rotation derivatives — no sampling noise.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from vqe_barren_plateaus import (  # noqa: E402
    H2_HAMILTONIAN,
    analytic_gradient,
    apply_ansatz,
    energy,
    fit_slope,
    global_cost,
    gradient_scale,
    hea_gates,
    ising_hamiltonian,
    normalize_local,
    verify_gradient,
    _base_state,
)


def test_analytic_gradient_matches_central_differences():
    """The reverse-adjoint analytic gradient equals central differences
    to ~1e-10 (machine precision for double precision)."""
    worst = verify_gradient(3, 2, ising_hamiltonian(3), atol=1e-6)
    assert worst < 1e-6


def test_analytic_gradient_h2_and_global_cost():
    """Also verified on the H2 Hamiltonian and the n-local global cost."""
    for terms in (H2_HAMILTONIAN, global_cost(4)):
        n = 2 if terms is H2_HAMILTONIAN else 4
        gates = hea_gates(n, 2)
        base = _base_state(n)
        rng = np.random.default_rng(42)
        for _ in range(2):
            theta = rng.uniform(-np.pi, np.pi, len(gates))
            g_an = analytic_gradient(theta, gates, terms, base)
            h = 1e-6
            g_fd = np.zeros_like(g_an)
            for j in range(len(theta)):
                tp, tm = theta.copy(), theta.copy()
                tp[j] += h
                tm[j] -= h
                g_fd[j] = (energy(tp, gates, terms, base)
                           - energy(tm, gates, terms, base)) / (2 * h)
            assert np.max(np.abs(g_an - g_fd)) < 1e-6


def test_width_effect_global_steeper_than_local():
    """The core scientific claim, at fixed seeds: over n = 2..10 the
    per-qubit log10 falloff of the global (n-local) cost is steeper
    (more negative) than that of the local Ising energy, on the same
    shallow L=2 HEA."""
    ns = list(range(2, 11))
    rng = np.random.default_rng(0)
    s_loc = [gradient_scale(n, 2, normalize_local(ising_hamiltonian(n)), 30, rng)
             for n in ns]
    rng = np.random.default_rng(0)
    s_glb = [gradient_scale(n, 2, global_cost(n), 30, rng) for n in ns]
    sl_loc, _, _ = fit_slope(ns, s_loc)
    sl_glb, _, r2_glb = fit_slope(ns, s_glb)
    assert sl_glb < sl_loc          # global falls faster per qubit
    assert sl_glb < -0.10           # and is a genuine decay
    assert sl_loc > -0.15           # local stays closer to flat
    assert r2_glb > 0.9             # clean log10-linear trend


def test_gradient_scale_does_not_grow_with_parameter_count():
    """The per-parameter RMS is the variance scale, not the full-vector
    norm: adding parameters must not inflate it (n=6, L=1 vs L=6)."""
    rng = np.random.default_rng(2)
    s1 = gradient_scale(6, 1, normalize_local(ising_hamiltonian(6)), 50, rng)
    rng = np.random.default_rng(2)
    s6 = gradient_scale(6, 6, normalize_local(ising_hamiltonian(6)), 50, rng)
    assert s6 < 2 * s1  # no sqrt(P) inflation


def test_no_barren_plateau_in_two_qubits():
    """H2 (2 qubits): the per-parameter gradient stays O(0.1) across
    depth — a flat landscape, no plateau (this is why small-molecule
    VQE trains)."""
    rng = np.random.default_rng(1)
    scales = [gradient_scale(2, L, H2_HAMILTONIAN, 100, rng) for L in range(1, 9)]
    sl, _, _ = fit_slope(list(range(1, 9)), scales)
    assert all(s > 0.05 for s in scales)
    assert sl > -0.05


def test_shallow_vqe_trains_at_n8():
    """The shallow local-cost circuit still descends most of the way to
    the exact Ising ground state at n=8 (gradient not yet barren)."""
    n = 8
    terms = ising_hamiltonian(n)
    gates = hea_gates(n, 2)
    base = _base_state(n)
    rng = np.random.default_rng(7)
    theta = rng.uniform(-np.pi, np.pi, len(gates))
    e0 = energy(theta, gates, terms, base)
    lr, b1, b2, eps = 0.2, 0.9, 0.999, 1e-8
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    for _ in range(300):
        g = analytic_gradient(theta, gates, terms, base)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        theta -= lr * (m / (1 - b1)) / (np.sqrt(v / (1 - b2)) + eps)
    e_f = energy(theta, gates, terms, base)

    # exact ground state by dense diagonalization
    m1 = {"I": np.eye(2, dtype=complex),
          "X": np.array([[0, 1], [1, 0]], dtype=complex),
          "Z": np.array([[1, 0], [0, -1]], dtype=complex)}
    H = np.zeros((2**n, 2**n), dtype=complex)
    for c, p in terms:
        M = np.array([[1.0]], dtype=complex)
        for ch in p:
            M = np.kron(M, m1[ch])
        H = H + c * M
    e_exact = np.linalg.eigvalsh(H)[0].real

    assert e_f < e0                      # it descends
    assert e_f < e_exact + 1.0           # and reaches within 1 Ha of the
                                         # exact ground state
