"""Tests for the classical pre-training mitigation of barren plateaus
(examples/vqe_barren_prewarm.py): the warm start restores the gradient
scale ~2 orders of magnitude at n=10 (and ~5 at n=12), fixed-step SGD
stays stuck for random initialization while the warm start descends, the
naive zero-filled warm start is a gradient-zero trap on the real-Pauli
local energy (RZZ slots exactly zero), and the perturbed protocol
escapes it.  All numbers reproducible with fixed seeds, exact analytic
gradients, and sparse-eigensolve references.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from vqe_barren_plateaus import hea_gates, ising_hamiltonian, _base_state  # noqa: E402
from vqe_barren_prewarm import (  # noqa: E402
    energy_gradient,
    fidelity_cost,
    fidelity_gradient,
    init_protocol,
    ising_ground_state,
    warm_start_angles,
)


N = 10


@pytest.fixture(scope="module")
def setup():
    gates = hea_gates(N, 2)
    base = _base_state(N)
    terms = ising_hamiltonian(N)
    E0, gs = ising_ground_state(N)
    warm = warm_start_angles(N, steps=800)
    rng = np.random.default_rng(0)
    th_r = init_protocol(N, 2, warm, "random", rng)
    th_n = init_protocol(N, 2, warm, "warm_naive", rng)
    th_p = init_protocol(N, 2, warm, "warm_perturbed", rng)
    return dict(gates=gates, base=base, terms=terms, E0=E0, gs=gs,
                warm=warm, th_r=th_r, th_n=th_n, th_p=th_p)


def test_fidelity_gradient_verified():
    """The reverse-adjoint fidelity gradient equals central differences
    to ~1e-10."""
    n6 = 6
    g6 = hea_gates(n6, 2)
    b6 = _base_state(n6)
    _, gs6 = ising_ground_state(n6)
    rng = np.random.default_rng(123)
    th = rng.uniform(-np.pi, np.pi, len(g6))
    g_an = fidelity_gradient(th, g6, b6, gs6)
    h = 1e-6
    g_fd = np.zeros_like(g_an)
    for j in range(len(g6)):
        tp, tm = th.copy(), th.copy()
        tp[j] += h
        tm[j] -= h
        g_fd[j] = (fidelity_cost(tp, g6, b6, gs6)
                   - fidelity_cost(tm, g6, b6, gs6)) / (2 * h)
    assert np.max(np.abs(g_an - g_fd)) < 1e-6


def test_warm_start_restores_gradient_scale(setup):
    """On the global fidelity cost at n=10, the warm start has a
    gradient scale > 100x the random one."""
    s = setup
    P = len(s["gates"])
    g_r = fidelity_gradient(s["th_r"], s["gates"], s["base"], s["gs"])
    g_p = fidelity_gradient(s["th_p"], s["gates"], s["base"], s["gs"])
    rms_r = np.sqrt(np.dot(g_r, g_r) / P)
    rms_p = np.sqrt(np.dot(g_p, g_p) / P)
    assert rms_p > 100 * rms_r
    assert rms_r < 1e-3   # random is (nearly) barren


def test_sgd_random_stuck_warm_descends(setup):
    """Fixed-step SGD (the classic plateau consequence): random stays at
    fidelity ~0 while the warm start reaches fidelity > 0.1 in 300
    steps."""
    s = setup
    lr = 0.5
    th = s["th_r"].copy()
    for _ in range(300):
        th = th - lr * fidelity_gradient(th, s["gates"], s["base"], s["gs"])
    f_r = 1 - fidelity_cost(th, s["gates"], s["base"], s["gs"])
    th = s["th_p"].copy()
    for _ in range(300):
        th = th - lr * fidelity_gradient(th, s["gates"], s["base"], s["gs"])
    f_p = 1 - fidelity_cost(th, s["gates"], s["base"], s["gs"])
    assert f_r < 0.01      # stuck
    assert f_p > 0.1       # warm start descends


def test_naive_warm_is_gradient_zero_trap(setup):
    """On the local (real-Pauli) Ising energy, the naive zero-filled
    warm start has exactly zero RZZ slots (machine precision) and an
    overall gradient scale far below random; the perturbed protocol
    escapes."""
    s = setup
    P = len(s["gates"])
    g_n = energy_gradient(s["th_n"], s["gates"], s["base"], s["terms"])
    g_p = energy_gradient(s["th_p"], s["gates"], s["base"], s["terms"])
    rms_n = np.sqrt(np.dot(g_n, g_n) / P)
    rms_p = np.sqrt(np.dot(g_p, g_p) / P)
    # RZZ slots of the naive point: indices n .. 2n-2 (first layer)
    rzz_slots = slice(N, 2 * N - 1)
    assert np.max(np.abs(g_n[rzz_slots])) < 1e-12   # exactly zero
    assert rms_n < 1e-3                             # trap
    assert rms_p > 10 * rms_n                       # perturbation escapes
