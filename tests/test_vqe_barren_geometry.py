"""Tests for the geometric-root analysis of barren plateaus
(examples/vqe_barren_geometry.py): the QFI is machine-verified, the
intrinsic (coordinate-free) gradient scale decays with width at the same
rate as the Euclidean one (the plateau is not a coordinate artifact),
the root decomposes into cost concentration x geometric alignment, the
natural gradient cannot walk out of the plateau, and the warm start
raises the intrinsic scale while the manifold rank drops (parameter
redundancy).
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from vqe_barren_plateaus import hea_gates, _base_state  # noqa: E402
from vqe_barren_geometry import (  # noqa: E402
    apply,
    deriv_states,
    fidelity_energy_direction,
    geometry,
    ising_ground_state,
    natural_sgd,
    qfi_and_gradient,
    verify_qfi,
    warm_start_angles,
)
from vqe_barren_prewarm import init_protocol  # noqa: E402


def test_qfi_matches_central_differences():
    err = verify_qfi(6, hea_gates(6, 2), _base_state(6), ising_ground_state(6)[1])
    assert err < 1e-6


def test_qfi_pseudoinverse_is_correct():
    """The regularized pseudo-inverse used for the natural gradient
    satisfies the defining property F F^+ F = F (machine precision)."""
    n = 6
    gates = hea_gates(n, 2)
    base = _base_state(n)
    _, gs = ising_ground_state(n)
    rng = np.random.default_rng(0)
    th = rng.uniform(-np.pi, np.pi, len(gates))
    v = fidelity_energy_direction(th, gates, base, gs)
    F, g, g_nat, rank = qfi_and_gradient(th, gates, base, v)
    assert rank == len(gates)  # small random point: full rank
    w, U = np.linalg.eigh(F)
    keep = w > 1e-7
    Fpinv = (U * np.where(keep, 1.0 / w, 0.0)) @ U.T
    assert np.max(np.abs(F @ Fpinv @ F - F)) < 1e-8
    # g_nat is exactly the pseudo-inverse action
    assert np.max(np.abs(g_nat - Fpinv @ g)) < 1e-10


def test_intrinsic_and_euclidean_decay_at_same_rate():
    """The coordinate-free intrinsic gradient scale decays with n at
    (almost) the same rate as the Euclidean one — the plateau is not a
    coordinate artifact."""
    ns = list(range(6, 13))
    eucs, intrs = [], []
    rng = np.random.default_rng(0)
    for nn in ns:
        gn = hea_gates(nn, 2)
        bn = _base_state(nn)
        _, gsn = ising_ground_state(nn)
        th0 = rng.uniform(-np.pi, np.pi, len(gn))
        v = fidelity_energy_direction(th0, gn, bn, gsn)
        e, i, _, _, _ = geometry(th0, gn, bn, v)
        eucs.append(e)
        intrs.append(i)
    sl_e, _ = np.polyfit(ns, np.log10(eucs), 1)
    sl_i, _ = np.polyfit(ns, np.log10(intrs), 1)
    assert abs(sl_e - sl_i) < 0.05
    assert sl_e < -0.1 and sl_i < -0.1  # both genuinely decay


def test_root_decomposes_into_concentration_and_alignment():
    """log10 gradient slope ~= log10|v| slope + log10 align slope."""
    ns = list(range(6, 13))
    eucs, nvs, aligns = [], [], []
    rng = np.random.default_rng(0)
    for nn in ns:
        gn = hea_gates(nn, 2)
        bn = _base_state(nn)
        _, gsn = ising_ground_state(nn)
        th0 = rng.uniform(-np.pi, np.pi, len(gn))
        v = fidelity_energy_direction(th0, gn, bn, gsn)
        e, i, nv, al, _ = geometry(th0, gn, bn, v)
        eucs.append(e)
        nvs.append(nv)
        aligns.append(al)
    sl_e, _ = np.polyfit(ns, np.log10(eucs), 1)
    sl_v, _ = np.polyfit(ns, np.log10(nvs), 1)
    sl_a, _ = np.polyfit(ns, np.log10(aligns), 1)
    assert abs((sl_v + sl_a) - sl_e) < 0.1  # multiplicative decomposition


def test_natural_gradient_cannot_escape_plateau():
    """At n=12, both euclidean and natural SGD stay stuck (fidelity
    ~0) after 300 steps — a coordinate-free gradient cannot cure the
    geometric decay."""
    n = 12
    gates = hea_gates(n, 2)
    base = _base_state(n)
    _, gs = ising_ground_state(n)
    rng = np.random.default_rng(0)
    th0 = rng.uniform(-np.pi, np.pi, len(gates))
    f0 = 1 - abs(np.vdot(gs, apply(th0, gates, base))) ** 2
    th_e = th0.copy()
    for _ in range(150):
        v = fidelity_energy_direction(th_e, gates, base, gs)
        _, g, _, _ = qfi_and_gradient(th_e, gates, base, v)
        th_e = th_e - 0.5 * g
    th_n = natural_sgd(th0, gates, base, gs, steps=150)
    f_e = 1 - abs(np.vdot(gs, apply(th_e, gates, base))) ** 2
    f_n = 1 - abs(np.vdot(gs, apply(th_n, gates, base))) ** 2
    assert 1 - f_e < 0.01   # fidelity ~0: stuck
    assert 1 - f_n < 0.01


def test_warm_start_raises_intrinsic_scale():
    """The warm start sits orders of magnitude above random points in
    the intrinsic scale, and the manifold rank drops (redundancy)."""
    n = 12
    layers = 2
    gates = hea_gates(n, layers)
    base = _base_state(n)
    _, gs = ising_ground_state(n)
    warm = warm_start_angles(n, steps=800)
    rng = np.random.default_rng(0)
    th_p = init_protocol(n, layers, warm, "warm_perturbed", rng)
    v_p = fidelity_energy_direction(th_p, gates, base, gs)
    _, i_p, _, _, rk_p = geometry(th_p, gates, base, v_p)
    # random median over points
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(5):
        th_r = rng.uniform(-np.pi, np.pi, len(gates))
        v_r = fidelity_energy_direction(th_r, gates, base, gs)
        _, i_r, _, _, _ = geometry(th_r, gates, base, v_r)
        vals.append(i_r)
    med = float(np.median(vals))
    assert i_p > 100 * med        # 2+ orders above the random median
    assert rk_p < len(gates)      # rank-deficient near a product state
