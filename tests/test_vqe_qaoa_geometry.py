"""Tests for the QAOA (MaxCut) gradient geometry
(examples/vqe_qaoa_geometry.py): the analytic gradient is verified
against central differences, the gradient scale does NOT decay
exponentially with width or depth (QAOA's deterministic |+> start and
2-local cost avoid barren plateaus — the exact-gradient measurement,
contrasting the HEA fidelity cost of feature 33), and the optimized
circuit reaches a good cut ratio vs the exhaustive MaxCut.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from vqe_qaoa_geometry import (  # noqa: E402
    adam_optimize,
    cut_values,
    cycle_plus_matching,
    gradient_scales,
    qaoa_energy,
    qaoa_gradient,
    qaoa_state,
)


def test_analytic_gradient_matches_central_differences():
    n, p = 8, 2
    C = cut_values(n, cycle_plus_matching(n))
    rng = np.random.default_rng(1)
    th = rng.uniform(0, np.pi, 2 * p)
    g_an = qaoa_gradient(th, n, C)
    h = 1e-6
    g_fd = np.zeros(2 * p)
    for j in range(2 * p):
        tp, tm = th.copy(), th.copy()
        tp[j] += h
        tm[j] -= h
        g_fd[j] = (qaoa_energy(tp, n, C) - qaoa_energy(tm, n, C)) / (2 * h)
    assert np.max(np.abs(g_an - g_fd)) < 1e-6


def test_gradient_does_not_decay_with_width():
    """QAOA (p=2) gradient scale vs n=6..14: no exponential decay —
    log10 slope is ~0 (contrast HEA fidelity -0.32 per qubit)."""
    ns = list(range(6, 15))
    eucs = []
    rng = np.random.default_rng(0)
    for n in ns:
        C = cut_values(n, cycle_plus_matching(n))
        e = 0.0
        for _ in range(5):
            th0 = rng.uniform(0, np.pi, 4)
            ee, _, _, _ = gradient_scales(th0, n, C)
            e += ee
        eucs.append(e / 5)
    sl, _ = np.polyfit(ns, np.log10(eucs), 1)
    assert abs(sl) < 0.1       # no exponential decay
    assert max(eucs) / min(eucs) < 10   # scale stays O(1-10)


def test_gradient_does_not_decay_with_depth():
    """n=10, p=1..5: intrinsic scale stays O(1-10)."""
    n10 = 10
    C10 = cut_values(n10, cycle_plus_matching(n10))
    rng = np.random.default_rng(0)
    scales = []
    for p in (1, 2, 3, 4, 5):
        s = 0.0
        for _ in range(5):
            th0 = rng.uniform(0, np.pi, 2 * p)
            _, i, _, _ = gradient_scales(th0, n10, C10)
            s += i
        scales.append(s / 5)
    assert max(scales) / min(scales) < 10


def test_maxcut_exhaustive_reference():
    """C.max() is the exact MaxCut (2^n cut counts)."""
    for n in (6, 8, 10):
        C = cut_values(n, cycle_plus_matching(n))
        # sanity: the cut count of a basis state is bounded by |E| = 3n/2
        assert C.max() <= 1.5 * n
        # the all-ones / all-zeros cuts cut 0 edges
        assert C[0] == 0.0
        assert C[-1] == 0.0


def test_optimized_cut_ratio():
    """p=2 Adam-optimized QAOA reaches > 0.7 of the exact MaxCut."""
    n = 8
    C = cut_values(n, cycle_plus_matching(n))
    mc = float(C.max())
    best = -1e9
    for s in range(4):
        rng = np.random.default_rng(s)
        th0 = rng.uniform(0, np.pi, 4)
        _, e = adam_optimize(th0, n, C, steps=200)
        best = max(best, e)
    assert best / mc > 0.7


def test_state_normalized_and_energy_bounded():
    """The QAOA state is normalized and E is a valid cut expectation."""
    n, p = 8, 2
    C = cut_values(n, cycle_plus_matching(n))
    rng = np.random.default_rng(0)
    th = rng.uniform(0, np.pi, 2 * p)
    psi = qaoa_state(th, n, C)
    assert abs(np.vdot(psi, psi) - 1.0) < 1e-12
    e = qaoa_energy(th, n, C)
    assert 0.0 <= e <= 1.5 * n
