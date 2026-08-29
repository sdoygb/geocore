"""Tests for the QEC diagnostics application layer: vectorized sweeps,
pseudo-thresholds (exactly pi/2 for every repetition code under coherent
X-noise), crossovers, and the diagnostic report with analytic
counterparts verified."""

import numpy as np
import pytest

from geocore.qec import (
    crossover,
    diagnose,
    logical_error_sweep,
    pseudo_threshold,
    repetition_closed_form,
    repetition_code_logical_error,
)

_THEtas = np.linspace(0.01, 0.5, 40)


def test_sweep_matches_statevector_simulation():
    """Vectorized closed-form sweep == the O(2^n) state-vector simulation
    to machine precision (the batch path is exact)."""
    for n in [3, 5, 7, 9]:
        sw = logical_error_sweep(n, _THEtas)
        ref = np.array([repetition_code_logical_error(t, n) for t in _THEtas])
        assert np.abs(sw - ref).max() < 1e-14, n


def test_sweep_matches_closed_form_loop():
    """Vectorized == per-point closed form (same formula, batched)."""
    for n in [3, 5, 7]:
        sw = logical_error_sweep(n, _THEtas)
        ref = np.array([repetition_closed_form(t, n) for t in _THEtas])
        assert np.abs(sw - ref).max() < 1e-15, n


def test_pseudo_threshold_is_exactly_pi_over_2():
    """For every repetition code under coherent X-noise the
    pseudo-threshold is exactly theta* = pi/2: at theta = pi/2,
    P_L(n) = 1/2^{n+1} * sum_{k>(n-1)/2} C(n,k) = 1/2 = P_phys."""
    for n in [3, 5, 7, 9, 11]:
        assert pseudo_threshold(n) == pytest.approx(np.pi / 2, abs=1e-12)
        assert repetition_closed_form(np.pi / 2, n) == pytest.approx(0.5, abs=1e-12)


def test_pseudo_threshold_substitution():
    """The root satisfies P_L(n, theta*) = P_phys(theta*) to machine
    precision (numerical root verified by substitution)."""
    for n in [5, 7, 9]:
        th = pseudo_threshold(n)
        assert abs(repetition_closed_form(th, n) - np.sin(th / 2) ** 2) < 1e-12


def test_encoding_helps_below_threshold():
    """For theta < pi/2 the code beats the physical error rate; above it
    loses (verified exactly for the distance-3 code: P_L = s^2(3-2s) vs
    P_phys = s with s = sin^2(theta/2))."""
    for th in [0.3, 0.9, 1.4]:
        assert repetition_closed_form(th, 3) < np.sin(th / 2) ** 2
    for th in [1.8, 2.4, 3.0]:
        assert repetition_closed_form(th, 3) > np.sin(th / 2) ** 2


def test_crossover_verified_by_substitution():
    """At the crossover both codes have equal logical error rates (verified
    by substitution to machine precision)."""
    for n1, n2 in [(3, 5), (5, 7), (3, 7)]:
        th = crossover(n1, n2)
        assert 0 < th < np.pi
        err = abs(repetition_closed_form(th, n1) - repetition_closed_form(th, n2))
        assert err < 1e-10, (n1, n2)


def test_diagnose_report_matches_analytic_law():
    """The diagnostic report reproduces the theta^{d+1} law: measured
    exponents within 0.01 of d+1, leading coefficients within 2% of
    C(n,(n+1)/2)/2^{n+1}, pseudo-thresholds = pi/2."""
    rep = diagnose((3, 5, 7))
    assert np.all(rep.exponent_errors < 0.01)
    assert np.all(rep.coefficient_relative_errors < 0.02)
    for n, th in rep.pseudo_thresholds.items():
        assert th == pytest.approx(np.pi / 2, abs=1e-12)
    assert rep.crossover is not None
    # the reported crossover is a verified root
    assert abs(
        repetition_closed_form(rep.crossover, 3)
        - repetition_closed_form(rep.crossover, 5)
    ) < 1e-10


def test_sweep_measured_speedup():
    """The vectorized sweep is measured faster than the per-point
    closed-form loop (the batch core path applied to QEC diagnostics)."""
    import time

    th = np.linspace(0.001, 0.5, 2000)
    t0 = time.perf_counter()
    for _ in range(10):
        logical_error_sweep(9, th)
    tv = (time.perf_counter() - t0) / 10
    t0 = time.perf_counter()
    for _ in range(10):
        for t in th:
            repetition_closed_form(t, 9)
    tl = (time.perf_counter() - t0) / 10
    assert tl / tv > 5.0
