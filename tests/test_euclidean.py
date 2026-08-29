"""Tests for the N-dimensional Euclidean space: arbitrary-dimension
optimization and classification (the fix for the 2D-only limitation)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from geocore import EuclideanSpace, minimize


def test_euclidean_geodesics_straight_lines():
    """Geodesics are straight lines; energy conserved; transport is the
    identity — in any dimension."""
    E = EuclideanSpace(7)
    p, v = np.random.default_rng(0).standard_normal((2, 7))
    sol = E.geodesic_closed_form(p, v, 1.3)
    assert np.abs(sol.point - (p + 1.3 * v)).max() < 1e-12
    assert abs(E.metric_norm_sq(p, v) - E.metric_norm_sq(sol.point, sol.velocity)) < 1e-12
    assert np.abs(E.parallel_transport(p, sol.point, v) - v).max() < 1e-12
    assert E.in_chart(p)


def test_highdim_optimization_converges():
    """minimize on EuclideanSpace(n) converges for random quadratic
    potentials in n = 3, 10, 50 dimensions."""
    for n in [3, 10, 50]:
        E = EuclideanSpace(n)
        target = np.linspace(1, n, n)
        f = lambda p: np.sum((np.asarray(p) - target) ** 2)
        res = minimize(E, f, np.zeros(n), lr=0.1, n_steps=2000, minimizer=target)
        assert res.converged
        assert res.minimizer_error < 1e-8, n


def test_highdim_logistic_regression():
    """Classification with d = 10 features on EuclideanSpace(11):
    recovers the true decision direction (cos ~ 1) with high accuracy."""
    from geocore import EuclideanSpace as ES

    rng = np.random.default_rng(0)
    d, n = 10, 500
    X = rng.standard_normal((n, d))
    w_star = rng.standard_normal(d)
    b_star = 0.5
    y = (X @ w_star + b_star > 0).astype(float)
    E = ES(d + 1)

    def bce(params):
        w, b = params[:d], params[d]
        z = X @ w + b
        return float(np.mean(np.logaddexp(0.0, z) - y * z))

    res = minimize(E, bce, np.zeros(d + 1), lr=0.05, n_steps=500, optimizer="adam")
    w, b = res.point[:d], res.point[d]
    acc = float(np.mean((X @ w + b > 0) == (y > 0)))
    cos = float(np.dot(w, w_star) / (np.linalg.norm(w) * np.linalg.norm(w_star)))
    assert acc > 0.9
    assert cos > 0.9


def test_highdim_matches_pytorch():
    """The d=10 logistic problem: geocore accuracy and direction agree
    with torch's nn.Linear + BCEWithLogits."""
    torch = pytest.importorskip("torch")
    from pytorch_comparison import highdim_logistic_geocore, highdim_logistic_torch  # noqa: E402

    acc_t, w_t, _, w_star = highdim_logistic_torch()
    acc_g, w_g, _, _ = highdim_logistic_geocore()
    assert abs(acc_t - acc_g) < 0.02
    cos_g = float(np.dot(w_g, w_star) / (np.linalg.norm(w_g) * np.linalg.norm(w_star)))
    cos_t = float(np.dot(w_t, w_star) / (np.linalg.norm(w_t) * np.linalg.norm(w_star)))
    assert cos_g > 0.9 and cos_t > 0.9
