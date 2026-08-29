"""Tests for manifold variance/covariance (≈ torch.std + tangent PCA):
the Frechet variance, the tangent-space covariance in an orthonormal
frame, and the principal directions — each verified against a closed
form."""

import numpy as np
import pytest

from geocore import (
    HyperbolicPlane,
    PolarPlane,
    Sphere,
)
from geocore.geostats import (
    frechet_mean,
    frechet_variance,
    principal_directions,
    tangent_covariance,
)

rng = np.random.default_rng(17)

_MANIFOLDS = [
    (PolarPlane(), (1.2, 3.0)),
    (Sphere(), (0.5, 2.5)),
    (HyperbolicPlane(), (0.2, 1.5)),
]


def _to_cart(p):
    return np.array([p[0] * np.cos(p[1]), p[0] * np.sin(p[1])])


def test_variance_zero_for_single_point():
    """A single (repeated) point has zero spread."""
    for manifold, p0 in [
        (PolarPlane(), [2.0, 0.8]),
        (Sphere(), [1.1, 0.6]),
        (HyperbolicPlane(), [0.3, 1.2]),
    ]:
        pts = np.array([p0, p0, p0])
        assert frechet_variance(manifold, pts) == pytest.approx(0.0, abs=1e-12)
        assert np.abs(tangent_covariance(manifold, pts)).max() < 1e-12


def test_trace_of_covariance_equals_variance():
    """tr(Cov) = Frechet variance to machine precision on every manifold
    (because |log_m(p_i)|_g = d(m, p_i))."""
    for manifold, (lo, hi) in _MANIFOLDS:
        pts = rng.uniform(lo, hi, (30, 2))
        var = frechet_variance(manifold, pts)
        cov = tangent_covariance(manifold, pts)
        assert abs(np.trace(cov) - var) < 1e-9, type(manifold).__name__


def test_covariance_symmetric_positive_semidefinite():
    for manifold, (lo, hi) in _MANIFOLDS:
        pts = rng.uniform(lo, hi, (30, 2))
        cov = tangent_covariance(manifold, pts)
        assert np.abs(cov - cov.T).max() < 1e-12
        assert np.linalg.eigvalsh(cov)[0] > -1e-10


def test_polar_variance_equals_cartesian_spread():
    """On the flat polar plane the Frechet variance is the mean squared
    Cartesian distance to the arithmetic mean."""
    P = PolarPlane()
    pts = rng.uniform(1.2, 3.0, (20, 2))
    m = frechet_mean(P, pts, lr=0.1, n_steps=300).point
    m_cart = _to_cart(m)
    var = frechet_variance(P, pts, mean=m)
    spread = np.mean(
        [np.sum((_to_cart(p) - m_cart) ** 2) for p in pts]
    )
    assert abs(var - spread) < 1e-9


def test_ellipse_exact_statistics():
    """A symmetric point distribution on an ellipse has exact statistics:
    variance (a^2 + b^2)/2, tangent-covariance eigenvalues (b^2/2, a^2/2),
    and the top principal direction is the long axis (|dot| = 1)."""
    P = PolarPlane()

    def to_polar(p):
        return np.array([np.linalg.norm(p), np.arctan2(p[1], p[0])])

    center = np.array([0.3, 0.8])
    a, b, th_ax = 0.5, 0.15, 0.4
    axis = np.array([np.cos(th_ax), np.sin(th_ax)])
    perp = np.array([-np.sin(th_ax), np.cos(th_ax)])
    N = 200
    psi = 2 * np.pi * np.arange(N) / N
    cart = center[:, None] + a * np.outer(axis, np.cos(psi)) + b * np.outer(perp, np.sin(psi))
    pts = np.array([to_polar(cart[:, i]) for i in range(N)])
    m = frechet_mean(P, pts, lr=0.1, n_steps=500).point
    var = frechet_variance(P, pts, mean=m)
    evals, evecs = principal_directions(P, pts, mean=m)
    # exact statistics for uniform discrete angles
    assert var == pytest.approx((a * a + b * b) / 2, abs=1e-9)
    assert evals[0] == pytest.approx(b * b / 2, abs=1e-9)
    assert evals[1] == pytest.approx(a * a / 2, abs=1e-9)
    # top eigenvector (orthonormal frame) -> Cartesian tangent
    y_m, r_m = m[1], m[0]
    e2 = evecs[:, 1]
    v_cart = e2[0] * np.array([np.cos(y_m), np.sin(y_m)]) + e2[1] * np.array(
        [-np.sin(y_m), np.cos(y_m)]
    )
    v_cart /= np.linalg.norm(v_cart)
    assert abs(v_cart @ axis) > 0.9999


def test_principal_directions_orthonormal_ascending():
    """The eigenvectors form an orthonormal basis and the eigenvalues are
    ascending."""
    for manifold, (lo, hi) in _MANIFOLDS:
        pts = rng.uniform(lo, hi, (25, 2))
        evals, evecs = principal_directions(manifold, pts)
        assert evals[0] <= evals[1] + 1e-12
        assert np.abs(evecs.T @ evecs - np.eye(2)).max() < 1e-12


def test_sphere_variance_bounded():
    """On the sphere the variance is non-negative and bounded by pi^2."""
    S = Sphere()
    pts = rng.uniform(0.5, 2.5, (20, 2))
    var = frechet_variance(S, pts)
    assert 0.0 <= var <= np.pi**2 + 1e-9
