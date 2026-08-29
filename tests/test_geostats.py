"""Tests for geometric statistics (Frechet mean, the analogue of
torch.mean) and the analytic-gradient optimizer integration."""

import numpy as np
import pytest

from geocore import (
    HyperbolicPlane,
    PolarPlane,
    Sphere,
    minimize,
)
from geocore.derivatives import log_map
from geocore.geostats import frechet_mean, geodesic_distance
from geocore.invariants import VerificationError

rng = np.random.default_rng(13)

_MANIFOLDS = [
    (PolarPlane(), (1.2, 3.0), (-1.0, 1.0)),
    (Sphere(), (0.5, 2.5), (0.1, 1.0)),
    (HyperbolicPlane(), (0.2, 1.5), (0.8, 2.0)),
]


def _to_cart(p):
    return np.array([p[0] * np.cos(p[1]), p[0] * np.sin(p[1])])


# ---------------------------------------------------------------------------
# Logarithmic map (inverse exponential)
# ---------------------------------------------------------------------------


def test_log_map_is_inverse_of_exp():
    """exp_p(log_p(q)) = q on every manifold, to ~1e-12 (using the
    numerically stable distance)."""
    for manifold, (lo, hi), _ in _MANIFOLDS:
        worst = 0.0
        for _ in range(50):
            p = rng.uniform(lo, hi, 2)
            q = rng.uniform(lo, hi, 2)
            d = geodesic_distance(manifold, p, q)
            if d < 1e-9 or (isinstance(manifold, Sphere) and abs(d - np.pi) < 0.1):
                continue
            v = log_map(manifold, p, q)
            sol = manifold.geodesic_closed_form(p, v, 1.0)
            worst = max(worst, geodesic_distance(manifold, sol.point, q))
        assert worst < 1e-9, type(manifold).__name__


def test_log_map_length_equals_distance():
    """|log_p(q)|_g = d(p, q): the log map has the right length."""
    for manifold, (lo, hi), _ in _MANIFOLDS:
        for _ in range(20):
            p = rng.uniform(lo, hi, 2)
            q = rng.uniform(lo, hi, 2)
            v = log_map(manifold, p, q)
            d = geodesic_distance(manifold, p, q)
            g = manifold.metric_norm_sq(p, v)
            assert abs(np.sqrt(g) - d) < 1e-9, type(manifold).__name__


# ---------------------------------------------------------------------------
# Frechet mean
# ---------------------------------------------------------------------------


def test_frechet_mean_polar_equals_cartesian_mean():
    """On the flat polar plane the Frechet mean is the Cartesian arithmetic
    mean — to machine precision (the strongest closed-form check)."""
    P = PolarPlane()
    pts = np.array([[2.0, 0.3], [1.2, -0.5], [1.9, 1.2], [1.1, 0.9]])
    res = frechet_mean(P, pts, lr=0.1, n_steps=500)
    cart_mean = np.mean([_to_cart(p) for p in pts], axis=0)
    assert np.abs(_to_cart(res.point) - cart_mean).max() < 1e-9
    assert res.converged


def test_frechet_mean_weighted_polar():
    """Weighted Frechet mean on the polar plane = weighted Cartesian mean."""
    P = PolarPlane()
    pts = np.array([[2.0, 0.3], [1.2, -0.5], [1.9, 1.2]])
    w = np.array([0.5, 0.3, 0.2])
    res = frechet_mean(P, pts, weights=w, lr=0.1, n_steps=500)
    cart_mean = np.average([_to_cart(p) for p in pts], axis=0, weights=w)
    assert np.abs(_to_cart(res.point) - cart_mean).max() < 1e-8


def test_frechet_mean_single_and_repeated_points():
    """The mean of identical points is that point (exact fixed point)."""
    for manifold, p0 in [
        (PolarPlane(), [2.0, 0.8]),
        (Sphere(), [1.1, 0.6]),
        (HyperbolicPlane(), [0.3, 1.2]),
    ]:
        res = frechet_mean(manifold, [p0, p0, p0], lr=0.1, n_steps=300)
        assert geodesic_distance(manifold, res.point, p0) < 1e-9


def test_frechet_mean_sphere_two_points_is_midpoint():
    """Two non-antipodal points: the mean is the geodesic midpoint
    (d(m, q1) = d(m, q2) = d(q1, q2)/2)."""
    S = Sphere()
    q1, q2 = np.array([1.0, 0.5]), np.array([1.3, 1.1])
    res = frechet_mean(S, [q1, q2], lr=0.1, n_steps=500)
    d12 = geodesic_distance(S, q1, q2)
    assert abs(geodesic_distance(S, res.point, q1) - d12 / 2) < 1e-7
    assert abs(geodesic_distance(S, res.point, q2) - d12 / 2) < 1e-7


def test_frechet_mean_hyperbolic_two_points_is_midpoint():
    """Two points: the mean is the midpoint of the connecting geodesic."""
    H = HyperbolicPlane()
    q1, q2 = np.array([0.3, 1.2]), np.array([0.8, 1.6])
    res = frechet_mean(H, [q1, q2], lr=0.1, n_steps=500)
    d12 = geodesic_distance(H, q1, q2)
    assert abs(geodesic_distance(H, res.point, q1) - d12 / 2) < 1e-7
    assert abs(geodesic_distance(H, res.point, q2) - d12 / 2) < 1e-7


def test_frechet_mean_analytic_gradient_verified():
    """minimize verifies the analytic gradient (from log_map) against
    finite differences on every step; the report carries the worst error."""
    H = HyperbolicPlane()
    pts = rng.uniform(0.2, 1.5, (5, 2)) + np.array([0.0, 0.8])
    res = frechet_mean(H, pts, lr=0.1, n_steps=200)
    assert res.max_grad_error is not None
    assert res.max_grad_error < 1e-7


def test_wrong_analytic_gradient_raises():
    """A deliberately wrong analytic gradient is caught by the
    verification and raises (surfaced, not hidden)."""

    def wrong_grad(p):
        return np.array([1.0, 0.0])  # constant, definitely wrong

    P = PolarPlane()
    f = lambda p: (p[0] - 1.5) ** 2 + (p[1] - 0.7) ** 2
    with pytest.raises(VerificationError, match="analytic gradient"):
        minimize(P, f, [2.0, 0.3], lr=0.1, n_steps=10, grad_f=wrong_grad)


def test_geodesic_distance_stable_for_near_coincident():
    """The distance formulas are numerically stable for nearly coincident
    points (no acos/acosh cancellation)."""
    S = Sphere()
    p = np.array([1.0, 0.5])
    q = p + np.array([1e-9, 1e-9])
    d = geodesic_distance(S, p, q)
    assert 1e-12 < d < 1e-6  # accurate, not 0 and not sqrt(machine-eps)
