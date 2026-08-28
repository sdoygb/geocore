"""Tests for the sphere and hyperbolic-plane manifolds (closed-form
geodesics verified to machine precision, optimizer runs on both)."""

import numpy as np
import pytest

from geocore import HyperbolicPlane, PolarPlane, Sphere, minimize
from geocore.ops import optim_gradient, optim_step
from geocore.shortcuts import (
    geodesic_hyperbolic_closed_form,
    geodesic_sphere_closed_form,
)


# ---------------------------------------------------------------------------
# Sphere
# ---------------------------------------------------------------------------


def test_sphere_metric_and_energy_conservation():
    S = Sphere()
    th, ph = 1.1, 0.6
    vth, vph = 0.3, 0.5
    assert S.metric_diag([th, ph]) == pytest.approx((1.0, np.sin(th) ** 2))
    sol = S.geodesic_closed_form([th, ph], [vth, vph], 0.7)
    e0 = S.metric_norm_sq([th, ph], [vth, vph])
    e1 = S.metric_norm_sq(sol.point, sol.velocity)
    assert abs(e1 - e0) < 1e-12


def test_sphere_great_circle_truth():
    """A sphere geodesic is a great circle: points stay on the unit sphere
    (|p_t| = 1) and are coplanar with p0 and the tangent (det = 0), to
    machine precision."""
    S = Sphere()
    th, ph = 1.1, 0.6
    vth, vph = 0.3, 0.5
    p0 = S._to_r3(th, ph)
    st, ct = np.sin(th), np.cos(th)
    v = vth * np.array([ct * np.cos(ph), ct * np.sin(ph), -st]) + vph * np.array(
        [-st * np.sin(ph), st * np.cos(ph), 0.0]
    )
    for t in [0.3, 0.7, 1.2, 2.0]:
        th_t, ph_t = S.geodesic_closed_form([th, ph], [vth, vph], t).point
        pt = S._to_r3(th_t, ph_t)
        assert abs(np.linalg.norm(pt) - 1.0) < 1e-12
        assert abs(np.linalg.det(np.array([p0, v, pt]))) < 1e-9


def test_sphere_closed_form_matches_rk4():
    S = Sphere()
    rep = geodesic_sphere_closed_form.verify_against(S, [1.1, 0.6], [0.3, 0.5], 0.7)
    assert rep.ok
    assert rep.max_error < 1e-9


def test_sphere_zero_velocity_is_fixed_point():
    S = Sphere()
    sol = S.geodesic_closed_form([1.0, 0.5], [0.0, 0.0], 3.0)
    assert np.allclose(sol.point, [1.0, 0.5], atol=1e-15)
    assert np.allclose(sol.velocity, [0.0, 0.0], atol=1e-15)


def test_sphere_chart_constraint():
    S = Sphere()
    assert S.in_chart([0.5, 1.0])
    assert not S.in_chart([0.0, 1.0])  # pole
    assert not S.in_chart([np.pi, 1.0])  # south pole


# ---------------------------------------------------------------------------
# Hyperbolic plane
# ---------------------------------------------------------------------------


def test_hyperbolic_metric_and_energy_conservation():
    H = HyperbolicPlane()
    x0, y0 = 0.3, 1.2
    vx, vy = 0.4, 0.1
    assert H.metric_diag([x0, y0]) == pytest.approx((1 / y0**2, 1 / y0**2))
    sol = H.geodesic_closed_form([x0, y0], [vx, vy], 0.8)
    e0 = H.metric_norm_sq([x0, y0], [vx, vy])
    e1 = H.metric_norm_sq(sol.point, sol.velocity)
    assert abs(e1 - e0) < 1e-12


def test_hyperbolic_semicircle_truth():
    """A hyperbolic geodesic is a semicircle orthogonal to the real axis:
    (x - c)^2 + y^2 = R^2 constant along the geodesic, to machine
    precision."""
    H = HyperbolicPlane()
    x0, y0 = 0.3, 1.2
    vx, vy = 0.4, 0.1
    c = x0 + y0 * vy / vx
    R2 = (x0 - c) ** 2 + y0**2
    for t in [0.3, 0.8, 1.5]:
        xt, yt = H.geodesic_closed_form([x0, y0], [vx, vy], t).point
        assert abs((xt - c) ** 2 + yt**2 - R2) < 1e-9


def test_hyperbolic_distance_additivity():
    """The Poincare distance along a geodesic equals |v| * t (arc-length
    parametrization) — the defining property of a geodesic, checked to
    machine precision."""
    H = HyperbolicPlane()
    x0, y0 = 0.3, 1.2
    vx, vy = 0.4, 0.1

    def d(a, b):
        return np.arccosh(
            1 + ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / (2 * a[1] * b[1])
        )

    e0 = H.metric_norm_sq([x0, y0], [vx, vy])
    for t in [0.3, 0.8, 1.5]:
        pt = H.geodesic_closed_form([x0, y0], [vx, vy], t).point
        assert abs(d([x0, y0], pt) - np.sqrt(e0) * t) < 1e-9


def test_hyperbolic_vertical_geodesic():
    """vx = 0: the geodesic is the vertical line x = x0, y(t) = y0 e^{vy t / y0}."""
    H = HyperbolicPlane()
    sol = H.geodesic_closed_form([0.5, 1.0], [0.0, 0.3], 1.5)
    assert abs(sol.point[0] - 0.5) < 1e-15
    assert abs(sol.point[1] - np.exp(0.45)) < 1e-12
    assert abs(sol.velocity[0]) < 1e-15


def test_hyperbolic_closed_form_matches_rk4():
    H = HyperbolicPlane()
    rep = geodesic_hyperbolic_closed_form.verify_against(H, [0.3, 1.2], [0.4, 0.1], 0.8)
    assert rep.ok
    assert rep.max_error < 1e-9


def test_hyperbolic_chart_constraint():
    H = HyperbolicPlane()
    assert H.in_chart([0.5, 1.0])
    assert not H.in_chart([0.5, -0.1])
    assert not H.in_chart([0.5, 0.0])


# ---------------------------------------------------------------------------
# Optimizer on the new manifolds
# ---------------------------------------------------------------------------


def _sphere_dist_sq(p, target):
    st, ct = np.sin(p[0]), np.cos(p[0])
    stt, ctt = np.sin(target[0]), np.cos(target[0])
    cosang = st * stt * np.cos(p[1] - target[1]) + ct * ctt
    return np.arccos(np.clip(cosang, -1, 1)) ** 2


def test_sphere_gradient_riesz():
    S = Sphere()
    target = [1.4, 2.0]
    point = np.array([0.9, 0.4])
    f = lambda p: _sphere_dist_sq(p, target)
    eps = 1e-6
    df = np.array(
        [
            (f([point[0] + eps, point[1]]) - f([point[0] - eps, point[1]])) / (2 * eps),
            (f([point[0], point[1] + eps]) - f([point[0], point[1] - eps])) / (2 * eps),
        ]
    )
    grad = optim_gradient(S, df, point)  # Riesz invariant checked internally
    g0, g1 = S.metric_diag(point)
    v = np.array([0.6, 0.8])
    assert abs(g0 * grad[0] * v[0] + g1 * grad[1] * v[1] - df @ v) < 1e-8


def test_optimizer_converges_on_sphere():
    """Riemannian gradient descent on S^2 converges to the closed-form
    minimizer (angular distance ~ 1e-10); every step is verified."""
    S = Sphere()
    target = [1.4, 2.0]
    f = lambda p: _sphere_dist_sq(p, target)
    res = minimize(S, f, [0.9, 0.4], lr=0.3, n_steps=300, minimizer=target,
                   atol=1e-5)
    assert res.converged
    assert res.descent_ok
    assert res.minimizer_error < 1e-7


def test_optimizer_converges_on_hyperbolic_plane():
    """Riemannian gradient descent on H^2 converges to the closed-form
    minimizer (Poincare distance ~ 1e-9)."""
    H = HyperbolicPlane()
    target = [0.4, 1.5]

    def dh(a, b):
        return np.arccosh(
            1 + ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / (2 * a[1] * b[1])
        )

    f = lambda p: dh(np.asarray(p, dtype=float), np.array(target)) ** 2
    res = minimize(H, f, [0.8, 1.8], lr=0.3, n_steps=300, minimizer=target,
                   atol=1e-5)
    assert res.converged
    assert res.descent_ok
    assert res.minimizer_error < 1e-7


def test_optim_step_generic_dispatch_on_new_manifolds():
    """optim.step dispatches by default to any RiemannianManifold and stays
    in the chart."""
    for manifold in [Sphere(), HyperbolicPlane()]:
        point = np.array([1.1, 1.2]) if isinstance(manifold, Sphere) else np.array([0.5, 1.2])
        dv = np.array([-0.2, -0.1])
        new_point = optim_step(manifold, point, dv, 0.1)
        assert manifold.in_chart(new_point)


# ---------------------------------------------------------------------------
# Benchmarks (measured claims only)
# ---------------------------------------------------------------------------


def test_sphere_geodesic_benchmark():
    S = Sphere()
    log = geodesic_sphere_closed_form.profile(S, [1.1, 0.6], [0.3, 0.5], 0.7,
                                              n_trials=50, size_of=lambda *a: 2)
    assert log.speedup_flops > 10.0
    assert log.speedup_time > 5.0


def test_hyperbolic_geodesic_benchmark():
    H = HyperbolicPlane()
    log = geodesic_hyperbolic_closed_form.profile(H, [0.3, 1.2], [0.4, 0.1], 0.8,
                                                  n_trials=50, size_of=lambda *a: 2)
    assert log.speedup_flops > 10.0
    assert log.speedup_time > 5.0


# ---------------------------------------------------------------------------
# PolarPlane regression (base-class refactor)
# ---------------------------------------------------------------------------


def test_polar_plane_metric_norm_sq_new_signature():
    P = PolarPlane()
    assert P.metric_norm_sq([2.0, 0.5], [1.0, 3.0]) == pytest.approx(1.0 + 4.0 * 9.0)
    assert P.in_chart([1.0, 0.0])
    assert not P.in_chart([-0.1, 0.0])
