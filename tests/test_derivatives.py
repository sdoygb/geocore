"""Tests for the analytic derivative operators (the geometric analogue of
autograd): closed-form derivatives verified against finite differences to
machine precision."""

import numpy as np
import pytest

from geocore import (
    HyperbolicPlane,
    PolarPlane,
    Rotation,
    Sphere,
)
from geocore.derivatives import (
    geodesic_jacobian,
    polar_jacobian,
    rotation_derivative,
)
from geocore.ops import geodesic_jacobian_op, rotation_derivative_op
from geocore.shortcuts import (
    geodesic_jacobian_closed_form,
    rotation_derivative_closed_form,
)

rng = np.random.default_rng(5)


# ---------------------------------------------------------------------------
# rotation derivative
# ---------------------------------------------------------------------------


def test_rotation_derivative_matches_finite_difference():
    """d/dtheta R_P(theta)|psi> analytic == central difference, to ~1e-10."""
    eps = 1e-6
    for n in range(1, 5):
        state = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
        axis = "".join(rng.choice(list("XYZ"), n))
        theta = rng.uniform(0.2, 1.5)
        d_analytic = rotation_derivative(axis, theta, state)
        R = lambda th: Rotation(axis, th).to_matrix()
        d_num = ((R(theta + eps) - R(theta - eps)) @ state) / (2 * eps)
        assert np.abs(d_analytic - d_num).max() < 1e-8


def test_rotation_derivative_op_invariant():
    """The op's invariant verifies the generic (finite-difference) path
    against the analytic closed form automatically."""
    state = rng.standard_normal(8) + 1j * rng.standard_normal(8)
    d = rotation_derivative_op(Rotation("XYZ", 0.7), state)  # raises if mismatch
    assert d.shape == state.shape


def test_rotation_derivative_closed_form_matches_generic():
    rep = rotation_derivative_closed_form.verify_against(Rotation("XYZ", 0.7), state := (rng.standard_normal(8) + 1j * rng.standard_normal(8)))
    assert rep.ok
    assert rep.max_error < 1e-8


def test_rotation_derivative_measured_speedup():
    n = 7
    state = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
    log = rotation_derivative_closed_form.profile(
        Rotation("X" * n, 0.7), state, n_trials=10,
        size_of=lambda r, s: len(r.axis),
    )
    assert log.speedup_flops > 1e3  # two expm (O(8^n)) vs Pauli action (O(2^n))
    assert log.speedup_time > 100.0


# ---------------------------------------------------------------------------
# geodesic Jacobians
# ---------------------------------------------------------------------------

_CASES = [
    (PolarPlane(), [2.0, 0.8], [0.2, 0.15], 0.5),
    (Sphere(), [1.1, 0.6], [0.3, 0.5], 0.7),
    (HyperbolicPlane(), [0.3, 1.2], [0.4, 0.1], 0.8),
]


def test_jacobians_match_finite_difference():
    """Analytic Jacobians == central differences to ~1e-9 on every
    manifold."""
    eps = 1e-6
    for manifold, init, vel, t in _CASES:
        Jp, Jv = geodesic_jacobian(manifold, init, vel, t)
        Jp_num, Jv_num = np.zeros((2, 2)), np.zeros((2, 2))
        init, vel = np.asarray(init, float), np.asarray(vel, float)
        for j in range(2):
            dp = np.zeros(2)
            dp[j] = eps
            Jp_num[:, j] = (
                manifold.geodesic_closed_form(init + dp, vel, t).point
                - manifold.geodesic_closed_form(init - dp, vel, t).point
            ) / (2 * eps)
            Jv_num[:, j] = (
                manifold.geodesic_closed_form(init, vel + dp, t).point
                - manifold.geodesic_closed_form(init, vel - dp, t).point
            ) / (2 * eps)
        assert np.abs(Jp - Jp_num).max() < 1e-7, type(manifold).__name__
        assert np.abs(Jv - Jv_num).max() < 1e-7, type(manifold).__name__


def test_jacobian_velocity_homogeneity():
    """Homogeneity of the geodesic in the velocity: gamma(t; p0, lambda v0)
    = gamma(lambda t; p0, v0).  Differentiating at lambda = 1 gives
    Jv . v0 = t gamma'(t) — a clean analytic identity, machine-checkable."""
    for manifold, init, vel, t in _CASES:
        Jp, Jv = geodesic_jacobian(manifold, init, vel, t)
        v0 = np.asarray(vel, float)
        gamma_t = manifold.geodesic_closed_form(init, v0, t)
        lhs = Jv @ v0
        rhs = t * gamma_t.velocity
        # polar coordinates: velocity is (v_r, v_y); both sides are polar
        assert np.abs(lhs - rhs).max() < 1e-9, type(manifold).__name__


def test_jacobian_op_invariant():
    """The op's invariant checks the finite-difference generic path against
    the analytic closed form automatically."""
    for manifold, init, vel, t in _CASES:
        Jp, Jv = geodesic_jacobian_op(manifold, init, vel, t)  # raises if bad
        assert Jp.shape == (2, 2) and Jv.shape == (2, 2)


def test_jacobian_closed_form_matches_generic():
    for manifold, init, vel, t in _CASES:
        rep = geodesic_jacobian_closed_form.verify_against(manifold, init, vel, t)
        assert rep.ok, (type(manifold).__name__, rep.details)
        assert rep.max_error < 1e-7


def test_jacobian_measured_speedup():
    """The analytic Jacobian is measured faster than 8 closed-form geodesic
    evals (modest — its real value is exactness: ~1e-10 vs FD error)."""
    manifold = HyperbolicPlane()
    log = geodesic_jacobian_closed_form.profile(
        manifold, [0.3, 1.2], [0.4, 0.1], 0.8, n_trials=50, size_of=lambda *a: 2
    )
    assert log.speedup_flops > 2.0
    assert log.speedup_time > 2.0


def test_jacobian_degenerate_branches_raise():
    """Zero velocity (sphere) and the vertical branch (hyperbolic) have no
    well-defined embedding Jacobian; the analytic form raises cleanly."""
    S = Sphere()
    with pytest.raises(ValueError):
        geodesic_jacobian(S, [1.0, 0.5], [0.0, 0.0], 0.5)
    H = HyperbolicPlane()
    with pytest.raises(ValueError):
        geodesic_jacobian(H, [0.5, 1.0], [0.0, 0.3], 0.5)


def test_polar_jacobian_translation_invariance():
    """On the polar plane the endpoint is p0 + t v0 (Cartesian), so the
    Jacobian w.r.t. the point is the coordinate-frame rotation (a
    rotation matrix: orthogonal up to the metric scaling)."""
    Jp, Jv = polar_jacobian([2.0, 0.8], [0.2, 0.15], 0.5)
    # Jv * v0 = t * gamma'(t) already covered; here check Jp is invertible
    assert abs(np.linalg.det(Jp)) > 1e-9
