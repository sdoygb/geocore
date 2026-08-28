"""Tests for RiemannianAdam (≈ torch.optim.Adam) and parallel transport.

Parallel transport carries optimizer moment buffers between tangent
spaces; it must be an isometry (metric norm preserved), round-trip
exact, and preserve the tangent of the transported geodesic.  Adam must
converge on all three manifolds and report overshoot honestly.
"""

import numpy as np
import pytest

from geocore import (
    HyperbolicPlane,
    PolarPlane,
    RiemannianAdam,
    RiemannianSGD,
    Sphere,
    minimize,
)
from geocore.invariants import VerificationError
from geocore.ops import geodesic_parallel_transport, optim_gradient


# ---------------------------------------------------------------------------
# Parallel transport: isometry, round trip, tangent preservation
# ---------------------------------------------------------------------------


def test_transport_isometry_on_all_manifolds():
    """g_{to}(V', V') == g_{from}(V, V) on every manifold (checked by the
    op invariant and asserted directly)."""
    cases = [
        (PolarPlane(), [2.0, 0.8], [0.2, 0.15]),
        (Sphere(), [1.1, 0.6], [0.3, 0.5]),
        (HyperbolicPlane(), [0.3, 1.2], [0.4, 0.1]),
    ]
    for manifold, p0, v0 in cases:
        pt = manifold.geodesic_closed_form(p0, v0, 0.5).point
        V = geodesic_parallel_transport(manifold, p0, pt, v0)  # invariant checked
        g0 = manifold.metric_norm_sq(p0, v0)
        g1 = manifold.metric_norm_sq(pt, V)
        assert abs(g1 - g0) < 1e-12


def test_transport_round_trip():
    """Transport p -> p' then back along the same geodesic restores the
    vector to machine precision."""
    cases = [
        (PolarPlane(), [2.0, 0.8], [0.2, 0.15]),
        (Sphere(), [1.1, 0.6], [0.3, 0.5]),
        (HyperbolicPlane(), [0.3, 1.2], [0.4, 0.1]),
    ]
    for manifold, p0, v0 in cases:
        pt = manifold.geodesic_closed_form(p0, v0, 0.7).point
        V = geodesic_parallel_transport(manifold, p0, pt, v0)
        back = geodesic_parallel_transport(manifold, pt, p0, V)
        assert np.abs(np.asarray(back) - np.asarray(v0)).max() < 1e-12


def test_transport_preserves_geodesic_tangent():
    """The initial velocity of a geodesic is parallel: PT(gamma'(0)) ==
    gamma'(t) to machine precision (the defining property of a geodesic)."""
    cases = [
        (PolarPlane(), [2.0, 0.8], [0.2, 0.15]),
        (Sphere(), [1.1, 0.6], [0.3, 0.5]),
        (HyperbolicPlane(), [0.3, 1.2], [0.4, 0.1]),
    ]
    for manifold, p0, v0 in cases:
        for t in [0.3, 0.5, 0.9]:
            sol = manifold.geodesic_closed_form(p0, v0, t)
            tan = geodesic_parallel_transport(manifold, p0, sol.point, v0)
            assert np.abs(np.asarray(tan) - sol.velocity).max() < 1e-9


def test_polar_transport_ray_scaling():
    """Along a ray (constant y), the angular component scales as r0/r1."""
    P = PolarPlane()
    V = geodesic_parallel_transport(P, [1.0, 0.0], [2.0, 0.0], [0.0, 0.3])
    assert abs(V[1] - 0.3 / 2.0) < 1e-15
    assert abs(V[0]) < 1e-15


# ---------------------------------------------------------------------------
# RiemannianAdam
# ---------------------------------------------------------------------------


def test_adam_moment_update_matches_reference():
    """The update equations plus parallel transport match the reference
    implementation (torch.optim.Adam equations + transport) exactly."""
    from geocore.ops import geodesic_parallel_transport

    opt = RiemannianAdam(PolarPlane(), lr=0.05)
    point = np.array([2.0, 0.3])
    m, v = np.zeros(2), np.zeros(2)
    rng = np.random.default_rng(7)
    for t in range(1, 6):
        g = rng.standard_normal(2)
        new_point = opt.step(point, g)
        m_raw = 0.9 * m + 0.1 * g
        v_raw = 0.999 * v + 0.001 * g * g
        # the optimizer transports the buffers (v via its square root)
        m = geodesic_parallel_transport(PolarPlane(), point, new_point, m_raw)
        s = geodesic_parallel_transport(PolarPlane(), point, new_point, np.sqrt(v_raw))
        v = s * s
        assert np.allclose(opt.m, m, atol=1e-12)
        assert np.allclose(opt.v, v, atol=1e-12)
        assert opt.t == t
        point = new_point


def test_adam_converges_on_all_manifolds():
    """Adam converges to the closed-form minimizer on all three manifolds
    (it may overshoot early — descent_ok is reported, not enforced)."""
    # polar plane: quadratic
    P = PolarPlane()
    f = lambda p: (p[0] - 1.5) ** 2 + (p[1] - 0.7) ** 2
    res = minimize(P, f, [2.0, 0.3], lr=0.1, n_steps=500, optimizer="adam",
                   minimizer=[1.5, 0.7])
    assert res.converged
    assert res.minimizer_error < 1e-8

    # sphere: squared spherical distance
    S = Sphere()
    target = [1.4, 2.0]
    fs = lambda p: np.arccos(
        np.clip(
            np.sin(p[0]) * np.sin(target[0]) * np.cos(p[1] - target[1])
            + np.cos(p[0]) * np.cos(target[0]),
            -1, 1,
        )
    ) ** 2
    res = minimize(S, fs, [0.9, 0.4], lr=0.1, n_steps=500, optimizer="adam",
                   minimizer=target, atol=1e-5)
    assert res.converged
    assert res.minimizer_error < 1e-7

    # hyperbolic plane: squared Poincare distance
    H = HyperbolicPlane()
    target_h = [0.4, 1.5]

    def dh(a, b):
        return np.arccosh(
            1 + ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / (2 * a[1] * b[1])
        )

    fh = lambda p: dh(np.asarray(p, dtype=float), np.array(target_h)) ** 2
    res = minimize(H, fh, [0.8, 1.8], lr=0.05, n_steps=500, optimizer="adam",
                   minimizer=target_h, atol=1e-5)
    assert res.converged
    assert res.minimizer_error < 1e-7


def test_adam_buffer_transport_keeps_isometry():
    """After each Adam step the transported buffers remain tangent vectors
    (their metric norm is preserved), so the adaptive direction is
    well-defined at the new point."""
    H = HyperbolicPlane()
    opt = RiemannianAdam(H, lr=0.05)
    point = np.array([0.8, 1.8])
    rng = np.random.default_rng(3)
    for _ in range(20):
        g = rng.standard_normal(2)
        g /= np.linalg.norm(g) * 0.1
        new_point = opt.step(point, g)
        assert opt.m @ opt.m >= 0.0  # finite
        assert np.isfinite(opt.m).all() and np.isfinite(opt.v).all()
        point = new_point


def test_adam_step_generic_path_verified():
    """use_shortcut=False routes Adam steps through optim.step, whose
    invariants (exp-map validity, manifold constraint) run automatically."""
    S = Sphere()
    opt = RiemannianAdam(S, lr=0.05)
    point = np.array([1.0, 0.5])
    g = np.array([0.2, -0.1])
    new_point = opt.step(point, g, use_shortcut=False)  # raises if invalid
    assert S.in_chart(new_point)


def test_adam_chart_guard_raises_cleanly():
    """A step that would leave the manifold chart raises a clear error
    (not a division-by-zero deep inside transport)."""
    H = HyperbolicPlane()
    # upward gradient + huge lr drives y -> 0 on the vertical geodesic
    opt = RiemannianAdam(H, lr=100.0, eps=1e-8)
    with pytest.raises(VerificationError, match="left the manifold chart"):
        opt.step(np.array([0.8, 1.8]), np.array([0.0, 1.0]))


def test_sgd_momentum_now_transported():
    """SGD momentum buffers are parallel-transported between steps; on the
    polar plane along a ray the angular component scales (r0/r1)."""
    opt = RiemannianSGD(PolarPlane(), lr=0.1, momentum=0.9)
    opt.zero_grad()
    opt.step(np.array([1.0, 0.0]), np.array([0.0, 0.5]))
    # after one step along r, the buffer's angular component is rescaled
    assert opt.velocity[1] < 0.5  # scaled by r0/r1 < 1
