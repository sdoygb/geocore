"""Tests for vectorized/batched core paths (the analogue of batched
tensor ops / vmap): batch geodesics (RK4 + closed form) and batch
parallel transport must equal the per-point paths to machine precision,
and the batch closed-form shortcut must be measured faster."""

import numpy as np
import pytest

from geocore import (
    HyperbolicPlane,
    PolarPlane,
    Sphere,
    minimize,
    minimize_batch,
)
from geocore.invariants import VerificationError
from geocore.ops import geodesic_batch
from geocore.shortcuts import geodesic_batch_closed_form

rng = np.random.default_rng(11)

_MANIFOLDS = [
    (PolarPlane(), (1.2, 3.0)),
    (Sphere(), (0.5, 2.5)),
    (HyperbolicPlane(), (0.2, 1.5)),
]


def _polar_cart_compare(generic, fast):
    """Geometric comparison in Cartesian coordinates (handles the periodic
    polar angle: y and y + 2pi are the same point)."""
    g = np.atleast_2d(np.asarray(generic, float))
    f = np.atleast_2d(np.asarray(fast, float))
    gx = np.column_stack([g[:, 0] * np.cos(g[:, 1]), g[:, 0] * np.sin(g[:, 1])])
    fx = np.column_stack([f[:, 0] * np.cos(f[:, 1]), f[:, 0] * np.sin(f[:, 1])])
    return np.abs(gx - fx).max()


def _batch_compare(manifold):
    return _polar_cart_compare if isinstance(manifold, PolarPlane) else None


# ---------------------------------------------------------------------------
# Batch geodesics
# ---------------------------------------------------------------------------


def test_batch_op_matches_vectorized_rk4():
    """geodesic.batch (per-point loop) agrees with the vectorized RK4 path
    to machine precision (the BatchConsistency invariant runs on the op)."""
    for manifold, (lo, hi) in _MANIFOLDS:
        init = rng.uniform(lo, hi, (40, 2))
        vel = rng.uniform(-0.3, 0.3, (40, 2))
        t = rng.uniform(0.1, 0.9, 40)
        pts = geodesic_batch(manifold, init, vel, t)  # raises if inconsistent
        assert pts.shape == (40, 2)


def test_batch_closed_form_matches_per_point():
    """The vectorized closed form equals the per-point generic path to
    machine precision (RK4 error), on every manifold."""
    for manifold, (lo, hi) in _MANIFOLDS:
        init = rng.uniform(lo, hi, (40, 2))
        vel = rng.uniform(-0.3, 0.3, (40, 2))
        t = rng.uniform(0.1, 0.9, 40)
        rep = geodesic_batch_closed_form.verify_against(
            manifold, init, vel, t, compare=_batch_compare(manifold)
        )
        assert rep.ok, (type(manifold).__name__, rep.details)
        assert rep.max_error < 1e-8


def test_batch_closed_form_matches_scalar_closed_form():
    """Vectorized closed form == per-point closed form, exactly (same
    formula, batched)."""
    for manifold, (lo, hi) in _MANIFOLDS:
        init = rng.uniform(lo, hi, (30, 2))
        vel = rng.uniform(-0.3, 0.3, (30, 2))
        t = rng.uniform(0.1, 0.9, 30)
        pb, _ = manifold.geodesic_closed_form_batch(init, vel, t)
        err = 0.0
        for i in range(30):
            sol = manifold.geodesic_closed_form(init[i], vel[i], float(t[i]))
            err = max(err, float(np.abs(pb[i] - sol.point).max()))
        assert err < 1e-14


def test_batch_rk4_matches_scalar_rk4():
    """Vectorized RK4 == per-point RK4, exactly (0.0 — same arithmetic)."""
    for manifold, (lo, hi) in _MANIFOLDS:
        init = rng.uniform(lo, hi, (30, 2))
        vel = rng.uniform(-0.3, 0.3, (30, 2))
        t = rng.uniform(0.1, 0.9, 30)
        pb, _ = manifold.geodesic_generic_batch(init, vel, t)
        err = 0.0
        for i in range(30):
            sol = manifold.geodesic_generic(init[i], vel[i], float(t[i]))
            err = max(err, float(np.abs(pb[i] - sol.point).max()))
        assert err < 1e-12


def test_batch_energy_conservation():
    """Energy is conserved per point along vectorized closed-form
    geodesics (the batch path preserves the Levi-Civita invariant)."""
    for manifold, (lo, hi) in _MANIFOLDS:
        init = rng.uniform(lo, hi, (30, 2))
        vel = rng.uniform(-0.3, 0.3, (30, 2))
        t = rng.uniform(0.1, 0.9, 30)
        pb, vb = manifold.geodesic_closed_form_batch(init, vel, t)
        e0 = manifold.metric_norm_sq(init, vel)
        e1 = manifold.metric_norm_sq(pb, vb)
        assert np.abs(e1 - e0).max() < 1e-12


def test_batch_transport_matches_scalar():
    """Vectorized parallel transport == per-point transport on every
    manifold."""
    for manifold, (lo, hi) in _MANIFOLDS:
        pf = rng.uniform(lo, hi, (30, 2))
        pt = rng.uniform(lo, hi, (30, 2))
        v = rng.uniform(-0.5, 0.5, (30, 2))
        vb = manifold.parallel_transport_batch(pf, pt, v)
        for i in range(30):
            assert np.abs(
                manifold.parallel_transport(pf[i], pt[i], v[i]) - vb[i]
            ).max() < 1e-14


def test_batch_transport_isometry():
    """Batch transport is an isometry per point: metric norm preserved."""
    for manifold, (lo, hi) in _MANIFOLDS:
        pf = rng.uniform(lo, hi, (30, 2))
        pt = rng.uniform(lo, hi, (30, 2))
        v = rng.uniform(-0.5, 0.5, (30, 2))
        vb = manifold.parallel_transport_batch(pf, pt, v)
        g0 = manifold.metric_norm_sq(pf, v)
        g1 = manifold.metric_norm_sq(pt, vb)
        assert np.abs(g1 - g0).max() < 1e-12


# ---------------------------------------------------------------------------
# Batch benchmark (measured claims only)
# ---------------------------------------------------------------------------


def test_batch_shortcut_measured_speedup():
    """The vectorized closed-form batch geodesic is measured faster than
    the per-point loop (the whole point of the batch core path)."""
    manifold = HyperbolicPlane()
    init = rng.uniform(0.2, 1.5, (200, 2))
    vel = rng.uniform(-0.3, 0.3, (200, 2))
    t = rng.uniform(0.1, 0.9, 200)
    log = geodesic_batch_closed_form.profile(
        manifold, init, vel, t, n_trials=3,
        size_of=lambda *a: np.atleast_2d(a[1]).shape[0],
    )
    assert log.speedup_flops > 10.0  # analytic: B*1200 vs B*40
    assert log.speedup_time > 100.0  # Python loop vs vectorized numpy


# ---------------------------------------------------------------------------
# Batch optimizer
# ---------------------------------------------------------------------------


def test_minimize_batch_matches_per_point():
    """minimize_batch (vectorized) equals running minimize per starting
    point — same closed-form steps, same result."""
    P = PolarPlane()
    f_scalar = lambda p: (p[0] - 1.5) ** 2 + (p[1] - 0.7) ** 2
    f_vec = lambda p: (p[..., 0] - 1.5) ** 2 + (p[..., 1] - 0.7) ** 2
    starts = np.array([[2.0, 0.3], [1.2, -0.5], [1.9, 1.2], [1.1, 0.9]])
    res = minimize_batch(P, f_vec, starts, lr=0.05, n_steps=500,
                         minimizer=[1.5, 0.7])
    assert res.converged.all()
    assert res.descent_ok
    for i, p0 in enumerate(starts):
        r = minimize(P, f_scalar, p0, lr=0.05, n_steps=500, minimizer=[1.5, 0.7])
        assert np.linalg.norm(res.points[i] - r.point) < 1e-9


def test_minimize_batch_sphere():
    """Batch optimizer converges on the sphere to the target (angular
    distance ~ 0)."""
    S = Sphere()
    target = np.array([1.4, 2.0])
    fs = lambda p: np.arccos(
        np.clip(
            np.sin(p[..., 0]) * np.sin(target[0]) * np.cos(p[..., 1] - target[1])
            + np.cos(p[..., 0]) * np.cos(target[0]),
            -1, 1,
        )
    ) ** 2
    starts = np.array([[0.9, 0.4], [1.8, 1.0], [1.0, 3.0]])
    res = minimize_batch(S, fs, starts, lr=0.3, n_steps=500,
                         minimizer=target, atol=1e-5)
    assert res.converged.all()
    assert np.all(res.minimizer_error < 1e-7)


def test_minimize_batch_chart_guard():
    """A batch step leaving the chart raises a clear error."""
    H = HyperbolicPlane()
    f = lambda p: p[..., 1] ** 2  # gradient points down: y -> 0 (boundary)
    starts = np.array([[0.5, 1.0], [0.5, 0.02]])  # one point near y ~ 0
    with pytest.raises(VerificationError, match="left the manifold chart"):
        minimize_batch(H, f, starts, lr=100.0, n_steps=10)
