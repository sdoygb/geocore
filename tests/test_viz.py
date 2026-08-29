"""Tests for the spread-ellipse visualization: the exponential image of
the tangent PCA ellipse is geometrically faithful (d(m, exp_m(v)) =
|v|_g to machine precision), and the figure renders under the Agg
backend."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from geocore import (
    HyperbolicPlane,
    PolarPlane,
    Sphere,
    frechet_mean,
)
from geocore.geostats import geodesic_distance, principal_directions
from geocore.viz import axis_endpoints, plot_spread, spread_ellipse_points

rng = np.random.default_rng(23)

_CASES = [
    (PolarPlane(), (1.5, 3.0)),
    (Sphere(), (0.8, 2.2)),
    (HyperbolicPlane(), (0.4, 1.8)),
]


def _mean_and_pca(manifold, pts):
    m = frechet_mean(manifold, pts, lr=0.1, n_steps=300).point
    evals, evecs = principal_directions(manifold, pts, mean=m)
    return m, evals, evecs


def test_ellipse_points_are_exp_images():
    """Every ellipse point satisfies d(m, exp_m(v(psi))) = |v(psi)|_g to
    machine precision (the ellipse is the true spread geometry, not a
    drawing convenience)."""
    scale = 2.0
    n = 32
    psi = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    for manifold, (lo, hi) in _CASES:
        pts = rng.uniform(lo, hi, (40, 2))
        m, evals, evecs = _mean_and_pca(manifold, pts)
        ell = spread_ellipse_points(manifold, m, evals, evecs, scale=scale, n=n)
        worst = 0.0
        for i, p in enumerate(ell):
            v_ortho = scale * (
                np.cos(psi[i]) * np.sqrt(evals[1]) * evecs[:, 1]
                + np.sin(psi[i]) * np.sqrt(evals[0]) * evecs[:, 0]
            )
            vlen = np.sqrt(v_ortho[0] ** 2 + v_ortho[1] ** 2)
            worst = max(worst, abs(geodesic_distance(manifold, m, p) - vlen))
        assert worst < 1e-12, type(manifold).__name__


def test_axis_endpoints_at_sigma_radii():
    """The principal-axis endpoints lie at distance scale·sqrt(lambda_j)
    from the mean (their exp images)."""
    scale = 2.0
    for manifold, (lo, hi) in _CASES:
        pts = rng.uniform(lo, hi, (40, 2))
        m, evals, evecs = _mean_and_pca(manifold, pts)
        ends = axis_endpoints(manifold, m, evals, evecs, scale=scale)
        for j in (0, 1):
            assert abs(
                geodesic_distance(manifold, m, ends[j]) - scale * np.sqrt(evals[j])
            ) < 1e-12, (type(manifold).__name__, j)


def test_plot_spread_returns_axes_and_saves():
    """plot_spread renders under Agg and returns the Axes; the PNG saves
    (the figure is decoration, the geometry is verified above)."""
    import matplotlib.pyplot as plt
    import os
    import tempfile

    P = PolarPlane()
    pts = rng.uniform(1.5, 3.0, (30, 2))
    fig, ax = plt.subplots(figsize=(4, 4))
    returned = plot_spread(P, pts, ax=ax, title="test")
    assert returned is ax
    assert len(ax.collections) >= 1  # scatter
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "spread.png")
        fig.savefig(path)
        assert os.path.getsize(path) > 1000


def test_ellipse_encloses_most_points():
    """The 2-sigma ellipse encloses the bulk of the point set (a sanity
    check that the PCA ellipse matches the actual spread)."""
    P = PolarPlane()
    pts = rng.uniform(1.5, 3.0, (60, 2))
    m, evals, evecs = _mean_and_pca(P, pts)
    ell = spread_ellipse_points(P, m, evals, evecs, scale=2.0, n=64)
    # inside-test via angular sweep is fiddly; use the tangent norm:
    # a point is inside the 2-sigma ellipse iff |log_m(p)|^T Cov^-1 |log_m(p)| < 4
    from geocore.derivatives import log_map
    from geocore.geostats import tangent_covariance

    cov = tangent_covariance(P, pts, mean=m)
    cov_inv = np.linalg.inv(cov)
    g0, g1 = P.metric_diag(m)
    sqrt_g = np.sqrt(np.array([g0, g1]))
    inside = 0
    for p in pts:
        v = log_map(P, m, p) * sqrt_g
        if v @ cov_inv @ v < 4.0:
            inside += 1
    assert inside / len(pts) > 0.6
