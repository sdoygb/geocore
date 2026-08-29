"""Visualization of point spread on manifolds (spread ellipses).

The tangent PCA of a point set gives principal directions in the
orthonormal tangent frame at the Frechet mean; the spread ellipse is the
image of the tangent-space ellipse (radii scale·sqrt(lambda_j) along the
principal axes) under the **exponential map** — a geometrically faithful
picture of the spread (every ellipse point satisfies
d(m, exp_m(v)) = |v|_g to machine precision).

Display charts are the most natural one per manifold: the polar plane in
Cartesian coordinates, the hyperbolic plane in the upper half-plane, and
the sphere in an azimuthal-equidistant projection centered at the mean
(where the displayed ellipse is exactly the tangent-space ellipse).

Pure presentation: the geometry is verified in tests, the figure is
decoration.
"""

from __future__ import annotations

import numpy as np

from .derivatives import log_map
from .geostats import frechet_mean, principal_directions
from .manifolds import RiemannianManifold

__all__ = ["spread_ellipse_points", "axis_endpoints", "plot_spread"]


def spread_ellipse_points(manifold, mean, evals, evecs, scale=2.0, n=64):
    """The spread ellipse on the manifold: the tangent-space ellipse with
    radii scale·sqrt(λ_j) along the principal axes, flowed through the
    exponential map.  Returns (n, 2) manifold coordinates."""
    g0, g1 = manifold.metric_diag(mean)
    sqrt_g = np.sqrt(np.array([g0, g1]))
    pts = []
    for psi in np.linspace(0.0, 2.0 * np.pi, n, endpoint=False):
        v_ortho = scale * (
            np.cos(psi) * np.sqrt(evals[1]) * evecs[:, 1]
            + np.sin(psi) * np.sqrt(evals[0]) * evecs[:, 0]
        )
        v_coord = v_ortho / sqrt_g
        pts.append(manifold.geodesic_closed_form(mean, v_coord, 1.0).point)
    return np.asarray(pts)


def axis_endpoints(manifold, mean, evals, evecs, scale=2.0):
    """The endpoints of the principal axes on the manifold (the images of
    ±scale·sqrt(λ_j)·e_j under the exponential map)."""
    g0, g1 = manifold.metric_diag(mean)
    sqrt_g = np.sqrt(np.array([g0, g1]))
    ends = []
    for j in (0, 1):
        v_ortho = scale * np.sqrt(evals[j]) * evecs[:, j]
        v_coord = v_ortho / sqrt_g
        ends.append(manifold.geodesic_closed_form(mean, v_coord, 1.0).point)
    return np.asarray(ends)


# ---------------------------------------------------------------------------
# Display charts
# ---------------------------------------------------------------------------


def _to_display(manifold, points, mean=None):
    """Manifold coordinates -> display coordinates (one point per row)."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    name = type(manifold).__name__
    if name == "PolarPlane":
        return np.column_stack(
            [points[:, 0] * np.cos(points[:, 1]), points[:, 0] * np.sin(points[:, 1])]
        )
    if name == "HyperbolicPlane":
        return points.copy()
    if name == "Sphere":
        # azimuthal equidistant projection centered at the mean
        if mean is None:
            raise ValueError("Sphere display needs the mean")
        m = np.asarray(mean, dtype=float)
        g0, g1 = manifold.metric_diag(m)
        sqrt_g = np.sqrt(np.array([g0, g1]))
        out = np.empty((len(points), 2))
        for i, p in enumerate(points):
            v = log_map(manifold, m, p) * sqrt_g  # orthonormal frame
            d = np.sqrt(v[0] ** 2 + v[1] ** 2)
            if d < 1e-12:
                out[i] = [0.0, 0.0]
            else:
                out[i] = [d * v[0] / d, d * v[1] / d]
        return out
    raise NotImplementedError(f"display chart: unknown manifold {name}")


def plot_spread(
    manifold,
    points,
    ax=None,
    scale=2.0,
    n=64,
    show_axes=True,
    mean=None,
    title=None,
):
    """Plot the point set, its Frechet mean, the principal axes and the
    spread ellipse (the exponential image of the tangent-space ellipse).

    Returns the matplotlib Axes (Agg-safe: no display needed)."""
    import matplotlib.pyplot as plt

    points = np.atleast_2d(np.asarray(points, dtype=float))
    if mean is None:
        mean = frechet_mean(manifold, points, lr=0.1, n_steps=300).point
    evals, evecs = principal_directions(manifold, points, mean=mean)

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.5))

    disp_points = _to_display(manifold, points, mean)
    disp_mean = _to_display(manifold, np.asarray(mean)[None, :], mean)[0]
    ell = _to_display(manifold, spread_ellipse_points(
        manifold, mean, evals, evecs, scale=scale, n=n), mean)

    ax.scatter(disp_points[:, 0], disp_points[:, 1], s=14, alpha=0.65,
               label="points")
    ax.plot(ell[:, 0], ell[:, 1], color="C1", lw=1.6, label="spread ellipse")
    ax.scatter(*disp_mean, marker="x", color="C1", s=90, label="Frechet mean")
    if show_axes:
        ends = _to_display(manifold, axis_endpoints(
            manifold, mean, evals, evecs, scale=scale), mean)
        for j, color in [(1, "C2"), (0, "C3")]:
            ax.plot([disp_mean[0], ends[j][0]], [disp_mean[1], ends[j][1]],
                    color=color, lw=1.4,
                    label=f"principal axis {j} (λ={evals[j]:.3f})")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title or type(manifold).__name__)
    ax.legend(fontsize=7, loc="best")
    return ax
