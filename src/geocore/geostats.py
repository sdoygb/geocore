"""Geometric statistics (the analogue of torch.mean/std on manifolds).

The Frechet mean of a point set is the point minimizing the weighted sum
of squared geodesic distances:

    m = argmin_p sum_i w_i d(p, p_i)^2

Its gradient is closed form (a standard Riemannian-geometry fact):

    grad_p d(p, q)^2 = -2 log_p(q)

so ``frechet_mean`` drives the Riemannian optimizer with the *analytic*
gradient (verified against finite differences on every step by
``minimize``), instead of numerical differentiation.  On the flat polar
plane the Frechet mean is the Euclidean (Cartesian) arithmetic mean — a
machine-checkable truth used in the tests.
"""

from __future__ import annotations

import numpy as np

from .derivatives import log_map
from .optim import minimize

__all__ = ["geodesic_distance", "frechet_mean"]


def geodesic_distance(manifold, p, q) -> float:
    """The geodesic distance between two points (closed form per manifold)."""
    name = type(manifold).__name__
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if name == "PolarPlane":
        dx = q[0] * np.cos(q[1]) - p[0] * np.cos(p[1])
        dy = q[0] * np.sin(q[1]) - p[0] * np.sin(p[1])
        return float(np.sqrt(dx * dx + dy * dy))
    if name == "Sphere":
        # numerically stable haversine form (acos(1 - eps) loses precision
        # for nearly coincident points; 2 asin(sqrt(h)) does not)
        dth = 0.5 * (p[0] - q[0])
        dph = 0.5 * (p[1] - q[1])
        h = np.sin(dth) ** 2 + np.sin(p[0]) * np.sin(q[0]) * np.sin(dph) ** 2
        return float(2.0 * np.arcsin(np.sqrt(min(1.0, h))))
    if name == "HyperbolicPlane":
        delta = ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) / (2.0 * p[1] * q[1])
        # acosh(1 + delta) = 2 asinh(sqrt(delta/2)) — stable for small delta
        return float(2.0 * np.arcsinh(np.sqrt(max(0.0, delta / 2.0))))
    raise NotImplementedError(f"geodesic_distance: unknown manifold {name}")


def _initial_guess(manifold, points, weights):
    """Weighted coordinate average — a crude but valid starting point (the
    optimizer converges from it; the average of theta stays in (0, pi),
    y > 0, r > 0 for valid inputs)."""
    return np.average(np.atleast_2d(np.asarray(points, dtype=float)), axis=0, weights=weights)


def frechet_mean(
    manifold,
    points,
    weights=None,
    lr: float = 0.1,
    n_steps: int = 300,
    atol: float = 1e-6,
):
    """The weighted Frechet mean of ``points`` on ``manifold``.

    Minimizes sum_i w_i d(p, p_i)^2 with the *analytic* gradient
    -2 sum_i w_i log_p(p_i) (verified against finite differences every
    step by ``minimize``).  Returns the ``OptimizationResult`` whose
    ``.point`` is the mean.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    N = points.shape[0]
    if weights is None:
        weights = np.ones(N) / N
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    def f(p):
        total = 0.0
        for i in range(N):
            total += weights[i] * geodesic_distance(manifold, p, points[i]) ** 2
        return total

    def grad_f(p):
        g = np.zeros(2)
        for i in range(N):
            g -= 2.0 * weights[i] * log_map(manifold, p, points[i])
        return g

    return minimize(
        manifold, f, _initial_guess(manifold, points, weights),
        lr=lr, n_steps=n_steps, atol=atol, grad_f=grad_f,
    )
