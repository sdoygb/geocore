"""Layer 0 — the hyperbolic plane in the upper-half-plane model.

Metric: ds^2 = (dx^2 + dy^2) / y^2  (y > 0).

  - Christoffels: Γ^x_xy = Γ^x_yx = -1/y, Γ^y_xx = 1/y, Γ^y_yy = -1/y
  - geodesic ODE (generic path):
        x'' = 2 x' y' / y
        y'' = ((y')^2 - (x')^2) / y
  - closed form (Layer-3 shortcut): geodesics are semicircles orthogonal
    to the real axis (or vertical lines when x' = 0).  A semicircle of
    center (c, 0) and radius R is parameterized by
        x(t) = c + R tanh(α t + β),   y(t) = R / cosh(α t + β)
    which satisfies (x - c)^2 + y^2 = R^2 and g(v, v) = α^2 (constant).
"""

from __future__ import annotations

import numpy as np

from .manifolds import GeodesicSolution, RiemannianManifold

__all__ = ["HyperbolicPlane"]


class HyperbolicPlane(RiemannianManifold):
    """The hyperbolic plane H^2, ds^2 = (dx^2 + dy^2) / y^2 (y > 0)."""

    def metric_diag(self, point) -> tuple[float, float]:
        y = float(point[1])
        inv = 1.0 / (y * y)
        return (inv, inv)

    def in_chart(self, point) -> bool:
        return float(point[1]) > 1e-12

    def geodesic_ode(self, state):
        """state = (x, y, x', y') -> d/dt(state)."""
        _x, y, v_x, v_y = state
        return np.array([v_x, v_y, 2.0 * v_x * v_y / y, (v_y * v_y - v_x * v_x) / y])

    def geodesic_closed_form(self, initial, velocity, t):
        """Closed form: a semicircle orthogonal to the real axis (or a
        vertical line when vx = 0).  Exact up to floating point.

        From (x0, y0) with velocity (vx, vy), the circle center is
        c = x0 + y0 vy / vx (the tangent is perpendicular to the radius),
        the radius R = sqrt((x0 - c)^2 + y0^2), and the parameters
        α = vx R / y0^2 (the g-norm of the velocity) and β = atanh((x0-c)/R)
        follow from the initial conditions.
        """
        x0, y0 = float(initial[0]), float(initial[1])
        v_x, v_y = float(velocity[0]), float(velocity[1])
        if abs(v_x) < 1e-15:
            # vertical geodesic: x constant, y(t) = y0 exp(vy t / y0)
            y_t = y0 * np.exp(v_y * t / y0)
            return GeodesicSolution(
                [x0, y_t], [0.0, v_y * np.exp(v_y * t / y0)]
            )
        c = x0 + y0 * v_y / v_x
        R = np.sqrt((x0 - c) ** 2 + y0 * y0)
        alpha = v_x * R / (y0 * y0)
        beta = np.arctanh((x0 - c) / R)
        s = alpha * t + beta
        sech_s = 1.0 / np.cosh(s)
        x_t = c + R * np.tanh(s)
        y_t = R * sech_s
        v_x_t = R * alpha * sech_s * sech_s
        v_y_t = -R * alpha * sech_s * np.tanh(s)
        return GeodesicSolution([x_t, y_t], [v_x_t, v_y_t])

    def __repr__(self):
        return "HyperbolicPlane(ds^2 = (dx^2 + dy^2) / y^2)"
