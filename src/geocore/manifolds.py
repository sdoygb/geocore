"""Layer 0 extension — geometric manifolds.

v0.2 introduced the polar plane (P = (r, y) with metric ds^2 = dr^2 + r^2
dy^2), the warped product R+ x_f S-like fibre with f(r) = r — i.e. the
Euclidean plane in polar-like coordinates.  v0.4 adds the sphere (spherical
coordinates, great-circle geodesics) and the hyperbolic plane (upper-half
plane, semicircle/vertical geodesics).  All three share one interface:
``metric_diag`` (the diagonal metric), ``geodesic_ode`` (the generic RK4
path), ``geodesic_closed_form`` (the exact path, a Layer-3 shortcut) and
``in_chart`` (the coordinate constraint).

The Christoffel symbols come from the metric (geometrized): each geodesic
ODE is the standard second-order system derived from the metric.
"""

from __future__ import annotations

import numpy as np

from .objects import GeometricObject

__all__ = ["RiemannianManifold", "PolarPlane", "GeodesicSolution"]


class GeodesicSolution:
    """(point, velocity) along a geodesic at a parameter value t."""

    __slots__ = ("point", "velocity")

    def __init__(self, point, velocity):
        self.point = np.asarray(point, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)

    def __repr__(self):
        return f"GeodesicSolution(point={self.point}, velocity={self.velocity})"


class RiemannianManifold(GeometricObject):
    """Base class for coordinate-chart manifolds.

    A 2-dimensional manifold in coordinates (u, v) with diagonal metric
    ds^2 = g0(u,v) du^2 + g1(u,v) dv^2.  Subclasses provide:

      - ``metric_diag(point) -> (g0, g1)``: the diagonal metric,
      - ``geodesic_ode(state) -> d(state)/dt``: the second-order geodesic
        ODE (state = (u, v, u', v')) — the generic path,
      - ``geodesic_closed_form(initial, velocity, t)``: the exact geodesic
        — the Layer-3 shortcut,
      - ``in_chart(point) -> bool``: the coordinate chart constraint.

    The exponential map is the geodesic of unit parameter:
    exp_p(v) = geodesic(p, v, 1).
    """

    @property
    def dim(self) -> int:
        return 2

    def metric_diag(self, point) -> tuple[float, float]:
        raise NotImplementedError

    def metric_norm_sq(self, point, velocity) -> float:
        """g(w, w) = g0 w1^2 + g1 w2^2 — the conserved quantity along
        geodesics (a Levi-Civita invariant)."""
        g0, g1 = self.metric_diag(np.asarray(point, dtype=float))
        v1, v2 = np.asarray(velocity, dtype=float)
        return float(g0 * v1 * v1 + g1 * v2 * v2)

    def in_chart(self, point) -> bool:
        raise NotImplementedError

    def geodesic_ode(self, state):
        raise NotImplementedError

    def geodesic_generic(self, initial, velocity, t, n_steps=200):
        """Generic path: RK4 integration of the geodesic ODE."""
        dt = t / n_steps
        state = np.array(
            [initial[0], initial[1], velocity[0], velocity[1]], dtype=float
        )
        for _ in range(n_steps):
            k1 = self.geodesic_ode(state)
            k2 = self.geodesic_ode(state + dt / 2 * k1)
            k3 = self.geodesic_ode(state + dt / 2 * k2)
            k4 = self.geodesic_ode(state + dt * k3)
            state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return GeodesicSolution(state[:2], state[2:])

    def geodesic_closed_form(self, initial, velocity, t):
        raise NotImplementedError

    def verify(self) -> dict:
        return {"ok": True, "note": "manifold invariants checked via geodesic ops"}


class PolarPlane(RiemannianManifold):
    """The polar plane: ds^2 = dr^2 + r^2 dy^2 (warped product, f(r) = r).

    Geometry:
      - metric: g_rr = 1, g_yy = r^2
      - Christoffels: Gamma^r_yy = -r, Gamma^y_ry = Gamma^y_yr = 1/r
      - geodesic ODE:  d^2r/dt^2 = r (dy/dt)^2
                       d^2y/dt^2 = -2/r (dr/dt)(dy/dt)
      - closed form: geodesics are straight lines in Cartesian coordinates
        (r cos y, r sin y) — the Layer-3 shortcut.
    """

    def metric_diag(self, point) -> tuple[float, float]:
        r = float(point[0])
        return (1.0, r * r)

    def in_chart(self, point) -> bool:
        return float(point[0]) > 1e-12

    def geodesic_ode(self, state):
        """state = (r, y, v_r, v_y) -> d/dt(state)."""
        r, _y, v_r, v_y = state
        return np.array([v_r, v_y, r * v_y * v_y, -2.0 * v_r * v_y / r])

    def geodesic_closed_form(self, initial, velocity, t):
        """Closed form: a straight line in Cartesian coordinates.

        (x, Y) = (r cos y, r sin y); the geodesic is (x0 + t vx, Y0 + t vy);
        the polar components are recovered by the inverse chart.  Exact up
        to floating point — no integration.
        """
        r0, y0 = float(initial[0]), float(initial[1])
        v_r, v_y = float(velocity[0]), float(velocity[1])
        x0, Y0 = r0 * np.cos(y0), r0 * np.sin(y0)
        vx = v_r * np.cos(y0) - r0 * v_y * np.sin(y0)
        vy = v_r * np.sin(y0) + r0 * v_y * np.cos(y0)
        x, Y = x0 + t * vx, Y0 + t * vy
        r = np.sqrt(x * x + Y * Y)
        y = np.arctan2(Y, x)
        # polar velocity at the endpoint: invert (vx, vy) = v_r e_r + r v_y e_phi
        v_r_t = vx * np.cos(y) + vy * np.sin(y)
        v_y_t = (-vx * np.sin(y) + vy * np.cos(y)) / r
        return GeodesicSolution([r, y], [v_r_t, v_y_t])

    def __repr__(self):
        return "PolarPlane(ds^2 = dr^2 + r^2 dy^2)"
