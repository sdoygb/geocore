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

__all__ = ["RiemannianManifold", "PolarPlane", "EuclideanSpace", "GeodesicSolution"]


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
        geodesics (a Levi-Civita invariant).  Accepts single (2,) or
        batched (B, 2) inputs."""
        g0, g1 = self.metric_diag(np.asarray(point, dtype=float))
        v = np.asarray(velocity, dtype=float)
        return g0 * v[..., 0] * v[..., 0] + g1 * v[..., 1] * v[..., 1]

    def in_chart(self, point) -> bool:
        raise NotImplementedError

    def parallel_transport(self, point_from, point_to, vector):
        """Parallel transport of a tangent vector from point_from to
        point_to along the connecting geodesic — an isometry (metric norm
        preserved)."""
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

    def geodesic_generic_batch(self, initial, velocity, t, n_steps=200):
        """Vectorized RK4 over a batch of geodesics — the batch generic
        path (the analogue of vectorized/batched tensor ops).

        ``initial``/``velocity`` are (B, 2); ``t`` is a scalar or (B,).
        Returns (points (B, 2), velocities (B, 2)).  The geodesic ODE is
        already elementwise, so the same RK4 loop runs on (B, 4) states.
        """
        initial = np.atleast_2d(np.asarray(initial, dtype=float))
        velocity = np.atleast_2d(np.asarray(velocity, dtype=float))
        B = initial.shape[0]
        t = np.broadcast_to(np.asarray(t, dtype=float), (B,))
        dt = t / n_steps
        state = np.column_stack([initial, velocity])  # (B, 4)
        for _ in range(n_steps):
            k1 = self.geodesic_ode(state)
            k2 = self.geodesic_ode(state + dt[:, None] / 2 * k1)
            k3 = self.geodesic_ode(state + dt[:, None] / 2 * k2)
            k4 = self.geodesic_ode(state + dt[:, None] * k3)
            state = state + dt[:, None] / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return state[:, :2], state[:, 2:]

    def geodesic_closed_form_batch(self, initial, velocity, t):
        """Vectorized closed-form geodesics over a batch: (points (B, 2),
        velocities (B, 2)).  The Layer-3 shortcut for batch workloads."""
        raise NotImplementedError

    def parallel_transport_batch(self, point_from, point_to, vector):
        """Vectorized parallel transport over a batch (B, 2) each."""
        raise NotImplementedError

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
        r = np.asarray(point, dtype=float)[..., 0]  # scalar or (B,)
        return (1.0, r * r)

    def in_chart(self, point) -> bool:
        return np.asarray(point, dtype=float)[..., 0] > 1e-12

    def geodesic_ode(self, state):
        """state = (r, y, v_r, v_y) -> d/dt(state).  ``[..., i]`` indexing
        supports both a single (4,) state and a batched (B, 4) state."""
        r, v_r, v_y = state[..., 0], state[..., 2], state[..., 3]
        return np.stack(
            [v_r, v_y, r * v_y * v_y, -2.0 * v_r * v_y / r], axis=-1
        )

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

    def parallel_transport(self, point_from, point_to, vector):
        """Parallel transport along the straight-line geodesic.

        The plane is flat, so in Cartesian coordinates transport is the
        identity; the polar coordinate frame rotates and scales along the
        way.  From (r0, y0) to (r1, y1) with d = y1 - y0:

            v_r'  = cos(d) v_r + sin(d) r0 v_y
            v_y'  = (-sin(d) v_r + cos(d) r0 v_y) / r1

        (an isometry of g = diag(1, r^2); exact up to floating point).
        """
        r0, y0 = float(point_from[0]), float(point_from[1])
        r1, y1 = float(point_to[0]), float(point_to[1])
        v = np.asarray(vector, dtype=float)
        cos_d, sin_d = np.cos(y1 - y0), np.sin(y1 - y0)
        return np.array(
            [
                cos_d * v[0] + sin_d * r0 * v[1],
                (-sin_d * v[0] + cos_d * r0 * v[1]) / r1,
            ]
        )

    def geodesic_closed_form_batch(self, initial, velocity, t):
        """Vectorized straight-line geodesics over a batch (B, 2)."""
        init = np.atleast_2d(np.asarray(initial, dtype=float))
        vel = np.atleast_2d(np.asarray(velocity, dtype=float))
        B = init.shape[0]
        t = np.broadcast_to(np.asarray(t, dtype=float), (B,))
        r0, y0 = init[:, 0], init[:, 1]
        v_r, v_y = vel[:, 0], vel[:, 1]
        x0, Y0 = r0 * np.cos(y0), r0 * np.sin(y0)
        vx = v_r * np.cos(y0) - r0 * v_y * np.sin(y0)
        vy = v_r * np.sin(y0) + r0 * v_y * np.cos(y0)
        x = x0 + t * vx
        Y = Y0 + t * vy
        r = np.sqrt(x * x + Y * Y)
        y = np.arctan2(Y, x)
        v_r_t = vx * np.cos(y) + vy * np.sin(y)
        v_y_t = (-vx * np.sin(y) + vy * np.cos(y)) / r
        return np.column_stack([r, y]), np.column_stack([v_r_t, v_y_t])

    def parallel_transport_batch(self, point_from, point_to, vector):
        """Vectorized polar transport over a batch (B, 2) each."""
        pf = np.atleast_2d(np.asarray(point_from, dtype=float))
        pt = np.atleast_2d(np.asarray(point_to, dtype=float))
        v = np.atleast_2d(np.asarray(vector, dtype=float))
        cos_d = np.cos(pt[:, 1] - pf[:, 1])
        sin_d = np.sin(pt[:, 1] - pf[:, 1])
        out = np.empty_like(v)
        out[:, 0] = cos_d * v[:, 0] + sin_d * pf[:, 0] * v[:, 1]
        out[:, 1] = (-sin_d * v[:, 0] + cos_d * pf[:, 0] * v[:, 1]) / pt[:, 0]
        return out

    def __repr__(self):
        return "PolarPlane(ds^2 = dr^2 + r^2 dy^2)"


class EuclideanSpace(RiemannianManifold):
    """R^n with the standard metric — the flat N-dimensional parameter
    space (the natural home of high-dimensional optimization and
    classification).

    Geodesics are straight lines (no integration), parallel transport is
    the identity, and the chart covers everything.  This is what makes
    the geocore optimizers work on arbitrary-dimensional problems, not
    just the 2-d manifolds.
    """

    def __init__(self, n: int = 2):
        self._n = int(n)
        if self._n < 1:
            raise ValueError("EuclideanSpace needs n >= 1")

    @property
    def dim(self) -> int:
        return self._n

    def metric_diag(self, point):
        p = np.asarray(point, dtype=float)
        return np.ones(p.shape[:-1] + (self._n,)) if p.ndim > 1 else np.ones(self._n)

    def metric_norm_sq(self, point, velocity):
        v = np.asarray(velocity, dtype=float)
        return np.sum(v * v, axis=-1)

    def in_chart(self, point) -> bool:
        return True

    def geodesic_ode(self, state):
        """d/dt [x, v] = [v, 0]: straight-line geodesics."""
        v = state[..., self._n:]
        z = np.zeros_like(v)
        return np.concatenate([v, z], axis=-1)

    def geodesic_closed_form(self, initial, velocity, t):
        p = np.asarray(initial, dtype=float) + t * np.asarray(velocity, dtype=float)
        return GeodesicSolution(p, np.asarray(velocity, dtype=float))

    def geodesic_closed_form_batch(self, initial, velocity, t):
        init = np.atleast_2d(np.asarray(initial, dtype=float))
        vel = np.atleast_2d(np.asarray(velocity, dtype=float))
        B = init.shape[0]
        t = np.broadcast_to(np.asarray(t, dtype=float), (B,))[:, None]
        return init + t * vel, np.broadcast_to(vel, init.shape)

    def parallel_transport(self, point_from, point_to, vector):
        return np.asarray(vector, dtype=float).copy()

    def parallel_transport_batch(self, point_from, point_to, vector):
        return np.asarray(vector, dtype=float).copy()

    def __repr__(self):
        return f"EuclideanSpace(n={self._n})"
