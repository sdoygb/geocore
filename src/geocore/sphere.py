"""Layer 0 — the unit sphere S^2 in spherical coordinates.

Metric: ds^2 = dθ^2 + sin^2θ dφ^2  (θ ∈ (0, π), φ ∈ [0, 2π)).

  - Christoffels: Γ^θ_φφ = -sinθ cosθ, Γ^φ_θφ = Γ^φ_φθ = cotθ
  - geodesic ODE (generic path):
        dθ'' = sinθ cosθ (φ')^2
        φ''  = -2 cotθ θ' φ'
  - closed form (Layer-3 shortcut): geodesics are great circles.  In the
    R^3 embedding, p(t) = p cos(|v| t) + (v/|v|) sin(|v| t) with p a unit
    vector and v its tangent; the spherical coordinates are recovered by
    the inverse chart.
"""

from __future__ import annotations

import numpy as np

from .manifolds import GeodesicSolution, RiemannianManifold

__all__ = ["Sphere"]


class Sphere(RiemannianManifold):
    """The unit sphere S^2, ds^2 = dθ^2 + sin^2θ dφ^2."""

    def metric_diag(self, point) -> tuple[float, float]:
        theta = np.asarray(point, dtype=float)[..., 0]  # scalar or (B,)
        return (1.0, np.sin(theta) ** 2)

    def in_chart(self, point) -> bool:
        theta = np.asarray(point, dtype=float)[..., 0]
        return (1e-12 < theta) & (theta < np.pi - 1e-12)

    def geodesic_ode(self, state):
        """state = (θ, φ, θ', φ') -> d/dt(state).  ``[..., i]`` indexing
        supports both a single (4,) state and a batched (B, 4) state."""
        theta, v_th, v_ph = state[..., 0], state[..., 2], state[..., 3]
        return np.stack(
            [
                v_th,
                v_ph,
                np.sin(theta) * np.cos(theta) * v_ph * v_ph,
                -2.0 * np.cos(theta) / np.sin(theta) * v_th * v_ph,
            ],
            axis=-1,
        )

    def _to_r3(self, theta, phi):
        st, ct = np.sin(theta), np.cos(theta)
        return np.array([st * np.cos(phi), st * np.sin(phi), ct])

    def geodesic_closed_form(self, initial, velocity, t):
        """Closed form: a great circle in the R^3 embedding.

        p(t) = p0 cos(|v| t) + (v/|v|) sin(|v| t); the spherical chart is
        recovered at the endpoint, with the coordinate velocity from the
        embedding tangent vector.  Exact up to floating point.
        """
        theta0, phi0 = float(initial[0]), float(initial[1])
        v_th, v_ph = float(velocity[0]), float(velocity[1])
        p0 = self._to_r3(theta0, phi0)
        # embedding tangent: v = θ' e_θ + φ' e_φ
        st, ct = np.sin(theta0), np.cos(theta0)
        e_th = np.array([ct * np.cos(phi0), ct * np.sin(phi0), -st])
        e_ph = np.array([-st * np.sin(phi0), st * np.cos(phi0), 0.0])
        v = v_th * e_th + v_ph * e_ph
        n = np.linalg.norm(v)
        if n < 1e-15:  # zero velocity: fixed point (the identity of exp)
            return GeodesicSolution([theta0, phi0], [v_th, v_ph])
        p_t = p0 * np.cos(n * t) + (v / n) * np.sin(n * t)
        v_t = -p0 * n * np.sin(n * t) + v * np.cos(n * t)
        x, Y, z = p_t
        theta_t = np.arctan2(np.sqrt(x * x + Y * Y), z)
        phi_t = np.arctan2(Y, x)
        # coordinate velocity at the endpoint: v_t = θ' e_θ + φ' e_φ,
        # with e_θ, e_φ orthogonal (metric diagonal) — project back.
        st_t, ct_t = np.sin(theta_t), np.cos(theta_t)
        e_th_t = np.array([ct_t * np.cos(phi_t), ct_t * np.sin(phi_t), -st_t])
        e_ph_t = np.array([-st_t * np.sin(phi_t), st_t * np.cos(phi_t), 0.0])
        v_th_t = float(v_t @ e_th_t)
        v_ph_t = float(v_t @ e_ph_t) / (st_t * st_t)
        return GeodesicSolution([theta_t, phi_t], [v_th_t, v_ph_t])

    def parallel_transport(self, point_from, point_to, vector):
        """Parallel transport along the connecting great circle: a rotation
        about the axis p0 x p1 by the angular distance.

        In the R^3 embedding, transport along the great circle from p0 to
        p1 is the rotation R(k, phi) with axis k = (p0 x p1)/|p0 x p1| and
        angle phi = acos(p0 . p1) — an isometry of the tangent spaces
        (metric norm preserved, exact up to floating point).
        """
        th0, ph0 = float(point_from[0]), float(point_from[1])
        th1, ph1 = float(point_to[0]), float(point_to[1])
        v = np.asarray(vector, dtype=float)
        p0 = self._to_r3(th0, ph0)
        p1 = self._to_r3(th1, ph1)
        # embedding tangent at p0
        st, ct = np.sin(th0), np.cos(th0)
        e_th = np.array([ct * np.cos(ph0), ct * np.sin(ph0), -st])
        e_ph = np.array([-st * np.sin(ph0), st * np.cos(ph0), 0.0])
        V = v[0] * e_th + v[1] * e_ph
        cos_phi = float(np.clip(p0 @ p1, -1.0, 1.0))
        phi = np.arccos(cos_phi)
        if phi < 1e-14:
            return v.copy()
        axis = np.cross(p0, p1)
        axis /= np.linalg.norm(axis)
        K = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        R = (
            np.cos(phi) * np.eye(3)
            + np.sin(phi) * K
            + (1.0 - np.cos(phi)) * np.outer(axis, axis)
        )
        V1 = R @ V
        # back to spherical coordinates at point_to
        st1, ct1 = np.sin(th1), np.cos(th1)
        e_th1 = np.array([ct1 * np.cos(ph1), ct1 * np.sin(ph1), -st1])
        e_ph1 = np.array([-st1 * np.sin(ph1), st1 * np.cos(ph1), 0.0])
        return np.array(
            [float(V1 @ e_th1), float(V1 @ e_ph1) / (st1 * st1)]
        )

    def geodesic_closed_form_batch(self, initial, velocity, t):
        """Vectorized great-circle geodesics over a batch (B, 2)."""
        init = np.atleast_2d(np.asarray(initial, dtype=float))
        vel = np.atleast_2d(np.asarray(velocity, dtype=float))
        B = init.shape[0]
        t = np.broadcast_to(np.asarray(t, dtype=float), (B,))
        th0, ph0 = init[:, 0], init[:, 1]
        v_th, v_ph = vel[:, 0], vel[:, 1]
        st, ct = np.sin(th0), np.cos(th0)
        p0 = np.column_stack([st * np.cos(ph0), st * np.sin(ph0), ct])
        e_th = np.column_stack([ct * np.cos(ph0), ct * np.sin(ph0), -st])
        e_ph = np.column_stack([-st * np.sin(ph0), st * np.cos(ph0), np.zeros(B)])
        v = v_th[:, None] * e_th + v_ph[:, None] * e_ph  # (B, 3)
        n = np.linalg.norm(v, axis=1)
        nz = n > 1e-15
        n_safe = np.where(nz, n, 1.0)
        ang = n * t
        p_t = p0 * np.cos(ang)[:, None] + (v / n_safe[:, None]) * np.sin(ang)[:, None]
        p_t = np.where(nz[:, None], p_t, p0)  # zero velocity: fixed point
        v_t = -p0 * n[:, None] * np.sin(ang)[:, None] + v * np.cos(ang)[:, None]
        x, Y, z = p_t[:, 0], p_t[:, 1], p_t[:, 2]
        th_t = np.arctan2(np.sqrt(x * x + Y * Y), z)
        ph_t = np.arctan2(Y, x)
        st_t, ct_t = np.sin(th_t), np.cos(th_t)
        e_th_t = np.column_stack([ct_t * np.cos(ph_t), ct_t * np.sin(ph_t), -st_t])
        e_ph_t = np.column_stack(
            [-st_t * np.sin(ph_t), st_t * np.cos(ph_t), np.zeros(B)]
        )
        v_th_t = np.sum(v_t * e_th_t, axis=1)
        v_ph_t = np.sum(v_t * e_ph_t, axis=1) / (st_t * st_t)
        return np.column_stack([th_t, ph_t]), np.column_stack([v_th_t, v_ph_t])

    def parallel_transport_batch(self, point_from, point_to, vector):
        """Vectorized great-circle transport over a batch (B, 2) each."""
        pf = np.atleast_2d(np.asarray(point_from, dtype=float))
        pt = np.atleast_2d(np.asarray(point_to, dtype=float))
        v = np.atleast_2d(np.asarray(vector, dtype=float))
        B = pf.shape[0]
        th0, ph0 = pf[:, 0], pf[:, 1]
        th1, ph1 = pt[:, 0], pt[:, 1]
        st, ct = np.sin(th0), np.cos(th0)
        p0 = np.column_stack([st * np.cos(ph0), st * np.sin(ph0), ct])
        st1, ct1 = np.sin(th1), np.cos(th1)
        p1 = np.column_stack([st1 * np.cos(ph1), st1 * np.sin(ph1), ct1])
        e_th = np.column_stack([ct * np.cos(ph0), ct * np.sin(ph0), -st])
        e_ph = np.column_stack([-st * np.sin(ph0), st * np.cos(ph0), np.zeros(B)])
        V = v[:, 0, None] * e_th + v[:, 1, None] * e_ph  # (B, 3)
        cos_phi = np.clip(np.sum(p0 * p1, axis=1), -1.0, 1.0)
        phi = np.arccos(cos_phi)
        same = phi < 1e-14
        axis = np.cross(p0, p1)
        axis_norm = np.linalg.norm(axis, axis=1)
        axis = axis / np.where(axis_norm > 0, axis_norm, 1.0)[:, None]
        # Rodrigues rotation by phi about axis, per row
        Kx = np.column_stack(
            [np.zeros(B), -axis[:, 2], axis[:, 1]]
        )
        Ky = np.column_stack([axis[:, 2], np.zeros(B), -axis[:, 0]])
        Kz = np.column_stack([-axis[:, 1], axis[:, 0], np.zeros(B)])
        c, s = np.cos(phi), np.sin(phi)
        # R v = c v + s (axis x v) + (1-c) axis (axis . v)
        cross = np.column_stack(
            [
                axis[:, 1] * V[:, 2] - axis[:, 2] * V[:, 1],
                axis[:, 2] * V[:, 0] - axis[:, 0] * V[:, 2],
                axis[:, 0] * V[:, 1] - axis[:, 1] * V[:, 0],
            ]
        )
        dot = np.sum(axis * V, axis=1)
        V1 = c[:, None] * V + s[:, None] * cross + (1.0 - c)[:, None] * axis * dot[:, None]
        V1 = np.where(same[:, None], V, V1)
        e_th1 = np.column_stack([ct1 * np.cos(ph1), ct1 * np.sin(ph1), -st1])
        e_ph1 = np.column_stack(
            [-st1 * np.sin(ph1), st1 * np.cos(ph1), np.zeros(B)]
        )
        out = np.empty_like(v)
        out[:, 0] = np.sum(V1 * e_th1, axis=1)
        out[:, 1] = np.sum(V1 * e_ph1, axis=1) / (st1 * st1)
        return out

    def __repr__(self):
        return "Sphere(ds^2 = dθ^2 + sin^2θ dφ^2)"
