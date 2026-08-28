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
        theta = float(point[0])
        return (1.0, np.sin(theta) ** 2)

    def in_chart(self, point) -> bool:
        theta = float(point[0])
        return 1e-12 < theta < np.pi - 1e-12

    def geodesic_ode(self, state):
        """state = (θ, φ, θ', φ') -> d/dt(state)."""
        theta, _phi, v_th, v_ph = state
        return np.array(
            [
                v_th,
                v_ph,
                np.sin(theta) * np.cos(theta) * v_ph * v_ph,
                -2.0 * np.cos(theta) / np.sin(theta) * v_th * v_ph,
            ]
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

    def __repr__(self):
        return "Sphere(ds^2 = dθ^2 + sin^2θ dφ^2)"
