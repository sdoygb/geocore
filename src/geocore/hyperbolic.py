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
        y = np.asarray(point, dtype=float)[..., 1]  # scalar or (B,)
        inv = 1.0 / (y * y)
        return (inv, inv)

    def in_chart(self, point) -> bool:
        return np.asarray(point, dtype=float)[..., 1] > 1e-12

    def geodesic_ode(self, state):
        """state = (x, y, x', y') -> d/dt(state).  ``[..., i]`` indexing
        supports both a single (4,) state and a batched (B, 4) state."""
        y, v_x, v_y = state[..., 1], state[..., 2], state[..., 3]
        return np.stack(
            [v_x, v_y, 2.0 * v_x * v_y / y, (v_y * v_y - v_x * v_x) / y],
            axis=-1,
        )

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
        beta = np.arctanh(np.clip((x0 - c) / R, -1.0 + 1e-15, 1.0 - 1e-15))
        s = alpha * t + beta
        sech_s = 1.0 / np.cosh(s)
        x_t = c + R * np.tanh(s)
        y_t = R * sech_s
        v_x_t = R * alpha * sech_s * sech_s
        v_y_t = -R * alpha * sech_s * np.tanh(s)
        return GeodesicSolution([x_t, y_t], [v_x_t, v_y_t])

    def parallel_transport(self, point_from, point_to, vector):
        """Parallel transport along the connecting geodesic (semicircle).

        Along the geodesic circle of center (c, 0), the transport of a
        tangent vector is a rotation by the swept angle dθ combined with a
        scaling sin θ_t / sin θ_0 = y_t / y_0:

            V' = (y_t / y_0) R(dθ) V

        (derived from the Christoffel equations; exact up to floating
        point).  For vertical geodesics (x constant) the circle degenerates
        and the transport is a pure scaling V' = (y_t / y_0) V.
        """
        x0, y0 = float(point_from[0]), float(point_from[1])
        xt, yt = float(point_to[0]), float(point_to[1])
        v = np.asarray(vector, dtype=float)
        if abs(x0 - xt) < 1e-15 and abs(y0 - yt) < 1e-15:
            return v.copy()
        scale = yt / y0
        if abs(x0 - xt) < 1e-15:
            return scale * v  # vertical geodesic: pure scaling
        c = (x0 * x0 + y0 * y0 - xt * xt - yt * yt) / (2.0 * (x0 - xt))
        th0 = np.arctan2(y0, x0 - c)
        th_t = np.arctan2(yt, xt - c)
        dth = (th_t - th0 + np.pi) % (2.0 * np.pi) - np.pi  # wrap to (-pi, pi]
        cos_d, sin_d = np.cos(dth), np.sin(dth)
        return scale * np.array(
            [cos_d * v[0] - sin_d * v[1], sin_d * v[0] + cos_d * v[1]]
        )

    def geodesic_closed_form_batch(self, initial, velocity, t):
        """Vectorized semicircle geodesics over a batch (B, 2)."""
        init = np.atleast_2d(np.asarray(initial, dtype=float))
        vel = np.atleast_2d(np.asarray(velocity, dtype=float))
        B = init.shape[0]
        t = np.broadcast_to(np.asarray(t, dtype=float), (B,))
        x0, y0 = init[:, 0], init[:, 1]
        v_x, v_y = vel[:, 0], vel[:, 1]
        # vertical branch: x constant, y(t) = y0 exp(vy t / y0)
        vertical = np.abs(v_x) < 1e-15
        y_t = np.where(vertical, y0 * np.exp(v_y * t / y0), y0)
        x_t = np.where(vertical, x0, x0)
        vx_t = np.where(vertical, 0.0, v_x)
        vy_t = np.where(vertical, v_y * np.exp(v_y * t / y0), v_y)
        # general branch: semicircle c + R tanh(alpha t + beta)
        vx_safe = np.where(vertical, 1.0, v_x)
        c = x0 + y0 * v_y / vx_safe
        R = np.sqrt((x0 - c) ** 2 + y0 * y0)
        alpha = v_x * R / (y0 * y0)
        beta = np.arctanh(np.clip((x0 - c) / R, -1.0 + 1e-15, 1.0 - 1e-15))
        s = alpha * t + beta
        sech_s = 1.0 / np.cosh(s)
        x_g = c + R * np.tanh(s)
        y_g = R * sech_s
        vx_g = R * alpha * sech_s * sech_s
        vy_g = -R * alpha * sech_s * np.tanh(s)
        x_t = np.where(vertical, x_t, x_g)
        y_t = np.where(vertical, y_t, y_g)
        vx_t = np.where(vertical, vx_t, vx_g)
        vy_t = np.where(vertical, vy_t, vy_g)
        return np.column_stack([x_t, y_t]), np.column_stack([vx_t, vy_t])

    def parallel_transport_batch(self, point_from, point_to, vector):
        """Vectorized semicircle transport over a batch (B, 2) each."""
        pf = np.atleast_2d(np.asarray(point_from, dtype=float))
        pt = np.atleast_2d(np.asarray(point_to, dtype=float))
        v = np.atleast_2d(np.asarray(vector, dtype=float))
        B = pf.shape[0]
        x0, y0 = pf[:, 0], pf[:, 1]
        xt, yt = pt[:, 0], pt[:, 1]
        same = (np.abs(x0 - xt) < 1e-15) & (np.abs(y0 - yt) < 1e-15)
        vertical = np.abs(x0 - xt) < 1e-15
        scale = yt / y0
        out = np.empty_like(v)
        out[:, 0] = np.where(same, v[:, 0], scale * v[:, 0])
        out[:, 1] = np.where(same, v[:, 1], scale * v[:, 1])
        # general branch: rotation by dth plus scaling
        dx = np.where(vertical, 1.0, x0 - xt)
        c = (x0 * x0 + y0 * y0 - xt * xt - yt * yt) / (2.0 * dx)
        th0 = np.arctan2(y0, x0 - c)
        th_t = np.arctan2(yt, xt - c)
        dth = (th_t - th0 + np.pi) % (2.0 * np.pi) - np.pi
        cos_d, sin_d = np.cos(dth), np.sin(dth)
        out_r0 = np.where(
            vertical,
            out[:, 0],
            scale * (cos_d * v[:, 0] - sin_d * v[:, 1]),
        )
        out_r1 = np.where(
            vertical,
            out[:, 1],
            scale * (sin_d * v[:, 0] + cos_d * v[:, 1]),
        )
        out[:, 0] = np.where(same, v[:, 0], out_r0)
        out[:, 1] = np.where(same, v[:, 1], out_r1)
        return out

    def __repr__(self):
        return "HyperbolicPlane(ds^2 = (dx^2 + dy^2) / y^2)"
