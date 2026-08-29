"""Analytic derivative operators (the geometric analogue of autograd's
gradient computation).

PyTorch's autograd computes gradients by reverse-mode automatic
differentiation; geocore provides *analytic* closed-form derivatives,
verified against finite differences to machine precision:

  - ``rotation_derivative``: d/dtheta R_P(theta)|psi> — closed form
        d/dtheta R_P(theta) = -(i/2) P R_P(theta)   (since P^2 = I)
    so d/dtheta R_P(theta)|psi> = -(1/2) sin(theta/2)|psi>
                                  - (i/2) cos(theta/2) P|psi>,
    an O(2^n) path instead of two dense matrix exponentials.
  - ``geodesic_jacobian``: the Jacobians of the geodesic endpoint w.r.t.
    the initial point and velocity (the sensitivity / tangent-propagation
    analogue of a forward-mode Jacobian).  Each manifold has a closed form
    derived from its geodesic formula; verified against central
    differences.

All derivatives are standard mathematics, verified to machine precision;
the derivation engine is the geometry theory.
"""

from __future__ import annotations

import numpy as np

from .clifford import pauli_action_on_state

__all__ = [
    "rotation_derivative",
    "geodesic_jacobian",
    "polar_jacobian",
    "sphere_jacobian",
    "hyperbolic_jacobian",
]


# ---------------------------------------------------------------------------
# rotation_derivative
# ---------------------------------------------------------------------------


def rotation_derivative(axis: str, theta: float, state) -> np.ndarray:
    """d/dtheta R_P(theta)|psi>, closed form.

    R_P(theta) = cos(theta/2) I - i sin(theta/2) P, so
    d/dtheta R_P(theta)|psi> = -(1/2) sin(theta/2)|psi>
                               - (i/2) cos(theta/2) P|psi>.
    """
    state = np.asarray(state, dtype=complex)
    psi = pauli_action_on_state(axis, state)
    return -(0.5) * np.sin(theta / 2) * state - 0.5j * np.cos(theta / 2) * psi


# ---------------------------------------------------------------------------
# Polar plane Jacobians
# ---------------------------------------------------------------------------


def polar_jacobian(initial, velocity, t):
    """Jacobians of the polar-plane geodesic endpoint w.r.t. the initial
    point and velocity.

    The geodesic is a straight line in Cartesian coordinates:
    (x, Y) = (x0 + t vx, Y0 + t vy) with (x0, Y0) = (r0 cos y0, r0 sin y0)
    and (vx, vy) the Cartesian velocity.  The endpoint polar coordinates
    are recovered by (r, y) = (sqrt(x^2+Y^2), atan2(Y, x)); the Jacobians
    follow by the chain rule through the coordinate map.
    """
    r0, y0 = float(initial[0]), float(initial[1])
    v_r, v_y = float(velocity[0]), float(velocity[1])
    x0, Y0 = r0 * np.cos(y0), r0 * np.sin(y0)
    vx = v_r * np.cos(y0) - r0 * v_y * np.sin(y0)
    vy = v_r * np.sin(y0) + r0 * v_y * np.cos(y0)
    x, Y = x0 + t * vx, Y0 + t * vy
    r = np.sqrt(x * x + Y * Y)
    # d(r, y)/d(x, Y) at the endpoint
    Dr = np.array([[x / r, Y / r], [-Y / (r * r), x / (r * r)]])
    # d(x, Y)/d(r0, y0)
    d_x_r0 = np.cos(y0) + t * (-v_y * np.sin(y0))
    d_Y_r0 = np.sin(y0) + t * (v_y * np.cos(y0))
    d_x_y0 = -r0 * np.sin(y0) + t * (-v_r * np.sin(y0) - r0 * v_y * np.cos(y0))
    d_Y_y0 = r0 * np.cos(y0) + t * (v_r * np.cos(y0) - r0 * v_y * np.sin(y0))
    Jp_cart = np.array([[d_x_r0, d_x_y0], [d_Y_r0, d_Y_y0]])
    # d(x, Y)/d(v_r, v_y)
    d_x_vr = t * np.cos(y0)
    d_Y_vr = t * np.sin(y0)
    d_x_vy = t * (-r0 * np.sin(y0))
    d_Y_vy = t * (r0 * np.cos(y0))
    Jv_cart = np.array([[d_x_vr, d_x_vy], [d_Y_vr, d_Y_vy]])
    return Dr @ Jp_cart, Dr @ Jv_cart


# ---------------------------------------------------------------------------
# Sphere Jacobians
# ---------------------------------------------------------------------------


def sphere_jacobian(initial, velocity, t):
    """Jacobians of the sphere geodesic endpoint (theta_t, phi_t) w.r.t.
    (theta0, phi0) and (v_theta, v_phi), via the R^3 embedding.

    p_t = p0 cos(nt) + u sin(nt), u = v/n, n = |v|; the endpoint chart is
    recovered by atan2; the Jacobians follow by the chain rule through the
    embedding (all standard differential geometry, exact up to rounding).
    """
    th0, ph0 = float(initial[0]), float(initial[1])
    v_th, v_ph = float(velocity[0]), float(velocity[1])
    st, ct = np.sin(th0), np.cos(th0)
    p0 = np.array([st * np.cos(ph0), st * np.sin(ph0), ct])
    e_th = np.array([ct * np.cos(ph0), ct * np.sin(ph0), -st])
    e_ph = np.array([-st * np.sin(ph0), st * np.cos(ph0), 0.0])
    v = v_th * e_th + v_ph * e_ph
    n = np.linalg.norm(v)
    if n < 1e-15:
        raise ValueError("sphere_jacobian: zero velocity (degenerate)")
    u = v / n
    ang = n * t
    p_t = p0 * np.cos(ang) + u * np.sin(ang)
    x, Y, z = p_t
    rho = np.sqrt(x * x + Y * Y)
    # d(theta_t, phi_t)/d p_t (unit sphere)
    Dchart = np.array(
        [
            [x * z / rho, Y * z / rho, -rho],
            [-Y / (rho * rho), x / (rho * rho), 0.0],
        ]
    )
    # d p_t / d p0 = cos(nt) I ;  d p0/d(theta0, phi0) = [e_th, e_ph]
    Dp0 = np.column_stack([e_th, e_ph])
    # d p_t / d v (embedding v)
    Dv = (
        -np.sin(ang) * t * np.outer(p0, u)
        + (np.sin(ang) / n) * (np.eye(3) - np.outer(u, u))
        + np.cos(ang) * t * np.outer(u, u)
    )
    # d v / d(v_th, v_ph) = [e_th, e_ph]
    Jv = Dchart @ Dv @ Dp0
    # d p_t / d(theta0, phi0): p_t depends on the initial frame through
    # both p0 and the embedding velocity v (its basis e_th, e_ph rotates):
    #   d v / d theta0 = v_th (-p0) + v_ph cot(th0) e_ph
    #   d v / d phi0   = v_th cot(th0) e_ph + v_ph (ct zhat - p0)
    zhat = np.array([0.0, 0.0, 1.0])
    dv_dth = v_th * (-p0) + v_ph * (ct / st) * e_ph
    dv_dph = v_th * (ct / st) * e_ph + v_ph * (ct * zhat - p0)
    Dvframe = np.column_stack([dv_dth, dv_dph])
    Jp = Dchart @ (np.cos(ang) * Dp0 + Dv @ Dvframe)
    return Jp, Jv


# ---------------------------------------------------------------------------
# Hyperbolic plane Jacobians
# ---------------------------------------------------------------------------


def hyperbolic_jacobian(initial, velocity, t):
    """Jacobians of the hyperbolic-plane geodesic endpoint (x_t, y_t)
    w.r.t. (x0, y0) and (vx, vy), from the semicircle parameterization

        x_t = c + R tanh s,  y_t = R / cosh s,  s = alpha t + beta
        c = x0 + y0 vy/vx,  R = sqrt((x0-c)^2 + y0^2),
        alpha = vx R / y0^2,  beta = atanh((x0 - c)/R).

    All partial derivatives follow by the chain rule (exact up to
    rounding; the vertical-geodesic branch vx = 0 is excluded).
    """
    x0, y0 = float(initial[0]), float(initial[1])
    v_x, v_y = float(velocity[0]), float(velocity[1])
    if abs(v_x) < 1e-15:
        raise ValueError("hyperbolic_jacobian: vertical branch (vx = 0) excluded")
    c = x0 + y0 * v_y / v_x
    R = np.sqrt((x0 - c) ** 2 + y0 * y0)
    alpha = v_x * R / (y0 * y0)
    w = (x0 - c) / R
    beta = np.arctanh(np.clip(w, -1.0 + 1e-15, 1.0 - 1e-15))
    s = alpha * t + beta
    tanh_s, sech_s = np.tanh(s), 1.0 / np.cosh(s)
    x_t = c + R * tanh_s
    y_t = R * sech_s

    # partials of c, R, alpha, beta w.r.t. (x0, y0, vx, vy)
    dc = np.array([1.0, v_y / v_x, -y0 * v_y / (v_x * v_x), y0 / v_x])
    dR = np.zeros(4)
    dR[0] = 0.0  # (x0-c)(1-dc/dx0)/R with dc/dx0=1
    dR[1] = (-(x0 - c) * dc[1] + y0) / R
    dR[2] = (-(x0 - c) * dc[2]) / R
    dR[3] = (-(x0 - c) * dc[3]) / R
    dalpha = np.zeros(4)
    dalpha[0] = 0.0
    dalpha[1] = v_x * dR[1] / (y0 * y0) - 2.0 * v_x * R / (y0**3)
    dalpha[2] = R / (y0 * y0) + v_x * dR[2] / (y0 * y0)
    dalpha[3] = v_x * dR[3] / (y0 * y0)
    dw = np.zeros(4)
    dw[0] = 0.0  # (1-dc/dx0)/R - w dR/dx0/R = 0
    dw[1] = (-dc[1]) / R - w * dR[1] / R
    dw[2] = (-dc[2]) / R - w * dR[2] / R
    dw[3] = (-dc[3]) / R - w * dR[3] / R
    dbeta = dw / (1.0 - w * w)
    ds = t * dalpha + dbeta
    # d(x_t, y_t)/d(x0, y0, vx, vy)
    dx = dc + dR * tanh_s + R * sech_s * sech_s * ds
    dy = dR * sech_s - R * sech_s * tanh_s * ds
    return np.column_stack([dx[:2], dy[:2]]).T, np.column_stack([dx[2:], dy[2:]]).T


def geodesic_jacobian(manifold, initial, velocity, t):
    """Analytic Jacobians (Jp, Jv) of the geodesic endpoint at time t with
    respect to the initial point and velocity — one closed form per
    manifold."""
    name = type(manifold).__name__
    if name == "PolarPlane":
        return polar_jacobian(initial, velocity, t)
    if name == "Sphere":
        return sphere_jacobian(initial, velocity, t)
    if name == "HyperbolicPlane":
        return hyperbolic_jacobian(initial, velocity, t)
    raise NotImplementedError(f"geodesic_jacobian: unknown manifold {name}")
