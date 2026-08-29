#!/usr/bin/env python3
"""PyTorch classic computation examples, re-run with geocore — side by
side.  Three official-tutorial problems:

1. LINEAR REGRESSION  (the PyTorch入门 tutorial): learn y = w x + b from
   noisy data.  PyTorch: nn.Linear + torch.optim.SGD.  geocore:
   minimize() on the (flat) polar plane.  (Local torch has a broken
   numpy bridge, so each side uses its own data with the same ground
   truth — both must recover (w, b) = (3, -2).)
2. JACOBIANS  (the autograd tutorial): d(gamma(t))/d(p0) for a sphere
   geodesic.  PyTorch: torch.autograd.functional.jacobian on a pure-
   torch geodesic.  geocore: the analytic geodesic.jacobian.  Same
   function, same result to 1e-9.
3. SPECTRUM  (torch.linalg): eigenvalues of the discrete circle
   Laplacian vs the closed form k^2 (geocore spectral).

Run:  PYTHONPATH=src python3 examples/pytorch_comparison.py
"""

import numpy as np


def linear_regression_torch(seed=0):
    import torch

    torch.manual_seed(seed)
    x = torch.linspace(-2, 2, 100).unsqueeze(1)
    y = 3 * x - 2 + 0.1 * torch.randn_like(x)
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = torch.nn.MSELoss()
    losses = []
    for _ in range(2000):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    w = float(model.weight[0, 0])
    b = float(model.bias[0])
    return w, b, losses[-1]


def linear_regression_geocore(seed=0):
    from geocore import PolarPlane, minimize

    rng = np.random.default_rng(seed)
    x = np.linspace(-2, 2, 100)
    y = 3 * x - 2 + 0.1 * rng.standard_normal(100)
    # parameters (w, b) live on the flat polar plane: the metric is
    # Euclidean, so minimize() here is ordinary gradient descent
    P = PolarPlane()

    def mse(params):
        w, b = params[0], params[1]
        return float(np.mean((w * x + b - y) ** 2))

    # p0 must stay away from r = 0 (the polar-plane chart singularity)
    res = minimize(P, mse, [1.5, -1.0], lr=0.1, n_steps=2000)
    w, b = res.point
    return w, b, res.f_history[-1]


def jacobian_torch():
    import torch
    from torch.autograd.functional import jacobian

    def geodesic(params):
        # sphere geodesic endpoint (theta_t, phi_t) from (theta0, phi0,
        # v_theta, v_phi) at t = 0.7, pure torch
        th0, ph0, v_th, v_ph = params[0], params[1], params[2], params[3]
        t = 0.7
        st, ct = torch.sin(th0), torch.cos(th0)
        p0 = torch.stack([st * torch.cos(ph0), st * torch.sin(ph0), ct])
        e_th = torch.stack([ct * torch.cos(ph0), ct * torch.sin(ph0), -st])
        e_ph = torch.stack([-st * torch.sin(ph0), st * torch.cos(ph0), torch.zeros_like(th0)])
        v = v_th * e_th + v_ph * e_ph
        n = torch.norm(v)
        ang = n * t
        p_t = p0 * torch.cos(ang) + (v / n) * torch.sin(ang)
        x, Y, z = p_t
        th_t = torch.atan2(torch.sqrt(x * x + Y * Y), z)
        ph_t = torch.atan2(Y, x)
        return torch.stack([th_t, ph_t])

    params = torch.tensor([1.1, 0.6, 0.3, 0.5], dtype=torch.float64)
    J = jacobian(geodesic, params)  # (2, 4)
    return np.array(J[:, :2].tolist())  # tolist: numpy bridge is broken


def jacobian_geocore():
    from geocore import Sphere
    from geocore.derivatives import geodesic_jacobian

    Jp, Jv = geodesic_jacobian(Sphere(), [1.1, 0.6], [0.3, 0.5], 0.7)
    return Jp


def hessian_geocore(P1, P2, th1, th2, state):
    """Analytic 2x2 Hessian of f(th1, th2) = Re <psi|R_P1(th1) R_P2(th2)|psi>.
    d^2/dth^2 R_P(th) = -R_P(th)/4  (closed form from P^2 = I); the mixed
    term is the product of the first derivatives."""
    from geocore.clifford import rotation_action_closed_form, pauli_action_on_state

    psi = np.asarray(state, dtype=complex)
    R2psi = rotation_action_closed_form(P2, th2, psi)
    R1R2psi = rotation_action_closed_form(P1, th1, R2psi)
    # dR/dth |psi> = -(1/2) sin(th/2)|psi> - (i/2) cos(th/2) P|psi>
    d2 = -(0.5) * np.sin(th2 / 2) * psi - 0.5j * np.cos(th2 / 2) * pauli_action_on_state(P2, psi)
    # mixed term: <psi| (dR1/dth1) d2 >  (A = dR1/dth1 is anti-Hermitian;
    # <psi|A B|psi> is NOT <d1|d2>)
    A_d2 = -(0.5) * np.sin(th1 / 2) * d2 - 0.5j * np.cos(th1 / 2) * pauli_action_on_state(P1, d2)
    f00 = -0.25 * float(np.real(np.vdot(psi, R1R2psi)))
    f11 = -0.25 * float(np.real(np.vdot(psi, R1R2psi)))
    f01 = float(np.real(np.vdot(psi, A_d2)))
    return np.array([[f00, f01], [f01, f11]])


def hessian_torch(P1, P2, th1, th2, state):
    import torch
    from torch.autograd.functional import hessian

    def pauli_matrix_torch(P):
        I = torch.eye(2, dtype=torch.complex128)
        X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
        Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
        M = torch.tensor([[1]], dtype=torch.complex128)
        for ch in P:
            M = torch.kron(M, {"I": I, "X": X, "Y": Y, "Z": Z}[ch])
        return M

    psi = torch.tensor(np.asarray(state, dtype=complex), dtype=torch.complex128)
    M1 = pauli_matrix_torch(P1)
    M2 = pauli_matrix_torch(P2)

    def f(ab):
        a, b = ab[0], ab[1]
        c1 = torch.cos(a / 2)
        s1 = torch.sin(a / 2)
        c2 = torch.cos(b / 2)
        s2 = torch.sin(b / 2)
        r1 = (c1 * torch.eye(M1.shape[0], dtype=torch.complex128) - 1j * s1 * M1)
        r2 = (c2 * torch.eye(M2.shape[0], dtype=torch.complex128) - 1j * s2 * M2)
        out = r1 @ r2 @ psi
        return torch.real(torch.vdot(psi, out))

    H = hessian(f, torch.tensor([th1, th2], dtype=torch.float64))
    return np.array(H.tolist())


def spectrum_torch(n_grid=64):
    import torch

    d = torch.zeros(n_grid, n_grid)
    for i in range(n_grid):
        d[i, i] = 2.0
        d[i, (i + 1) % n_grid] = -1.0
        d[i, (i - 1) % n_grid] = -1.0
    evals = np.array(torch.linalg.eigvalsh(d).tolist())
    h = 2 * np.pi / n_grid
    return evals / h**2  # discrete Laplacian -> k^2 in the continuum


def spectrum_geocore(n_grid=64, n_evals=5):
    from geocore import Circle

    return Circle().laplacian_eigenvalues_closed(n_evals)


def logistic_regression_torch(seed=0):
    import torch

    torch.manual_seed(seed)
    x = torch.linspace(-3, 3, 200).unsqueeze(1)
    y = (2 * x - 1 > 0).float()  # boundary w=2, b=-1
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(1500):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    return float(model.weight[0, 0]), float(model.bias[0])


def logistic_regression_geocore(seed=0):
    from geocore import PolarPlane, minimize

    rng = np.random.default_rng(seed)
    x = np.linspace(-3, 3, 200)
    y = (2 * x - 1 > 0).astype(float)
    P = PolarPlane()

    def bce(params):
        w, b = params[0], params[1]
        z = w * x + b
        return float(np.mean(np.logaddexp(0.0, z) - y * z))  # BCE(logits)

    res = minimize(P, bce, [0.5, 0.0], lr=0.3, n_steps=4000)
    return float(res.point[0]), float(res.point[1])


def rosenbrock(params):
    x, y = params[0], params[1]
    return (1 - x) ** 2 + 100 * (y - x * x) ** 2


def adam_rosenbrock_torch(seed=0, steps=3000):
    import torch

    torch.manual_seed(seed)
    p = torch.tensor([-0.5, 0.5], requires_grad=True)
    opt = torch.optim.Adam([p], lr=0.01)
    for _ in range(steps):
        opt.zero_grad()
        f = (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2
        f.backward()
        opt.step()
    return float(p[0]), float(p[1]), float(f)


def adam_rosenbrock_geocore(seed=0, steps=3000):
    from geocore import PolarPlane, minimize

    P = PolarPlane()

    def f(params):
        r, ph = params[0], params[1]
        return rosenbrock([r * np.cos(ph), r * np.sin(ph)])

    res = minimize(P, f, [0.7, 2.4], lr=0.01, n_steps=steps, optimizer="adam")
    x, y = res.point[0] * np.cos(res.point[1]), res.point[0] * np.sin(res.point[1])
    return float(x), float(y), float(rosenbrock([x, y]))


def main():
    print("=== 1. LINEAR REGRESSION (recover y = 3x - 2) ===")
    w_t, b_t, l_t = linear_regression_torch()
    w_g, b_g, l_g = linear_regression_geocore()
    print(f"PyTorch : w = {w_t:+.4f}  b = {b_t:+.4f}  (final loss {l_t:.4e})")
    print(f"geocore : w = {w_g:+.4f}  b = {b_g:+.4f}  (final loss {l_g:.4e})")
    print(f"          |w-3| = {abs(w_t-3):.2e} vs {abs(w_g-3):.2e}  "
          f"|b+2| = {abs(b_t+2):.2e} vs {abs(b_g+2):.2e}")

    print("\n=== 2. JACOBIAN of the sphere geodesic endpoint ===")
    J_t = jacobian_torch()
    J_g = jacobian_geocore()
    err = np.abs(J_t - J_g).max()
    print("torch autograd (numerical convention):")
    print(np.round(J_t, 4))
    print("geocore analytic:")
    print(np.round(J_g, 4))
    print(f"max |difference|: {err:.2e}")

    print("\n=== 3. SPECTRUM of the circle Laplacian ===")
    ev_num = spectrum_torch()
    ev_closed = spectrum_geocore()
    print("closed form (k^2):", np.round(ev_closed, 3))
    print("torch eigvalsh  :", np.round(ev_num[:5], 3))
    err_s = float(np.abs(ev_num[:5] - np.array(ev_closed)).max())
    print(f"discrete-vs-continuum error: {err_s:.3f} (converges O(1/N^2))")


def main_extra():
    print("\n=== 2b. LOGISTIC REGRESSION (classify y = (2x-1 > 0)) ===")
    w_t, b_t = logistic_regression_torch()
    w_g, b_g = logistic_regression_geocore()
    print(f"PyTorch : w = {w_t:+.3f}  b = {b_t:+.3f}")
    print(f"geocore : w = {w_g:+.3f}  b = {b_g:+.3f}")

    print("\n=== 2c. HESSIAN of Re<psi|R_P1 R_P2|psi> ===")
    import numpy as _np
    _rng = _np.random.default_rng(0)
    state = _rng.standard_normal(4) + 1j * _rng.standard_normal(4)
    H_g = hessian_geocore("XY", "YX", 0.4, 0.9, state)
    H_t = hessian_torch("XY", "YX", 0.4, 0.9, state)
    err_h = _np.abs(H_g - H_t).max()
    print("geocore analytic Hessian:")
    print(_np.round(H_g, 4))
    print("torch autograd Hessian:")
    print(_np.round(H_t, 4))
    print(f"max |difference|: {err_h:.2e}")

    print("\n=== 2d. ADAM on Rosenbrock (torch.optim.Adam vs RiemannianAdam) ===")
    w_t, b_t, f_t = adam_rosenbrock_torch()
    w_g, b_g, f_g = adam_rosenbrock_geocore()
    print(f"PyTorch : ({w_t:.4f}, {b_t:.4f})  f = {f_t:.3e}")
    print(f"geocore : ({w_g:.4f}, {b_g:.4f})  f = {f_g:.3e}")


if __name__ == "__main__":
    main()
    main_extra()
