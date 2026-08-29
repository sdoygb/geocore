"""Side-by-side with PyTorch's classic computation examples: geocore
re-runs linear regression (torch.optim), the autograd Jacobian of a
geodesic, and the circle-Laplacian spectrum (torch.linalg).  The
torch-dependent parts are skipped when torch is unavailable."""

import numpy as np
import pytest


def test_linear_regression_recovers_parameters():
    """The PyTorch入门 problem (recover y = 3x - 2 from noisy data) run
    with geocore's minimize on the flat polar plane: parameters
    recovered to the noise level."""
    from geocore import PolarPlane, minimize

    rng = np.random.default_rng(0)
    x = np.linspace(-2, 2, 100)
    y = 3 * x - 2 + 0.1 * rng.standard_normal(100)
    P = PolarPlane()

    def mse(params):
        w, b = params[0], params[1]
        return float(np.mean((w * x + b - y) ** 2))

    res = minimize(P, mse, [1.5, -1.0], lr=0.1, n_steps=2000)
    w, b = res.point
    assert abs(w - 3) < 0.02
    assert abs(b + 2) < 0.02


def test_analytic_jacobian_matches_pytorch_autograd():
    """The autograd-tutorial example (Jacobian of a sphere geodesic
    endpoint): torch.autograd.functional.jacobian == geocore's analytic
    geodesic.jacobian to machine precision."""
    torch = pytest.importorskip("torch")
    from torch.autograd.functional import jacobian as t_jacobian

    from geocore import Sphere
    from geocore.derivatives import geodesic_jacobian

    def geodesic(params):
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
        return torch.stack([torch.atan2(torch.sqrt(x * x + Y * Y), z), torch.atan2(Y, x)])

    params = torch.tensor([1.1, 0.6, 0.3, 0.5], dtype=torch.float64)
    J_t = np.array(t_jacobian(geodesic, params)[:, :2].tolist())
    J_g = geodesic_jacobian(Sphere(), [1.1, 0.6], [0.3, 0.5], 0.7)[0]
    assert np.abs(J_t - J_g).max() < 1e-9


def test_circle_spectrum_closed_form_vs_discrete():
    """torch.linalg example: discrete circle-Laplacian eigenvalues
    converge to the closed form k^2 (geocore spectral) as O(1/N^2)."""
    from geocore import Circle

    closed = np.array(Circle().laplacian_eigenvalues_closed(5))
    assert np.allclose(closed, [0, 1, 1, 4, 4], atol=1e-9)
    # discrete Laplacian converges to the continuum
    torch = pytest.importorskip("torch")
    n_grid = 64
    d = torch.zeros(n_grid, n_grid)
    for i in range(n_grid):
        d[i, i] = 2.0
        d[i, (i + 1) % n_grid] = -1.0
        d[i, (i - 1) % n_grid] = -1.0
    ev = np.array(torch.linalg.eigvalsh(d).tolist())
    h = 2 * np.pi / n_grid
    err = np.abs(ev[:5] / h**2 - closed).max()
    assert err < 0.02
