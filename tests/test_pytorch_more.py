"""More PyTorch-example comparisons: logistic regression (classification),
the analytic Hessian of a two-rotation expectation vs torch autograd
(1.1e-16), and Adam on Rosenbrock (torch.optim.Adam vs RiemannianAdam)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from geocore import PolarPlane, minimize


def test_logistic_regression_recovers_decision_boundary():
    """The classification problem (y = (2x - 1 > 0)): the learned
    decision boundary x = -b/w approaches the true 0.5."""
    rng = np.random.default_rng(0)
    x = np.linspace(-3, 3, 200)
    y = (2 * x - 1 > 0).astype(float)
    P = PolarPlane()

    def bce(params):
        w, b = params[0], params[1]
        z = w * x + b
        return float(np.mean(np.logaddexp(0.0, z) - y * z))

    res = minimize(P, bce, [0.5, 0.0], lr=0.3, n_steps=4000)
    w, b = res.point
    boundary = -b / w
    assert 0.3 <= boundary <= 0.6


def test_analytic_hessian_matches_pytorch():
    """The analytic 2x2 Hessian of Re<psi|R_P1 R_P2|psi> equals
    torch.autograd.functional.hessian to machine precision (the mixed
    term is <psi|A d2> with A anti-Hermitian — torch caught a wrong
    naive derivation)."""
    torch = pytest.importorskip("torch")
    from pytorch_comparison import hessian_geocore, hessian_torch  # noqa: E402

    rng = np.random.default_rng(0)
    state = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    H_g = hessian_geocore("XY", "YX", 0.4, 0.9, state)
    H_t = hessian_torch("XY", "YX", 0.4, 0.9, state)
    assert np.abs(H_g - H_t).max() < 1e-9
    assert np.abs(H_g - H_g.T).max() < 1e-12  # symmetric


def test_adam_rosenbrock_both_converge():
    """torch.optim.Adam and RiemannianAdam both drive Rosenbrock to
    (1, 1); both final values below 1e-4."""
    torch = pytest.importorskip("torch")
    from pytorch_comparison import adam_rosenbrock_geocore, adam_rosenbrock_torch  # noqa: E402

    xt, yt, ft = adam_rosenbrock_torch()
    xg, yg, fg = adam_rosenbrock_geocore()
    assert ft < 1e-4
    assert fg < 1e-4
    assert abs(xt - 1) < 0.01 and abs(yt - 1) < 0.01
    assert abs(xg - 1) < 0.01 and abs(yg - 1) < 0.01
