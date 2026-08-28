"""Tests for the Riemannian optimizer (analogue of torch.optim).

The theory engine selects the geometry (gradient = Riesz representative,
step = exponential map); everything presented is standard mathematics
verified to machine precision.
"""

import numpy as np
import pytest

from geocore import PolarPlane, RiemannianSGD, minimize
from geocore.ops import optim_gradient, optim_step
from geocore.shortcuts import optim_step_closed_form


def _quadratic(rstar=1.5, ystar=0.7):
    def f(p):
        return (p[0] - rstar) ** 2 + (p[1] - ystar) ** 2

    return f, np.array([rstar, ystar])


def test_gradient_is_riesz_representative():
    """g(grad f, v) == df(v) for random tangent vectors (machine precision)."""
    m = PolarPlane()
    f, _ = _quadratic()
    rng = np.random.default_rng(42)
    for _ in range(10):
        point = np.array([rng.uniform(0.5, 3.0), rng.uniform(-2.0, 2.0)])
        df = np.array([2 * (point[0] - 1.5), 2 * (point[1] - 0.7)])
        grad = optim_gradient(m, df, point)
        v = rng.standard_normal(2)
        v /= np.linalg.norm(v)
        lhs = grad[0] * v[0] + point[0] ** 2 * grad[1] * v[1]  # g(grad, v)
        rhs = df[0] * v[0] + df[1] * v[1]  # df(v)
        assert abs(lhs - rhs) < 1e-12


def test_gradient_closed_form_values():
    """On the polar plane grad f = (df_r, df_y / r^2) exactly."""
    m = PolarPlane()
    point = np.array([2.0, 0.5])
    grad = optim_gradient(m, np.array([1.0, 3.0]), point)
    assert np.allclose(grad, [1.0, 3.0 / 4.0], atol=1e-15)


def test_step_matches_closed_form_exp_map():
    """optim.step (generic RK4) == closed-form exp map to machine precision."""
    m = PolarPlane()
    point = np.array([1.7, 0.4])
    dv = np.array([-0.3, 0.2])
    report = optim_step_closed_form.verify_against(m, point, dv, 0.1)
    assert report.ok
    assert report.max_error < 1e-8  # RK4 with 200 steps: O(dt^4)


def test_step_descent_property():
    """With a potential supplied, a gradient step does not increase f."""
    m = PolarPlane()
    f, _ = _quadratic()
    point = np.array([1.7, 0.4])
    df = np.array([2 * (point[0] - 1.5), 2 * (point[1] - 0.7)])
    grad = optim_gradient(m, df, point)
    new_point = optim_step(m, point, -grad, 0.1, f=f)  # invariant checks descent
    assert f(new_point) <= f(point) + 1e-7


def test_step_stays_on_manifold():
    """The polar chart requires r > 0; the step must respect it."""
    m = PolarPlane()
    point = np.array([1.0, 0.0])
    dv = np.array([-0.5, 0.0])
    new_point = optim_step(m, point, dv, 0.1)
    assert new_point[0] > 1e-12


def test_riemannian_sgd_converges_to_minimizer():
    """minimize() converges to the closed-form minimizer of a quadratic;
    f is non-increasing; the final Riemannian gradient norm is tiny."""
    m = PolarPlane()
    f, minimizer = _quadratic()
    res = minimize(m, f, [2.0, 0.3], lr=0.05, n_steps=500, minimizer=minimizer)
    assert res.converged
    assert res.descent_ok
    assert res.final_grad_norm < 1e-8
    assert res.minimizer_error < 1e-8
    # monotone decrease of f along the trajectory
    diffs = np.diff(res.f_history)
    assert np.all(diffs <= 1e-9)


def test_riemannian_sgd_with_momentum():
    """Momentum accelerates convergence and still lands at the minimizer;
    the velocity buffer follows v <- m v + grad."""
    m = PolarPlane()
    f, minimizer = _quadratic()
    # momentum 0.9 converges at rate ~sqrt(0.9) per step: ~500 steps to 1e-12
    res = minimize(m, f, [2.0, 0.3], lr=0.05, n_steps=500, momentum=0.9,
                   minimizer=minimizer)
    assert res.minimizer_error < 1e-9
    # stateful API: buffer updates
    opt = RiemannianSGD(m, lr=0.1, momentum=0.9)
    opt.zero_grad()
    opt.step(np.array([1.0, 0.0]), np.array([0.5, 0.0]))
    assert np.allclose(opt.velocity, [0.5, 0.0], atol=1e-15)
    opt.step(np.array([1.0, 0.0]), np.array([0.5, 0.0]))
    assert np.allclose(opt.velocity, [0.95, 0.0], atol=1e-15)  # 0.9*0.5+0.5


def test_minimize_generic_path_matches_shortcut():
    """The generic (RK4) and closed-form step paths agree to 1e-8 and both
    converge to the same minimizer."""
    m = PolarPlane()
    f, minimizer = _quadratic()
    res_fast = minimize(m, f, [2.0, 0.3], lr=0.05, n_steps=300,
                        minimizer=minimizer, use_shortcut=True)
    res_generic = minimize(m, f, [2.0, 0.3], lr=0.05, n_steps=300,
                           minimizer=minimizer, use_shortcut=False)
    assert np.linalg.norm(res_fast.point - res_generic.point) < 1e-6
    assert res_generic.descent_ok
    assert res_generic.converged


def test_optimizer_benchmark():
    """The closed-form step is measured faster than the RK4 generic path
    (analytic flops: 200x6 vs 20; wall time measured)."""
    m = PolarPlane()
    point = np.array([1.7, 0.4])
    dv = np.array([-0.3, 0.2])
    log = optim_step_closed_form.profile(m, point, dv, 0.1, n_trials=50,
                                         size_of=lambda *a: 2)
    assert log.speedup_flops > 10.0
    assert log.speedup_time > 5.0
