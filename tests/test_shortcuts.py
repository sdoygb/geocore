"""Tests: Layer 3 reduce-computation (shortcuts + benchmarks)."""

import numpy as np

from geocore import Pauli, Rotation, get_op
from geocore.shortcuts import registry


def _state(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)


def test_shortcut_matches_generic_to_machine_precision():
    r = Rotation("XYZ", 0.7)
    s = _state(3)
    res, report = registry.apply("rotation.closed_form", r, s, verify=True)
    assert report.ok, report.details
    assert report.max_error < 1e-9


def test_shortcut_reduces_computation_measured():
    """The first measured claim: closed-form Pauli rotation beats expm."""
    for n in [3, 4, 5]:
        r = Rotation("XYZ"[: n % 3 + 1].ljust(n, "X"), 0.7)
        s = _state(n)
        log = registry.benchmark(
            "rotation.closed_form", r, s, n_trials=20, size_of=lambda a, b: len(a.axis)
        )
        assert log.speedup_time > 1.0, f"n={n}: no time speedup: {log}"
        assert log.speedup_flops > 1.0, f"n={n}: no flops speedup: {log}"


def test_shortcut_scales_exponentially_better():
    """FLOPs ratio grows like 4^n (d^3 vs d with d = 2^n)."""
    log_small = registry.get("rotation.closed_form").profile(
        Rotation("XX", 0.7), _state(2), n_trials=5, size_of=lambda a, b: len(a.axis)
    )
    log_big = registry.get("rotation.closed_form").profile(
        Rotation("XX" * 2, 0.7), _state(4), n_trials=5, size_of=lambda a, b: len(a.axis)
    )
    assert log_big.speedup_flops > log_small.speedup_flops


def test_rotation_apply_operator_with_invariant():
    """The generic op self-checks against the closed form (Layer 2)."""
    r = Rotation("XY", 0.4)
    s = _state(2)
    result = get_op("rotation.apply_to_state")(r, s)
    assert result.shape == s.shape


def test_geodesic_shortcut_matches_generic():
    """Closed-form geodesic == RK4 integration to machine precision."""
    from geocore import PolarPlane, get_op
    from geocore.shortcuts import registry

    m = PolarPlane()
    rng = np.random.default_rng(3)
    for _ in range(10):
        init = np.array([rng.uniform(1.2, 3.0), rng.uniform(-1.0, 1.0)])
        vel = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.2, 0.2)])
        t = float(rng.uniform(0.1, 0.9))
        generic = get_op("geodesic.polar_point")(m, init, vel, t)
        fast, report = registry.apply("geodesic.polar_closed_form", m, init, vel, t, verify=True)
        assert report.ok, report.details
        assert report.max_error < 1e-9
        assert np.allclose(generic.point, fast.point, atol=1e-9)


def test_geodesic_energy_conservation_invariant():
    """The generic ODE path self-checks energy conservation (Layer 2)."""
    from geocore import PolarPlane, get_op

    m = PolarPlane()
    init = np.array([2.0, 0.8])
    vel = np.array([0.2, 0.15])
    res = get_op("geodesic.polar_point")(m, init, vel, 0.5)  # raises if invariant fails
    e0 = m.metric_norm_sq(init[0], vel[0], vel[1])
    e1 = m.metric_norm_sq(res.point[0], res.velocity[0], res.velocity[1])
    assert abs(e1 - e0) < 1e-9


def test_geodesic_shortcut_measured_speedup():
    from geocore import PolarPlane
    from geocore.shortcuts import registry

    m = PolarPlane()
    init = np.array([2.0, 0.8])
    vel = np.array([0.2, 0.15])
    log = registry.benchmark("geodesic.polar_closed_form", m, init, vel, 0.5,
                             n_trials=100, size_of=lambda *a: 2)
    assert log.speedup_time > 1.0
    assert log.speedup_flops > 1.0
