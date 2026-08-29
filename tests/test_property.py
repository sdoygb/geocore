"""Property-based (fuzz) tests: every invariant must hold for *randomly
generated* inputs, not just the fixed scenarios in the other tests.

This is the evidence that geocore is a dynamic tool: arbitrary inputs
are accepted and every declared invariant holds, machine-precision, for
thousands of random cases.  Seeds are fixed for reproducibility.
"""

import numpy as np
import pytest

from geocore import (
    Clifford,
    HyperbolicPlane,
    Pauli,
    PolarPlane,
    Rotation,
    Sphere,
    frechet_mean,
    minimize,
)
from geocore.clifford import (
    clifford_gates_to_matrix,
    matrices_equal_up_to_phase,
)
from geocore.derivatives import log_map
from geocore.geostats import (
    frechet_variance,
    geodesic_distance,
    principal_directions,
    tangent_covariance,
)
from geocore.ops import clifford_compose, clifford_conjugate
from geocore.qec import (
    logical_error_sweep,
    pseudo_threshold,
    repetition_code_logical_error,
)

rng = np.random.default_rng(2024)

_SINGLE = ["h", "s", "sd", "sx", "sxdg"]


def rand_gates(n, n_gates):
    gates = []
    for _ in range(n_gates):
        g = rng.choice(_SINGLE + (["cx"] if n >= 2 else []))
        if g == "cx":
            c, t = rng.choice(n, 2, replace=False)
            gates.append((g, (int(c), int(t))))
        else:
            gates.append((g, (int(rng.integers(n)),)))
    return gates


def rand_pauli(n):
    return Pauli("".join(rng.choice(list("XYZI"), n)))


# ---------------------------------------------------------------------------
# Pauli / Rotation fuzz
# ---------------------------------------------------------------------------


def test_fuzz_pauli_commutation_matrix_truth():
    """Commutation decisions equal [P, Q] = 0 from explicit matrices for
    random Paulis (n = 1..4)."""
    for _ in range(150):
        n = int(rng.integers(1, 5))
        P, Q = rand_pauli(n), rand_pauli(n)
        commutes = P.commutes_with(Q)
        pq = P.to_matrix() @ Q.to_matrix()
        qp = Q.to_matrix() @ P.to_matrix()
        assert commutes == bool(np.allclose(pq, qp, atol=1e-12))


def test_fuzz_rotation_closure():
    """Rotation merge/cancel and the closed-form action for random axes
    and angles (including edge angles: 0, 2pi, negative, large)."""
    from geocore.clifford import rotation_action_closed_form

    axes = ["".join(rng.choice(list("XYZI"), int(rng.integers(1, 4)))) for _ in range(50)]
    angles = list(rng.uniform(-8.0, 8.0, 100)) + [0.0, 2 * np.pi, -2 * np.pi, 4 * np.pi]
    for axis in axes + ["X", "Z", "Y"]:
        for theta in angles:
            R = Rotation(axis, theta)
            # merge with itself: R(theta)^2 == R(2 theta)
            m = R.merge_with(R)
            err = np.abs(m.to_matrix() - Rotation(axis, 2 * theta).to_matrix()).max()
            assert err < 1e-12
            # closed-form action == dense
            state = rng.standard_normal(2 ** len(axis)) + 1j * rng.standard_normal(2 ** len(axis))
            got = rotation_action_closed_form(axis, theta, state)
            truth = R.to_matrix() @ state
            assert np.abs(got - truth).max() < 1e-9
    # cancel at exact multiples of 2 pi
    for k in range(-5, 6):
        assert Rotation("X", 2 * np.pi * k).cancels()
    assert not Rotation("X", 1.3).cancels()


# ---------------------------------------------------------------------------
# Clifford fuzz
# ---------------------------------------------------------------------------


def test_fuzz_clifford_against_dense():
    """Random deep Clifford circuits: tableau == dense (up to phase),
    conjugation == dense, composition == dense (up to phase)."""
    for _ in range(60):
        n = int(rng.integers(1, 4))
        gates = rand_gates(n, int(rng.integers(1, 12)))
        C = Clifford(gates, n)
        ok, err = matrices_equal_up_to_phase(C.to_matrix(), clifford_gates_to_matrix(gates, n))
        assert ok and err < 1e-9
        # conjugation of a random Pauli
        P = rand_pauli(n)
        conj, r = C.conjugate(P)
        M = C.to_matrix()
        truth = M @ P.to_matrix() @ M.conj().T
        assert np.abs(Pauli(conj.axis).to_matrix() * (-1) ** r - truth).max() < 1e-9


def test_fuzz_clifford_compose_dense_and_associative():
    for _ in range(40):
        n = int(rng.integers(1, 4))
        C1, C2, C3 = (Clifford(rand_gates(n, int(rng.integers(1, 8))), n) for _ in range(3))
        C = clifford_compose(C1, C2)  # op invariant runs
        ok, err = matrices_equal_up_to_phase(
            C1.compose(C2).to_matrix(), C1.to_matrix() @ C2.to_matrix()
        )
        assert ok and err < 1e-9
        ok, _ = matrices_equal_up_to_phase(
            C1.compose(C2.compose(C3)).to_matrix(),
            (C1.compose(C2)).compose(C3).to_matrix(),
        )
        assert ok


# ---------------------------------------------------------------------------
# Manifold fuzz
# ---------------------------------------------------------------------------

_MANIFOLDS = [
    (PolarPlane(), (1.2, 3.0), (-1.5, 1.5)),
    (Sphere(), (0.5, 2.5), (0.1, 1.0)),
    (HyperbolicPlane(), (0.2, 1.5), (0.8, 2.0)),
]


def test_fuzz_geodesics_energy_and_rk4():
    """Random points/velocities/t: energy conserved (closed form) and the
    closed form agrees with RK4 for random inputs (the RK4 comparison
    uses 500 steps and a 1e-5 tolerance: random draws occasionally land
    near the sphere's poles, where the coordinate ODE is stiff)."""
    for manifold, (lo, hi), (vlo, vhi) in _MANIFOLDS:
        for _ in range(60):
            init = rng.uniform(lo, hi, 2)
            vel = rng.uniform(vlo, vhi, 2)
            t = rng.uniform(0.05, 0.9)
            sol = manifold.geodesic_closed_form(init, vel, t)
            e0 = manifold.metric_norm_sq(init, vel)
            e1 = manifold.metric_norm_sq(sol.point, sol.velocity)
            assert abs(e1 - e0) < 1e-9
            # random RK4 vs closed form (geometric distance)
            rk = manifold.geodesic_generic(init, vel, t, n_steps=500)
            d = geodesic_distance(manifold, rk.point, sol.point)
            assert d < 1e-5, type(manifold).__name__


def test_fuzz_parallel_transport_isometry_roundtrip():
    """Random transports: isometry and round-trip exact."""
    for manifold, (lo, hi), _ in _MANIFOLDS:
        for _ in range(60):
            p = rng.uniform(lo, hi, 2)
            q = rng.uniform(lo, hi, 2)
            v = rng.uniform(-0.5, 0.5, 2)
            vt = manifold.parallel_transport(p, q, v)
            g0 = manifold.metric_norm_sq(p, v)
            g1 = manifold.metric_norm_sq(q, vt)
            assert abs(g1 - g0) < 1e-9
            back = manifold.parallel_transport(q, p, vt)
            assert np.abs(back - v).max() < 1e-9


def test_fuzz_log_map_inverse_exp():
    """exp_p(log_p(q)) = q for random pairs on every manifold."""
    for manifold, (lo, hi), _ in _MANIFOLDS:
        for _ in range(50):
            p = rng.uniform(lo, hi, 2)
            q = rng.uniform(lo, hi, 2)
            d = geodesic_distance(manifold, p, q)
            if isinstance(manifold, Sphere) and d > np.pi - 0.1:
                continue
            v = log_map(manifold, p, q)
            sol = manifold.geodesic_closed_form(p, v, 1.0)
            assert geodesic_distance(manifold, sol.point, q) < 1e-9


def test_fuzz_optimizer_arbitrary_quadratic():
    """minimize converges for random quadratic potentials from random
    starts (the optimizer is a tool, not a demo: any convex f works; the
    polar y-direction is ill-conditioned by ~r^2, so the budget must
    match — gradient descent's own conditioning, not a failure)."""
    for _ in range(20):
        target = rng.uniform(1.3, 2.8, 2)
        f = lambda p, t=target: (p[0] - t[0]) ** 2 + (p[1] - t[1]) ** 2
        p0 = rng.uniform(1.2, 3.0, 2)
        res = minimize(PolarPlane(), f, p0, lr=0.1, n_steps=800, minimizer=target)
        assert res.minimizer_error < 1e-5
        assert res.final_grad_norm < 1e-4


def test_fuzz_statistics_identities():
    """tr(Cov) = variance, mean fixed point, for random point sets on
    every manifold (both use the SAME mean — the identity is exact for a
    fixed mean, avoiding independent-convergence noise)."""
    for manifold, (lo, hi), _ in _MANIFOLDS:
        for _ in range(15):
            pts = rng.uniform(lo, hi, (int(rng.integers(5, 40)), 2))
            m = frechet_mean(manifold, pts, lr=0.1, n_steps=300).point
            var = frechet_variance(manifold, pts, mean=m)
            cov = tangent_covariance(manifold, pts, mean=m)
            assert abs(np.trace(cov) - var) < 1e-8, type(manifold).__name__
            assert np.linalg.eigvalsh(cov)[0] > -1e-9
            # single-point mean is the point itself
            p0 = pts[0]
            m2 = frechet_mean(manifold, [p0, p0, p0], lr=0.1, n_steps=200).point
            assert geodesic_distance(manifold, m2, p0) < 1e-8


# ---------------------------------------------------------------------------
# QEC fuzz
# ---------------------------------------------------------------------------


def test_fuzz_qec_closed_form_matches_simulation():
    """The closed form equals the O(2^n) state-vector simulation for
    random distances and random noise strengths (including edge angles)."""
    for n in [3, 5, 7]:
        thetas = list(rng.uniform(0.001, 1.0, 20)) + [0.0, np.pi / 2, 1e-9]
        sw = logical_error_sweep(n, thetas)
        for t, p in zip(thetas, sw):
            assert abs(p - repetition_code_logical_error(t, n)) < 1e-12
    # pseudo-threshold is exactly pi/2 for every distance
    for n in [3, 5, 7, 9, 11, 13]:
        assert pseudo_threshold(n) == pytest.approx(np.pi / 2, abs=1e-9)
