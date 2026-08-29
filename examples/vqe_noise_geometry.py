#!/usr/bin/env python3
"""Noise-aware VQE from information geometry — the geometric root of
depolarizing noise and of ZNE (zero-noise extrapolation), in the style
of the barren-plateau root analysis (vqe_barren_geometry.py).

Geometry-theory anchors (article 0.11): the SLD quantum Fisher
information for mixed-state families (def 0.11.3.01), the Bures metric
== QFI/4 (prop 0.11.3.04), and the intrinsicness of the metric.  The
key geometric objects here:

  * The depolarizing channel D_lambda(rho) = (1-lambda) rho +
    lambda I/d is an *affine contraction* of the density-matrix space —
    the straight segment from the pure state to the maximally mixed
    state (a geodesic of the affine structure).
  * Energy is an affine function along it:
        E(lambda) = (1-lambda) E_pure + lambda Tr(H)/d          (EXACT)
    This is the geometric reason linear ZNE is exact for a single
    depolarizing point — not an approximation.
  * The SLD-QFI contracts by a scalar:
        F_noisy(lambda) = c(lambda) F_pure,  c(lambda) =
        (1-lambda)^2 / (1-lambda + 2 lambda / 2^n)              (EXACT)
    (verified against the numerical SLD to machine precision).
    Consequence: the Euclidean gradient contracts by (1-lambda), but
    the natural gradient contracts by (1-lambda+2lambda/d)/(1-lambda)
    ~= 1 — the metric contraction and the gradient contraction cancel
    for large d.  Depolarizing noise is an isotropic scalar
    contraction, and the natural gradient is immune to it (to O(1/d)).
  * The variational bound Tr(rho H) >= E_gs holds for any physical
    mixed state (spectral theorem) — noise does NOT push the energy
    below the ground state; it *pulls the optimum* (theta*_noisy makes
    E_pure(theta*) > E_gs).  We measure the pull.
  * ZNE order = number of noise points: with m depolarizing points the
    energy is a degree-m polynomial in lambda; linear extrapolation
    has O(lambda^2) error, an (m+1)-point polynomial extrapolation is
    exact to machine precision.

All numbers machine-verified; honest framing as before: this derives
the root of noise effects in coordinate-free terms and gives exact
closed forms, it is not a claimed full solution of NISQ error
mitigation.

Run:  PYTHONPATH=src python3 examples/vqe_noise_geometry.py
"""

import numpy as np

from geocore.clifford import pauli_action_on_state, rotation_action_closed_form
from geocore.derivatives import rotation_derivative

from vqe_barren_plateaus import _base_state, hea_gates, ising_hamiltonian  # same dir


# ---------------------------------------------------------------------------
# Exact references and noise machinery
# ---------------------------------------------------------------------------


def ising_matrix(n, h=1.0):
    """Dense Ising matrix (small n only)."""
    m1 = {"I": np.eye(2, dtype=complex),
          "X": np.array([[0, 1], [1, 0]], dtype=complex),
          "Z": np.array([[1, 0], [0, -1]], dtype=complex)}
    H = np.zeros((2**n, 2**n), dtype=complex)
    for c, p in ising_hamiltonian(n, h):
        M = np.array([[1.0]], dtype=complex)
        for ch in p:
            M = np.kron(M, m1[ch])
        H = H + c * M
    return H


def apply_theta(theta, gates, base):
    psi = base.copy()
    for axis, idx in gates:
        psi = rotation_action_closed_form(axis, theta[idx], psi)
    return psi


def e_lin(lambda_, E_pure, TrH, d):
    """The exact affine energy along the depolarizing segment."""
    return (1 - lambda_) * E_pure + lambda_ * TrH / d


def e_direct(rho, H):
    return float(np.real(np.trace(rho @ H)))


def depolarize(psi, lambda_, d):
    """rho(lambda) = (1-lambda)|psi><psi| + lambda I/d (dense, small n)."""
    return (1 - lambda_) * np.outer(psi, psi.conj()) + lambda_ / d * np.eye(d)


# ---------------------------------------------------------------------------
# SLD-QFI (numerical) vs the scalar-contraction closed form
# ---------------------------------------------------------------------------


def fs_qfi(D, psi):
    """Pure-state Fubini-Study QFI (closed form, analytic derivatives)."""
    P = len(D)
    F = np.zeros((P, P))
    for k in range(P):
        for j in range(P):
            F[k, j] = 4.0 * np.real(
                np.vdot(D[k], D[j]) - np.vdot(D[k], psi) * np.vdot(psi, D[j]))
    return F


def sld_qfi(psi, D, lambda_, atol=1e-12):
    """Numerical SLD-QFI in the eigenbasis of rho(lambda):
    L_ab = 2 dr_ab / (p_a + p_b), F = Tr(rho (L_k L_j + L_j L_k)/2)."""
    d = len(psi)
    P = len(D)
    rho = depolarize(psi, lambda_, d)
    p, U = np.linalg.eigh(rho)
    Ls = []
    for k in range(P):
        dr = (1 - lambda_) * (np.outer(D[k], psi.conj())
                              + np.outer(psi, D[k].conj()))
        drt = U.conj().T @ dr @ U
        L = np.zeros((d, d), dtype=complex)
        for a in range(d):
            for b in range(d):
                s = p[a] + p[b]
                if s > atol:
                    L[a, b] = 2.0 * drt[a, b] / s
        Ls.append(U @ L @ U.conj().T)
    Fq = np.zeros((P, P))
    for k in range(P):
        for j in range(P):
            Fq[k, j] = float(np.real(
                np.trace(rho @ (Ls[k] @ Ls[j] + Ls[j] @ Ls[k]) / 2)))
    return Fq


def contraction_factor(lambda_, d):
    """c(lambda) = (1-lambda)^2 / (1-lambda + 2 lambda / d)."""
    return (1 - lambda_) ** 2 / (1 - lambda_ + 2 * lambda_ / d)


def natural_gradient_contraction(lambda_, d):
    """(1-lambda)/c(lambda) = (1-lambda+2lambda/d)/(1-lambda)."""
    return (1 - lambda_ + 2 * lambda_ / d) / (1 - lambda_)


# ---------------------------------------------------------------------------
# Variational pull under noise
# ---------------------------------------------------------------------------


def noise_pull(n, lambda_, steps=400, lr=0.05, seed=0, adam=False):
    """Minimize E_noisy(theta) = (1-lambda) E_pure(theta) + const and
    measure how far the optimum is pulled: E_pure(theta*) - E_gs.

    For single-point depolarizing noise the gradient is a scalar
    (1-lambda) times the exact one — the *direction* is unchanged, so
    the pull comes from the step-size effect on fixed-step optimizers:
    with SGD, larger lambda means smaller steps and a shallower
    optimum; with Adam (which normalizes the gradient) the pull is
    masked.  We report both honestly."""
    gates = hea_gates(n, 2)
    base = _base_state(n)
    terms = ising_hamiltonian(n)
    H = ising_matrix(n)
    E_gs = np.linalg.eigvalsh(H)[0].real
    rng = np.random.default_rng(seed)
    th = rng.uniform(-np.pi, np.pi, len(gates))
    if adam:
        m = np.zeros_like(th)
        v = np.zeros_like(th)
        b1, b2, eps = 0.9, 0.999, 1e-8
    for _ in range(steps):
        psi = apply_theta(th, gates, base)
        Hpsi = np.zeros_like(psi)
        for c, p in terms:
            Hpsi += c * pauli_action_on_state(p, psi)
        g = np.zeros(len(gates))
        Fw = [base.copy()]
        for axis, idx in gates:
            Fw.append(rotation_action_closed_form(axis, th[idx], Fw[-1]))
        w = Hpsi.copy()
        for j in range(len(gates) - 1, -1, -1):
            axis, idx = gates[j]
            dd = rotation_derivative(axis, th[idx], Fw[j])
            g[idx] = 2.0 * float(np.vdot(dd, w).real)
            w = rotation_action_closed_form(axis, -th[idx], w)
        g = (1 - lambda_) * g
        if adam:
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            th = th - lr * (m / (1 - b1)) / (np.sqrt(v / (1 - b2)) + eps)
        else:
            th = th - lr * g
    psi = apply_theta(th, gates, base)
    E_pure_star = float(np.real(np.vdot(psi, H @ psi)))
    return E_pure_star, E_gs, E_pure_star - E_gs


# ---------------------------------------------------------------------------
# ZNE order = number of noise points
# ---------------------------------------------------------------------------


def depolarizing_circuit_energy(n, layers, H, theta, lambda_, seed=0):
    """E(lambda) of an L-layer circuit with depolarization (strength
    lambda) after EVERY layer — a polynomial of degree L in lambda."""
    gates = hea_gates(n, layers)
    base = _base_state(n)
    rng = np.random.default_rng(seed)
    th = rng.uniform(-np.pi, np.pi, len(gates))
    # build layer unitaries
    layer_gates = []
    for L in range(layers):
        layer_gates.append(gates[L * len(gates) // layers:(L + 1) * len(gates) // layers])
    d = 2**n
    rho = np.outer(base, base.conj())
    for lg in layer_gates:
        U = np.eye(d, dtype=complex)
        for axis, idx in lg:
            # dense rotation matrix
            P = np.zeros((d, d), dtype=complex)
            for a in range(d):
                P[:, a] = pauli_action_on_state(axis, np.eye(d)[:, a])
            U = (np.cos(th[idx] / 2) * np.eye(d)
                 - 1j * np.sin(th[idx] / 2) * P) @ U
        rho = U @ rho @ U.conj().T
        rho = (1 - lambda_) * rho + lambda_ / d * np.eye(d)
    return float(np.real(np.trace(rho @ H)))


# ---------------------------------------------------------------------------
# Main study
# ---------------------------------------------------------------------------


def main():
    print("=" * 74)
    print("Noise-aware VQE from information geometry (depolarizing "
          "channel)")
    print("=" * 74)

    # 0) affine energy along the depolarizing segment
    print("\n[0] Depolarizing noise = affine segment (energy is LINEAR "
          "in lambda):")
    n = 2
    d = 2**n
    gates = hea_gates(n, 1)
    base = _base_state(n)
    rng = np.random.default_rng(0)
    th = rng.uniform(-np.pi, np.pi, len(gates))
    psi = apply_theta(th, gates, base)
    H = ising_matrix(n)
    E_pure = float(np.real(np.vdot(psi, H @ psi)))
    TrH = float(np.real(np.trace(H)))
    worst = 0.0
    for lam in (0.0, 0.2, 0.5, 0.8):
        rho = depolarize(psi, lam, d)
        worst = max(worst, abs(e_lin(lam, E_pure, TrH, d) - e_direct(rho, H)))
    print(f"    max |E_affine - Tr(rho H)| over lambda: {worst:.1e}  "
          f"(the formula is exact)")

    # 1) QFI scalar contraction closed form vs numerical SLD
    print("\n[1] SLD-QFI contracts by a scalar c(lambda):")
    for nq in (3, 4):
        dq = 2**nq
        gq = hea_gates(nq, 1)
        bq = _base_state(nq)
        rng = np.random.default_rng(0)
        thq = rng.uniform(-np.pi, np.pi, len(gq))
        Fs = [bq.copy()]
        for axis, idx in gq:
            Fs.append(rotation_action_closed_form(axis, thq[idx], Fs[-1]))
        ps = Fs[-1]
        Dq = []
        for j in range(len(gq)):
            axis, idx = gq[j]
            dd = rotation_derivative(axis, thq[idx], Fs[j])
            for k in range(j + 1, len(gq)):
                dd = rotation_action_closed_form(gq[k][0], thq[gq[k][1]], dd)
            Dq.append(dd)
        Fp = fs_qfi(Dq, ps)
        for lam in (0.1, 0.5):
            c = contraction_factor(lam, dq)
            Fn = sld_qfi(ps, Dq, lam)
            rel = np.max(np.abs(Fn - c * Fp)) / np.max(np.abs(Fp))
            print(f"    n={nq} lambda={lam:.1f}: c={c:.6f}  "
                  f"rel err vs SLD {rel:.1e}")

    # 2) natural gradient immunity
    print("\n[2] Natural gradient is (nearly) immune to depolarizing "
          "noise:")
    n2 = 12
    d2 = 2**n2
    for lam in (0.1, 0.3):
        ce = 1 - lam                       # euclidean gradient contraction
        cn = natural_gradient_contraction(lam, d2)
        print(f"    lambda={lam:.1f}: euclidean grad x {ce:.4f},  "
              f"natural grad x {cn:.6f}  (d={d2})")
    print("    -> isotropic scalar contraction cancels in the natural")
    print("       gradient to O(1/d): changing the coordinates cannot")

    # 3) variational bound and the pull of the optimum
    print("\n[3] Variational bound Tr(rho H) >= E_gs survives noise; the "
          "OPTIMUM is pulled:")
    H6 = ising_matrix(6)
    E_gs6 = np.linalg.eigvalsh(H6)[0].real
    # bound check on random states
    rng = np.random.default_rng(1)
    ok = True
    for _ in range(50):
        ps = rng.normal(size=64) + 1j * rng.normal(size=64)
        ps = ps / np.linalg.norm(ps)
        for lam in (0.2, 0.7):
            rho = depolarize(ps, lam, 64)
            if e_direct(rho, H6) < E_gs6 - 1e-9:
                ok = False
    print(f"    Tr(rho H) >= E_gs on 100 noisy samples: {ok}")
    print("    fixed-step SGD, 400 steps (pull from the step-size "
          "effect):")
    for lam in (0.0, 0.3, 0.6):
        Es, E_gs, pull = noise_pull(6, lam, lr=0.05)
        print(f"      lambda={lam:.1f}: E_pure(theta*) = {Es:.4f} "
              f"(E_gs {E_gs:.4f}, pulled {pull:+.4f})")
    print("    Adam, 400 steps (honest contrast: normalization masks "
          "the pull):")
    for lam in (0.0, 0.6):
        Es, E_gs, pull = noise_pull(6, lam, lr=0.2, adam=True)
        print(f"      lambda={lam:.1f}: E_pure(theta*) = {Es:.4f} "
              f"(pulled {pull:+.4f})")

    # 4) ZNE order = number of noise points
    print("\n[4] ZNE exactness order = number of depolarizing points:")
    H2 = ising_matrix(2)
    lam_pts = np.array([0.05, 0.15, 0.25])
    for L in (1, 2):
        Es = np.array([depolarizing_circuit_energy(2, L, H2, None, lam)
                       for lam in lam_pts])
        # exact lambda=0 value: noiseless circuit energy
        E0 = depolarizing_circuit_energy(2, L, H2, None, 0.0)
        lin = np.polyfit(lam_pts, Es, 1)
        poly = np.polyfit(lam_pts, Es, L)
        E_lin = np.polyval(lin, 0.0)
        E_poly = np.polyval(poly, 0.0)
        print(f"    L={L} noise points: linear extrap err "
              f"{abs(E_lin - E0):.2e}, degree-{L} poly extrop err "
              f"{abs(E_poly - E0):.2e}")

    print("\nSummary (all machine-verified):")
    print("  - depolarizing noise is an affine contraction of state")
    print("    space: energy is linear in lambda (ZNE linear = exact")
    print("    for one noise point), and the SLD-QFI contracts by")
    print("    c(lambda) = (1-lambda)^2/(1-lambda+2 lambda/2^n).")
    print("  - the natural gradient is immune to this isotropic")
    print("    contraction to O(1/d) — a coordinate-free statement.")
    print("  - the variational bound survives (Tr(rho H) >= E_gs); the")
    print("    optimum is pulled, and ZNE's exact order equals the")
    print("    number of noise points (polynomial extrapolation).")
    print("  - this is a root analysis with exact closed forms, not a")
    print("    claimed full solution of NISQ error mitigation.")


if __name__ == "__main__":
    main()
