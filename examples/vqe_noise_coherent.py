#!/usr/bin/env python3
"""Coherent (unitary rotation) noise from information geometry — the
fourth and last fingerprint in the noise-channel spectrum (features
34-36).

Coherent noise replaces a gate by U(th) = cos(th/2) I + i sin(th/2) E
(a rotation about a Pauli axis E).  Unlike depolarizing, amplitude
damping and phase damping, it is UNITARY: pure states stay pure
(rank-1), so the mixed-state SLD machinery is not needed — the pure
Fubini-Study metric applies directly.

Geometry-theory anchors (article 0.11): the intrinsicness of the
metric (Sec 2.1) and the fact that unitary maps preserve the
Fubini-Study metric (the U(2^n) isometry of state space).  Machine-
verified findings:

  1. PURITY: coherent noise keeps rho rank-1 to machine precision
     (the orbit is a geodesic arc of CP^N, not an affine segment into
     the interior).
  2. ENERGY TRACK closed form (single noise point, axis E):
        E(th) = A cos^2(th/2) + B sin^2(th/2) + C sin(th),
     A = <H>, B = <E H E>, C = (i/2)<[P_E, H]>        (2e-16)
     The channel constant of the QEC geometry (eps = sin^2(th/2))
     enters: with x = sin^2(th/2), E(x) = A(1-x) + B x +
     C*2*sqrt(x(1-x)) — linear in x ONLY when C = 0.
  3. METRIC fingerprint: ZERO contraction.  A fixed unitary (the noise
     does not depend on the circuit parameters) preserves the
     FS-QFI exactly (1e-16), unlike depolarizing (scalar c), amplitude
     damping (scalar 1-g) and phase damping (anisotropic).  This is
     the fourth fingerprint: unitary noise does not shrink the metric.
  4. NATURAL GRADIENT is NOT immune: the metric is unchanged but the
     gradient g = dE/dtheta changes (the energy landscape rotates), so
     g_nat = F^-1 g changes — the opposite of depolarizing noise.
  5. ZNE: in the eps = sin^2(th/2) space, linear extrapolation is
     exact when C = 0 (commuting axis / special state) and has the
     O(sqrt(eps(1-eps))) residual when C != 0; in the th space the
     error is O(th^2).  Measured for three axes.
  6. The variational bound survives trivially (pure states).

Honest framing: the fourth fingerprint completes the noise-channel
spectrum (affine-isotropic / metric-scalar-anisotropic-energy /
metric-anisotropic / metric-zero unitary); it is a root classification,
machine-verified, not a claimed full solution of NISQ mitigation.

Run:  PYTHONPATH=src python3 examples/vqe_noise_coherent.py
"""

import numpy as np

from geocore.clifford import pauli_action_on_state, rotation_action_closed_form
from geocore.derivatives import rotation_derivative

from vqe_barren_plateaus import _base_state, hea_gates  # same directory
from vqe_noise_geometry import ising_matrix  # same directory


def pauli_mat(axis, n):
    d = 2**n
    P = np.zeros((d, d), dtype=complex)
    for a in range(d):
        e = np.zeros(d, dtype=complex)
        e[a] = 1
        P[:, a] = pauli_action_on_state(axis, e)
    return P


def rot_U(P, th):
    return np.cos(th / 2) * np.eye(P.shape[0]) - 1j * np.sin(th / 2) * P


def purity_max_eig(psi):
    return float(np.linalg.eigvalsh(np.outer(psi, psi.conj()))[-1])


def deriv_states(theta, gates, base):
    F = [base.copy()]
    for axis, idx in gates:
        F.append(rotation_action_closed_form(axis, theta[idx], F[-1]))
    psi = F[-1]
    D = []
    for j in range(len(gates)):
        axis, idx = gates[j]
        dd = rotation_derivative(axis, theta[idx], F[j])
        for k in range(j + 1, len(gates)):
            dd = rotation_action_closed_form(gates[k][0], theta[gates[k][1]], dd)
        D.append(dd)
    return psi, D


def fs_qfi(D, psi):
    P = len(D)
    F = np.zeros((P, P))
    for k in range(P):
        for j in range(P):
            F[k, j] = 4.0 * np.real(
                np.vdot(D[k], D[j]) - np.vdot(D[k], psi) * np.vdot(psi, D[j]))
    return F


def energy_track(psi, H, P, ths):
    """E(th) numerically vs the closed form A cos^2+B sin^2+C sin."""
    A = float(np.real(np.vdot(psi, H @ psi)))
    B = float(np.real(np.vdot(psi, P @ H @ P @ psi)))
    C = float(np.real(0.5j * np.vdot(psi, (P @ H - H @ P) @ psi)))
    worst = 0.0
    for th in ths:
        psip = rot_U(P, th) @ psi
        E_num = float(np.real(np.vdot(psip, H @ psip)))
        E_form = (A * np.cos(th / 2) ** 2 + B * np.sin(th / 2) ** 2
                  + C * np.sin(th))
        worst = max(worst, abs(E_num - E_form))
    return A, B, C, worst


def zne_eps_extrap(psi, H, P, ths):
    """Linear ZNE in the eps = sin^2(th/2) space vs exact E(0)."""
    eps = np.sin(ths / 2) ** 2
    Es = np.array([float(np.real(np.vdot(rot_U(P, t) @ psi,
                                         H @ (rot_U(P, t) @ psi))))
                   for t in ths])
    E0 = float(np.real(np.vdot(psi, H @ psi)))
    lin = np.polyval(np.polyfit(eps, Es, 1), 0.0)
    lin_th = np.polyval(np.polyfit(ths, Es, 1), 0.0)
    return abs(lin - E0), abs(lin_th - E0)


def main():
    print("=" * 74)
    print("Coherent (unitary) rotation noise from information geometry")
    print("=" * 74)

    n = 2
    gates = hea_gates(n, 1)
    base = _base_state(n)
    rng = np.random.default_rng(0)
    th0 = rng.uniform(-np.pi, np.pi, len(gates))
    psi = base.copy()
    for axis, idx in gates:
        psi = rotation_action_closed_form(axis, th0[idx], psi)
    H = ising_matrix(n)

    # 1) purity
    print("\n[1] Coherent noise keeps the state pure (rank-1):")
    for ax in ("XI", "IZ", "ZZ"):
        P = pauli_mat(ax, n)
        for th in (0.1, 0.9):
            pur = purity_max_eig(rot_U(P, th) @ psi)
            print(f"    axis {ax} th={th}: max eig {pur:.15f} "
                  f"(rank-1: {abs(pur - 1) < 1e-12})")

    # 2) energy closed form
    print("\n[2] Energy track closed form "
          "E(th) = A cos^2 + B sin^2 + C sin (exact):")
    ths = np.array([0.05, 0.2, 0.5, 0.9, 1.3])
    for ax in ("XI", "IZ", "ZZ"):
        P = pauli_mat(ax, n)
        A, B, C, worst = energy_track(psi, H, P, ths)
        print(f"    axis {ax}: max err {worst:.1e}  (A={A:+.4f} "
              f"B={B:+.4f} C={C:+.4f})")

    # 3) metric fingerprint: zero contraction
    print("\n[3] Metric fingerprint: ZERO contraction (FS-QFI is "
          "preserved by a fixed unitary):")
    _, D = deriv_states(th0, gates, base)
    F0 = fs_qfi(D, psi)
    for ax in ("XI", "IZ", "ZZ"):
        P = pauli_mat(ax, n)
        for th in (0.2, 0.7):
            Un = rot_U(P, th)
            F1 = fs_qfi([Un @ dd for dd in D], Un @ psi)
            err = np.max(np.abs(F1 - F0))
            print(f"    axis {ax} th={th}: max|F - F0| = {err:.1e} "
                  f"(zero contraction: {err < 1e-10})")

    # 4) natural gradient NOT immune
    print("\n[4] Natural gradient is NOT immune (metric unchanged, "
          "gradient rotated):")
    g0 = 2.0 * np.real(np.array([np.vdot(dd, H @ psi) for dd in D]))
    P = pauli_mat("XI", n)
    Un = rot_U(P, 0.4)
    psip = Un @ psi
    gN = 2.0 * np.real(np.array([np.vdot(Un @ dd, H @ psip) for dd in D]))
    Fn = fs_qfi([Un @ dd for dd in D], psip)
    reg = 1e-10 * np.eye(len(gates))
    nat0 = np.linalg.solve(F0 + reg, g0)
    natN = np.linalg.solve(Fn + reg, gN)
    print(f"    |g|: {np.linalg.norm(g0):.4f} -> {np.linalg.norm(gN):.4f}")
    print(f"    |g_nat|: {np.linalg.norm(nat0):.4f} -> "
          f"{np.linalg.norm(natN):.4f}  (changes, unlike depolarizing)")

    # 5) ZNE: eps = sin^2(th/2) space
    print("\n[5] ZNE in the eps = sin^2(th/2) space (the QEC channel "
          "constant):")
    thz = np.array([0.3, 0.5, 0.7])
    for ax in ("XI", "IZ", "ZZ"):
        P = pauli_mat(ax, n)
        e_eps, e_th = zne_eps_extrap(psi, H, P, thz)
        _, _, C, _ = energy_track(psi, H, P, ths)
        print(f"    axis {ax} (C={C:+.4f}): eps-space lin err {e_eps:.2e}"
              f", th-space lin err {e_th:.2e}")

    # 6) variational bound (trivial for pure states)
    print("\n[6] Variational bound Tr(rho H) >= E_gs (trivial for pure "
          "states):")
    E_gs = np.linalg.eigvalsh(H)[0].real
    ok = True
    for ax in ("XI", "IZ", "ZZ"):
        P = pauli_mat(ax, n)
        for th in (0.2, 0.7):
            psip = rot_U(P, th) @ psi
            if float(np.real(np.vdot(psip, H @ psip))) < E_gs - 1e-9:
                ok = False
    print(f"    holds on all noisy pure states: {ok}")

    print("\nSummary: the four fingerprints of the noise spectrum:")
    print("  depolarizing : affine segment, scalar QFI contraction c,")
    print("                 natural gradient immune")
    print("  amp. damping : basis-dependent energy, scalar QFI (1-g),")
    print("                 partial natural-gradient immunity")
    print("  phase damping: anisotropic QFI contraction")
    print("  coherent     : geodesic arc (pure), ZERO metric")
    print("                 contraction, natural gradient NOT immune")
    print("  Linear ZNE is exact in the eps=sin^2(th/2) space iff the")
    print("  rotation axis commutes with H at the state (C = 0).")
    print("  Honest: a root classification, not a full solution.")


if __name__ == "__main__":
    main()
