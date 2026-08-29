#!/usr/bin/env python3
"""Discrete dynamic evolution — the zero-gradient solver that is immune
to barren plateaus (the "evolve, don't optimize" route).

The root finding (article 10.86): barren plateaus are a disease of the
CONTINUOUS parameterization — random parameterized circuits flood the
Hilbert space and high-dimensional statistics flatten the landscape, so
the gradient decays exponentially.  The geometric answer is the
discrete dynamic evolution of the geometry theory (0.13 zero-motion /
0.14 dynamic evolution): solve by EVOLUTION STEPS, not by continuous
gradient descent.  With a fixed adiabatic schedule there are zero
parameters and zero gradients, so there is nothing to plateau.

The circuit is the QAOA structure (layers e^{-i beta H_B} e^{-i gamma
H_C}) but with the parameters FIXED to the discretized adiabatic path:

    H(s) = s * H_Ising + (1-s) * (-sum X)      (s: 0 -> 1)
         = s * (ZZ part) + (2s-1) * (X part)   (h = 1)

    gamma_k = dt * s_k        (diagonal ZZ part, s_k = (k+0.5)/p)
    beta_k  = dt * (2 s_k - 1)(transverse X part, sign changes)

Cost: O(p * 2^n), p polynomial (n=10: p=4000, dt=0.1); T = p*dt grows
polynomially with n (adiabatic, not exponential).  Measured fidelity:

    n= 4: 0.993   n= 6: 0.989   n= 8: 0.984   n=10: 0.979
    (p = 2000-4000, T = 200-400)

vs the continuous-gradient HEA VQE, which at n=10-12 has gradient RMS
~1e-4..1e-7 and gets stuck (barren).

Honest framing: this is the adiabatic/quantum-simulation route (it
leaves the variational framework, no variational-advantage claim); the
T ~ 1/gap^2 growth with n is adiabatic-typical (polynomial, not the
exponential of the plateau); convergence plateaus at ~0.98 (adiabatic /
Trotter residual).  What it demonstrates: the plateau is a disease of
continuous parameterization — discrete evolution does not enter that
framework at all.

Run:  PYTHONPATH=src python3 examples/vqe_discrete_evolution.py
"""

import numpy as np

from geocore.clifford import rotation_action_closed_form

from vqe_barren_plateaus import _base_state, ising_hamiltonian  # same dir
from vqe_barren_prewarm import ising_ground_state  # same dir


def diag_values(n, terms):
    """Diagonal (Z-only) values of the Ising Hamiltonian."""
    vals = np.zeros(2**n)
    for c, p in terms:
        if "X" in p:
            continue
        idxs = [i for i, ch in enumerate(p) if ch == "Z"]
        sign = np.ones(2**n)
        for i in idxs:
            b = ((np.arange(2**n) >> (n - 1 - i)) & 1).astype(float)
            sign *= (1 - 2 * b)
        vals += c * sign
    return vals


def discrete_adiabatic(n, p, T, C, base=None):
    """Zero-gradient discrete adiabatic evolution with the fixed
    schedule gamma_k = dt s_k, beta_k = dt (2 s_k - 1)."""
    if base is None:
        base = _base_state(n)
    dt = T / p
    psi = np.ones(2**n, dtype=complex) / np.sqrt(2**n)  # |+>^n
    for k in range(p):
        s = (k + 0.5) / p
        g = dt * s
        b = dt * (2 * s - 1)
        psi = psi * np.exp(-1j * g * C)  # e^{-i gamma H_C}
        for q in range(n):               # e^{-i beta H_B}
            axis = ["I"] * n
            axis[q] = "X"
            psi = rotation_action_closed_form("".join(axis), 2 * b, psi)
    return psi


def ising_energy(psi, n, terms):
    """<psi|H|psi> with the full Ising H (ZZ + X parts)."""
    d = 2**n
    m1 = {"I": np.eye(2, dtype=complex),
          "X": np.array([[0, 1], [1, 0]], dtype=complex),
          "Z": np.array([[1, 0], [0, -1]], dtype=complex)}
    H = np.zeros((d, d), dtype=complex)
    for c, p in terms:
        M = np.array([[1.0]], dtype=complex)
        for ch in p:
            M = np.kron(M, m1[ch])
        H = H + c * M
    return float(np.real(np.vdot(psi, H @ psi)))


def hea_gradient_scale(n, gates, base, gs, npts=10, seed=0):
    """HEA gradient RMS on the fidelity cost (the barren baseline)."""
    from geocore.derivatives import rotation_derivative
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(npts):
        th = rng.uniform(-np.pi, np.pi, len(gates))
        psi = base.copy()
        for axis, idx in gates:
            psi = rotation_action_closed_form(axis, th[idx], psi)
        v = gs * np.vdot(gs, psi)
        gs_ = []
        for j in range(len(gates)):
            phi = base.copy()
            for k in range(j):
                ax2, idx2 = gates[k]
                phi = rotation_action_closed_form(ax2, th[idx2], phi)
            d = rotation_derivative(gates[j][0], th[gates[j][1]], phi)
            for k in range(j + 1, len(gates)):
                ax2, idx2 = gates[k]
                d = rotation_action_closed_form(ax2, th[idx2], d)
            gs_.append(-2.0 * np.real(np.vdot(d, v)))
        rs.append(np.sqrt(np.mean([g * g for g in gs_])))
    return float(np.median(rs))


def main():
    print("=" * 74)
    print("Discrete dynamic evolution (zero-gradient adiabatic) vs")
    print("continuous-gradient VQE — immune to barren plateaus")
    print("=" * 74)

    # 0) mechanism statement
    print("\n[0] Fixed adiabatic schedule, ZERO parameters, ZERO "
          "gradients:")
    print("    gamma_k = dt * s_k,  beta_k = dt * (2 s_k - 1),  "
          "s_k = (k+0.5)/p")
    print("    -> there is no gradient to plateau.")

    # 1) convergence scan
    print("\n[1] Convergence (fidelity to the exact ground state), "
          "p = Trotter steps, T = adiabatic time:")
    for n in (4, 6, 8, 10):
        base = _base_state(n)
        E0, gs = ising_ground_state(n)
        C = diag_values(n, ising_hamiltonian(n))
        print(f"    n={n:2d}:", end="")
        for p, T in [(200, 20), (1000, 100), (2000, 200), (4000, 400)]:
            psi = discrete_adiabatic(n, p, T, C, base)
            fid = abs(np.vdot(gs, psi)) ** 2
            print(f"  p={p:4d} fid={fid:.3f}", end="")
        print()

    # 2) energy accuracy at the plateau point
    print("\n[2] Energy error vs exact (p=4000, T=400):")
    for n in (4, 6, 8, 10):
        base = _base_state(n)
        E0, gs = ising_ground_state(n)
        C = diag_values(n, ising_hamiltonian(n))
        psi = discrete_adiabatic(n, 4000, 400, C, base)
        E = ising_energy(psi, n, ising_hamiltonian(n))
        print(f"    n={n:2d}: E = {E:.4f}  exact {E0:.4f}  "
              f"err {E - E0:+.4f}")

    # 3) contrast with the continuous-gradient baseline
    print("\n[3] Continuous-gradient HEA VQE baseline (fidelity cost, "
          "the barren case):")
    from vqe_barren_plateaus import hea_gates
    for n in (8, 10, 12):
        gates = hea_gates(n, 2)
        base = _base_state(n)
        _, gs = ising_ground_state(n)
        g = hea_gradient_scale(n, gates, base, gs)
        print(f"    n={n:2d}: HEA gradient RMS = {g:.1e}  "
              f"({'barren' if g < 1e-3 else 'trainable'})")

    print("\nSummary: with a fixed adiabatic schedule the discrete")
    print("evolution has zero parameters and zero gradients, so the")
    print("barren plateau (a disease of continuous parameterization)")
    print("never enters; fidelity 0.98+ at n=10 with polynomial cost.")
    print("Honest: adiabatic route (T ~ 1/gap^2), 0.98 plateau, no")
    print("variational-advantage claim.")


if __name__ == "__main__":
    main()
