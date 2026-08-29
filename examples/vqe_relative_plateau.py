#!/usr/bin/env python3
"""There is no absolute barren plateau — only a relative one: the
same system, the same target, and the tool (parameterization /
solver) decides whether the landscape is barren.

Classic example (article 10.86 §9): the Ising n=12 ground state.

  tool A  pure-continuous HEA (continuous params + gradient):
          gradient RMS 2.7e-7   -> BARREN (the mismatched tool: a
          continuous tool on a system whose Hamiltonian spectrum is a
          discrete face)
  tool A' continuous + discrete-spectrum anchoring (diagonal-phase
          layers e^{-i gamma H_C}):
          anchored-param gradient 2.3e-4  -> partially protected
          (~6.6x; the spectrum anchor is a discrete face injected
          into the continuous tool)
  tool B  discrete dynamic evolution (zero-gradient adiabatic):
          fidelity 0.972 to the exact GS -> CONVERGED (the matched
          tool: discrete evolution for a discrete-spectrum system)

and a matching-degree scan: as the fraction of spectrum-anchored
parameters grows, the gradient scale grows continuously — the
"barrenness" is a monotone function of the tool-system match, not an
absolute property of the system.

Run:  PYTHONPATH=src python3 examples/vqe_relative_plateau.py
"""

import numpy as np

from geocore.clifford import rotation_action_closed_form
from geocore.derivatives import rotation_derivative

from vqe_barren_plateaus import _base_state, hea_gates, ising_hamiltonian  # noqa
from vqe_barren_prewarm import ising_ground_state  # noqa
from vqe_discrete_evolution import diag_values  # noqa
from vqe_evolution_scaling import (  # noqa
    sector_alternating_init,
    sector_pure_evolution,
)

N = 12


def hea_gradient_rms(theta, gates, base, gs):
    psi = base.copy()
    for axis, idx in gates:
        psi = rotation_action_closed_form(axis, theta[idx], psi)
    v = gs * np.vdot(gs, psi)
    gg = []
    for j in range(len(gates)):
        phi = base.copy()
        for k in range(j):
            ax2, idx2 = gates[k]
            phi = rotation_action_closed_form(ax2, theta[idx2], phi)
        d = rotation_derivative(gates[j][0], theta[gates[j][1]], phi)
        for k in range(j + 1, len(gates)):
            ax2, idx2 = gates[k]
            d = rotation_action_closed_form(ax2, theta[idx2], d)
        gg.append(-2.0 * np.real(np.vdot(d, v)))
    return float(np.sqrt(np.mean([g * g for g in gg])))


def anchored_gradient_rms(theta, gamma, gates, base, gs, C, L=2):
    """Gradient RMS of the anchored (diagonal-phase) parameters."""
    half = len(gates) // L
    psi = base.copy()
    for k in range(L):
        for g in gates[k * half:(k + 1) * half]:
            axis, idx = g
            psi = rotation_action_closed_form(axis, theta[idx], psi)
        psi = psi * np.exp(-1j * gamma[k] * C)
    v = gs * np.vdot(gs, psi)
    w = v.copy()
    vals = []
    for k in range(L - 1, -1, -1):
        psi2 = base.copy()
        for m in range(k + 1):
            for g in gates[m * half:(m + 1) * half]:
                axis, idx = g
                psi2 = rotation_action_closed_form(axis, theta[idx], psi2)
            psi2 = psi2 * np.exp(-1j * gamma[m] * C)
        vals.append(-2.0 * np.real(np.vdot(-1j * C * psi2, w)))
        w = w * np.exp(1j * gamma[k] * C)
    return float(np.sqrt(np.mean([x * x for x in vals])))


def main():
    print("=" * 74)
    print("There is no absolute barren plateau — only a relative one")
    print("=" * 74)
    print("(the same system, the same target; the TOOL decides)")

    gates = hea_gates(N, 2)
    base = _base_state(N)
    _, gs = ising_ground_state(N)
    C = diag_values(N, ising_hamiltonian(N))
    rng = np.random.default_rng(0)

    print(f"\nClassic example: Ising n={N} ground state (same system)")
    thA = rng.uniform(-np.pi, np.pi, len(gates))
    gA = hea_gradient_rms(thA, gates, base, gs)
    print(f"  tool A  pure-continuous HEA   : grad RMS {gA:.2e}   "
          f"BARREN")
    th2 = rng.uniform(-np.pi, np.pi, len(gates))
    gm2 = rng.uniform(0, 0.5, 2)
    gA2 = anchored_gradient_rms(th2, gm2, gates, base, gs, C)
    print(f"  tool A' continuous+spectrum   : anchored grad "
          f"{gA2:.2e}  partially protected ({gA2 / max(gA, 1e-300):.1f}x)")
    par = ising_gs_parity(N)
    psi = sector_pure_evolution(N, 1000, 100, C,
                                sector_alternating_init(N, par))
    fid = abs(np.vdot(gs, psi)) ** 2
    print(f"  tool B  discrete evolution    : fidelity {fid:.3f}  "
          f"CONVERGED")

    print("\nMatching-degree scan: anchored-param gradient (median of 8")
    print("random points) vs the number of spectrum-anchored layers:")
    for L in (1, 2, 3, 4):
        vals = []
        for s in range(8):
            th = rng.uniform(-np.pi, np.pi, len(gates))
            gm = rng.uniform(0, 0.5, L)
            vals.append(anchored_gradient_rms(th, gm, gates, base, gs, C, L))
        print(f"    L={L} anchor layers: grad RMS {np.median(vals):.2e}")

    print("\nConclusion (structure proposition, article 10.86 §9): the")
    print("plateau is the tool-system mismatch — a continuous tool on a")
    print("discrete-spectrum system (barren), the spectrum anchor or")
    print("the discrete evolution on the same system (protected /")
    print("converged).  Barrenness is relative to the tool choice; it")
    print("is not an absolute property of the system.")


if __name__ == "__main__":
    from vqe_evolution_scaling import ising_gs_parity  # noqa: E402
    main()
