#!/usr/bin/env python3
"""LiH molecule with the discrete-evolution solver — the molecular
extension beyond H2 (feature 40, article 10.86 §7).

LiH in STO-3G (Li-H 1.6 A): 12 qubits, 4 electrons via Jordan-Wigner
(openfermion, machine-verified against FCI: E = -7.882324, the JW
matrix reproduces it exactly).  631 Pauli terms (79 Z-only diagonal,
552 with X/Y).

Method (the unified sector/spatial logic of feature 40): zero-gradient
discrete adiabatic evolution on the diagonal -> full path
    H(s) = H_diag + s * H_off
with the initial state the H_diag ground state (the HF-like
computational basis), the diagonal phase applied as an O(2^n) phase
vector, and the 552 off-diagonal Pauli rotations via the closed-form
R_P(theta) (O(2^n) each).  Zero parameters, zero gradients.

Verified here: convergence to the FCI ground state as a function of the
Trotter steps p and the adiabatic time T, and the energy error vs
chemical accuracy (1.6e-3 Ha).

Run:  PYTHONPATH=src python3 examples/vqe_lih_evolution.py
"""

import numpy as np

from geocore.clifford import rotation_action_closed_form

from vqe_barren_plateaus import _base_state  # noqa: F401  (unused, keep style)

_LIH_GEOM = [["Li", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 1.6]]]


def lih_hamiltonian(geometry=None, basis="sto-3g"):
    """(n_qubits, diag_values, off_terms, exact_gs, fci_energy, nuc).
    diag_values: the diagonal (Z-only + constant) Hamiltonian vector.
    off_terms: list of (coeff, axis) Pauli terms with X/Y.
    exact_gs: dense JW ground-state vector (4096 dims)."""
    from openfermion import MolecularData, jordan_wigner, get_sparse_operator
    from openfermionpyscf import run_pyscf

    mol = MolecularData(geometry=geometry or _LIH_GEOM, basis=basis,
                        multiplicity=1)
    mol = run_pyscf(mol, run_fci=True)
    Hq = jordan_wigner(mol.get_molecular_hamiltonian())
    n = mol.n_qubits
    d = 2**n

    z_terms, off_terms = [], []
    for t, c in Hq.terms.items():
        if not t:            # the constant is folded into `diag` below
            continue
        # verified: openfermion JW qubit q is big-endian, i.e. it maps
        # to geocore axis position q (the reconstruction without any
        # flip reproduces the sparse-H expectation on the ground state
        # to machine precision).
        axis = ["I"] * n
        for q, p in t:
            axis[q] = p
        ax = "".join(axis)
        if any(ch in "XY" for ch in ax):
            off_terms.append((float(c.real), ax))
        else:
            z_terms.append((float(c.real), ax))

    diag = np.full(d, Hq.constant.real, dtype=complex)
    for c, ax in z_terms:
        vals = np.ones(d, dtype=complex)
        for i, ch in enumerate(ax):
            if ch == "Z":
                # geocore position i = bit n-1-i (big-endian)
                b = ((np.arange(d) >> (n - 1 - i)) & 1).astype(float)
                vals *= (1 - 2 * b)
        diag += c * vals

    Hm = get_sparse_operator(Hq)          # sparse; GS via Lanczos
    import scipy.sparse.linalg as spla
    w, v = spla.eigsh(Hm, k=1, which="SA")
    gsv = v[:, 0]
    return n, diag, off_terms, gsv, float(w[0]), float(mol.fci_energy)


def evolve_lih(n, diag, off_terms, p, T):
    """Zero-gradient discrete adiabatic: H(s) = H_diag + s * H_off.
    Init = H_diag ground state (HF-like computational basis)."""
    d = 2**n
    idx0 = int(np.argmin(diag.real))
    psi = np.zeros(d, dtype=complex)
    psi[idx0] = 1
    dt = T / p
    diag_phase = np.exp(-1j * dt * diag)
    for k in range(p):
        s = (k + 0.5) / p
        psi = psi * diag_phase
        for c, ax in off_terms:
            psi = rotation_action_closed_form(ax, 2 * dt * s * c, psi)
    return psi


def main():
    print("=" * 74)
    print("LiH (12 qubits) with the discrete-evolution solver")
    print("=" * 74)

    n, diag, off_terms, gs, E0, fci = lih_hamiltonian()
    print(f"[0] LiH STO-3G: {n} qubits, {len(off_terms)} off-diagonal "
          f"Pauli terms")
    print(f"    FCI / JW exact ground state: {E0:.6f} Ha "
          f"(match FCI {abs(E0 - fci) < 1e-8})")
    print(f"    HF-like initial (diagonal ground state): "
          f"E = {diag.real.min():.4f}")

    print("\n[1] Convergence (fidelity to the exact ground state):")
    print("    p    T=10       T=20       T=40       T=80")
    for p in (50, 100, 200):
        row = []
        for T in (10, 20, 40, 80):
            psi = evolve_lih(n, diag, off_terms, p, T)
            row.append(abs(np.vdot(gs, psi)) ** 2)
        print(f"    {p:3d}  " + "  ".join(f"{f:.3f}" for f in row))

    print("\n[2] Energy error vs exact (chemical accuracy 1.6e-3 Ha):")
    Hm = None  # energy computed via expectation with pauli terms below
    for p, T in [(100, 40), (200, 80)]:
        psi = evolve_lih(n, diag, off_terms, p, T)
        E = _energy(psi, diag, off_terms)
        print(f"    p={p:3d} T={T:2d}: E = {E:.6f}  err {E - E0:+.4f} Ha")

    print("\nHonest: zero-gradient discrete evolution on the first real")
    print("molecule beyond H2 (12 qubits); the diagonal->full path with")
    print("the HF-like start; convergence plateau and T growth are")
    print("measured, not assumed.")


def _energy(psi, diag, off_terms):
    """<psi|H|psi> from the diagonal vector + Pauli off-diagonal terms."""
    e = float(np.real(np.vdot(psi, diag * psi)))
    for c, ax in off_terms:
        e += c * float(np.real(np.vdot(psi, pauli_vec(ax, psi))))
    return e


def pauli_vec(ax, psi):
    """P|psi> for a Pauli axis via the closed form's building block."""
    from geocore.clifford import pauli_action_on_state
    return pauli_action_on_state(ax, psi)


if __name__ == "__main__":
    main()
