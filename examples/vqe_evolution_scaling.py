#!/usr/bin/env python3
"""The discrete-evolution solver at scale: the adiabatic scaling law
T(n) and the molecular extension — feature 40 of the barren-plateau
series (article 10.86: evolve, don't optimize).

Part A — T(n) scaling law (Ising chain): the ground-state gap of the
transverse-field Ising chain at h=1 decays polynomially, Delta ~ 3/n
(measured), so the adiabatic time needed for fidelity >= 0.90 follows
T ~ const / Delta^2 ~ O(n^2): polynomial, not the exponential of the
plateau.

Part B — molecular extension: the H2 STO-3G two-qubit Hamiltonian
(Qiskit standard coefficients, exact ground state -1.857275, verified
in feature 30).  The adiabatic path from the diagonal part (its ground
state is the computational-basis/HF-like state |01>) to the full
Hamiltonian converges with energy error ~1e-4 Ha — 16x inside chemical
accuracy (1.6e-3 Ha) — with zero parameters, zero gradients.

Machine-verified:
  Ising:  n*Delta = 2.78, 2.89, 2.95, 2.99, 3.01 (n=4..12) -> Delta ~ 3/n
          T_req(0.90) = 51, 102, 102, 205, 205; T*Delta^2 ~ 13-25
  H2:     fidelity 0.988, energy err +0.0001 Ha (chemical accuracy 1.6e-3)

Honest: T ~ 1/Delta^2 is adiabatic-typical (polynomial for this
family, but small gaps elsewhere would demand longer T); the H2 test is
the 2-qubit reduction (the full spin-orbital Jordan-Wigner pipeline is
future work); no variational-advantage claim.

Run:  PYTHONPATH=src python3 examples/vqe_evolution_scaling.py
"""

import numpy as np

from geocore.clifford import rotation_action_closed_form

from vqe_barren_plateaus import _base_state, ising_hamiltonian  # same dir
from vqe_barren_prewarm import ising_ground_state  # same dir
from vqe_discrete_evolution import diag_values, discrete_adiabatic  # same dir


# ---------------------------------------------------------------------------
# Part A: Ising gap and adiabatic time scaling
# ---------------------------------------------------------------------------


def ising_gap(n):
    """Ground-to-first-excited gap (sparse Lanczos, descending order)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    I2 = sp.eye(2, format="csc")
    Z = sp.diags([1.0, -1.0], format="csc")
    X = sp.csc_matrix(np.array([[0, 1], [1, 0]]))

    def kron2(A, B):
        return sp.kron(A, B, format="csc")

    H = sp.csc_matrix((2**n, 2**n))
    for i in range(n - 1):
        M = sp.eye(1, format="csc")
        for q in range(n):
            M = kron2(M, Z if q in (i, i + 1) else I2)
        H = H + M
    for i in range(n):
        M = sp.eye(1, format="csc")
        for q in range(n):
            M = kron2(M, X if q == i else I2)
        H = H + M
    w = spla.eigsh(H, k=2, which="SA", return_eigenvectors=False)
    return float(w[-2] - w[-1])  # descending order


def adiabatic_time_req(n, dt=0.1, target=0.90, Tmax=1000):
    """Smallest T (power-of-2 scan) with fidelity >= target."""
    base = _base_state(n)
    _, gs = ising_ground_state(n)
    C = diag_values(n, ising_hamiltonian(n))
    T = dt
    while T <= Tmax:
        p = max(int(round(T / dt)), 1)
        psi = discrete_adiabatic(n, p, T, C, base)
        if abs(np.vdot(gs, psi)) ** 2 >= target:
            return T
        T *= 2
    return None


# ---------------------------------------------------------------------------
# Part B: H2 molecule
# ---------------------------------------------------------------------------

# H2 STO-3G, 2-qubit Pauli decomposition (Qiskit standard; exact GS
# -1.857275 verified in examples/vqe_h2.py, feature 30).
H2_HAMILTONIAN = [
    (-1.052373245772859, "II"),
    (0.39793742484318045, "IZ"),
    (-0.39793742484318045, "ZI"),
    (-0.01128010425623538, "ZZ"),
    (0.18093119978423156, "XX"),
]


def pauli_matrix(axis, n=2):
    m = {"I": np.eye(2, dtype=complex),
         "X": np.array([[0, 1], [1, 0]], dtype=complex),
         "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
         "Z": np.array([[1, 0], [0, -1]], dtype=complex)}
    M = np.array([[1.0]], dtype=complex)
    for ch in axis:
        M = np.kron(M, m[ch])
    return M


def h2_adiabatic(p, T):
    """Discrete adiabatic evolution: H(s) = H_diag + s * H_XX.
    Diagonal phase is s-independent (H_diag), the XX term grows with s.
    Initial state |z0> = ground state of H_diag (HF-like |01>)."""
    Hdiag = np.zeros((4, 4), dtype=complex)
    cXX = 0.0
    for c, pa in H2_HAMILTONIAN:
        if "X" in pa:
            cXX = c
        else:
            Hdiag = Hdiag + c * pauli_matrix(pa)
    _, v0 = np.linalg.eigh(Hdiag)
    z0 = v0[:, 0]
    dt = T / p
    diag_phase = np.exp(-1j * dt * np.diag(Hdiag).real)
    psi = z0.copy()
    for k in range(p):
        s = (k + 0.5) / p
        psi = psi * diag_phase
        psi = rotation_action_closed_form("XX", 2 * dt * s * cXX, psi)
    return psi


def main():
    print("=" * 74)
    print("Discrete evolution at scale: T(n) scaling law + molecule")
    print("=" * 74)

    # A) Ising gap and T(n)
    print("\n[A] Ising chain: gap and adiabatic time vs n")
    print("    (even n: transverse-field path, clean; odd n: the path")
    print("     has a spurious degeneracy at s=0.5 where the X terms")
    print("     cancel — honest limitation, needs a symmetry-aware")
    print("     path, future work)")
    print("    n    Delta      T_req(0.90)   n*Delta   T*Delta^2")
    ns = list(range(4, 13))
    for n in ns:
        d = ising_gap(n)
        if n % 2 == 0:
            T = adiabatic_time_req(n)
            print(f"    {n:2d}   {d:.4f}   {T or -1:6.0f}      "
                  f"{n * d:.2f}      {T * d * d if T else -1:.1f}")
        else:
            print(f"    {n:2d}   {d:.4f}   (X-cancel degeneracy)")
    print("    -> Delta ~ 3/n (n*Delta -> 3.0), T ~ const/Delta^2 ~ "
          "O(n^2)")
    print("       polynomial, NOT the exponential of the plateau")

    # B) H2 molecule
    print("\n[B] H2 molecule (STO-3G, 2 qubits): discrete adiabatic "
          "from the HF-like diagonal state")
    H = np.zeros((4, 4), dtype=complex)
    for c, pa in H2_HAMILTONIAN:
        H = H + c * pauli_matrix(pa)
    E0 = np.linalg.eigvalsh(H)[0].real
    _, gsv = np.linalg.eigh(H)
    gs = gsv[:, 0]
    for p, T in [(200, 20), (1000, 100)]:
        psi = h2_adiabatic(p, T)
        fid = abs(np.vdot(gs, psi)) ** 2
        E = float(np.real(np.vdot(psi, H @ psi)))
        print(f"    p={p:4d} T={T:3d}: fidelity {fid:.4f}, "
              f"E = {E:.6f}  (exact {E0:.6f}, err {E - E0:+.4f} Ha, "
              f"chemical accuracy 1.6e-3)")

    print("\nSummary: the discrete-evolution solver is polynomial in n")
    print("(T ~ O(n^2) on even n, not exponential) and reaches chemical")
    print("accuracy on the H2 molecule with zero gradients.  Honest:")
    print("T ~ 1/Delta^2 is family-dependent; odd-n Ising needs a")
    print("symmetry-aware path (the s=0.5 X-cancellation degeneracy);")
    print("the H2 test uses the 2-qubit reduction (full JW pipeline")
    print("future work).")


if __name__ == "__main__":
    main()
