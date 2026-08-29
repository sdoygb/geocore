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


def ising_gs_parity(n):
    """Z2-flip parity of the ground state (+1 even sector, -1 odd)."""
    _, gs = ising_ground_state(n)
    rev = gs[::-1]  # flip all bits: index i <-> 2^n-1-i
    return int(round(np.vdot(rev, gs).real))


def boundary_correlation(n):
    """<Z_0 Z_{n-1}> in the ground state: + (odd n, frustrated
    boundary) vs - (even n, matched boundary)."""
    _, gs = ising_ground_state(n)
    Z = np.diag([1.0, -1.0]).astype(complex)
    M0 = np.array([[1.0]], dtype=complex)
    Mn = np.array([[1.0]], dtype=complex)
    for i in range(n):
        M0 = np.kron(M0, Z if i == 0 else np.eye(2, dtype=complex))
        Mn = np.kron(Mn, Z if i == n - 1 else np.eye(2, dtype=complex))
    return float(np.real(np.vdot(gs, M0 @ Mn @ gs)))


def sector_adiabatic(n, parity, p, T):
    """Symmetry-reduced adiabatic evolution INSIDE the Z2 sector of the
    ground state: project H(s) to the sector, start from the sector's
    alternating (diagonal) ground state.  This fixes the odd-n case
    (the plain path starts from |+> in the even sector and can never
    reach an odd-sector ground state)."""
    from scipy.linalg import expm

    def Hmat(nn):
        m1 = {"I": np.eye(2, dtype=complex),
              "X": np.array([[0, 1], [1, 0]], dtype=complex),
              "Z": np.array([[1, 0], [0, -1]], dtype=complex)}
        H = np.zeros((2**nn, 2**nn), dtype=complex)
        for c, p_ in ising_hamiltonian(nn):
            M = np.array([[1.0]], dtype=complex)
            for ch in p_:
                M = np.kron(M, m1[ch])
            H = H + c * M
        return H

    H = Hmat(n)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Xs = np.zeros_like(H)
    for i in range(n):
        M = np.array([[1.0]], dtype=complex)
        for q in range(n):
            M = np.kron(M, X if q == i else np.eye(2, dtype=complex))
        Xs = Xs + M
    Zsum = H - Xs
    # sector basis: symmetrized/antisymmetrized bit pairs
    Pidx = lambda i: 2**n - 1 - i  # noqa: E731
    basis = []
    used = set()
    for i in range(2**n):
        j = Pidx(i)
        if i in used or j in used or i == j:
            continue
        used.add(i)
        used.add(j)
        ei = np.zeros(2**n, dtype=complex)
        ei[i] = 1
        ej = np.zeros(2**n, dtype=complex)
        ej[j] = 1
        basis.append((ei + parity * ej) / np.sqrt(2))
    B = np.array(basis).T
    Hp = B.conj().T @ H @ B
    Zp = B.conj().T @ Zsum @ B
    _, v0 = np.linalg.eigh((Zp + Zp.conj().T) / 2)
    psi = v0[:, 0]
    dt = T / p
    for k in range(p):
        s = (k + 0.5) / p
        Hs = Hp * s + Zp * (1 - s)
        psi = expm(-1j * dt * ((Hs + Hs.conj().T) / 2)) @ psi
    return B @ psi


def sector_alternating_init(n, parity):
    """(A +/- B)/sqrt(2): the alternating-state combination in the
    given Z2 sector (A = |0101..>, B = |1010..>).  Pure-sector start:
    the Sigma-ZZ diagonal ground state inside the sector (unique)."""
    ia = 0
    for i in range(n):
        if i % 2 == 1:
            ia |= 1 << (n - 1 - i)
    ib = 0
    for i in range(n):
        if i % 2 == 0:
            ib |= 1 << (n - 1 - i)
    a = np.zeros(2**n, dtype=complex)
    b = np.zeros(2**n, dtype=complex)
    a[ia] = 1
    b[ib] = 1
    return (a + parity * b) / np.sqrt(2)


def sector_pure_evolution(n, p, T, C, init):
    """Zero-gradient Pauli-Trotter evolution on the Sigma-ZZ -> Ising
    path H(s) = Sigma-ZZ + s * Xsum: the diagonal phase is constant,
    the X term grows with s.  With a pure-sector init the Z2 symmetry
    keeps the state inside the sector automatically — O(p*2^n*n), no
    matrix projection, works for odd AND even n (the fix of the odd-n
    symmetry-forbidden case at scale)."""
    dt = T / p
    psi = init.copy()
    diag_phase = np.exp(-1j * dt * C)
    for k in range(p):
        s = (k + 0.5) / p
        psi = psi * diag_phase
        for q in range(n):
            axis = ["I"] * n
            axis[q] = "X"
            psi = rotation_action_closed_form("".join(axis), 2 * dt * s, psi)
    return psi


def adiabatic_time_req(n, dt=0.1, target=0.90, Tmax=1000):
    """Smallest T (power-of-2 scan) with fidelity >= target, on the
    unified sector-pure path (works for odd and even n)."""
    _, gs = ising_ground_state(n)
    C = diag_values(n, ising_hamiltonian(n))
    par = ising_gs_parity(n)
    init = sector_alternating_init(n, par)
    T = dt
    while T <= Tmax:
        p = max(int(round(T / dt)), 1)
        psi = sector_pure_evolution(n, p, T, C, init)
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

    # A) Ising gap and T(n) — unified sector-pure path (odd AND even)
    print("\n[A] Ising chain: gap and adiabatic time vs n")
    print("    (unified Sigma-ZZ -> Ising path with a PURE-SECTOR")
    print("     alternating init: works for odd and even n, no")
    print("     symmetry obstruction, Pauli-Trotter O(p*2^n*n))")
    print("    n    Delta      T_req(0.90)   n*Delta   T*Delta^2   "
          "gs sector   <Z0 Z_{n-1}>")
    ns = list(range(4, 13))
    for n in ns:
        d = ising_gap(n)
        par = ising_gs_parity(n)
        corr = boundary_correlation(n)
        T = adiabatic_time_req(n)
        print(f"    {n:2d}   {d:.4f}   {T or -1:6.1f}      "
              f"{n * d:.2f}      {T * d * d if T else -1:.1f}   "
              f"{par:+2d}         {corr:+.3f}")
    print("    -> Delta ~ 3/n (n*Delta -> 3.0), T ~ const/Delta^2 ~ "
          "O(n^2)")
    print("       polynomial, NOT the exponential of the plateau; the")
    print("       pure-sector init fixes the odd-n case (the plain |+>")
    print("       path is symmetry-forbidden: gs in the odd sector)")

    # A2) the odd/even spatial diagnosis and the sector fix at scale
    print("\n[A2] Odd/even spatial property and the sector fix at "
          "scale:")
    for n in (5, 7, 12, 14):
        par = ising_gs_parity(n)
        _, gs = ising_ground_state(n)
        C = diag_values(n, ising_hamiltonian(n))
        init = sector_alternating_init(n, par)
        psi = sector_pure_evolution(n, 1000, 100, C, init)
        fid = abs(np.vdot(gs, psi)) ** 2
        print(f"    n={n:2d} (parity {par:+d}): sector-pure path fid "
              f"{fid:.3f}")

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
    print("(T ~ O(n^2), now on the UNIFIED sector-pure path for odd and")
    print("even n) and reaches chemical accuracy on the H2 molecule with")
    print("zero gradients.  The odd/even difference is a spatial")
    print("property: the frustrated boundary on odd chains puts the")
    print("ground state in the Z2-odd sector, so a |+>-based path is")
    print("symmetry-forbidden; a pure-sector alternating init fixes it")
    print("with the same Pauli-Trotter cost (fid 0.99 at n=5,7; 0.97 at")
    print("n=12; 0.966 at n=14).  Honest: T ~ 1/Delta^2 is")
    print("family-dependent; the H2 test uses the 2-qubit reduction.")


if __name__ == "__main__":
    main()
