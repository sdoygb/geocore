#!/usr/bin/env python3
"""Exterior-algebra (Clifford) sector construction — the computation
layer itself speaks the geometry (feature 47; article 10.86 §9.06).

The N-electron sector of any molecular Hamiltonian is the degree-N
piece of the exterior algebra:

    Fock space = exterior algebra  Lambda(C^n) = (+)_{N} Lambda^N(C^n),
    creation  a+_p = wedge with e_p   (e_p ^ .),
    annihilation a_p = contraction by e_p  (iota_{e_p} .),

and the fermion anticommutation signs ARE the exterior-algebra
grading: every wedge/contraction past an occupied orbital flips a
sign, (-1)^{# occupied orbitals below p}.  The molecular Hamiltonian

    H = const + sum o[p,q] a+_p a_q
              + 1/2 sum t[p,q,r,s] a+_p a+_q a_r a_s

is therefore an even element of the Clifford algebra Cl(2n), and its
action on a sector state is pure exterior algebra — no Jordan-Wigner
strings, no Pauli expansion, no 7.7e4 Pauli terms.  The particle
number is conserved automatically (wedge/contraction move one or two
particles), so the N-sector Hamiltonian is built directly in
Lambda^N(C^n) from the one/two-body integrals.

What this replaces (the honest accounting):
  - the JW/Pauli pipeline (openfermion jordan_wigner + 77212 Pauli
    terms for LiH cc-pVDZ) is not the physics, it is one representation
    of Cl(2n); the exterior action computes the same matrix elements
    with the grading signs arising from wedge anticommutation;
  - the only "standard" input left is the SCF integrals o/t (the
    physics data), and the reference energies used for verification;
    every step from integrals to the sector Hamiltonian is exterior
    algebra.

Machine-verified:
  - the exterior sector Hamiltonian equals the JW-Pauli sector
    Hamiltonian element-wise (LiH/H2O STO-3G, machine precision);
  - LiH cc-pVDZ (38 qubits, N=4): exterior build with integral
    truncation (|t|,|o| > 1e-3) reproduces the Pauli-truncated
    ground state -8.01502588 Ha (2e-4-level agreement, both inside
    chemical accuracy of CCSD -8.01473762).

Run:  PYTHONPATH=src python3 examples/vqe_exterior_algebra.py
"""

import numpy as np
from itertools import combinations

from vqe_sector_reduction import sector_states  # noqa: E402


def exterior_sign(z, p, n):
    """Exterior-algebra grading sign for acting on orbital p of the
    big-endian bitstring z: (-1)^{# occupied orbitals q < p}.  This IS
    the fermion anticommutation sign — the wedge/contraction must
    pass the occupied orbitals below p."""
    return -1.0 if ((z >> (n - p)).bit_count() & 1) else 1.0


def _bit(q, n):
    return 1 << (n - 1 - q)


def exterior_hamiltonian(n, N, o, t, const, eps=0.0):
    """(hd, H_off) in the N-sector basis, built by the exterior
    (fermionic) action directly — no Pauli expansion.  o: one-body
    (n x n, Hermitian), t: two-body (n^4, physicist notation with
    t[p,q,r,s] = t[r,s,p,q]); const: constant.  eps: drop integrals
    with |value| <= eps (spectral truncation on the integrals, the
    exterior-algebra analogue of the Pauli truncation)."""
    from scipy import sparse
    states = sector_states(n, N)
    idx = {z: i for i, z in enumerate(states)}
    dim = len(states)
    hd = np.zeros(dim, dtype=complex)
    rows_l, cols_l, vals_l = [], [], []

    def emit(zt, z, v):
        rows_l.append(idx[zt])
        cols_l.append(idx[z])
        vals_l.append(v)

    # --- diagonal: const + one-body diagonal o[p,p] (creation a+_p a_p
    # keeps the state, sign +1) ---
    for zi, z in enumerate(states):
        d = const + sum(o[p, p] for p in range(n) if (z >> (n - 1 - p)) & 1)
        hd[zi] = d

    # --- one-body off-diagonal: o[p,q] a+_p a_q, q occupied, p empty.
    # For each (p,q) the sources are: q occupied, p empty, and the
    # other N-1 occupied among the remaining n-2 orbitals.
    for p in range(n):
        for q in range(n):
            if p == q or abs(o[p, q]) <= eps:
                continue
            rest = [r for r in range(n) if r not in (p, q)]
            bitq = _bit(q, n)
            bitp = _bit(p, n)
            for oc in combinations(range(n - 2), N - 1):
                z = bitq
                for i in oc:
                    z |= _bit(rest[i], n)
                # contraction a_q then wedge a+_p (exterior grading)
                s = exterior_sign(z, q, n)
                z1 = z ^ bitq
                s *= exterior_sign(z1, p, n)
                emit(z1 ^ bitp, z, o[p, q] * s)

    # --- two-body: t[p,q,r,s] a+_p a+_q a_r a_s with p<q, r<s (the
    # antisymmetry of a+_p a+_q and a_r a_s lets the full sum reduce
    # to  2 sum_{p<q,r<s} (t[p,q,r,s] - t[p,q,s,r]) a+_p a+_q a_r a_s
    # — machine-verified against the JW-Pauli build).  p,q,r,s may
    # overlap ({p,q} ∩ {r,s} non-empty: the same orbital annihilated
    # then re-created, e.g. a+_p a+_q a_q a_s); the sequential action
    # a_s, a_r, a+_q, a+_p with per-step occupancy checks handles all
    # cases automatically.  Sources: r,s occupied, p,q empty (exactly
    # N-2 others occupied among the remaining orbitals).
    for p in range(n):
        for q in range(p + 1, n):
            for r in range(n):
                for s in range(r + 1, n):
                    c2 = 2.0 * (t[p, q, r, s] - t[p, q, s, r])
                    if abs(c2) <= eps:
                        continue
                    rest = [u for u in range(n) if u not in (p, q, r, s)]
                    bitr = _bit(r, n)
                    bits = _bit(s, n)
                    bitq = _bit(q, n)
                    bitp = _bit(p, n)
                    for oc in combinations(range(len(rest)), N - 2):
                        z = bitr | bits
                        for i in oc:
                            z |= _bit(rest[i], n)
                        # a_s, a_r (contractions), then a+_q, a+_p
                        # (wedges), each with its grading sign
                        sgn = exterior_sign(z, s, n)
                        z1 = z ^ bits
                        sgn *= exterior_sign(z1, r, n)
                        z2 = z1 ^ bitr
                        sgn *= exterior_sign(z2, q, n)
                        z3 = z2 ^ bitq
                        sgn *= exterior_sign(z3, p, n)
                        emit(z3 ^ bitp, z, c2 * sgn)

    rows = np.array(rows_l, dtype=np.int64)
    cols = np.array(cols_l, dtype=np.int64)
    vals = np.array(vals_l, dtype=complex)
    H_off = sparse.coo_matrix((vals, (rows, cols)),
                              shape=(dim, dim)).tocsr()
    return hd, H_off


def integrals_from_openfermion(geometry, basis, run_fci=False):
    """(n, o, t, const, fci_energy) — the SCF integrals (the physics
    data) from openfermion; everything after this point in
    exterior_hamiltonian is exterior algebra."""
    from openfermion import MolecularData
    from openfermionpyscf import run_pyscf
    mol = MolecularData(geometry=geometry, basis=basis, multiplicity=1)
    mol = run_pyscf(mol, run_scf=True, run_fci=run_fci)
    Hf = mol.get_molecular_hamiltonian()
    n = mol.n_qubits
    o = np.asarray(Hf.one_body_tensor)
    t = np.asarray(Hf.two_body_tensor)
    return n, o, t, float(Hf.constant), (float(mol.fci_energy)
                                         if run_fci else None)


def main():
    from scipy import sparse
    import scipy.sparse.linalg as spla
    import time
    from vqe_sector_reduction import sector_hamiltonian
    from vqe_lih_evolution import lih_hamiltonian

    print("=" * 74)
    print("Exterior-algebra (Clifford) sector construction — the")
    print("computation layer speaks the geometry (no JW, no Pauli)")
    print("=" * 74)

    # [0] machine-precision cross-check: exterior == JW-Pauli sector
    cases = [
        ("LiH STO-3G", [["Li", [0, 0, 0]], ["H", [0, 0, 1.6]]], 4),
        ("H2O STO-3G", [["O", [0, 0, 0]], ["H", [0.757, 0.586, 0]],
                         ["H", [-0.757, 0.586, 0]]], 10),
    ]
    for name, geom, ne in cases:
        from scipy import sparse as _sp
        n, o, t, const, fci = integrals_from_openfermion(geom, "sto-3g",
                                                         run_fci=True)
        hd1, H1 = exterior_hamiltonian(n, ne, o, t, const)   # exterior
        M1 = (_sp.diags(hd1) + H1).tocsr()
        M1.eliminate_zeros()
        # JW-Pauli reference
        n2, diag, off, gs, E0, fci2 = lih_hamiltonian(geometry=geom)
        hd2, H2 = sector_hamiltonian(n2, ne, diag, off)
        M2 = (_sp.diags(hd2) + H2).tocsr()
        M2.eliminate_zeros()
        same = (np.allclose(M1.toarray(), M2.toarray(), atol=1e-10)
                and M1.nnz == M2.nnz)
        print(f"  {name:<14}: exterior == JW-Pauli sector "
              f"(machine precision: {same})")
        assert same
        ev = np.linalg.eigvalsh(np.diag(hd1) + H1.toarray())
        print(f"    sector GS {ev[0]:.6f} == full-space FCI "
              f"{E0:.6f} (|d|={abs(ev[0] - E0):.1e})")
        assert abs(ev[0] - E0) < 1e-8

    # [1] LiH cc-pVDZ: exterior build with integral truncation
    n, o, t, const, _ = integrals_from_openfermion(
        [["Li", [0, 0, 0]], ["H", [0, 0, 1.6]]], "cc-pVDZ")
    N = 4
    eps = 1e-3
    t0 = time.time()
    hd, H_off = exterior_hamiltonian(n, N, o, t, const, eps)
    t_build = time.time() - t0
    H = sparse.diags(hd) + H_off
    print(f"\n  LiH cc-pVDZ (exterior, |int|>{eps:.0e}): build "
          f"{t_build:.0f}s, H_off nnz={H_off.nnz}")
    t0 = time.time()
    w, v = spla.eigsh(H, k=1, which="SA")
    t_eig = time.time() - t0
    print(f"    exterior sector GS: {w[0]:.8f} Ha ({t_eig:.0f}s)")
    try:
        with open("/tmp/lih_ccpVDZ_ccsd.txt") as f:
            ref = float(f.read().strip())
        print(f"    reference CCSD: {ref:.8f} Ha; "
              f"|exterior GS - CCSD| = {abs(w[0] - ref):.2e} Ha")
        print(f"    Pauli-truncated GS was -8.01502588 Ha "
              f"(cross-check |d| = {abs(w[0] - (-8.01502588)):.2e})")
    except OSError:
        pass

    print("\n  Honest boundaries: the SCF integrals o/t are the physics")
    print("  input (openfermion/pyscf); everything from integrals to")
    print("  the sector Hamiltonian — grading signs, N conservation,")
    print("  truncation — is exterior algebra.  The eps=1e-3 integral")
    print("  truncation is an approximation like the Pauli one, each")
    print("  calibrated separately.")


if __name__ == "__main__":
    main()
