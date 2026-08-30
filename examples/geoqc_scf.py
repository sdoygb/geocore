#!/usr/bin/env python3
"""Grassmann-manifold SCF and the full geometrised pipeline (M2/M3 of
the geoqc project; article 10.86 §9.10-11).

Hartree-Fock is a fixed point on the Grassmannian Gr(N, n): the
occupied subspace C (n x N) is a point of the manifold, the RHF
energy is a function on it, its gradient is 2(1-P)FP, and the fixed
point [F, P] = 0 is exactly the vanishing of that gradient.  The
iteration C <- lowest-N eigenspace of F is the manifold's steepest
jump (the Roothaan step), monitored by the geometric gradient norm
and the Fubini-Study path length of the iterates on Gr(N,n).

The demo closes the loop of the geoqc project: AO integrals (physics
input, honestly labelled) -> Grassmann SCF (this module) -> MO
transform -> exterior-algebra N-sector Hamiltonian (geoqc.exterior)
-> sector ground state.  Every step is machine-verified against the
standard library (pyscf RHF for the SCF, FCI for the ground state).

Run:  PYTHONPATH=src python3 examples/geoqc_scf.py
"""

import numpy as np

from geoqc.integrals import ao_integrals, mo_transform
from geoqc.scf import grassmann_scf
from geoqc import exterior


def mo_integrals(C, h, eri):
    """AO -> MO integrals: o = C^T h C, t = C^4 eri (C orthonormal)."""
    o = C.T @ h @ C
    t = np.einsum("ia,jb,kc,ld,ijkl->abcd", C, C, C, C, eri)
    return o, t


def main():
    from pyscf import gto, scf
    from scipy import sparse
    import scipy.sparse.linalg as spla

    print("=" * 74)
    print("Grassmann-manifold SCF + the full geometrised pipeline")
    print("(geoqc M2/M3: AO integrals -> SCF -> exterior sector)")
    print("=" * 74)

    cases = [
        ("LiH STO-3G", [["Li", [0, 0, 0]], ["H", [0, 0, 1.6]]], 4, 4),
        ("H2O STO-3G", [["O", [0, 0, 0]], ["H", [0.757, 0.586, 0]],
                         ["H", [-0.757, 0.586, 0]]], 10, 10),
    ]
    for name, geom, ne, n_electrons in cases:
        n, h, eri, S, nuc = ao_integrals(geom, "sto-3g")
        E, P, C, C_o, grads, dists = grassmann_scf(h, eri, S, ne // 2)
        mol = gto.M(atom=geom, basis="sto-3g")
        mf = scf.RHF(mol)
        mf.kernel()
        print(f"\n  {name}: Grassmann SCF on Gr({n // 2}, {n})")
        print(f"    E = {E + nuc:.10f} Ha, pyscf RHF = {mf.e_tot:.10f} "
              f"(|dE| {abs(E + nuc - mf.e_tot):.1e})")
        print(f"    gradient norm {grads[0]:.1e} -> {grads[-1]:.1e} "
              f"({len(grads)} iterations); FS path length on the "
              f"manifold {dists.sum():.4f}")
        assert abs(E + nuc - mf.e_tot) < 1e-8

        # [pipeline] full MO set (occupied + virtual) from the
        # converged Fock, then exterior N-sector Hamiltonian
        from scipy.linalg import sqrtm
        from geoqc.scf import fock_matrix
        X = np.asarray(sqrtm(np.linalg.inv(S)).real)
        h_o = X.T @ h @ X
        eri_o = mo_transform(X, eri)
        F = fock_matrix(h_o, eri_o, 2.0 * C_o @ C_o.T)
        ev, C_all = np.linalg.eigh(F)          # full orthonormal MO set
        o = C_all.T @ h_o @ C_all
        t = np.einsum("ia,jb,kc,ld,ijkl->abcd",
                      C_all, C_all, C_all, C_all, eri_o)
        from geoqc.integrals import spin_orbital_integrals
        o_s, t_s = spin_orbital_integrals(o, t)   # spatial -> spin-orbital,
        # chemist -> physicist (the exterior module's convention)
        const = float(nuc)
        N = n_electrons
        hd, H_off = exterior.exterior_hamiltonian(2 * n, N, o_s, t_s, const)
        H = sparse.diags(hd) + H_off
        w, _ = spla.eigsh(H, k=1, which="SA")
        print(f"    exterior sector ({2 ** (2 * n)} -> C({2 * n},{N})="
              f"{hd.size}) GS = {w[0]:.8f} Ha")

        # FCI reference (independent benchmark)
        from openfermion import MolecularData
        from openfermionpyscf import run_pyscf
        from openfermion import jordan_wigner, get_sparse_operator
        mref = run_pyscf(MolecularData(geometry=geom, basis="sto-3g",
                                       multiplicity=1), run_fci=True)
        print(f"    FCI reference = {mref.fci_energy:.8f} Ha "
              f"(|d| = {abs(w[0] - mref.fci_energy):.1e})")
        assert abs(w[0] - mref.fci_energy) < 1e-6

    print("\n  Honest boundaries: the AO integrals are physics input")
    print("  (standard GTO numerical integrals); the SCF is now the")
    print("  Grassmann fixed point; the sector ground state is the")
    print("  exterior algebra; FCI is the independent benchmark.")


if __name__ == "__main__":
    main()
