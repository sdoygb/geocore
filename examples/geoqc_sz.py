#!/usr/bin/env python3
"""S_z (spin) sector projection — the first symmetry extension of the
geoqc sector build (article 10.86 §9.14).

Any molecular Hamiltonian conserves S_z, so the N-sector decomposes
into spin sectors of fixed S_z = (n_alpha - n_beta)/2.  For singlet
ground states the S_z = 0 sector contains the ground state, with
dimension

    C(n/2, n_alpha) * C(n/2, n_beta)     instead of   C(n, N).

geoqc.exterior.exterior_hamiltonian_sz builds the S_z-sector
Hamiltonian directly: states = alpha-choose x beta-choose, terms
filtered by S_z conservation (#alpha created == #alpha annihilated),
source combinations decomposed into alpha x beta outer products.
Machine-verified equal to the full-N-sector ground state.

Scale (honest): S_z alone shrinks LiH cc-pVDZ 73815 -> 29241 and
N2 STO-3G 38760 -> 14400, but cc-pVDZ N2 (C(18,7)^2 = 1.01e9) and
H2O (C(19,5)^2 = 1.35e8) remain too large — the point-group
projection (next step) is needed for those.

Run:  PYTHONPATH=src python3 examples/geoqc_sz.py
"""

import time
import numpy as np
from math import comb

from geoqc.integrals import ao_integrals, mo_transform, spin_orbital_integrals
from geoqc.scf import grassmann_scf, fock_matrix
from geoqc import exterior
from scipy.linalg import sqrtm
from scipy import sparse
import scipy.sparse.linalg as spla


def sector_gs(geom, basis, N, sz=None, eps=0.0):
    """Ground state via the (S_z) sector build."""
    n, h, eri, S, nuc = ao_integrals(geom, basis)
    E, P, C, C_o, _, _ = grassmann_scf(h, eri, S, N // 2)
    X = np.asarray(sqrtm(np.linalg.inv(S)).real)
    h_o = X.T @ h @ X
    eri_o = mo_transform(X, eri)
    F = fock_matrix(h_o, eri_o, 2.0 * C_o @ C_o.T)
    _, C_all = np.linalg.eigh(F)
    o = C_all.T @ h_o @ C_all
    t = mo_transform(C_all, eri_o)
    o_s, t_s = spin_orbital_integrals(o, t)
    if sz is None:
        hd, H_off = exterior.exterior_hamiltonian(2 * n, N, o_s, t_s,
                                                  float(nuc), eps)
    else:
        hd, H_off = exterior.exterior_hamiltonian_sz(2 * n, N, sz, o_s,
                                                     t_s, float(nuc), eps)
    H = sparse.diags(hd) + H_off
    w, _ = spla.eigsh(H, k=1, which="SA")
    return w[0], H_off.shape[0]


def main():
    print("=" * 74)
    print("S_z (spin) sector projection — symmetry extension of the")
    print("geoqc sector build (singlet ground states in S_z = 0)")
    print("=" * 74)

    cases = [
        ("LiH STO-3G", [["Li", [0, 0, 0]], ["H", [0, 0, 1.6]]],
         "sto-3g", 4),
        ("N2 STO-3G", [["N", [0, 0, 0]], ["N", [0, 0, 1.1]]],
         "sto-3g", 14),
        ("LiH cc-pVDZ", [["Li", [0, 0, 0]], ["H", [0, 0, 1.6]]],
         "cc-pVDZ", 4),
    ]
    for name, geom, basis, N in cases:
        n, h, eri, S, nuc = ao_integrals(geom, basis)
        t0 = time.time()
        w_full, d_full = sector_gs(geom, basis, N, sz=None, eps=1e-4)
        t_full = time.time() - t0
        t0 = time.time()
        w_sz, d_sz = sector_gs(geom, basis, N, sz=0, eps=1e-4)
        t_sz = time.time() - t0
        n_a = N // 2
        print(f"\n  {name}: N={N}, n={2 * n} spin orbitals")
        print(f"    full-N:  dim C({2 * n},{N})={d_full}, GS {w_full:.8f} "
              f"({t_full:.0f}s)")
        print(f"    S_z=0:   dim C({n},{n_a})^2={d_sz}, GS {w_sz:.8f} "
              f"({t_sz:.0f}s)  |dGS| {abs(w_sz - w_full):.1e}")
        assert abs(w_sz - w_full) < 1e-8

    print("\n  Scale table (S_z = 0 singlet sectors, honest):")
    print("    LiH cc-pVDZ (38, N=4):  73815 -> 29241   (feasible, 2.5x)")
    print("    N2 STO-3G   (20, N=14): 38760 -> 14400   (feasible, 2.7x)")
    print("    N2 cc-pVDZ  (36, N=14): 5.6e9 -> 1.01e9  (needs point group)")
    print("    H2O cc-pVDZ (38, N=10): 4.7e8 -> 1.35e8  (needs point group)")
    print("  -> S_z alone is a real gain where feasible; the cc-pVDZ")
    print("     N2/H2O FCI benchmarks need the point-group projection,")
    print("     the next symmetry step.")


if __name__ == "__main__":
    main()
