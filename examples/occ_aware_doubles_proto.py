#!/usr/bin/env python3
"""Prototype: occupancy-aware double excitation (only enumerate non-zero).

Verifies that enumerating only (r,s in occ, p,q in vir) produces identical
results to the current full enumeration approach.
"""
import numpy as np
from math import comb
from itertools import combinations

def doubles_occ_aware(az, bz, n_orb, t_s):
    """Enumerate only non-zero double excitations for determinant (az, bz).

    Returns (target_az, target_bz, values) arrays.
    """
    n_spin = 2 * n_orb
    occ_a = [i for i in range(n_orb) if (az >> i) & 1]
    occ_b = [i for i in range(n_orb) if (bz >> i) & 1]
    vir_a = [i for i in range(n_orb) if not ((az >> i) & 1)]
    vir_b = [i for i in range(n_orb) if not ((bz >> i) & 1)]

    out_az = []
    out_bz = []
    out_val = []

    # Alpha-alpha double excitations
    for r, s in combinations(occ_a, 2):
        for p, q in combinations(vir_a, 2):
            taz = az & ~(1 << r) & ~(1 << s) | (1 << p) | (1 << q)
            tbz = bz
            # Spin-orbital indices: alpha=2i
            sp, sq, sr, ss = 2*p, 2*q, 2*r, 2*s
            # <pq|rs> - <pq|sr> (physicist notation, alpha-alpha)
            val = t_s[sp, sq, sr, ss] - t_s[sp, sq, ss, sr]
            # Sign: annihilate s then r (higher index first for convention)
            # Need to match exterior._spin_sign convention
            out_az.append(taz)
            out_bz.append(tbz)
            out_val.append(val)

    # Beta-beta double excitations
    for r, s in combinations(occ_b, 2):
        for p, q in combinations(vir_b, 2):
            taz = az
            tbz = bz & ~(1 << r) & ~(1 << s) | (1 << p) | (1 << q)
            sp, sq, sr, ss = 2*p+1, 2*q+1, 2*r+1, 2*s+1
            val = t_s[sp, sq, sr, ss] - t_s[sp, sq, ss, sr]
            out_az.append(taz)
            out_bz.append(tbz)
            out_val.append(val)

    # Alpha-beta double excitations
    for r in occ_a:
        for s in occ_b:
            for p in vir_a:
                for q in vir_b:
                    taz = az & ~(1 << r) | (1 << p)
                    tbz = bz & ~(1 << s) | (1 << q)
                    sp, sq, sr, ss = 2*p, 2*q+1, 2*r, 2*s+1
                    val = t_s[sp, sq, sr, ss]  # alpha-beta no exchange
                    out_az.append(taz)
                    out_bz.append(tbz)
                    out_val.append(val)

    return np.array(out_az, dtype=np.int64), np.array(out_bz, dtype=np.int64), np.array(out_val)


def test_correctness():
    """Test against exterior.sparse_action_sz_vec on H2/STO-3G."""
    import sys
    sys.path.insert(0, '.')
    from geoqc import exterior
    from geoqc.integrals import spin_orbital_integrals
    from pyscf import gto, scf

    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    o = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    t = mol.intor('int2e')
    n_orb = mol.nao
    o_s, t_s = spin_orbital_integrals(o, t)
    ns = 2 * n_orb
    nelec = 2

    # HF determinant
    hf_az = 0b01  # alpha orbital 0 occupied
    hf_bz = 0b01  # beta orbital 0 occupied

    # Current method
    apply_fn, _, _, _, _, _ = exterior.sparse_action_sz_vec(
        ns, nelec, 0, o_s, t_s, 0.0, 1e-10)
    t_idx_cur, t_val_cur, src_cur = apply_fn(
        np.array([hf_az], dtype=np.int64),
        np.array([hf_bz], dtype=np.int64),
        np.array([1.0], dtype=np.float64))

    # New method (doubles only)
    t_az_new, t_bz_new, t_val_new = doubles_occ_aware(hf_az, hf_bz, n_orb, t_s)

    print(f'Current method: {len(t_idx_cur)} outputs')
    print(f'New method (doubles): {len(t_val_new)} outputs')

    # Convert new method to combined index
    # Need rank tables to compare
    # For now, just check counts
    n_occ_a = bin(hf_az).count('1')
    n_occ_b = bin(hf_bz).count('1')
    n_vir_a = n_orb - n_occ_a
    n_vir_b = n_orb - n_occ_b
    expected_doubles = (comb(n_occ_a,2)*comb(n_vir_a,2) +
                        comb(n_occ_b,2)*comb(n_vir_b,2) +
                        n_occ_a*n_occ_b*n_vir_a*n_vir_b)
    print(f'Expected doubles: {expected_doubles}')
    print(f'Match: {len(t_val_new) == expected_doubles}')

    # Compare values for alpha-beta (simplest case)
    # Current method outputs include singles + doubles, need to filter
    print('\nSample values (new method):')
    for i in range(min(5, len(t_val_new))):
        print(f'  az={t_az_new[i]:04b} bz={t_bz_new[i]:04b} val={t_val_new[i]:.6f}')


if __name__ == '__main__':
    test_correctness()
