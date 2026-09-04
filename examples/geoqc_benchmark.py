#!/usr/bin/env python3
"""Benchmark: WCI (5-layer optimized) vs pyscf FCI (direct_spin1 Davidson).

Compares wall-clock time and final energy on systems of increasing dimension.
"""
import sys, time, math
import numpy as np
from pyscf import gto, scf, ao2mo, fci
from geoqc import wci, exterior

SYSTEMS = [
    # (name, atom, basis, nelec, expected_dim)
    ('H2O/STO-3G',  'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', 'sto-3g', (5,5), 441),
    ('LiH/6-31G',   'Li 0 0 0; H 0 0 1.595', '6-31g', (2,2), 3025),
    ('N2/STO-3G',   'N 0 0 0; N 0 0 1.098', 'sto-3g', (7,7), 14400),
    ('CH4/STO-3G',  'C 0 0 0; H 0.627 0.627 0.627; H 0.627 -0.627 -0.627; H -0.627 0.627 -0.627; H -0.627 -0.627 0.627', 'sto-3g', (5,5), 15876),
]

def run_pyscf_fci(mol, mo_coeff, h1e, eri, nelec, n_orb):
    """Run pyscf direct_spin1 FCI and return (energy, time)."""
    cis = fci.direct_spin1.FCISolver(mol)
    cis.verbose = 0
    cis.conv_tol = 1e-10
    cis.nroots = 1
    t0 = time.time()
    e, c = cis.kernel(h1e, eri, n_orb, nelec)
    dt = time.time() - t0
    return float(e), dt

def run_wci(mol, mo_coeff, h1e, eri, nelec, n_orb, max_wp=30, tol=1e-8):
    """Run WCI (library module) and return (energy, time, n_wp, dim_var)."""
    from geoqc import exterior, integrals
    from geoqc.wci import build_rank_tables

    # spin-orbital integrals (MUST use spin_orbital_integrals, not hand-rolled)
    o_s, t_s = integrals.spin_orbital_integrals(h1e, eri)
    nuc = mol.energy_nuc()
    n_spin = 2 * n_orb

    n_a, n_b = nelec
    apply_fn, _, _, _, da, db = exterior.sparse_action_sz(
        n_spin, nelec[0]+nelec[1], 0, o_s, t_s, nuc, 1e-4)
    dim = da * db
    rt_a, rt_b, az_of, bz_of = build_rank_tables(n_orb, n_a, n_b, da, db)

    # diagonal
    DIAG_FULL_LIMIT = 2_000_000
    if dim <= DIAG_FULL_LIMIT:
        hd, *_ = exterior.sector_diagonal_sz(
            n_spin, nelec[0]+nelec[1], 0, o_s, t_s, nuc, 1e-4, two_body=False)
        _hd_arr = hd
        def hd_fn(idxs):
            return _hd_arr[np.asarray(idxs, dtype=np.int64)]
    else:
        def hd_fn(idxs):
            return exterior.sector_diagonal_at(
                n_spin, nelec[0]+nelec[1], 0, o_s, t_s, nuc, 1e-4,
                idxs=np.asarray(idxs, dtype=np.int64))

    # HF seed
    e_a = np.array([o_s[2*k, 2*k].real for k in range(n_orb)])
    e_b = np.array([o_s[2*k+1, 2*k+1].real for k in range(n_orb)])
    hf_a = np.sort(np.argsort(e_a)[:n_a])
    hf_b = np.sort(np.argsort(e_b)[:n_b])
    hf_az = int(np.sum(1 << hf_a))
    hf_bz = int(np.sum(1 << hf_b))
    seed = int(rt_a[hf_az]) * db + int(rt_b[hf_bz])

    t0 = time.time()
    E, unique_idx, coeffs, wavepackets, history = wci.wci(
        apply_fn, hd_fn, seed, db, az_of, bz_of, rt_a, rt_b,
        max_wavepackets=max_wp, tol=tol, verbose=False)
    dt = time.time() - t0
    return float(E), dt, len(wavepackets), len(unique_idx)

def main():
    print('=== WCI vs pyscf FCI Benchmark ===')
    print(f'{"System":<14} {"dim":>7} {"FCI E":>12} {"FCI t":>8} {"WCI E":>12} {"WCI t":>8} {"speedup":>8} {"n_wp":>5} {"V/dim":>7}')
    print('-' * 100)

    results = []
    for name, atom, basis, nelec, expected_dim in SYSTEMS:
        print(f'\n--- {name} (expected dim={expected_dim}) ---')
        try:
            mol = gto.M(atom=atom, basis=basis, symmetry=False, verbose=0, spin=0)
            mf = scf.RHF(mol)
            mf.verbose = 0
            mf.kernel()
            mo_coeff = mf.mo_coeff
            n_orb = mo_coeff.shape[1]
            h1e = mo_coeff.T @ (mol.intor('int1e_kin') + mol.intor('int1e_nuc')) @ mo_coeff
            eri = ao2mo.kernel(mol, mo_coeff, compact=False).reshape(n_orb, n_orb, n_orb, n_orb)

            actual_dim = math.comb(n_orb, nelec[0]) * math.comb(n_orb, nelec[1])
            print(f'  n_orb={n_orb}, nelec={nelec}, actual dim={actual_dim}')

            # pyscf FCI
            e_fci, t_fci = run_pyscf_fci(mol, mo_coeff, h1e, eri, nelec, n_orb)
            print(f'  FCI: E={e_fci:.10f}, t={t_fci:.2f}s')

            # WCI
            e_wci, t_wci, n_wp, v_dim = run_wci(mol, mo_coeff, h1e, eri, nelec, n_orb)
            print(f'  WCI: E={e_wci:.10f}, t={t_wci:.2f}s, n_wp={n_wp}, V={v_dim}')

            speedup = t_fci / t_wci if t_wci > 0 else float('inf')
            err = abs(e_wci - e_fci)
            print(f'  speedup={speedup:.2f}x, |E_WCI-E_FCI|={err:.2e}')

            results.append({
                'name': name, 'dim': actual_dim,
                'e_fci': e_fci, 't_fci': t_fci,
                'e_wci': e_wci, 't_wci': t_wci,
                'speedup': speedup, 'n_wp': n_wp, 'v_dim': v_dim,
                'err': err,
            })

            print(f'{name:<14} {actual_dim:>7} {e_fci:>12.6f} {t_fci:>8.2f} {e_wci:>12.6f} {t_wci:>8.2f} {speedup:>8.2f} {n_wp:>5} {v_dim/actual_dim:>7.1%}')

        except Exception as e:
            print(f'  ERROR: {e}')
            import traceback
            traceback.print_exc()

    print('\n=== Summary ===')
    print(f'{"System":<14} {"dim":>7} {"FCI t":>8} {"WCI t":>8} {"speedup":>8} {"n_wp":>5} {"V/dim":>7} {"error":>10}')
    print('-' * 80)
    for r in results:
        print(f'{r["name"]:<14} {r["dim"]:>7} {r["t_fci"]:>8.2f} {r["t_wci"]:>8.2f} {r["speedup"]:>8.2f} {r["n_wp"]:>5} {r["v_dim"]/r["dim"]:>7.1%} {r["err"]:>10.2e}')

if __name__ == '__main__':
    main()
