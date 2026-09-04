#!/usr/bin/env python3
"""GPU-accelerated WCI test: verify correctness and measure speedup.

Compares CPU vs GPU WCI on H2O/STO-3G (or other system).
Usage:
    PYTHONPATH=. python3 examples/geoqc_wci_gpu.py [npz] [nelec] [--ref-e0 X.XX]
"""
import sys, time
import numpy as np
from itertools import combinations
from geoqc import exterior, wci

def load_or_generate(npz_path, mol_name='h2o', basis='sto-3g'):
    """Load integrals from npz, or generate with pyscf if missing."""
    import os
    if os.path.exists(npz_path):
        d = np.load(npz_path)
        return int(d['n']), d['o_s'], d['t_s'], float(d['nuc'])

    print(f'  Generating integrals for {mol_name}/{basis}...')
    from pyscf import gto, scf, ao2mo
    from geoqc import integrals

    mol_configs = {
        'h2o': ('O 0 0 0; H 0 0 1.0; H 0 1.0 0', 10),
        'lih': ('Li 0 0 0; H 0 0 1.5', 4),
    }
    atom, nelec = mol_configs.get(mol_name, ('H 0 0 0; H 0 0 0.74', 2))
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).run()
    mo = mf.mo_coeff
    n_orb = mo.shape[1]
    h1e = mo.T @ (mol.intor('int1e_kin') + mol.intor('int1e_nuc')) @ mo
    eri = ao2mo.kernel(mol, mo, compact=False).reshape(n_orb, n_orb, n_orb, n_orb)
    o_s, t_s = integrals.spin_orbital_integrals(h1e, eri)
    nuc = mol.energy_nuc()
    np.savez(npz_path, n=n_orb, o_s=o_s, t_s=t_s, nuc=nuc)
    print(f'  Saved to {npz_path}')
    return n_orb, o_s, t_s, nuc

def build_rank_tables(n_orb, n_a, n_b, da, db):
    rt_a = np.full(1 << n_orb, -1, dtype=np.int64)
    rt_b = np.full(1 << n_orb, -1, dtype=np.int64)
    for i, c in enumerate(combinations(range(n_orb), n_a)):
        rt_a[sum(1 << j for j in c)] = i
    for i, c in enumerate(combinations(range(n_orb), n_b)):
        rt_b[sum(1 << j for j in c)] = i
    az_of = np.full(da, -1, dtype=np.int64)
    bz_of = np.full(db, -1, dtype=np.int64)
    for i, c in enumerate(combinations(range(n_orb), n_a)):
        az_of[i] = sum(1 << j for j in c)
    for i, c in enumerate(combinations(range(n_orb), n_b)):
        bz_of[i] = sum(1 << j for j in c)
    return rt_a, rt_b, az_of, bz_of

def run_wci(npz_path, nelec, use_gpu=False, max_wp=15, energy_tol=1e-4, mol_name='h2o', basis='sto-3g'):
    n_orb, o_s, t_s, nuc = load_or_generate(npz_path, mol_name, basis)
    n_a = n_b = nelec // 2
    ns = 2 * n_orb  # spin-orbital count
    dim_sector = None

    # Sparse action (vectorized) — expects ns (spin orbitals)
    apply_fn, n_a2, n_b2, n_orb2, da, db = exterior.sparse_action_sz_vec(
        ns, nelec, 0, o_s, t_s, nuc, 1e-4)
    dim = da * db

    # Rank tables
    rt_a, rt_b, az_of, bz_of = build_rank_tables(n_orb, n_a, n_b, da, db)

    # Diagonal (one-body only, two_body=False)
    hd, *_ = exterior.sector_diagonal_sz(
        ns, nelec, 0, o_s, t_s, nuc, 1e-4, two_body=False)
    _hd_arr = hd
    def hd_fn(idxs):
        return _hd_arr[np.asarray(idxs, dtype=np.int64)]

    # HF seed: lowest-energy orbitals by diagonal elements
    e_a = np.array([o_s[2*k, 2*k].real for k in range(n_orb)])
    e_b = np.array([o_s[2*k+1, 2*k+1].real for k in range(n_orb)])
    hf_a = np.sort(np.argsort(e_a)[:n_a])
    hf_b = np.sort(np.argsort(e_b)[:n_b])
    hf_az = int(np.sum(1 << hf_a))
    hf_bz = int(np.sum(1 << hf_b))
    seed_idx = int(rt_a[hf_az]) * db + int(rt_b[hf_bz])

    print(f'\n{"="*60}')
    print(f'WCI {"GPU" if use_gpu else "CPU"}: {npz_path}')
    print(f'  n_orb={n_orb}, nelec={nelec}, dim={dim}')
    print(f'  max_wp={max_wp}, energy_tol={energy_tol}')
    print(f'{"="*60}')

    t0 = time.time()
    E, unique_idx, coeffs, wavepackets, history = wci.wci(
        apply_fn, hd_fn, seed_idx, db, az_of, bz_of, rt_a, rt_b,
        max_wavepackets=max_wp, tol=1e-6, verbose=True,
        energy_tol=energy_tol, use_gpu=use_gpu)
    t_total = time.time() - t0

    print(f'\n  Final: E={E:.10f} Ha, V={len(unique_idx)}, '
          f'n_wp={len(wavepackets)}, time={t_total:.2f}s')
    return E, t_total, len(unique_idx), len(wavepackets)

def main():
    npz_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/_h2o_sto3g.npz'
    nelec = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    ref_e0 = float(sys.argv[3]) if len(sys.argv) > 3 else None
    mol_name = 'h2o' if 'h2o' in npz_path else 'lih'
    if 'cc-pvdz' in npz_path or 'ccpvdz' in npz_path:
        basis = 'cc-pvdz'
    elif '631g' in npz_path or '6-31g' in npz_path:
        basis = '6-31g'
    else:
        basis = 'sto-3g'

    # CPU run
    E_cpu, t_cpu, V_cpu, nwp_cpu = run_wci(npz_path, nelec, use_gpu=False, mol_name=mol_name, basis=basis)

    # GPU run
    E_gpu, t_gpu, V_gpu, nwp_gpu = run_wci(npz_path, nelec, use_gpu=True, mol_name=mol_name, basis=basis)

    # Comparison
    print(f'\n{"="*60}')
    print('COMPARISON')
    print(f'{"="*60}')
    print(f'  CPU: E={E_cpu:.10f}, V={V_cpu}, n_wp={nwp_cpu}, time={t_cpu:.2f}s')
    print(f'  GPU: E={E_gpu:.10f}, V={V_gpu}, n_wp={nwp_gpu}, time={t_gpu:.2f}s')
    print(f'  Energy diff: {abs(E_cpu - E_gpu):.2e} Ha')
    print(f'  Speedup: {t_cpu/t_gpu:.2f}x')
    if ref_e0 is not None:
        print(f'  Ref: {ref_e0:.10f}')
        print(f'  CPU err: {abs(E_cpu - ref_e0)*1000:.3f} mHa')
        print(f'  GPU err: {abs(E_gpu - ref_e0)*1000:.3f} mHa')

if __name__ == '__main__':
    main()
