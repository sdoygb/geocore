#!/usr/bin/env python3
"""Verify GPU doubles correctness on H2O/STO-3G.

Compares CPU-only WCI energy vs GPU-doubles WCI energy.
"""
import sys, time
from math import comb
import numpy as np
from geoqc import exterior
from geoqc.wci import wci

def load(npz_path):
    d = np.load(npz_path)
    return int(d['n']), d['o_s'], d['t_s'], float(d['nuc'])

def build_rank_tables(n_orb, n_a, n_b, da, db):
    from itertools import combinations
    az_of = np.array([sum(1<<j for j in c) for c in combinations(range(n_orb), n_a)], dtype=np.int64)
    bz_of = np.array([sum(1<<j for j in c) for c in combinations(range(n_orb), n_b)], dtype=np.int64)
    rt_a = np.full(1<<n_orb, -1, dtype=np.int64); rt_a[az_of] = np.arange(da, dtype=np.int64)
    rt_b = np.full(1<<n_orb, -1, dtype=np.int64); rt_b[bz_of] = np.arange(db, dtype=np.int64)
    return rt_a, rt_b, az_of, bz_of

def main():
    npz_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/_h2o_sto3g.npz'
    max_wp = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    n_orb, o_s, t_s, nuc = load(npz_path)
    nelec = 10; n_a = n_b = nelec // 2; ns = 2 * n_orb

    rt_a, rt_b, az_of, bz_of = build_rank_tables(n_orb, n_a, n_b,
        comb(n_orb, n_a), comb(n_orb, n_b))

    # HF seed
    e_a = np.array([o_s[2*k, 2*k].real for k in range(n_orb)])
    e_b = np.array([o_s[2*k+1, 2*k+1].real for k in range(n_orb)])
    hf_a = np.sort(np.argsort(e_a)[:n_a])
    hf_b = np.sort(np.argsort(e_b)[:n_b])
    hf_az = int(np.sum(1 << hf_a))
    hf_bz = int(np.sum(1 << hf_b))
    db = comb(n_orb, n_b)
    seed_idx = int(rt_a[hf_az]) * db + int(rt_b[hf_bz])

    dim = comb(n_orb, n_a) * db
    print(f'H2O/STO-3G: n_orb={n_orb}, dim={dim:,}')

    # Diagonal
    hd, *_ = exterior.sector_diagonal_sz(
        ns, nelec, 0, o_s, t_s, nuc, 1e-4, two_body=False)
    _hd_arr = hd
    def hd_fn(idxs):
        return _hd_arr[np.asarray(idxs, dtype=np.int64)]

    # --- CPU run ---
    print(f'\n--- CPU (max_wp={max_wp}) ---')
    apply_fn_cpu, _, _, _, da, db = exterior.sparse_action_sz_vec(
        ns, nelec, 0, o_s, t_s, nuc, 1e-4)
    t0 = time.time()
    E_cpu, _, _, _, _ = wci(
        apply_fn_cpu, hd_fn, seed_idx, db, az_of, bz_of, rt_a, rt_b,
        max_wavepackets=max_wp, tol=1e-6, verbose=False, energy_tol=None)
    t_cpu = time.time() - t0
    print(f'  E={E_cpu:.10f} Ha, time={t_cpu:.2f}s')

    # --- GPU run (doubles on GPU) ---
    print(f'\n--- GPU doubles (max_wp={max_wp}) ---')
    from geoqc.gpu import GPUApply
    gpu_apply = GPUApply(n_orb, o_s, t_s, eps=1e-4, chunk_size=64)
    apply_fn_gpu, _, _, _, da, db = exterior.sparse_action_sz_vec(
        ns, nelec, 0, o_s, t_s, nuc, 1e-4, gpu_apply=gpu_apply)
    t0 = time.time()
    E_gpu, _, _, _, _ = wci(
        apply_fn_gpu, hd_fn, seed_idx, db, az_of, bz_of, rt_a, rt_b,
        max_wavepackets=max_wp, tol=1e-6, verbose=False, energy_tol=None)
    t_gpu = time.time() - t0
    gpu_apply.release()
    print(f'  E={E_gpu:.10f} Ha, time={t_gpu:.2f}s')

    # --- Compare ---
    print(f'\n{"="*50}')
    print(f'CPU E: {E_cpu:.10f} Ha')
    print(f'GPU E: {E_gpu:.10f} Ha')
    print(f'dE:    {abs(E_cpu - E_gpu):.2e} Ha')
    print(f'Speedup: {t_cpu/t_gpu:.2f}x')
    if abs(E_cpu - E_gpu) < 1e-6:
        print('SUCCESS: GPU doubles energy matches CPU!')
    else:
        print('MISMATCH!')

if __name__ == '__main__':
    main()
