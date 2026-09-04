#!/usr/bin/env python3
"""Test GPU doubles alone (without GPU matvec) on H2O/cc-pVDZ."""
import sys, time
import numpy as np
from geoqc import exterior
from geoqc.wci import wci
from geoqc.gpu import GPUApply, GPUArgsort, GPUArgsort64, _get_global_gpu
from math import comb
from itertools import combinations

def load(npz_path):
    d = np.load(npz_path)
    return int(d['n']), d['o_s'], d['t_s'], float(d['nuc'])

def build_rank_tables(n_orb, n_a, n_b, da, db):
    az_of = np.array([sum(1<<j for j in c) for c in combinations(range(n_orb), n_a)], dtype=np.int64)
    bz_of = np.array([sum(1<<j for j in c) for c in combinations(range(n_orb), n_b)], dtype=np.int64)
    rt_a = np.full(1<<n_orb, -1, dtype=np.int64); rt_a[az_of] = np.arange(da, dtype=np.int64)
    rt_b = np.full(1<<n_orb, -1, dtype=np.int64); rt_b[bz_of] = np.arange(db, dtype=np.int64)
    return rt_a, rt_b, az_of, bz_of

def main():
    npz_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/_h2o_ccpvdz.npz'
    max_wp = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    n_orb, o_s, t_s, nuc = load(npz_path)
    nelec = 10; n_a = n_b = nelec // 2; ns = 2 * n_orb
    da = comb(n_orb, n_a); db = comb(n_orb, n_b)
    dim = da * db

    rt_a, rt_b, az_of, bz_of = build_rank_tables(n_orb, n_a, n_b, da, db)

    # Diagonal (on-demand for large systems)
    def hd_fn(idxs):
        return exterior.sector_diagonal_at(
            ns, nelec, 0, o_s, t_s, nuc, 1e-4,
            idxs=np.asarray(idxs, dtype=np.int64))

    # HF seed
    e_a = np.array([o_s[2*k, 2*k].real for k in range(n_orb)])
    e_b = np.array([o_s[2*k+1, 2*k+1].real for k in range(n_orb)])
    hf_a = np.sort(np.argsort(e_a)[:n_a])
    hf_b = np.sort(np.argsort(e_b)[:n_b])
    hf_az = int(np.sum(1 << hf_a))
    hf_bz = int(np.sum(1 << hf_b))
    seed_idx = int(rt_a[hf_az]) * db + int(rt_b[hf_bz])

    print(f'H2O/cc-pVDZ: n_orb={n_orb}, dim={dim:,}')

    # --- CPU run ---
    print(f'\n--- CPU (max_wp={max_wp}) ---')
    apply_fn_cpu, _, _, _, _, _ = exterior.sparse_action_sz_vec(
        ns, nelec, 0, o_s, t_s, nuc, 1e-4)
    t0 = time.time()
    E_cpu, idx_cpu, _, _, hist_cpu = wci(
        apply_fn_cpu, hd_fn, seed_idx, db, az_of, bz_of, rt_a, rt_b,
        max_wavepackets=max_wp, tol=1e-6, verbose=True,
        energy_tol=None, use_gpu=False)
    t_cpu = time.time() - t0
    print(f'  CPU: E={E_cpu:.10f} Ha, V={len(idx_cpu):,}, time={t_cpu:.1f}s')

    # --- GPU doubles + GPU argsort run ---
    print(f'\n--- GPU doubles + GPU argsort (max_wp={max_wp}) ---')
    gpu_ctx, gpu_queue, _ = _get_global_gpu()
    gpu_apply = GPUApply(n_orb, o_s, t_s, eps=1e-4, chunk_size=64)
    # Choose int32 or int64 argsort based on dim
    max_residual = 100_000_000  # 3.2GB GPU memory for int64 argsort (safe for 8GB GPU)
    if dim < (1 << 31):
        gpu_argsort = GPUArgsort(gpu_ctx, gpu_queue, max_residual)
        print(f'  Using GPUArgsort (int32, dim={dim:,} < 2^31)')
    else:
        gpu_argsort = GPUArgsort64(gpu_ctx, gpu_queue, max_residual)
        print(f'  Using GPUArgsort64 (int64, dim={dim:,} >= 2^31)')
    apply_fn_gpu, _, _, _, _, _ = exterior.sparse_action_sz_vec(
        ns, nelec, 0, o_s, t_s, nuc, 1e-4, gpu_apply=gpu_apply)
    t0 = time.time()
    E_gpu, idx_gpu, _, _, hist_gpu = wci(
        apply_fn_gpu, hd_fn, seed_idx, db, az_of, bz_of, rt_a, rt_b,
        max_wavepackets=max_wp, tol=1e-6, verbose=True,
        energy_tol=None, use_gpu=False, gpu_argsort=gpu_argsort)
    t_gpu = time.time() - t0
    gpu_apply.release()
    gpu_argsort.release()
    print(f'  GPU doubles+argsort: E={E_gpu:.10f} Ha, V={len(idx_gpu):,}, time={t_gpu:.1f}s')

    # --- Compare ---
    print(f'\n{"="*50}')
    print(f'CPU E:         {E_cpu:.10f} Ha')
    print(f'GPU doubles E: {E_gpu:.10f} Ha')
    print(f'dE:            {abs(E_cpu - E_gpu):.2e} Ha')
    print(f'CPU V:         {len(idx_cpu):,}')
    print(f'GPU doubles V: {len(idx_gpu):,}')
    print(f'Speedup:       {t_cpu/t_gpu:.2f}x')
    if abs(E_cpu - E_gpu) < 1e-6:
        print('SUCCESS: GPU doubles energy matches CPU!')
    else:
        print('MISMATCH!')

if __name__ == '__main__':
    main()
