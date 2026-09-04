#!/usr/bin/env python3
"""H2O/cc-pVDZ GPU acceleration benchmark — lightweight (5 wavepackets).

Separates CPU and GPU runs for clearer timing. Prints per-iteration
breakdown (H build, diagonalization, residual).
"""
import sys, time
import numpy as np
from itertools import combinations
from geoqc import exterior, wci

def load(npz_path):
    d = np.load(npz_path)
    return int(d['n']), d['o_s'], d['t_s'], float(d['nuc'])

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

def run_wci(npz_path, use_gpu=False, max_wp=5):
    n_orb, o_s, t_s, nuc = load(npz_path)
    nelec = 10
    n_a = n_b = nelec // 2
    ns = 2 * n_orb

    gpu_apply = None
    if use_gpu:
        from geoqc.gpu import GPUApply
        gpu_apply = GPUApply(n_orb, o_s, t_s, eps=1e-4, chunk_size=64)

    apply_fn, _, _, _, da, db = exterior.sparse_action_sz_vec(
        ns, nelec, 0, o_s, t_s, nuc, 1e-4,
        gpu_apply=gpu_apply if use_gpu else None)
    dim = da * db
    rt_a, rt_b, az_of, bz_of = build_rank_tables(n_orb, n_a, n_b, da, db)

    # Diagonal: use on-demand for large systems (sector_diagonal_sz would
    # allocate dim*8 bytes = 14.4 GB for H2O/cc-pVDZ, causing swap)
    DIAG_FULL_LIMIT = 2_000_000
    if dim <= DIAG_FULL_LIMIT:
        hd, *_ = exterior.sector_diagonal_sz(
            ns, nelec, 0, o_s, t_s, nuc, 1e-4, two_body=False)
        _hd_arr = hd
        def hd_fn(idxs):
            return _hd_arr[np.asarray(idxs, dtype=np.int64)]
    else:
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

    tag = "GPU" if use_gpu else "CPU"
    print(f'\n{"="*60}')
    print(f'WCI {tag}: H2O/cc-pVDZ (n_orb={n_orb}, dim={dim:,})')
    print(f'  max_wp={max_wp}, use_gpu={use_gpu}')
    print(f'{"="*60}')

    t_start = time.time()
    E, unique_idx, coeffs, wavepackets, history = wci.wci(
        apply_fn, hd_fn, seed_idx, db, az_of, bz_of, rt_a, rt_b,
        max_wavepackets=max_wp, tol=1e-6, verbose=True,
        energy_tol=None, use_gpu=use_gpu)  # no energy_tol = run all max_wp
    t_total = time.time() - t_start

    print(f'\n  {tag} Final: E={E:.10f} Ha, V={len(unique_idx):,}, '
          f'n_wp={len(wavepackets)}, time={t_total:.1f}s')
    if gpu_apply is not None:
        gpu_apply.release()
    return E, t_total, len(unique_idx), len(wavepackets)

def main():
    npz_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/_h2o_ccpvdz.npz'
    max_wp = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    # CPU run
    E_cpu, t_cpu, V_cpu, nwp_cpu = run_wci(npz_path, use_gpu=False, max_wp=max_wp)

    # Cool down
    print('\n  Cooling down 5s...')
    time.sleep(5)

    # GPU run
    E_gpu, t_gpu, V_gpu, nwp_gpu = run_wci(npz_path, use_gpu=True, max_wp=max_wp)

    # Summary
    print(f'\n{"="*60}')
    print('SUMMARY: H2O/cc-pVDZ GPU Acceleration')
    print(f'{"="*60}')
    print(f'  {"":>6} {"CPU":>12} {"GPU":>12} {"Speedup":>10}')
    print(f'  {"Time":>6} {t_cpu:>11.1f}s {t_gpu:>11.1f}s {t_cpu/t_gpu:>9.2f}x')
    print(f'  {"V":>6} {V_cpu:>12,} {V_gpu:>12,}')
    print(f'  {"E":>6} {E_cpu:>12.6f} {E_gpu:>12.6f}')
    print(f'  Energy diff: {abs(E_cpu - E_gpu):.2e} Ha')

if __name__ == '__main__':
    main()
