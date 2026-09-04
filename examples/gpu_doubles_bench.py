#!/usr/bin/env python3
"""GPU doubles performance benchmark on H2O/cc-pVDZ scale.

Compares CPU apply_fn (singles+doubles) vs GPU doubles only.
"""
import sys, time
import numpy as np
from itertools import combinations
from geoqc import exterior
from geoqc.gpu import GPUApply

def main():
    npz_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/_h2o_ccpvdz.npz'
    S = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    d = np.load(npz_path)
    n_orb, o_s, t_s, nuc = int(d['n']), d['o_s'], d['t_s'], float(d['nuc'])
    nelec = 10
    n_a = n_b = nelec // 2
    ns = 2 * n_orb

    print(f'H2O/cc-pVDZ: n_orb={n_orb}, nelec={nelec}')
    print(f'Testing with S={S} source determinants')

    # CPU apply function
    apply_fn, _, _, _, da, db = exterior.sparse_action_sz_vec(
        ns, nelec, 0, o_s, t_s, nuc, 1e-4)
    dim = da * db
    print(f'FCI dim={dim:,}')

    # Generate random source determinants (valid alpha/beta bitstrings)
    rng = np.random.default_rng(42)
    az_list = [sum(1<<j for j in c) for c in combinations(range(n_orb), n_a)]
    bz_list = [sum(1<<j for j in c) for c in combinations(range(n_orb), n_b)]
    azs = rng.choice(az_list, size=S, replace=True).astype(np.int64)
    bzs = rng.choice(bz_list, size=S, replace=True).astype(np.int64)
    vals = rng.standard_normal(S).astype(np.float64)

    # --- CPU apply_fn (singles + doubles) ---
    print(f'\n--- CPU apply_fn (singles+doubles) ---')
    # warmup
    _ = apply_fn(azs[:100], bzs[:100], vals[:100])
    t0 = time.time()
    cpu_result = apply_fn(azs, bzs, vals)
    t_cpu = time.time() - t0
    if len(cpu_result) == 4:
        cpu_az, cpu_bz, cpu_v, cpu_src = cpu_result
    else:
        cpu_az, cpu_bz, cpu_v = cpu_result
        cpu_src = None
    n_cpu = len(cpu_az)
    print(f'  Time: {t_cpu:.3f}s')
    print(f'  Outputs: {n_cpu:,}')

    # Separate doubles from singles for comparison
    diff = np.array([bin(azs[cpu_src[i]]^cpu_az[i]).count('1') +
                      bin(bzs[cpu_src[i]]^cpu_bz[i]).count('1')
                      for i in range(n_cpu)])
    doubles_mask = diff != 2
    n_cpu_doubles = np.sum(doubles_mask)
    n_cpu_singles = n_cpu - n_cpu_doubles
    print(f'  Singles: {n_cpu_singles:,}, Doubles+diag: {n_cpu_doubles:,}')

    # --- GPU doubles ---
    print(f'\n--- GPU doubles only ---')
    gpu_apply = GPUApply(n_orb, o_s, t_s, eps=1e-4, chunk_size=64)
    print(f'  T={gpu_apply.T:,} double-excitation terms')

    # warmup
    _ = gpu_apply.doubles(azs[:100], bzs[:100], vals[:100])
    t0 = time.time()
    gpu_az, gpu_bz, gpu_v, gpu_src = gpu_apply.doubles(azs, bzs, vals)
    t_gpu = time.time() - t0
    n_gpu = len(gpu_az)
    print(f'  Time: {t_gpu:.3f}s')
    print(f'  Outputs: {n_gpu:,}')

    # Verify correctness (compare doubles part)
    cpu_d_az = cpu_az[doubles_mask]
    cpu_d_bz = cpu_bz[doubles_mask]
    cpu_d_v = np.asarray(cpu_v[doubles_mask]).real
    cpu_d_src = cpu_src[doubles_mask]

    idx_cpu = np.lexsort((cpu_d_bz, cpu_d_az, cpu_d_src))
    idx_gpu = np.lexsort((gpu_bz, gpu_az, gpu_src))
    cpu_d_az, cpu_d_bz, cpu_d_v, cpu_d_src = cpu_d_az[idx_cpu], cpu_d_bz[idx_cpu], cpu_d_v[idx_cpu], cpu_d_src[idx_cpu]
    gpu_az, gpu_bz, gpu_v, gpu_src = gpu_az[idx_gpu], gpu_bz[idx_gpu], gpu_v[idx_gpu], gpu_src[idx_gpu]

    if len(cpu_d_az) == len(gpu_az):
        err = np.max(np.abs(cpu_d_v - gpu_v))
        print(f'  Max value error: {err:.2e}')
        print(f'  Correct: {err < 1e-8}')
    else:
        print(f'  Count mismatch: CPU={len(cpu_d_az)}, GPU={len(gpu_az)}')

    gpu_apply.release()

    # Summary
    print(f'\n{"="*50}')
    print(f'SUMMARY (S={S})')
    print(f'{"="*50}')
    print(f'  CPU apply_fn (singles+doubles): {t_cpu*1000:.1f}ms')
    print(f'  GPU doubles only:               {t_gpu*1000:.1f}ms')
    print(f'  GPU doubles speedup:            {t_cpu/t_gpu:.2f}x')
    print(f'  (If doubles were 100% of apply time)')
    print(f'  Estimated apply speedup if doubles=70%: {1/(0.3 + 0.7/(t_cpu/t_gpu)):.2f}x')

if __name__ == '__main__':
    main()
