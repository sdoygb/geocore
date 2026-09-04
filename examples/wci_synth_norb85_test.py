"""
WCI test on H2CO/cc-pVDZ (n_orb=38, n_occ=8, dim ≈ 2.39×10^15) — real molecule.

Key optimisation: skips spin-orbital two-body tensor t_s — GPU doubles use
spatial integrals directly, and sector_diagonal_at only needs one-body o_s.
n_occ=8 keeps n_orb=38 < 64, so bitstrings fit in int64.
"""
import sys, os, time
import numpy as np
from math import comb

sys.path.insert(0, '/Users/oygb/Downloads/GeometryAI-Mac-Build/geocore')
from geoqc import exterior
from geoqc.wci import wci

sys.path.insert(0, '/Users/oygb/Downloads/GeometryAI-Mac-Build/geocore/examples')
from gpu_occ_aware_doubles_sp import GPUApplyOccAwareSP
from geoqc.gpu import GPUArgsort64, _get_global_gpu
from wci_h2o_ccpvtz_test import RankTable, build_rank_tables


def main():
    # Load H2CO/cc-pVDZ integrals (n_orb=38, n_occ=8, dim=2.39e15)
    d = np.load('data/h2co_ccpvdz_integrals.npz')
    n_orb = int(d['n'])
    n_occ = 8  # H2CO has 16 electrons
    h_spatial = d['h']
    t_spatial = d['t']
    nuc = float(d['nuc'])
    e_rhf = float(d['e_rhf'])
    e_ccsdt = float(d['e_ccsdt'])

    n_a = n_b = n_occ  # closed-shell
    nelec = n_a + n_b
    ns = 2 * n_orb
    da = comb(n_orb, n_a)
    db = comb(n_orb, n_b)
    dim = da * db

    print(f'=== H2CO/cc-pVDZ WCI Test (n_orb={n_orb}, nelec={nelec}) ===')
    print(f'dim = C({n_orb},{n_a}) * C({n_orb},{n_b}) = {da:,} * {db:,} = {dim:,} = {dim:.3e}')
    print(f'Reference: RHF={e_rhf:.6f}, CCSD(T)={e_ccsdt:.6f} Ha (PySCF)')
    print(f'Spatial integrals: h={h_spatial.shape}, t={t_spatial.shape}, t size={t_spatial.nbytes/1e6:.1f} MB')

    # --- ONLY one-body spin-orbital integral (o_s), NO t_s ---
    # t_s would be (2n)^4 * 16B = 13.4 GB for n_orb=85 — skip it entirely.
    # GPU doubles use spatial t_spatial; sector_diagonal_at only uses o_s.
    print(f'\nBuilding one-body spin-orbital integral (o_s only, skipping t_s)...')
    t0 = time.time()
    o_s = np.zeros((ns, ns), dtype=complex)
    o_s[0::2, 0::2] = h_spatial  # alpha-alpha
    o_s[1::2, 1::2] = h_spatial  # beta-beta
    print(f'  o_s: {o_s.shape}, size={o_s.nbytes/1e6:.2f} MB (vs t_s=13,400 MB)')
    print(f'  Done in {time.time()-t0:.1f}s')

    # Rank tables
    print(f'\nBuilding rank tables (da={da:,}, db={db:,})...')
    t0 = time.time()
    rt_a, rt_b, az_of, bz_of = build_rank_tables(n_orb, n_a, n_b, da, db)
    print(f'  Built in {time.time()-t0:.1f}s')

    # Diagonal (on-demand) — t_s=None is safe, sector_diagonal_at only uses o_s
    def hd_fn(idxs):
        return exterior.sector_diagonal_at(
            ns, nelec, 0, o_s, None, nuc, 1e-4,
            idxs=np.asarray(idxs, dtype=np.int64))

    # HF seed (lowest 5 orbitals)
    hf_az = int(np.sum(1 << np.arange(n_a)))
    hf_bz = int(np.sum(1 << np.arange(n_b)))
    seed_idx = int(rt_a.rank(np.array([hf_az]))[0]) * db + int(rt_b.rank(np.array([hf_bz]))[0])
    print(f'\nHF seed: az={hf_az:0{n_orb}b}, bz={hf_bz:0{n_orb}b}')
    print(f'  seed_idx = {seed_idx:,}')
    print(f'  HF energy = {hd_fn([seed_idx])[0]:.6f} Ha')

    # GPU setup
    max_wp = 1
    max_wp_size = 5000
    print(f'\n=== GPU occupancy-aware doubles (max_wp={max_wp}, max_wp_size={max_wp_size}, NO t_s) ===')
    gpu_ctx, gpu_queue, _ = _get_global_gpu()
    print(f'Creating GPUApplyOccAwareSP...')
    gpu_apply = GPUApplyOccAwareSP(n_orb, n_a, t_spatial, eps=1e-4, chunk_size=32)
    print(f'  Spatial integrals: {t_spatial.size:,} elements, {t_spatial.nbytes/1e6:.1f} MB')

    max_residual = 100_000_000
    print(f'Creating GPUArgsort64 (max_residual={max_residual:,})...')
    gpu_argsort = GPUArgsort64(gpu_ctx, gpu_queue, max_residual)

    # Apply function (singles CPU + doubles GPU, t_s=None)
    apply_fn, _, _, _, _, _ = exterior.sparse_action_sz_vec(
        ns, nelec, 0, o_s, None, nuc, 1e-4, gpu_apply=gpu_apply)

    # Run WCI
    print('\nStarting WCI...')
    t0 = time.time()
    # === WCI: dim=2.39e15 端到端验证 ===
    # 关键参数说明：
    #   max_wp=1          : 单波包可行性验证（多波包收敛见后续实验）
    #   max_wp_size=5000  : 变分空间 5000 态（内存安全上限，6000态触发 swap）
    #   h_build_chunk=200 : H 构建分块大小（每块~240MB输出，25块处理5000态）
    #                       chunk=1000 实测慢7.6x（kernel启动开销反而增大）
    #   t_s=None          : 跳过自旋轨道二体积分（n_orb=85时13.4GB），GPU doubles
    #                       直接用空间积分，sector_diagonal_at 只用 one-body o_s
    #   residual_chunk=50M: 残差计算分块大小（GPU argsort 缓冲区上限）
    E_gpu, idx_gpu, C_gpu, _, hist_gpu = wci(
        apply_fn, hd_fn, seed_idx, db, az_of, bz_of, rt_a, rt_b,
        max_wavepackets=max_wp, tol=1e-6, verbose=True,
        energy_tol=1e-4, use_gpu=False, gpu_argsort=gpu_argsort,
        max_wp_size=max_wp_size, h_build_chunk=200, residual_chunk=50_000_000)
    t_gpu = time.time() - t0

    print(f'\n=== Results ===')
    print(f'WCI energy = {E_gpu:.10f} Ha')
    print(f'V = {len(idx_gpu):,} determinants')
    print(f'Time = {t_gpu:.1f}s = {t_gpu/60:.2f}min')
    print(f'Wavepackets used: {len(hist_gpu)}')
    print(f'\n--- Comparison (H2CO/cc-pVDZ) ---')
    print(f'RHF:      {e_rhf:.10f} Ha')
    print(f'CCSD(T):  {e_ccsdt:.10f} Ha')
    print(f'WCI:      {E_gpu:.10f} Ha')
    print(f'WCI - CCSD(T) = {E_gpu - e_ccsdt:+.6f} Ha')
    print(f'\n=== dim={dim:.2e} (H2CO/cc-pVDZ, n_orb={n_orb}) test complete ===')


if __name__ == '__main__':
    main()
