#!/usr/bin/env python3
"""End-to-end WCI test at dim=10^12 (n_orb=44) with occupancy-aware GPU doubles.

Uses synthetic Hermitian integrals for controlled scaling verification.
Integrates GPUApplyOccAwareSP (spatial integrals) + GPUArgsort64 into WCI.
"""
import sys, time
import numpy as np
from math import comb
from itertools import combinations

from geoqc import exterior
from geoqc.wci import wci
from geoqc.gpu import GPUArgsort64, _get_global_gpu
from geoqc.integrals import spin_orbital_integrals

sys.path.insert(0, '/Users/oygb/Downloads/GeometryAI-Mac-Build/geocore/examples')
from gpu_occ_aware_doubles_sp import GPUApplyOccAwareSP


def build_rank_tables(n_orb, n_a, n_b, da, db):
    """Build rank tables for large systems using combinatorial rank (no full lookup).
    az_of/bz_of are arrays (rank -> bitstring), rt_a/rt_b are RankTable objects
    that compute rank on-the-fly using the combinatorial number system."""
    az_of = np.array([sum(1<<j for j in c) for c in combinations(range(n_orb), n_a)], dtype=np.int64)
    bz_of = np.array([sum(1<<j for j in c) for c in combinations(range(n_orb), n_b)], dtype=np.int64)
    rt_a = RankTable(n_orb, n_a)
    rt_b = RankTable(n_orb, n_b)
    return rt_a, rt_b, az_of, bz_of


class RankTable:
    """On-the-fly combinatorial rank table (bitstring -> rank).
    Uses LEXICOGRAPHIC order (matching Python itertools.combinations):
    rank = sum_{i=0}^{k-1} [C(n-c_{i-1}-1, k-i) - C(n-c_i, k-i)]
    where c_i are bit positions from LSB to MSB (ascending), c_{-1}=-1.
    Avoids the O(2^n_orb) full lookup table needed by np.full(1<<n_orb)."""

    def __init__(self, n_orb, n_occ):
        self.n_orb = n_orb
        self.n_occ = n_occ
        # Precompute C(n, k) for n up to n_orb+1, k up to n_occ+1
        self.comb_table = np.zeros((n_orb + 2, n_occ + 2), dtype=np.int64)
        for n in range(n_orb + 2):
            for k in range(min(n, n_occ + 1) + 1):
                self.comb_table[n, k] = comb(n, k)

    def _comb(self, n, k):
        if k < 0 or k > n or n < 0:
            return 0
        return self.comb_table[n, k]

    def __getitem__(self, key):
        key = np.asarray(key, dtype=np.int64)
        scalar = key.ndim == 0
        if scalar:
            key = key.reshape(1)
        n = len(key)
        n_orb = self.n_orb
        n_occ = self.n_occ
        ct = self.comb_table

        # Extract bit positions from LSB to MSB
        positions = np.zeros((n_occ, n), dtype=np.int64)
        remaining = key.copy()
        for i in range(n_occ):
            lowbit = remaining & -remaining
            # Safe bit position: log2 with correction for float precision
            pos = np.floor(np.log2(lowbit.astype(np.float64))).astype(np.int64)
            # Correct off-by-one: if 2^pos != lowbit, increment
            wrong = (np.int64(1) << pos) != lowbit
            pos = np.where(wrong, pos + 1, pos)
            positions[i] = pos
            remaining &= remaining - 1  # clear lowest set bit

        # Compute lexicographic rank
        ranks = np.zeros(n, dtype=np.int64)
        prev_pos = np.full(n, -1, dtype=np.int64)  # c_{-1} = -1
        for i in range(n_occ):
            c_i = positions[i]
            # C(n - prev_pos - 1, n_occ - i) - C(n - c_i, n_occ - i)
            term1_n = np.clip(n_orb - prev_pos - 1, 0, n_orb + 1)
            term1 = ct[term1_n, n_occ - i]
            term2_n = np.clip(n_orb - c_i, 0, n_orb + 1)
            term2 = ct[term2_n, n_occ - i]
            ranks += term1 - term2
            prev_pos = c_i

        if scalar:
            return ranks[0]
        return ranks


def make_synthetic_integrals(n_orb, seed=42):
    """Generate Hermitian synthetic integrals (chemist notation)."""
    rng = np.random.default_rng(seed)
    # 1e: Hermitian, diagonally dominant
    o = rng.standard_normal((n_orb, n_orb))
    o = (o + o.T) / 2
    o += np.diag(rng.standard_normal(n_orb) * 3.0)
    # 2e: chemist notation (ij|kl), symmetric under (ij)<->(kl) and particle exchange
    t = rng.standard_normal((n_orb, n_orb, n_orb, n_orb)) * 0.1
    t = 0.5 * (t + t.transpose(1, 0, 3, 2))  # (ij|kl) = (ji|lk)
    t = 0.5 * (t + t.transpose(2, 3, 0, 1))  # (ij|kl) = (kl|ij)
    return o, t


def main():
    n_orb = 44
    nelec = 10
    n_a = n_b = nelec // 2
    ns = 2 * n_orb
    da = comb(n_orb, n_a)
    db = comb(n_orb, n_b)
    dim = da * db

    print(f'=== dim=10^12 WCI Test (n_orb={n_orb}, nelec={nelec}) ===')
    print(f'dim = C({n_orb},{n_a}) * C({n_orb},{n_b}) = {da:,} * {db:,} = {dim:,} = {dim:.3e}')
    print(f'Non-zero doubles per state: aa+bb+ab = {2*comb(5,2)*comb(n_orb-5,2) + 25*(n_orb-5)**2:,}')
    print(f'All double terms (current method): {comb(n_orb,2)**2*4:,}')
    print(f'Computation reduction: {comb(n_orb,2)**2*4 / (2*comb(5,2)*comb(n_orb-5,2) + 25*(n_orb-5)**2):.1f}x')

    # Generate integrals
    print(f'\nGenerating synthetic integrals (n_orb={n_orb})...')
    t0 = time.time()
    o_spatial, t_spatial = make_synthetic_integrals(n_orb, seed=42)
    o_s, t_s = spin_orbital_integrals(o_spatial, t_spatial)
    nuc = 0.0  # synthetic, no nuclear repulsion
    print(f'  o_s: {o_s.shape}, t_s: {t_s.shape}, t_spatial: {t_spatial.shape}')
    print(f'  t_spatial size: {t_spatial.nbytes/1e6:.1f} MB')
    print(f'  t_s size: {t_s.nbytes/1e6:.1f} MB')
    print(f'  Generated in {time.time()-t0:.1f}s')

    # Rank tables
    print(f'\nBuilding rank tables (da={da:,}, db={db:,})...')
    t0 = time.time()
    rt_a, rt_b, az_of, bz_of = build_rank_tables(n_orb, n_a, n_b, da, db)
    print(f'  Built in {time.time()-t0:.1f}s')

    # Diagonal (on-demand)
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
    print(f'\nHF seed: az={hf_az:0{n_orb}b}, bz={hf_bz:0{n_orb}b}')
    print(f'  seed_idx = {seed_idx:,}')
    print(f'  HF energy = {hd_fn([seed_idx])[0]:.6f} Ha')

    # --- GPU occupancy-aware doubles + GPU argsort64 ---
    max_wp = 2
    print(f'\n=== GPU occupancy-aware doubles + GPU argsort64 (max_wp={max_wp}) ===')

    gpu_ctx, gpu_queue, _ = _get_global_gpu()

    # Occupancy-aware GPU doubles (spatial integrals)
    print('Creating GPUApplyOccAwareSP...')
    gpu_apply = GPUApplyOccAwareSP(n_orb, n_a, t_spatial, eps=1e-4, chunk_size=32)

    # GPU argsort64 (dim > 2^31)
    max_residual = 100_000_000  # increased from 50M (safe for 8GB VRAM)
    print(f'Creating GPUArgsort64 (max_residual={max_residual:,})...')
    gpu_argsort = GPUArgsort64(gpu_ctx, gpu_queue, max_residual)

    # Build apply_fn with GPU doubles
    apply_fn_gpu, _, _, _, _, _ = exterior.sparse_action_sz_vec(
        ns, nelec, 0, o_s, t_s, nuc, 1e-4, gpu_apply=gpu_apply)

    # Run WCI
    print('\nStarting WCI...')
    t0 = time.time()
    E_gpu, idx_gpu, C_gpu, _, hist_gpu = wci(
        apply_fn_gpu, hd_fn, seed_idx, db, az_of, bz_of, rt_a, rt_b,
        max_wavepackets=max_wp, tol=1e-6, verbose=True,
        energy_tol=1e-4, use_gpu=False, gpu_argsort=gpu_argsort,
        max_wp_size=2000, h_build_chunk=200, residual_chunk=50_000_000)
    t_gpu = time.time() - t0

    print(f'\n=== Results ===')
    print(f'E = {E_gpu:.10f} Ha')
    print(f'V = {len(idx_gpu):,} determinants')
    print(f'Time = {t_gpu:.1f}s = {t_gpu/60:.2f}min')
    print(f'Wavepackets used: {len(hist_gpu)}')
    for i, h in enumerate(hist_gpu):
        print(f'  WP{i+1}: n_var={h[0]}, E={h[1]:.10f}, r_in={h[2]:.2e}, r_out={h[3]:.2e}')

    # Cleanup
    gpu_apply.release()
    print(f'\n=== dim=10^12 test complete ===')


if __name__ == '__main__':
    main()
