"""WCI test on real molecule H2O/cc-pVTZ (n_orb=58, dim=2.1e13).

Loads PySCF-generated integrals from npz, runs WCI with occupancy-aware
GPU doubles, and compares against CCSD(T) reference energy.
"""
import sys, os, time
import numpy as np
from math import comb

sys.path.insert(0, '/Users/oygb/Downloads/GeometryAI-Mac-Build/geocore')
from geoqc import exterior
from geoqc.wci import wci
from geoqc.integrals import spin_orbital_integrals
from itertools import combinations

sys.path.insert(0, '/Users/oygb/Downloads/GeometryAI-Mac-Build/geocore/examples')
from gpu_occ_aware_doubles_sp import GPUApplyOccAwareSP
from geoqc.gpu import GPUArgsort64, _get_global_gpu


class RankTable:
    """On-the-fly combinatorial rank table (bitstring -> rank)."""
    def __init__(self, n_orb, n_occ):
        self.n_orb = n_orb
        self.n_occ = n_occ
        self.comb_table = np.zeros((n_orb + 2, n_occ + 2), dtype=np.int64)
        for n in range(n_orb + 2):
            for k in range(min(n, n_occ + 1) + 1):
                self.comb_table[n, k] = comb(n, k)

    def _comb(self, n, k):
        if k < 0 or k > n or n < 0:
            return 0
        return int(self.comb_table[n, k])

    def __getitem__(self, key):
        """Support rt_a[az] and rt_a[az_array] subscript access."""
        key = np.asarray(key, dtype=np.int64)
        return self.rank(key)

    def rank(self, bitstrings):
        bitstrings = np.asarray(bitstrings, dtype=np.int64)
        scalar = bitstrings.ndim == 0
        if scalar:
            bitstrings = bitstrings.reshape(1)
        n = len(bitstrings)
        n_orb = self.n_orb
        n_occ = self.n_occ
        ct = self.comb_table
        # Vectorised bit extraction: (n, n_orb) matrix of 0/1
        bits = ((bitstrings[:, None] >> np.arange(n_orb, dtype=np.int64)) & 1).astype(np.int64)
        # Occupied orbitals sort first (0 before 1), take first n_occ, then sort ascending
        occ_idx = np.argsort(~bits.astype(bool), axis=1)[:, :n_occ]
        positions = np.sort(occ_idx, axis=1).T  # shape (n_occ, n)
        ranks = np.zeros(n, dtype=np.int64)
        for i in range(n_occ):
            c_i = positions[i]
            prev_pos = positions[i-1] if i > 0 else np.full(n, -1, dtype=np.int64)
            term1_n = np.clip(n_orb - prev_pos - 1, 0, n_orb + 1)
            term1 = ct[term1_n, n_occ - i]
            term2_n = np.clip(n_orb - c_i, 0, n_orb + 1)
            term2 = ct[term2_n, n_occ - i]
            ranks += term1 - term2
        if scalar:
            return int(ranks[0])
        return ranks


def build_rank_tables(n_orb, n_a, n_b, da, db):
    az_of = np.array([sum(1<<j for j in c) for c in combinations(range(n_orb), n_a)], dtype=np.int64)
    bz_of = np.array([sum(1<<j for j in c) for c in combinations(range(n_orb), n_b)], dtype=np.int64)
    rt_a = RankTable(n_orb, n_a)
    rt_b = RankTable(n_orb, n_b)
    return rt_a, rt_b, az_of, bz_of


def main():
    # Load real molecule integrals
    data = np.load(os.path.join(os.path.dirname(__file__), '..', 'data', 'h2o_ccpvtz_integrals.npz'))
    n_orb = int(data['n'])
    h_spatial = data['h']
    t_spatial = data['t']  # chemist notation (ij|kl)
    nuc = float(data['nuc'])
    e_rhf = float(data['e_rhf'])
    e_ccsdt = float(data['e_ccsdt'])

    nelec = 10
    n_a = n_b = nelec // 2
    ns = 2 * n_orb
    da = comb(n_orb, n_a)
    db = comb(n_orb, n_b)
    dim = da * db

    print(f'=== H2O/cc-pVTZ WCI Test (n_orb={n_orb}, nelec={nelec}) ===')
    print(f'dim = C({n_orb},{n_a}) * C({n_orb},{n_b}) = {da:,} * {db:,} = {dim:,} = {dim:.3e}')
    print(f'Reference: RHF={e_rhf:.10f}, CCSD(T)={e_ccsdt:.10f} Ha')
    print(f'Non-zero doubles per state: aa+bb+ab = {2*comb(n_a,2)*comb(n_orb-n_a,2) + n_a*n_b*(n_orb-n_a)*(n_orb-n_b):,}')

    # Convert to spin-orbital integrals
    print(f'\nConverting to spin-orbital integrals...')
    t0 = time.time()
    o_s, t_s = spin_orbital_integrals(h_spatial, t_spatial)
    print(f'  o_s: {o_s.shape}, t_s: {t_s.shape}, t_s size: {t_s.nbytes/1e6:.1f} MB')
    print(f'  Done in {time.time()-t0:.1f}s')

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
    seed_idx = int(rt_a.rank(np.array([hf_az]))[0]) * db + int(rt_b.rank(np.array([hf_bz]))[0])
    print(f'\nHF seed: az={hf_az:0{n_orb}b}, bz={hf_bz:0{n_orb}b}')
    print(f'  seed_idx = {seed_idx:,}')
    print(f'  HF energy = {hd_fn([seed_idx])[0]:.6f} Ha')

    # GPU setup
    max_wp = 3
    print(f'\n=== GPU occupancy-aware doubles + GPU argsort64 (max_wp={max_wp}, max_wp_size=5000, H_cols |v|>1e-5, incremental chunked) ===')
    gpu_ctx, gpu_queue, _ = _get_global_gpu()
    print(f'Creating GPUApplyOccAwareSP...')
    gpu_apply = GPUApplyOccAwareSP(n_orb, n_a, t_spatial, eps=1e-4, chunk_size=32)
    print(f'  Spatial integrals: {t_spatial.size:,} elements, {t_spatial.nbytes/1e6:.1f} MB')

    max_residual = 100_000_000
    print(f'Creating GPUArgsort64 (max_residual={max_residual:,})...')
    gpu_argsort = GPUArgsort64(gpu_ctx, gpu_queue, max_residual)

    # Apply function (singles CPU + doubles GPU)
    apply_fn, _, _, _, _, _ = exterior.sparse_action_sz_vec(
        ns, nelec, 0, o_s, t_s, nuc, 1e-4, gpu_apply=gpu_apply)

    # Run WCI
    print('\nStarting WCI...')
    t0 = time.time()
    E_gpu, idx_gpu, C_gpu, _, hist_gpu = wci(
        apply_fn, hd_fn, seed_idx, db, az_of, bz_of, rt_a, rt_b,
        max_wavepackets=max_wp, tol=1e-6, verbose=True,
        energy_tol=1e-4, use_gpu=False, gpu_argsort=gpu_argsort,
        max_wp_size=5000, h_build_chunk=200, residual_chunk=50_000_000)
    t_gpu = time.time() - t0

    print(f'\n=== Results ===')
    print(f'WCI energy = {E_gpu:.10f} Ha')
    print(f'V = {len(idx_gpu):,} determinants')
    print(f'Time = {t_gpu:.1f}s = {t_gpu/60:.2f}min')
    print(f'Wavepackets used: {len(hist_gpu)}')
    print(f'\n--- Comparison ---')
    print(f'RHF:      {e_rhf:.10f} Ha')
    print(f'CCSD(T):  {e_ccsdt:.10f} Ha')
    print(f'WCI:      {E_gpu:.10f} Ha')
    print(f'WCI - CCSD(T) = {E_gpu - e_ccsdt:+.6f} Ha = {(E_gpu - e_ccsdt)*627.5:+.2f} kcal/mol')
    print(f'WCI - RHF = {E_gpu - e_rhf:+.6f} Ha (correlation energy)')
    print(f'\n=== H2O/cc-pVTZ test complete ===')


if __name__ == '__main__':
    main()
