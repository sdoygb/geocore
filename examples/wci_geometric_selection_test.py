"""
Quick comparison: plain CIPSI-style vs geometric (Bruhat-distance weighted)
wavepacket centre selection on H2CO/cc-pVDZ (dim=2.39e15).

Small-scale test first: max_wp_size=2000, max_wp=5, to verify the
geometric selection logic runs correctly and improves convergence.
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


def run_one(select_geometric, geometric_alpha, max_wp=5, max_wp_size=2000):
    # Load H2CO/cc-pVDZ integrals
    d = np.load('data/h2co_ccpvdz_integrals.npz')
    n_orb = int(d['n'])
    n_occ = 8
    h_spatial = d['h']
    t_spatial = d['t']
    nuc = float(d['nuc'])
    e_rhf = float(d['e_rhf'])
    e_ccsdt = float(d['e_ccsdt'])

    n_a = n_b = n_occ
    nelec = n_a + n_b
    ns = 2 * n_orb
    da = comb(n_orb, n_a)
    db = comb(n_orb, n_b)
    dim = da * db

    # one-body spin-orbital integral only (skip t_s)
    o_s = np.zeros((ns, ns), dtype=complex)
    o_s[0::2, 0::2] = h_spatial
    o_s[1::2, 1::2] = h_spatial

    # Rank tables
    rt_a, rt_b, az_of, bz_of = build_rank_tables(n_orb, n_a, n_b, da, db)

    # HF seed
    hf_az = int(np.sum(1 << np.arange(n_a)))
    hf_bz = int(np.sum(1 << np.arange(n_b)))
    seed_idx = int(rt_a.rank(np.array([hf_az]))[0] * db +
                    rt_b.rank(np.array([hf_bz]))[0])

    # GPU apply
    gpu_ctx, gpu_queue, _ = _get_global_gpu()
    gpu_apply = GPUApplyOccAwareSP(n_orb, n_a, t_spatial, eps=1e-4, chunk_size=32)
    gpu_argsort = GPUArgsort64(gpu_ctx, gpu_queue, 50_000_000)

    # Apply function (singles CPU + doubles GPU, t_s=None)
    apply_fn, _, _, _, _, _ = exterior.sparse_action_sz_vec(
        ns, nelec, 0, o_s, None, nuc, 1e-4, gpu_apply=gpu_apply)

    def hd_fn(idxs):
        return exterior.sector_diagonal_at(ns, nelec, 0, o_s, None,
                                            nuc, idxs=np.asarray(idxs))

    t0 = time.time()
    E, idx, coeffs, wps, hist = wci(
        apply_fn, hd_fn, seed_idx, db, az_of, bz_of, rt_a, rt_b,
        max_wavepackets=max_wp, tol=1e-6, verbose=True,
        energy_tol=None, use_gpu=False, gpu_argsort=gpu_argsort,
        max_wp_size=max_wp_size, h_build_chunk=200, residual_chunk=50_000_000,
        select_geometric=select_geometric, geometric_alpha=geometric_alpha)
    elapsed = time.time() - t0

    return E, hist, elapsed, e_ccsdt


def main():
    print('=' * 70)
    print('H2CO/cc-pVDZ (dim=2.39e15): plain vs geometric selection')
    print('Small-scale: max_wp_size=2000, max_wp=5')
    print('=' * 70)

    results = {}

    # --- Plain CIPSI-style ---
    print('\n--- [1/2] Plain CIPSI-style (argmax |r|) ---')
    E_plain, hist_plain, t_plain, e_ccsdt = run_one(
        select_geometric=False, geometric_alpha=1.0)
    results['plain'] = (E_plain, hist_plain, t_plain)

    # --- Geometric (α=1.0) ---
    print('\n--- [2/2] Geometric (Bruhat-distance weighted, α=1.0) ---')
    E_geo, hist_geo, t_geo, _ = run_one(
        select_geometric=True, geometric_alpha=1.0)
    results['geometric'] = (E_geo, hist_geo, t_geo)

    # --- Comparison ---
    print('\n' + '=' * 70)
    print('COMPARISON')
    print('=' * 70)
    print(f'CCSD(T) reference: {e_ccsdt:.6f} Ha')
    print()
    print(f'{"WP":>3} {"plain E":>12} {"plain ΔE":>10} {"geo E":>12} {"geo ΔE":>10} {"geo-plain":>10}')
    print('-' * 65)
    for i in range(max(len(hist_plain), len(hist_geo))):
        if i < len(hist_plain):
            n_v, E_p, r_in, r_out = hist_plain[i]
            dE_p = E_p - e_ccsdt
        else:
            E_p, dE_p = float('nan'), float('nan')
        if i < len(hist_geo):
            n_v, E_g, r_in, r_out = hist_geo[i]
            dE_g = E_g - e_ccsdt
        else:
            E_g, dE_g = float('nan'), float('nan')
        diff = E_g - E_p if not (np.isnan(E_p) or np.isnan(E_g)) else float('nan')
        print(f'{i+1:>3} {E_p:>12.6f} {dE_p:>10.6f} {E_g:>12.6f} {dE_g:>10.6f} {diff:>10.6f}')

    print()
    print(f'Plain:     E={E_plain:.6f}, error={E_plain-e_ccsdt:+.6f}, time={t_plain:.1f}s')
    print(f'Geometric: E={E_geo:.6f}, error={E_geo-e_ccsdt:+.6f}, time={t_geo:.1f}s')
    print(f'Improvement: {(E_plain-e_ccsdt) - (E_geo-e_ccsdt):+.6f} Ha')
    if abs(E_plain - e_ccsdt) > 1e-10:
        ratio = abs(E_plain - e_ccsdt) / max(abs(E_geo - e_ccsdt), 1e-10)
        print(f'Error reduction factor: {ratio:.2f}x')


if __name__ == '__main__':
    main()
