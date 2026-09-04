"""
Comparison: plain CIPSI-style vs fully-geometric Bruhat-ball greedy-cover
wavepacket centre selection on H2CO/cc-pVDZ (dim=2.39e15).

Small-scale test: max_wp_size=2000, max_wp=5, to verify the ball-cover
selection logic runs correctly and improves convergence.
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


def run_one(select_ball_cover=False, select_geometric=False,
            geometric_alpha=1.0, max_wp=5, max_wp_size=2000,
            ball_cover_topk=100):
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
        select_ball_cover=select_ball_cover,
        ball_cover_topk=ball_cover_topk,
        select_geometric=select_geometric,
        geometric_alpha=geometric_alpha)
    elapsed = time.time() - t0

    return E, hist, elapsed, e_ccsdt


def main():
    print('=' * 70)
    print('H2CO/cc-pVDZ (dim=2.39e15): plain CIPSI vs ball-cover selection')
    print('Small-scale: max_wp_size=2000, max_wp=5')
    print('=' * 70)

    results = {}

    # --- Plain CIPSI-style ---
    print('\n--- [1/2] Plain CIPSI-style (argmax |r|) ---')
    E_plain, hist_plain, t_plain, e_ccsdt = run_one(
        select_ball_cover=False, select_geometric=False)
    results['plain'] = (E_plain, hist_plain, t_plain)

    # --- Ball-cover (fully geometric) ---
    print('\n--- [2/2] Bruhat-ball greedy-cover (fully geometric) ---')
    E_ball, hist_ball, t_ball, _ = run_one(
        select_ball_cover=True, ball_cover_topk=100)
    results['ball_cover'] = (E_ball, hist_ball, t_ball)

    # --- Comparison ---
    print('\n' + '=' * 70)
    print('COMPARISON')
    print('=' * 70)
    print(f'CCSD(T) reference: {e_ccsdt:.6f} Ha')
    print()
    print(f'{"WP":>3} {"plain E":>12} {"plain err":>10} {"ball E":>12} {"ball err":>10} {"improve":>10}')
    print('-' * 65)
    for i in range(max(len(hist_plain), len(hist_ball))):
        if i < len(hist_plain):
            n_v, E_p, r_in, r_out = hist_plain[i]
            err_p = E_p - e_ccsdt
        else:
            E_p, err_p = float('nan'), float('nan')
        if i < len(hist_ball):
            n_v, E_b, r_in, r_out = hist_ball[i]
            err_b = E_b - e_ccsdt
        else:
            E_b, err_b = float('nan'), float('nan')
        imp = err_p - err_b if not (np.isnan(err_p) or np.isnan(err_b)) else float('nan')
        print(f'{i+1:>3} {E_p:>12.6f} {err_p:>10.6f} {E_b:>12.6f} {err_b:>10.6f} {imp:>10.6f}')

    print()
    print(f'Plain:      E={E_plain:.6f}, error={E_plain-e_ccsdt:+.6f}, time={t_plain:.1f}s')
    print(f'Ball-cover: E={E_ball:.6f}, error={E_ball-e_ccsdt:+.6f}, time={t_ball:.1f}s')
    print(f'Improvement: {(E_plain-e_ccsdt) - (E_ball-e_ccsdt):+.6f} Ha')
    if abs(E_plain - e_ccsdt) > 1e-10:
        ratio = abs(E_plain - e_ccsdt) / max(abs(E_ball - e_ccsdt), 1e-10)
        print(f'Error reduction factor: {ratio:.2f}x')


if __name__ == '__main__':
    main()
