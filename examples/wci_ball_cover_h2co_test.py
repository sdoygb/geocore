"""
H2CO/cc-pVDZ (dim=2.39e15): ball-cover selection only (no comparison
to avoid memory doubling). Tests if geometric ball-cover selection
improves convergence over CIPSI's false convergence on large systems.

Config: max_wp_size=3000, max_wp=3 (memory-safe).
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
    print('=' * 70)
    print('H2CO/cc-pVDZ (dim=2.39e15): ball-cover selection only')
    print('max_wp_size=3000, max_wp=3')
    print('=' * 70)

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

    print(f'n_orb={n_orb}, n_occ={n_occ}, dim={dim:.3e}')
    print(f'RHF={e_rhf:.6f}, CCSD(T)={e_ccsdt:.6f} Ha')

    # one-body spin-orbital integral only (skip t_s)
    o_s = np.zeros((ns, ns), dtype=complex)
    o_s[0::2, 0::2] = h_spatial
    o_s[1::2, 1::2] = h_spatial

    # Rank tables
    print(f'\nBuilding rank tables (da={da:,}, db={db:,})...')
    t0 = time.time()
    rt_a, rt_b, az_of, bz_of = build_rank_tables(n_orb, n_a, n_b, da, db)
    print(f'  Built in {time.time()-t0:.1f}s')

    # HF seed
    hf_az = int(np.sum(1 << np.arange(n_a)))
    hf_bz = int(np.sum(1 << np.arange(n_b)))
    seed_idx = int(rt_a.rank(np.array([hf_az]))[0] * db +
                    rt_b.rank(np.array([hf_bz]))[0])

    # GPU apply
    gpu_ctx, gpu_queue, _ = _get_global_gpu()
    gpu_apply = GPUApplyOccAwareSP(n_orb, n_a, t_spatial, eps=1e-4, chunk_size=32)
    gpu_argsort = GPUArgsort64(gpu_ctx, gpu_queue, 50_000_000)

    apply_fn, _, _, _, _, _ = exterior.sparse_action_sz_vec(
        ns, nelec, 0, o_s, None, nuc, 1e-4, gpu_apply=gpu_apply)

    def hd_fn(idxs):
        return exterior.sector_diagonal_at(ns, nelec, 0, o_s, None,
                                            nuc, idxs=np.asarray(idxs))

    # Run WCI with ball-cover selection
    print(f'\n=== WCI with ball-cover selection ===')
    t0 = time.time()
    E, idx, coeffs, wps, hist = wci(
        apply_fn, hd_fn, seed_idx, db, az_of, bz_of, rt_a, rt_b,
        max_wavepackets=3, tol=1e-6, verbose=True,
        energy_tol=None, use_gpu=False, gpu_argsort=gpu_argsort,
        max_wp_size=3000, h_build_chunk=200, residual_chunk=50_000_000,
        select_ball_cover=True, ball_cover_topk=100)
    elapsed = time.time() - t0

    print(f'\n=== Results ===')
    print(f'WCI energy = {E:.10f} Ha')
    print(f'CCSD(T)    = {e_ccsdt:.10f} Ha')
    print(f'Error      = {E - e_ccsdt:+.6f} Ha')
    print(f'RHF        = {e_rhf:.10f} Ha')
    print(f'Correlation energy recovered = {(e_rhf - E) / (e_rhf - e_ccsdt) * 100:.1f}%')
    print(f'Time = {elapsed:.1f}s = {elapsed/60:.2f}min')
    print(f'Wavepackets = {len(hist)}, final n_var = {hist[-1][0]}')

    print(f'\n=== Convergence history ===')
    print(f'{"WP":>3} {"n_var":>6} {"E (Ha)":>14} {"ΔE":>10} {"err vs CCSD(T)":>16} {"||r_out||":>10}')
    print('-' * 65)
    E_prev = None
    for i, (n_v, E_wp, r_in, r_out) in enumerate(hist):
        dE = E_wp - E_prev if E_prev is not None else 0
        err = E_wp - e_ccsdt
        print(f'{i+1:>3} {n_v:>6} {E_wp:>14.10f} {dE:>+10.6f} {err:>+16.6f} {r_out:>10.3e}')
        E_prev = E_wp


if __name__ == '__main__':
    main()
