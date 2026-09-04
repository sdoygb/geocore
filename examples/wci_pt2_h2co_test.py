"""
H2CO/cc-pVDZ (dim=2.39e15): PT2 correction with ball-cover selection.

Tests if Epstein-Nesbet PT2 (diagonal approximation of geometric
normal-Hessian correction) brings the energy close to chemical accuracy.

Config: max_wp_size=3000, max_wp=3, ball-cover selection, PT2 top-100k.
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
    print('H2CO/cc-pVDZ (dim=2.39e15): PT2 correction + ball-cover selection')
    print('max_wp_size=3000, max_wp=3, PT2 top-100k')
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
    print(f'Chemical accuracy = 1.59 mHa (1 kcal/mol)')

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

    # Run WCI with ball-cover selection + PT2
    print(f'\n=== WCI: ball-cover + PT2 correction ===')
    t0 = time.time()
    E, idx, coeffs, wps, hist = wci(
        apply_fn, hd_fn, seed_idx, db, az_of, bz_of, rt_a, rt_b,
        max_wavepackets=3, tol=1e-6, verbose=True,
        energy_tol=None, use_gpu=False, gpu_argsort=gpu_argsort,
        max_wp_size=3000, h_build_chunk=200, residual_chunk=50_000_000,
        select_ball_cover=True, ball_cover_topk=100,
        compute_pt2=True, pt2_top_n=100000, pt2_chunk=500)
    elapsed = time.time() - t0

    # Compute final PT2 for the last iteration (already computed in loop,
    # but extract from the last verbose output is not available; recompute)
    from geoqc.wci import compute_pt2_correction
    from geoqc.wci import compute_residual_incremental
    # Actually PT2 was already computed in the loop; we just need the final
    # corrected energy. But we don't store it in history. Let's recompute
    # the residual and PT2 for the final state.
    print(f'\n=== Recomputing final PT2 for detailed analysis ===')
    H_cols_final = {}
    from geoqc.wci import build_H_matrix
    H_mat_final = build_H_matrix(
        idx, apply_fn, hd_fn, db, az_of, bz_of, rt_a, rt_b,
        H_cols_final, chunk_size=200)
    from geoqc.wci import _eigh_ground
    E_final, coeffs_final = _eigh_ground(H_mat_final)
    r_in, r_out_idx, r_out_vals = compute_residual_incremental(
        coeffs_final, idx, H_cols_final, E_final, gpu_argsort=gpu_argsort,
        residual_chunk=50_000_000)
    E_pt2_final, n_pt2_final = compute_pt2_correction(
        r_out_idx, r_out_vals, E_final, apply_fn, hd_fn, db,
        az_of, bz_of, rt_a, rt_b, top_n=100000, chunk_size=500)
    E_corr_final = E_final + E_pt2_final

    print(f'\n=== Final Results ===')
    print(f'Variational E   = {E_final:.10f} Ha')
    print(f'PT2 correction  = {E_pt2_final:+.10f} Ha (top-{n_pt2_final})')
    print(f'E + PT2         = {E_corr_final:.10f} Ha')
    print(f'CCSD(T)         = {e_ccsdt:.10f} Ha')
    print(f'Error (var)     = {E_final - e_ccsdt:+.6f} Ha')
    print(f'Error (E+PT2)   = {E_corr_final - e_ccsdt:+.6f} Ha')
    print(f'Chemical acc.   = 1.59 mHa')
    print(f'Within chem acc? {"YES" if abs(E_corr_final - e_ccsdt) < 0.00159 else "NO"}')
    print(f'RHF             = {e_rhf:.10f} Ha')
    print(f'Corr. rec. (var)= {(e_rhf - E_final) / (e_rhf - e_ccsdt) * 100:.1f}%')
    print(f'Corr. rec. (PT2)= {(e_rhf - E_corr_final) / (e_rhf - e_ccsdt) * 100:.1f}%')
    print(f'Total time      = {elapsed:.1f}s = {elapsed/60:.2f}min')
    print(f'n_var           = {len(idx)}')


if __name__ == '__main__':
    main()
