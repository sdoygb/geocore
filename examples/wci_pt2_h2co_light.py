"""
H2CO/cc-pVDZ (dim=2.39e15): PT2 correction (lightweight version).

No redundant H-matrix rebuild at the end. max_wp_size=5000, max_wp=2
for memory safety. PT2 is computed per-iteration in the WCI loop.
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
    print('H2CO/cc-pVDZ (dim=2.39e15): PT2 correction (lightweight)')
    print('max_wp_size=5000, max_wp=2, PT2 top-100k')
    print('=' * 70)

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

    o_s = np.zeros((ns, ns), dtype=complex)
    o_s[0::2, 0::2] = h_spatial
    o_s[1::2, 1::2] = h_spatial

    print(f'\nBuilding rank tables...')
    t0 = time.time()
    rt_a, rt_b, az_of, bz_of = build_rank_tables(n_orb, n_a, n_b, da, db)
    print(f'  Built in {time.time()-t0:.1f}s')

    hf_az = int(np.sum(1 << np.arange(n_a)))
    hf_bz = int(np.sum(1 << np.arange(n_b)))
    seed_idx = int(rt_a.rank(np.array([hf_az]))[0] * db +
                    rt_b.rank(np.array([hf_bz]))[0])

    gpu_ctx, gpu_queue, _ = _get_global_gpu()
    gpu_apply = GPUApplyOccAwareSP(n_orb, n_a, t_spatial, eps=1e-4, chunk_size=32)
    gpu_argsort = GPUArgsort64(gpu_ctx, gpu_queue, 50_000_000)

    apply_fn, _, _, _, _, _ = exterior.sparse_action_sz_vec(
        ns, nelec, 0, o_s, None, nuc, 1e-4, gpu_apply=gpu_apply)

    def hd_fn(idxs):
        return exterior.sector_diagonal_at(
            ns, nelec, 0, o_s, None, nuc,
            idxs=np.asarray(idxs),
            lookup_tables=(az_of, bz_of, db))

    print(f'\n=== WCI: ball-cover + PT2 ===')
    t0 = time.time()
    E, idx, coeffs, wps, hist = wci(
        apply_fn, hd_fn, seed_idx, db, az_of, bz_of, rt_a, rt_b,
        max_wavepackets=2, tol=1e-6, verbose=True,
        energy_tol=None, use_gpu=False, gpu_argsort=gpu_argsort,
        max_wp_size=5000, h_build_chunk=200, residual_chunk=50_000_000,
        select_ball_cover=True, ball_cover_topk=100,
        compute_pt2=True, pt2_top_n=100000, pt2_chunk=200)
    elapsed = time.time() - t0

    # PT2 was computed per-iteration; the last iter line shows E+PT2.
    # Extract final corrected energy from the last iteration's output.
    # Since we don't store it in history, recompute just the PT2 for the
    # final state using the already-computed residual (but we don't have it
    # either).  Instead, just report the variational E and note that PT2
    # was printed per-iteration.
    print(f'\n=== Final Results ===')
    print(f'Variational E   = {E:.10f} Ha')
    print(f'CCSD(T)         = {e_ccsdt:.10f} Ha')
    print(f'Error (var)     = {E - e_ccsdt:+.6f} Ha')
    print(f'RHF             = {e_rhf:.10f} Ha')
    print(f'Corr. rec. (var)= {(e_rhf - E) / (e_rhf - e_ccsdt) * 100:.1f}%')
    print(f'Total time      = {elapsed:.1f}s = {elapsed/60:.2f}min')
    print(f'n_var           = {len(idx)}')
    print(f'\nNote: PT2-corrected energy was printed per-iteration above.')
    print(f'      The last iter line shows E+PT2 for the final state.')


if __name__ == '__main__':
    main()
