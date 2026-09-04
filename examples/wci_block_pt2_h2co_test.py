"""
H2CO/cc-pVDZ (dim=2.392e15): geometric block-Newton (10.91 §6) vs diagonal
EN-PT2 vs CCSD(T).  Memory-safe, meant to run in the background.

Strategy
--------
1. WCI variational space (2 wavepackets, ball-cover selection), return the
   final out-of-space residual WITHOUT recomputing H columns.
2. Diagonal EN-PT2 on top-200k residual determinants (the historical best
   diagonal setting, error floor ~0.010 Ha).
3. Geometric block-Newton: disjoint Bruhat 2-balls around the largest
   residual determinants, dense intra-ball normal-Hessian inversion.
   Sweep n_centers and watch residual-weight coverage, block sizes, time,
   and RSS memory.  The point: can a HANDFUL of dense balls (capturing
   intra-ball coupling) beat 200k independent diagonal denominators and
   break the 0.010 Ha floor toward chemical accuracy (1.59 mHa)?

Reference: CCSD(T) = -114.182146412 Ha, RHF = -113.841823791 Ha.
"""
import os, sys, time, resource, gc
import numpy as np
from math import comb

sys.path.insert(0, '/Users/oygb/Downloads/GeometryAI-Mac-Build/geocore')
from geoqc import exterior
from geoqc.integrals import spin_orbital_integrals
from geoqc.wci import (wci, compute_pt2_correction,
                       compute_block_pt2_correction)

sys.path.insert(0, '/Users/oygb/Downloads/GeometryAI-Mac-Build/geocore/examples')
from gpu_occ_aware_doubles_sp import GPUApplyOccAwareSP
from geoqc.gpu import GPUArgsort64, _get_global_gpu
from wci_h2o_ccpvtz_test import build_rank_tables


def mem_mb():
    """peak RSS in MB (macOS ru_maxrss is in bytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def stamp(tag):
    print(f'  [{tag}] peak RSS = {mem_mb():.0f} MB', flush=True)


def main():
    t_start = time.time()
    print('=' * 74, flush=True)
    print('H2CO/cc-pVDZ block-Newton (10.91) vs diagonal PT2 vs CCSD(T)',
          flush=True)
    print('=' * 74, flush=True)

    d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'h2co_ccpvdz_integrals.npz'))
    n_orb = int(d['n']); n_occ = 8
    h_spatial = d['h']; t_spatial = d['t']; nuc = float(d['nuc'])
    e_rhf = float(d['e_rhf']); e_ref = float(d['e_ccsdt'])
    n_a = n_b = n_occ; nelec = n_a + n_b; ns = 2 * n_orb
    da = comb(n_orb, n_a); db = comb(n_orb, n_b); dim = da * db
    lvl = os.environ.get('H2CO_SMOKE')
    if lvl == '1':                      # smoke: 1-WP path check
        max_wp, topk, top_n, ncs = 1, 20, 20_000, [2, 5]
        max_wp_size, wci_only = 1500, False
        lvl_tag = 'smoke (1-WP path check)'
    elif lvl == '2':                    # level A: 2-WP decision point
        max_wp, topk, top_n, ncs = 2, 100, 50_000, [10]
        max_wp_size, wci_only = 1500, False
        lvl_tag = 'A (2-WP decision)'
    elif lvl == '3':                    # level W: enlarged wp ball diag
        max_wp, topk, top_n, ncs = 2, 100, 50_000, []
        max_wp_size, wci_only = 5000, True
        lvl_tag = 'W (2-WP wp-size 5000, WCI-only)'
    else:                               # full sweep
        max_wp, topk, top_n, ncs = 2, 100, 200_000, [10, 25, 50]
        max_wp_size, wci_only = 1500, False
        lvl_tag = 'full'
    print(f'gear: {lvl_tag}  (max_wp={max_wp}, topk={topk}, top_n={top_n}, '
          f'wp_size={max_wp_size}, ncs={ncs})', flush=True)
    print(f'n_orb={n_orb}, dim={dim:.3e}', flush=True)
    print(f'RHF={e_rhf:.9f}  CCSD(T)={e_ref:.9f}', flush=True)

    # CPU path must use the true spin-orbital two-body tensor.  A hand-rolled
    # alpha/beta embedding of h alone miscounts the two-body diagonal
    # (silently ~11 Ha off); spin_orbital_integrals does the interleaved
    # alpha/beta embedding plus adbc folding.

    t0 = time.time()
    rt_a, rt_b, az_of, bz_of = build_rank_tables(n_orb, n_a, n_b, da, db)
    print(f'rank tables {time.time()-t0:.1f}s', flush=True)

    hf_az = int(np.sum(1 << np.arange(n_a))); hf_bz = int(np.sum(1 << np.arange(n_b)))
    seed_idx = int(rt_a.rank(np.array([hf_az]))[0] * db +
                   rt_b.rank(np.array([hf_bz]))[0])

    o_s, t_s = spin_orbital_integrals(h_spatial, t_spatial)
    print(f'spin-orbital tensors o_s={o_s.shape} t_s={t_s.shape} '
          f'({t_s.nbytes/1e6:.0f} MB)', flush=True)

    gpu_ctx = gpu_queue = None
    gpu_apply = None
    gpu_argsort = None
    try:  # AMD-GPU box only; this Mac has no pyopencl / no AMD GPU
        gpu_ctx, gpu_queue, _ = _get_global_gpu()
        gpu_apply = GPUApplyOccAwareSP(n_orb, n_a, t_spatial, eps=1e-4,
                                       chunk_size=32)
        gpu_argsort = GPUArgsort64(gpu_ctx, gpu_queue, 50_000_000)
    except RuntimeError:
        print('no AMD GPU found -> CPU fallback '
              '(gpu_apply=None, gpu_argsort=None)', flush=True)
    apply_fn, _, _, _, _, _ = exterior.sparse_action_sz_vec(
        ns, nelec, 0, o_s, t_s, nuc, 1e-4, gpu_apply=gpu_apply)

    def hd_fn(idxs):
        return exterior.sector_diagonal_at(
            ns, nelec, 0, o_s, t_s, nuc, idxs=np.asarray(idxs),
            lookup_tables=(az_of, bz_of, db))

    # ---- 1. WCI variational space (return final residual) ----------------
    print(f'\n--- WCI variational space ({max_wp} WP, max_wp_size={max_wp_size}, '
          f'ball-cover) ---', flush=True)
    t0 = time.time()
    (E_var, idx, coeffs, wps, hist,
     r_out_idx, r_out_vals) = wci(
        apply_fn, hd_fn, seed_idx, db, az_of, bz_of, rt_a, rt_b,
        max_wavepackets=max_wp, tol=1e-6, verbose=True, energy_tol=None,
        use_gpu=False, gpu_argsort=gpu_argsort, max_wp_size=max_wp_size,
        h_build_chunk=200, residual_chunk=50_000_000,
        select_ball_cover=True, ball_cover_topk=topk,
        compute_pt2=False, return_final_residual=True)
    print(f'WCI {time.time()-t0:.1f}s, |V|={len(idx)}, E_var={E_var:.10f}',
          flush=True)
    print(f'n_out residual points = {len(r_out_idx)}, '
          f'||r_out||={np.linalg.norm(r_out_vals):.5f}', flush=True)
    stamp('after WCI'); gc.collect()

    if wci_only:
        print(f'\n[WCI-only] E_var={E_var:.10f}  '
              f'||r_out||={np.linalg.norm(r_out_vals):.5f}', flush=True)
        print(f'total wall = {(time.time()-t_start)/60:.2f} min', flush=True)
        return

    # ---- 2. diagonal EN-PT2 (top 200k, historical best) ------------------
    print(f'\n--- diagonal EN-PT2 (top {top_n}) ---', flush=True)
    t0 = time.time()
    E_diag, n_eval = compute_pt2_correction(
        r_out_idx, r_out_vals, E_var, apply_fn, hd_fn, db,
        az_of, bz_of, rt_a, rt_b, top_n=top_n, chunk_size=200)
    print(f'diag PT2={E_diag:+.8f} (n={n_eval}, {time.time()-t0:.1f}s)',
          flush=True)
    stamp('after diag PT2'); gc.collect()

    # ---- 3. geometric block-Newton sweep ---------------------------------
    print('\n--- geometric block-Newton (disjoint Bruhat 2-balls) ---', flush=True)
    block_results = []
    for nc in ncs:
        t0 = time.time()
        E_blk, info = compute_block_pt2_correction(
            r_out_idx, r_out_vals, E_var, apply_fn, hd_fn, db,
            az_of, bz_of, rt_a, rt_b, in_space_idx=idx,
            n_centers=nc, h_build_chunk=200, verbose=False)
        dt = time.time() - t0
        bs = np.array(info["block_sizes"]) if info["block_sizes"] else np.array([0])
        block_results.append((nc, E_blk, info))
        print(f'nc={nc:3d}: E_block={E_blk:+.8f}  blocks={info["n_blocks"]} '
              f'assigned={info["n_assigned"]} cov={info["residual_coverage"]:.4f} '
              f'block size min/med/max={bs.min()}/{int(np.median(bs))}/{bs.max()} '
              f'({dt:.1f}s)', flush=True)
        stamp(f'block nc={nc}'); gc.collect()

    # ---- 4. summary vs CCSD(T) -------------------------------------------
    chem = 0.00159
    print('\n' + '=' * 74, flush=True)
    print(f'{"method":28s} {"E_corr":>14s} {"E_est":>16s} {"err/mHa":>10s}',
          flush=True)
    print('-' * 74, flush=True)

    def row(name, ec):
        est = E_var + ec
        err = (est - e_ref) * 1000
        flag = '  <-- CHEM' if abs(err) < chem * 1000 else ''
        print(f'{name:28s} {ec:+14.8f} {est:16.10f} {err:+10.4f}{flag}',
              flush=True)

    row('variational', 0.0)
    row('diagonal PT2 200k', E_diag)
    for nc, E_blk, info in block_results:
        row(f'block-Newton nc={nc}', E_blk)
    print(f'{"CCSD(T) reference":28s} {"":>14s} {e_ref:16.10f} {0.0:10.4f}',
          flush=True)
    print('=' * 74, flush=True)
    print(f'chemical accuracy = {chem*1000:.2f} mHa; '
          f'total wall = {(time.time()-t_start)/60:.2f} min', flush=True)


if __name__ == '__main__':
    main()
