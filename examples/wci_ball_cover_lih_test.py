"""
Quick validation of Bruhat-ball greedy-cover selection on LiH/STO-3G (dim=225).

Small system for fast logic verification: compares plain CIPSI-style vs
fully-geometric ball-cover wavepacket centre selection.
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

# PySCF for integral generation
from pyscf import gto, scf, cc


def generate_lih_sto3g():
    """Generate LiH/STO-3G MO-basis integrals with PySCF."""
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.595', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    e_rhf = mf.e_tot

    # CCSD(T) reference
    mycc = cc.CCSD(mf)
    mycc.kernel()
    e_ccsdt = mycc.e_tot + mycc.ccsd_t()

    # MO-basis integrals
    mo = mf.mo_coeff
    h_ao = mf.get_hcore()
    h_mo = mo.T @ h_ao @ mo
    t_ao = mol.intor('int2e')
    t_mo = np.einsum('ap,bq,cr,ds,abcd->pqrs', mo, mo, mo, mo, t_ao, optimize=True)
    nuc = mol.energy_nuc()

    return mol.nao_nr(), mol.nelectron // 2, h_mo, t_mo, nuc, e_rhf, e_ccsdt


def run_one(n_orb, n_occ, h_spatial, t_spatial, nuc, e_rhf, e_ccsdt,
            select_ball_cover=False, max_wp=5, max_wp_size=100,
            ball_cover_topk=50):
    """Run WCI with given selection criterion."""
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

    # GPU apply (small system, but use same code path)
    gpu_ctx, gpu_queue, _ = _get_global_gpu()
    gpu_apply = GPUApplyOccAwareSP(n_orb, n_a, t_spatial, eps=1e-10, chunk_size=32)
    gpu_argsort = GPUArgsort64(gpu_ctx, gpu_queue, 1_000_000)

    apply_fn, _, _, _, _, _ = exterior.sparse_action_sz_vec(
        ns, nelec, 0, o_s, None, nuc, 1e-10, gpu_apply=gpu_apply)

    def hd_fn(idxs):
        return exterior.sector_diagonal_at(ns, nelec, 0, o_s, None,
                                            nuc, idxs=np.asarray(idxs))

    t0 = time.time()
    E, idx, coeffs, wps, hist = wci(
        apply_fn, hd_fn, seed_idx, db, az_of, bz_of, rt_a, rt_b,
        max_wavepackets=max_wp, tol=1e-8, verbose=True,
        energy_tol=None, use_gpu=False, gpu_argsort=gpu_argsort,
        max_wp_size=max_wp_size, h_build_chunk=50, residual_chunk=100_000,
        select_ball_cover=select_ball_cover,
        ball_cover_topk=ball_cover_topk)
    elapsed = time.time() - t0

    return E, hist, elapsed, e_ccsdt, dim


def main():
    print('=' * 70)
    print('LiH/STO-3G (dim=225): plain CIPSI vs ball-cover selection')
    print('=' * 70)

    # Generate integrals
    print('\nGenerating LiH/STO-3G integrals...')
    n_orb, n_occ, h_spatial, t_spatial, nuc, e_rhf, e_ccsdt = generate_lih_sto3g()
    dim = comb(n_orb, n_occ) ** 2
    print(f'n_orb={n_orb}, n_occ={n_occ}, dim={dim}')
    print(f'RHF={e_rhf:.6f}, CCSD(T)={e_ccsdt:.6f} Ha')

    results = {}

    # --- Plain CIPSI-style ---
    print('\n--- [1/2] Plain CIPSI-style (argmax |r|) ---')
    E_plain, hist_plain, t_plain, _, _ = run_one(
        n_orb, n_occ, h_spatial, t_spatial, nuc, e_rhf, e_ccsdt,
        select_ball_cover=False, max_wp=5, max_wp_size=100)
    results['plain'] = (E_plain, hist_plain, t_plain)

    # --- Ball-cover ---
    print('\n--- [2/2] Bruhat-ball greedy-cover (fully geometric) ---')
    E_ball, hist_ball, t_ball, _, _ = run_one(
        n_orb, n_occ, h_spatial, t_spatial, nuc, e_rhf, e_ccsdt,
        select_ball_cover=True, max_wp=5, max_wp_size=100,
        ball_cover_topk=50)
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
