"""
LiH/6-31G (dim=3025): PT2 correction validation.

Small system for fast PT2 logic verification. Compares variational E vs
E+PT2 against CCSD(T) reference.
"""
import sys, os, time
import numpy as np
from math import comb

sys.path.insert(0, '/Users/oygb/Downloads/GeometryAI-Mac-Build/geocore')
from geoqc import exterior
from geoqc.wci import wci, compute_pt2_correction

sys.path.insert(0, '/Users/oygb/Downloads/GeometryAI-Mac-Build/geocore/examples')
from gpu_occ_aware_doubles_sp import GPUApplyOccAwareSP
from geoqc.gpu import GPUArgsort64, _get_global_gpu
from wci_h2o_ccpvtz_test import RankTable, build_rank_tables

from pyscf import gto, scf, cc


def generate_lih_631g():
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.595', basis='6-31g', verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    e_rhf = mf.e_tot
    mycc = cc.CCSD(mf)
    mycc.kernel()
    e_ccsdt = mycc.e_tot + mycc.ccsd_t()
    mo = mf.mo_coeff
    h_ao = mf.get_hcore()
    h_mo = mo.T @ h_ao @ mo
    t_ao = mol.intor('int2e')
    t_mo = np.einsum('ap,bq,cr,ds,abcd->pqrs', mo, mo, mo, mo, t_ao, optimize=True)
    nuc = mol.energy_nuc()
    return mol.nao_nr(), mol.nelectron // 2, h_mo, t_mo, nuc, e_rhf, e_ccsdt


def main():
    print('=' * 70)
    print('LiH/6-31G (dim=3025): PT2 correction validation')
    print('=' * 70)

    n_orb, n_occ, h_spatial, t_spatial, nuc, e_rhf, e_ccsdt = generate_lih_631g()
    dim = comb(n_orb, n_occ) ** 2
    print(f'n_orb={n_orb}, n_occ={n_occ}, dim={dim}')
    print(f'RHF={e_rhf:.6f}, CCSD(T)={e_ccsdt:.6f} Ha')

    n_a = n_b = n_occ
    nelec = n_a + n_b
    ns = 2 * n_orb
    da = comb(n_orb, n_a)
    db = comb(n_orb, n_b)

    o_s = np.zeros((ns, ns), dtype=complex)
    o_s[0::2, 0::2] = h_spatial
    o_s[1::2, 1::2] = h_spatial

    rt_a, rt_b, az_of, bz_of = build_rank_tables(n_orb, n_a, n_b, da, db)

    hf_az = int(np.sum(1 << np.arange(n_a)))
    hf_bz = int(np.sum(1 << np.arange(n_b)))
    seed_idx = int(rt_a.rank(np.array([hf_az]))[0] * db +
                    rt_b.rank(np.array([hf_bz]))[0])

    gpu_ctx, gpu_queue, _ = _get_global_gpu()
    gpu_apply = GPUApplyOccAwareSP(n_orb, n_a, t_spatial, eps=1e-10, chunk_size=32)
    gpu_argsort = GPUArgsort64(gpu_ctx, gpu_queue, 5_000_000)

    apply_fn, _, _, _, _, _ = exterior.sparse_action_sz_vec(
        ns, nelec, 0, o_s, None, nuc, 1e-10, gpu_apply=gpu_apply)

    def hd_fn(idxs):
        return exterior.sector_diagonal_at(
            ns, nelec, 0, o_s, None, nuc,
            idxs=np.asarray(idxs),
            lookup_tables=(az_of, bz_of, db))

    # Run WCI with PT2
    print(f'\n=== WCI with PT2 (max_wp=5, max_wp_size=200) ===')
    E, idx, coeffs, wps, hist = wci(
        apply_fn, hd_fn, seed_idx, db, az_of, bz_of, rt_a, rt_b,
        max_wavepackets=5, tol=1e-8, verbose=True,
        energy_tol=None, use_gpu=False, gpu_argsort=gpu_argsort,
        max_wp_size=200, h_build_chunk=100, residual_chunk=500_000,
        select_ball_cover=True, ball_cover_topk=50,
        compute_pt2=True, pt2_top_n=50000, pt2_chunk=200)

    print(f'\n=== Final ===')
    print(f'Variational E = {E:.10f} Ha')
    print(f'CCSD(T)       = {e_ccsdt:.10f} Ha')
    print(f'Error (var)   = {E - e_ccsdt:+.6f} Ha')

    # Note: PT2 was computed per-iteration in the loop; the final corrected
    # energy is printed in the last iter line. Let's recompute it explicitly.
    from geoqc.wci import compute_residual_incremental, build_H_matrix, _eigh_ground
    H_cols = {}
    H_mat = build_H_matrix(idx, apply_fn, hd_fn, db, az_of, bz_of, rt_a, rt_b,
                            H_cols, chunk_size=100)
    E_final, c_final = _eigh_ground(H_mat)
    r_in, r_out_idx, r_out_vals = compute_residual_incremental(
        c_final, idx, H_cols, E_final, gpu_argsort=gpu_argsort,
        residual_chunk=500_000)
    E_pt2, n_pt2 = compute_pt2_correction(
        r_out_idx, r_out_vals, E_final, apply_fn, hd_fn, db,
        az_of, bz_of, rt_a, rt_b, top_n=50000, chunk_size=200)
    E_corr = E_final + E_pt2

    print(f'\n=== PT2 Analysis (final state) ===')
    print(f'Variational E   = {E_final:.10f} Ha')
    print(f'PT2 correction  = {E_pt2:+.10f} Ha (top-{n_pt2})')
    print(f'E + PT2         = {E_corr:.10f} Ha')
    print(f'CCSD(T)         = {e_ccsdt:.10f} Ha')
    print(f'Error (var)     = {E_final - e_ccsdt:+.6f} Ha')
    print(f'Error (E+PT2)   = {E_corr - e_ccsdt:+.6f} Ha')
    print(f'Chemical acc.   = 1.59 mHa')
    print(f'Within chem acc? {"YES" if abs(E_corr - e_ccsdt) < 0.00159 else "NO"}')
    print(f'||r_out||       = {np.linalg.norm(r_out_vals):.6f}')
    print(f'n_out           = {len(r_out_vals)}')


if __name__ == '__main__':
    main()
