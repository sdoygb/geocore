"""
LiH/6-31G (dim=3025): validate the geometric block-Newton correction
(article 10.91 §6 eq 6.3, §9.1 block-Jacobi) against diagonal PT2 and the
EXACT 2nd-order Schur-complement correction on a dense Hamiltonian, at TWO
variational-space sizes (small V=large residual, larger V=small residual).

Hierarchy of normal-Hessian approximations (10.91 §6.2):
  diagonal PT2 : H_N approximated by its diagonal (1x1 blocks)
  block Newton : H_N block-diagonal on disjoint Bruhat 2-balls (this work)
  exact 2nd ord: H_N inverted as ONE block over V^perp (rigorous anchor)
Block-Jacobi (10.91 §9.1) removes the residual inter-block coupling.
"""
import sys, time
import numpy as np
from math import comb

sys.path.insert(0, '/Users/oygb/Downloads/GeometryAI-Mac-Build/geocore')
from geoqc import exterior
from geoqc.integrals import spin_orbital_integrals
from geoqc.wci import (wci, build_H_matrix,
                       compute_block_pt2_correction)

sys.path.insert(0, '/Users/oygb/Downloads/GeometryAI-Mac-Build/geocore/examples')
from wci_h2o_ccpvtz_test import build_rank_tables as _brt
from pyscf import gto, scf, fci


def build_system():
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.595', basis='6-31g', verbose=0)
    mf = scf.RHF(mol); mf.kernel()
    mo = mf.mo_coeff
    h_mo = mo.T @ mf.get_hcore() @ mo
    t_mo = np.einsum('ap,bq,cr,ds,abcd->pqrs', mo, mo, mo, mo,
                     mol.intor('int2e'), optimize=True)
    e_fci = fci.FCI(mf).kernel()[0]
    return mol.nao_nr(), mol.nelectron // 2, h_mo, t_mo, mol.energy_nuc(), e_fci


def analyse(V_cfg, apply_fn, hd_fn, db, az_of, bz_of, rt_a, rt_b,
            H_full, all_idx, e_fci, gpu_argsort, seed_idx):
    n_wp, wp_size, tag = V_cfg
    print('\n' + '#' * 74)
    print(f'# VARIATIONAL SPACE: {tag}  (max_wp={n_wp}, max_wp_size={wp_size})')
    print('#' * 74)
    _, idx, _, _, _ = wci(
        apply_fn, hd_fn, seed_idx, db, az_of, bz_of, rt_a, rt_b,
        max_wavepackets=n_wp, tol=1e-9, verbose=False, energy_tol=None,
        use_gpu=False, gpu_argsort=gpu_argsort, max_wp_size=wp_size,
        h_build_chunk=100, residual_chunk=500_000,
        select_ball_cover=True, ball_cover_topk=50, compute_pt2=False)
    dim = len(all_idx)
    posV = np.searchsorted(all_idx, idx)
    maskV = np.zeros(dim, dtype=bool); maskV[posV] = True
    posO = np.where(~maskV)[0]
    H_VV = H_full[np.ix_(posV, posV)]
    ev, Cv = np.linalg.eigh(H_VV)
    E_var = float(ev[0]); c_var = Cv[:, 0]
    H_OV = H_full[np.ix_(posO, posV)]; H_OO = H_full[np.ix_(posO, posO)]
    rO = H_OV @ c_var
    O_idx = all_idx[posO]
    Hn = H_OO - E_var * np.eye(len(posO))
    x_exact = np.linalg.solve(Hn, rO)
    E_exact2 = -float(rO @ x_exact)
    diag_O = np.diag(H_OO)
    dn = E_var - diag_O
    dn = np.where(np.abs(dn) < 1e-10, 1e-10, dn)
    E_diag = float(np.sum(rO ** 2 / dn))
    print(f'|V|={len(idx)}, n_out={len(posO)}, ||r||={np.linalg.norm(rO):.5f}, '
          f'min eig H_N={np.linalg.eigvalsh(Hn)[0]:+.3e}')

    # block-Newton sweep + same-set diagonal
    for nc in [8, 32, 128]:
        E_blk, info = compute_block_pt2_correction(
            O_idx, rO, E_var, apply_fn, hd_fn, db, az_of, bz_of, rt_a, rt_b,
            in_space_idx=idx, n_centers=nc, h_build_chunk=200, verbose=False)
        asg = info["assigned_idx"]; pa = np.searchsorted(O_idx, asg)
        Eds = float(np.sum(rO[pa] ** 2 / dn[pa]))
        if info["residual_coverage"] > 0.999:
            # block-Jacobi on the fully-covered subspace (10.91 §9.1)
            ap = np.searchsorted(O_idx, asg)
            Hs = Hn[np.ix_(ap, ap)]; rs = rO[ap]
            po = {int(g): j for j, g in enumerate(asg)}
            D = np.zeros_like(Hs)
            for mem in info["block_members"]:
                bp = np.array([po[int(g)] for g in mem])
                D[np.ix_(bp, bp)] = Hs[np.ix_(bp, bp)]
            Om = Hs - D
            rho = max(abs(np.linalg.eigvals(np.linalg.solve(D, Om))))
            x = np.linalg.solve(D, rs); gaps = [-rs @ x - (-rs @ np.linalg.solve(Hs, rs))]
            for _ in range(3):
                x = np.linalg.solve(D, rs - Om @ x)
                gaps.append(-rs @ x - (-rs @ np.linalg.solve(Hs, rs)))
            jac = f'rho={rho:.3f}, jac gaps={[f"{g:+.1e}" for g in gaps]}'
        else:
            jac = ''
        print(f'  nc={nc:3d} cov={info["residual_coverage"]:.4f}: '
              f'block={E_blk:+.8f} diag(same)={Eds:+.8f} {jac}')

    def mha(ec):  # error vs FCI in mHa
        return (E_var + ec - e_fci) * 1000
    print(f'  -> err vs FCI (mHa): variational={mha(0):+.4f}  '
          f'diagPT2={mha(E_diag):+.4f}  exact2nd={mha(E_exact2):+.4f}')
    return len(idx), mha(0.0), mha(E_diag), mha(E_exact2)


def main():
    print('=' * 74); print('LiH/6-31G block-Newton validation at two V sizes'); print('=' * 74)
    n_orb, n_occ, h_sp, t_sp, nuc, e_fci = build_system()
    n_a = n_b = n_occ; nelec = n_a + n_b; ns = 2 * n_orb
    da = comb(n_orb, n_a); db = comb(n_orb, n_b); dim = da * db
    print(f'n_orb={n_orb}, dim={dim}, FCI={e_fci:.10f}')
    rt_a, rt_b, az_of, bz_of = _brt(n_orb, n_a, n_b, da, db)
    hf_az = int(np.sum(1 << np.arange(n_a))); hf_bz = int(np.sum(1 << np.arange(n_b)))
    seed_idx = int(rt_a.rank(np.array([hf_az]))[0] * db + rt_b.rank(np.array([hf_bz]))[0])
    gpu_ctx, gpu_queue, gpu_apply, gpu_argsort = None, None, None, None  # CPU fallback (no pyopencl)
    # CPU path MUST have the spin-orbital two-body tensor. GPU path skips
    # t_s; a GPU-style call with t=None is a crash on the sparse-action side.
    o_s, t_s = spin_orbital_integrals(h_sp, t_sp)
    apply_fn, _, _, _, _, _ = exterior.sparse_action_sz_vec(
        ns, nelec, 0, o_s, t_s, nuc, 1e-10, gpu_apply=None)  # CPU fallback

    def hd_fn(idxs):
        return exterior.sector_diagonal_at(
            ns, nelec, 0, o_s, t_s, nuc, idxs=np.asarray(idxs),
            lookup_tables=(az_of, bz_of, db))

    all_idx = np.arange(dim, dtype=np.int64)
    t0 = time.time()
    H_full = build_H_matrix(all_idx, apply_fn, hd_fn, db, az_of, bz_of,
                            rt_a, rt_b, H_cols=None, chunk_size=300)
    print(f'dense H_full built {H_full.shape} in {time.time()-t0:.1f}s, '
          f'symm err {np.abs(H_full-H_full.T).max():.1e}')

    summary = []
    for cfg in [(1, 120, 'SMALL V (1 wavepacket, large residual)'),
                (3, 200, 'LARGER V (3 wavepackets, small residual)')]:
        summary.append(analyse(cfg, apply_fn, hd_fn, db, az_of, bz_of,
                               rt_a, rt_b, H_full, all_idx, e_fci,
                               gpu_argsort, seed_idx))

    print('\n' + '=' * 74)
    print(f'{"|V|":>5s} {"err var/mHa":>12s} {"err diagPT2":>12s} {"err exact2":>12s}')
    for nv, ev, ed, ee in summary:
        print(f'{nv:5d} {ev:>+12.4f} {ed:>+12.4f} {ee:>+12.4f}')
    print('=' * 74)


if __name__ == '__main__':
    main()
