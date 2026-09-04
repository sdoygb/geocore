#!/usr/bin/env python3
"""Candidate ③ — higher-order correction to the RS-1 perturbation
spectrum.  Does the RS-1 residual (measured |c_D| vs |H_DF|/ΔE_D) get
explained by the RS-2 term?

  c_D^(1) = -H_DF / ΔE_D                      (RS-1, the 10.87 §6.04 form)
  c_D^(2) = Σ_{E≠D,F} H_DE H_EF / (ΔE_E ΔE_D)   (RS-2, H0 = diagonal
            convention: no -H_DF(H_DD-H_FF)/ΔE² term, V_DD=0)

where F = HF reference.  H_DE needs one apply from each D (the same
matrix-free action; ~k apply calls).  Comparison:
  corr(|c|, |RS1|)          vs
  corr(|c|, |RS1 + RS2|)
A large gain means the residual is higher-order perturbative; no gain
means it is non-perturbative (true multireference / basis effect).

Usage: PYTHONPATH=src:. python3 examples/geoqc_rs2.py <system>
  system = h2o631g | n2_631g_11 | n2_631g_30_no
"""
import sys
import numpy as np
from itertools import combinations
from pyscf import gto, scf, ao2mo, fci
from pyscf.fci import direct_spin1, cistring
from geoqc import exterior
from geoqc.integrals import spin_orbital_integrals


def build_system(name):
    if name == 'h2o631g':
        mol = gto.M(atom='O 0 0 0; H 0.758 0.587 0; H -0.758 0.587 0',
                    basis='6-31g', verbose=0)
        fc, nelec = 0, 10
        R = None
    elif name == 'n2_631g_11':
        mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='6-31g', verbose=0)
        fc, nelec = 2, 10
        R = 1.1
    elif name == 'n2_631g_14':
        mol = gto.M(atom='N 0 0 0; N 0 0 1.4', basis='6-31g', verbose=0)
        fc, nelec = 2, 10
        R = 1.4
    elif name == 'n2_631g_30_no':
        mol = gto.M(atom='N 0 0 0; N 0 0 3.0', basis='6-31g', verbose=0)
        fc, nelec = 2, 10
        R = 3.0
    else:
        raise ValueError(name)
    return mol, fc, nelec, R


def fci_solve(mol, h1e, eri, n_act, na, nb):
    cis = direct_spin1.FCISolver(mol)
    cis.verbose = 0
    cis.max_space = 4
    cis.conv_tol = 1e-8
    E, c = cis.kernel(h1e, eri, n_act, (na, nb))
    return E, c


def analyze(name):
    mol, fc, nelec, R = build_system(name)
    mf = scf.RHF(mol).run()
    n_orb = mol.nao_nr()
    n_act = n_orb - fc
    na = nb = nelec // 2
    C = mf.mo_coeff[:, fc:]
    h1e = C.T @ mf.get_hcore() @ C
    eri = ao2mo.kernel(mol, C, compact=False).reshape(n_act, n_act, n_act, n_act)
    nuc = float(mol.energy_nuc())

    # NO basis for the R=3.0 system (true multireference check)
    if name == 'n2_631g_30_no':
        E1, c1 = fci_solve(mol, h1e, eri, n_act, na, nb)
        dm1a, dm1b = direct_spin1.make_rdm1s(c1, n_act, (na, nb))
        dm1 = np.asarray(dm1a) + np.asarray(dm1b)
        ev, U = np.linalg.eigh(dm1)
        order = np.argsort(ev)[::-1]
        U = U[:, order]
        h1e = U.T @ h1e @ U
        eri = np.einsum('pqrs,pi,qj,rk,sl->ijkl', eri, U, U, U, U)

    E_fci, c = fci_solve(mol, h1e, eri, n_act, na, nb)
    o_s, t_s = spin_orbital_integrals(h1e, eri)
    apply, n_a2, n_b2, n_orb2, da, db = exterior.sparse_action_sz_vec(
        2 * n_act, nelec, 0, o_s, t_s, nuc, 1e-6)
    hd, *_ = exterior.sector_diagonal_sz(2 * n_act, nelec, 0, o_s, t_s,
                                         nuc, 1e-6, two_body=True)

    # rank tables
    rt_a = np.full(1 << n_orb2, -1, dtype=np.int64)
    rt_b = np.full(1 << n_orb2, -1, dtype=np.int64)
    az_of = np.full(da, -1, dtype=np.int64)
    bz_of = np.full(db, -1, dtype=np.int64)
    for i, cc in enumerate(combinations(range(n_orb2), na)):
        rt_a[sum(1 << j for j in cc)] = i
        az_of[i] = sum(1 << j for j in cc)
    for i, cc in enumerate(combinations(range(n_orb2), nb)):
        rt_b[sum(1 << j for j in cc)] = i
        bz_of[i] = sum(1 << j for j in cc)

    # FCI vector to sector map
    strs = np.asarray(cistring.gen_strings4orblist(range(n_act), na),
                      dtype=np.int64)
    C = c.reshape(len(strs), len(strs))
    c_map = np.zeros(da * db, dtype=complex)
    for ia_p in range(len(strs)):
        ra = rt_a[int(strs[ia_p])]
        for ib_p in range(len(strs)):
            rb = rt_b[int(strs[ib_p])]
            c_map[ra * db + rb] = C[ia_p, ib_p]

    hf = (1 << na) - 1
    ra_hf = rt_a[hf]
    i_hf = ra_hf * db + ra_hf
    hd_hf = hd[i_hf]

    # RS-1: one apply from HF
    t_az, t_bz, t_v = apply(np.array([az_of[ra_hf]]), np.array([bz_of[ra_hf]]),
                            np.array([1.0 + 0j]))
    ti = rt_a[t_az] * db + rt_b[t_bz]
    H_off = np.zeros(da * db, dtype=complex)
    np.add.at(H_off, ti, t_v)
    dE = hd - hd_hf
    cpl = (np.abs(H_off) > 1e-8) & (np.abs(dE) > 1e-6)
    idxD = np.nonzero(cpl)[0]
    print(f'{name}: dim={da*db}  FCI E={E_fci:.8f}  coupled={len(idxD)}',
          flush=True)

    # RS-1 prediction
    H_DF = H_off[idxD]
    dE_D = dE[idxD]
    pred1 = np.abs(H_DF) / np.abs(dE_D)
    obs = np.abs(c_map[idxD])

    # RS-2: for each D, one apply to collect H_DE
    # c_D^(2) = sum_{E != D, F} H_DE H_EF / (dE_E dE_D)
    #   (H0 = diagonal convention: no -H_DF(H_DD-H_FF)/dE_D^2 term)
    # H_EF = H_off[E]  (coupling of E to HF, from the RS-1 apply)
    # H_DE = apply from D
    rs2 = np.zeros(len(idxD), dtype=complex)
    H_EF_map = {}
    for j, ti_v in enumerate(ti):
        if abs(t_v[j]) > 1e-10:
            H_EF_map[int(ti_v)] = t_v[j]
    for j, D in enumerate(idxD):
        azD = az_of[D // db]
        bzD = bz_of[D % db]
        ta, tb, tv = apply(np.array([azD]), np.array([bzD]),
                           np.array([1.0 + 0j]))
        te = rt_a[ta] * db + rt_b[tb]
        s = 0.0 + 0.0j
        for k in range(len(te)):
            E = int(te[k])
            if E == D or E == i_hf:
                continue
            H_EF = H_EF_map.get(E, 0.0)
            if abs(H_EF) < 1e-12 or abs(dE[E]) < 1e-8:
                continue
            s += tv[k] * H_EF / (dE[E] * dE_D[j])
        rs2[j] = s

    c1_full = -H_DF / dE_D
    pred2 = np.abs(c1_full + rs2)
    ok = (obs > 1e-10) & (pred1 > 1e-10) & (pred2 > 1e-10)
    c1 = np.corrcoef(obs[ok], pred1[ok])[0, 1]
    c2 = np.corrcoef(obs[ok], pred2[ok])[0, 1]
    # rel-err on the wavepacket support only (|c| >= 1e-3) — rel-err on
    # near-zero amplitudes is meaningless (H_DF ~ 0, |c| ~ 1e-7)
    sup = ok & (obs >= 1e-3)
    if sup.sum() >= 10:
        r1 = np.abs(obs[sup] - pred1[sup]) / obs[sup]
        r2 = np.abs(obs[sup] - pred2[sup]) / obs[sup]
        print(f'  RS-1 corr = {c1:+.4f}   RS-1+RS-2 corr = {c2:+.4f}', flush=True)
        print(f'  [support |c|>=1e-3, n={sup.sum()}] mean rel-err: RS-1 = '
              f'{r1.mean():.4f}   RS-1+RS-2 = {r2.mean():.4f}   '
              f'({100*(r1.mean()-r2.mean())/r1.mean():+.1f}%)', flush=True)
        print(f'  [support] median rel-err: RS-1 = {np.median(r1):.4f}   '
              f'RS-1+RS-2 = {np.median(r2):.4f}', flush=True)
    else:
        print(f'  RS-1 corr = {c1:+.4f}   RS-1+RS-2 corr = {c2:+.4f}   '
              f'(support n={sup.sum()} < 10, rel-err skipped)', flush=True)
    # top-10 |c| coverage (where the wavepacket weight lives)
    top = np.argsort(obs)[::-1][:10]
    print('  top-10 |c|:  |c|  RS-1 (err)  RS-1+RS-2 (err)')
    for jj in top:
        e1 = abs(obs[jj]-pred1[jj])/obs[jj]
        e2 = abs(obs[jj]-pred2[jj])/obs[jj]
        print(f'    {obs[jj]:.5f}  {pred1[jj]:.5f} ({e1:.2f})  '
              f'{pred2[jj]:.5f} ({e2:.2f})', flush=True)


if __name__ == '__main__':
    analyze(sys.argv[1])
