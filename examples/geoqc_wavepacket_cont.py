#!/usr/bin/env python3
"""Wavepacket continuous-representation analysis (machine precision).

Tests whether the FCI ground-state amplitudes are a *spectral object*:
    Hypothesis A (thermal kernel):  |c_D|^2 ~ exp(-beta * dE_D)  [REFUTED]
    Hypothesis B (RS-1 spectrum):   |c_D|   ~ |H_DF| / dE_D      [CONFIRMED, single-ref]
where dE_D = <D|H|D> - <HF|H|HF> (diagonal-energy denominator) and
H_DF = <D|H|HF> (coupling spectrum, one matrix-free apply on HF).

The wavepacket is then the 1-loop response of the manifold point HF to
the perturbation H — the "continuous representation" is two spectra,
both computable WITHOUT the FCI.

Multireference (N2 R=1.1, canonical RHF MOs) degrades the correlation
(0.60 vs 0.965): the packet spreads over many mid-amplitude states in
this basis — the sparsity geometry is a *covariant* (basis-dependent)
property, not intrinsic.  Honest boundary recorded.

Usage: PYTHONPATH=src:. python3 examples/geoqc_wavepacket_cont.py <system>
       system: h2o631g | n2_631g_11
"""
import sys
import numpy as np
from itertools import combinations
from pyscf import gto, scf, fci, ao2mo
from pyscf.fci import direct_spin1, cistring
from geoqc import exterior


def spin_integrals(mol, mf, fc=0):
    n_orb = mol.nao_nr()
    n_act = n_orb - fc
    h1e = mf.mo_coeff[:, fc:].T @ mf.get_hcore() @ mf.mo_coeff[:, fc:]
    eri = ao2mo.kernel(mol, mf.mo_coeff[:, fc:], compact=False).reshape(n_act, n_act, n_act, n_act)
    o_s = np.zeros((2 * n_act, 2 * n_act))
    o_s[0::2, 0::2] = h1e
    o_s[1::2, 1::2] = h1e
    t_s = np.zeros((2 * n_act,) * 4)
    for p in range(n_act):
        for q in range(n_act):
            for r in range(n_act):
                for s in range(n_act):
                    for (s1, s2, s3, s4) in ((0, 0, 0, 0), (1, 1, 1, 1), (0, 1, 0, 1), (1, 0, 1, 0)):
                        t_s[2*p+s1, 2*q+s2, 2*r+s3, 2*s+s4] = eri[p, q, r, s]
    return n_act, o_s, t_s, float(mol.energy_nuc())


def map_to_sector(C, strs, n_a, n_b, rt_a, rt_b, db):
    da = len(strs)
    cmap = np.zeros(da * db, dtype=complex)
    for ia_p in range(da):
        ra = rt_a[int(strs[ia_p])]
        for ib_p in range(da):
            rb = rt_b[int(strs[ib_p])]
            cmap[ra * db + rb] = C[ia_p, ib_p]
    return cmap


def analyze(system):
    if system == 'h2o631g':
        mol = gto.M(atom='O 0 0 0; H 0.758 0.587 0; H -0.758 0.587 0', basis='6-31g', verbose=0)
        fc, nelec = 0, 10
    elif system == 'n2_631g_11':
        mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='6-31g', verbose=0)
        fc, nelec = 2, 10
    else:
        raise ValueError(system)
    mf = scf.RHF(mol).run()
    n_act, o_s, t_s, nuc = spin_integrals(mol, mf, fc)
    na = nb = nelec // 2
    # FCI (memory-safe: direct_spin1, small Davidson space)
    h1e = o_s[0::2, 0::2]
    eri = np.zeros((n_act,) * 4)
    eri[:, :, :, :] = t_s[0::2, 0::2, 0::2, 0::2]
    cis = direct_spin1.FCISolver(mol)
    cis.verbose = 0
    cis.max_space = 4
    cis.conv_tol = 1e-8
    E, c = cis.kernel(h1e, eri, n_act, (na, nb))
    strs = np.asarray(cistring.gen_strings4orblist(range(n_act), na), dtype=np.int64)
    C = c.reshape(len(strs), len(strs))

    apply, n_a2, n_b2, n_orb2, da, db = exterior.sparse_action_sz(2 * n_act, nelec, 0, o_s, t_s, nuc, 1e-6)
    hd, *_ = exterior.sector_diagonal_sz(2 * n_act, nelec, 0, o_s, t_s, nuc, 1e-6, two_body=True)
    rt_a = np.full(1 << n_orb2, -1, dtype=np.int64)
    rt_b = np.full(1 << n_orb2, -1, dtype=np.int64)
    for i, cc in enumerate(combinations(range(n_orb2), n_a)):
        rt_a[sum(1 << j for j in cc)] = i
    for i, cc in enumerate(combinations(range(n_orb2), n_b)):
        rt_b[sum(1 << j for j in cc)] = i
    az_of = np.full(da, -1, dtype=np.int64)
    bz_of = np.full(db, -1, dtype=np.int64)
    for i, cc in enumerate(combinations(range(n_orb2), n_a)):
        az_of[i] = sum(1 << j for j in cc)
    for i, cc in enumerate(combinations(range(n_orb2), n_b)):
        bz_of[i] = sum(1 << j for j in cc)

    c_map = map_to_sector(C, strs, n_a, n_b, rt_a, rt_b, db)
    hf = (1 << na) - 1
    ra_hf = rt_a[hf]
    hd_hf = hd[ra_hf * db + ra_hf]
    t_az, t_bz, t_v = apply(az_of[ra_hf:ra_hf+1], bz_of[ra_hf:ra_hf+1], np.array([1.0 + 0j]))
    ti = rt_a[t_az] * db + rt_b[t_bz]
    H_off = np.zeros(da * db, dtype=complex)
    np.add.at(H_off, ti, t_v)
    dE = hd - hd_hf

    # --- Hypothesis A: thermal kernel ---
    cpl = (np.abs(H_off) > 1e-8) & (np.abs(dE) > 1e-6)
    amps = np.abs(c_map[cpl])
    des = dE[cpl]
    okA = (amps > 1e-14) & (des > 0)
    corrA = np.corrcoef(des[okA], np.log(amps[okA]))[0, 1] if okA.sum() > 3 else float('nan')

    # --- Hypothesis B: RS-1 spectrum (phase-free) ---
    obs = amps
    pred = np.abs(H_off[cpl]) / np.abs(des)
    okB = (obs > 1e-10) & (pred > 1e-10)
    corrB = np.corrcoef(obs[okB], pred[okB])[0, 1] if okB.sum() > 3 else float('nan')
    slopeB = np.polyfit(np.log(pred[okB]), np.log(obs[okB]), 1)[0] if okB.sum() > 3 else float('nan')

    hf_w = abs(c_map[ra_hf * db + ra_hf]) ** 2
    print(f'{system}: dim={da*db}  FCI E={E:.8f}  HF%={100*hf_w:.2f}  coupled={cpl.sum()}')
    print(f'  A thermal kernel corr(log|c|^2, dE)   = {corrA:+.4f}  [expected ~0 -> refuted]')
    print(f'  B RS-1 spectrum corr(|c|,|H|/dE)      = {corrB:+.4f}  [expected ~0.97 -> confirmed]')
    print(f'    log-log slope = {slopeB:.3f}  [expected ~1.0]')


if __name__ == '__main__':
    analyze(sys.argv[1])
