#!/usr/bin/env python3
"""N₂ 6-31G frozen-core in natural-orbital basis — basis-covariance test.

Q: is the wavepacket sparsity geometry basis-covariant?  In canonical RHF
MOs, N₂ R=1.1 shows HF=21%, RS-1 perturbation-spectrum corr 0.60 (vs 0.965
for H₂O).  In the FCI natural-orbital basis (NOON-sorted), the single-
reference weight should rise and the RS-1 form should be restored — if the
sparsity geometry is a covariant (basis-dependent) property.

Pipeline: FCI (direct_spin1) -> 1-RDM -> natural orbitals -> transform
h1e/eri -> FCI again in NO basis -> wavepacket + RS-1 analysis.
Memory-safe: max_space=4, one FCI at a time.
"""
import numpy as np
from itertools import combinations
from pyscf import gto, scf, fci, ao2mo
from pyscf.fci import direct_spin1, cistring
from geoqc import exterior


def run(system='n2_631g_11', R=1.1):
    mol = gto.M(atom=f'N 0 0 0; N 0 0 {R}', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run()
    n_orb = mol.nao_nr()
    fc = 2
    n_act = n_orb - fc
    nelec = 10
    na = nb = 5
    hcore = mf.get_hcore()
    h1e_can = mf.mo_coeff[:, fc:].T @ hcore @ mf.mo_coeff[:, fc:]
    eri_can = ao2mo.kernel(mol, mf.mo_coeff[:, fc:], compact=False).reshape(n_act, n_act, n_act, n_act)

    def fci_energy(h1e, eri):
        cis = direct_spin1.FCISolver(mol)
        cis.verbose = 0
        cis.max_space = 4
        cis.conv_tol = 1e-8
        return cis, cis.kernel(h1e, eri, n_act, (na, nb))

    # --- FCI #1 in canonical RHF MOs ---
    cis1, (E1, c1) = fci_energy(h1e_can, eri_can)
    dm1a, dm1b = direct_spin1.make_rdm1s(c1, n_act, (na, nb))
    dm1 = np.asarray(dm1a) + np.asarray(dm1b)  # total 1-RDM
    ev, U = np.linalg.eigh(dm1)
    order = np.argsort(ev)[::-1]   # NOON descending
    noon = ev[order]
    U = U[:, order]                # natural orbitals (columns)
    print(f'R={R}: FCI#1 E={E1:.8f}  NOON top-8 = {np.round(noon[:8], 4)}', flush=True)
    print(f'  NOON mid (multiref measure) = {np.round(noon[na-1:na+3], 4)}', flush=True)

    # --- transform to natural-orbital basis ---
    # h1e_NO = U^T h1e_can U ; eri_NO[p,q,r,s] = sum U[...] 4-fold
    h1e_NO = U.T @ h1e_can @ U
    eri_NO = np.einsum('pqrs,pi,qj,rk,sl->ijkl', eri_can, U, U, U, U)

    # --- FCI #2 in natural-orbital basis ---
    cis2, (E2, c2) = fci_energy(h1e_NO, eri_NO)
    print(f'  FCI#2 (NO basis) E={E2:.8f}  (must equal E1 within tol: {abs(E2-E1):.1e})', flush=True)
    np.savez('_no_cache.npz', c1=c1, c2=c2, noon=noon, U=U)
    print('  cached _no_cache.npz', flush=True)

    # --- wavepacket + RS-1 analysis in NO basis ---
    from geoqc.integrals import spin_orbital_integrals
    o_s, t_s = spin_orbital_integrals(h1e_NO, eri_NO)
    nuc = mol.energy_nuc()
    strs = np.asarray(cistring.gen_strings4orblist(range(n_act), na), dtype=np.int64)
    C = c2.reshape(len(strs), len(strs))
    apply, n_a2, n_b2, n_orb2, da, db = exterior.sparse_action_sz(2 * n_act, nelec, 0, o_s, t_s, nuc, 1e-6)
    hd, *_ = exterior.sector_diagonal_sz(2 * n_act, nelec, 0, o_s, t_s, nuc, 1e-6, two_body=True)
    rt_a = np.full(1 << n_orb2, -1, dtype=np.int64)
    rt_b = np.full(1 << n_orb2, -1, dtype=np.int64)
    for i, cc in enumerate(combinations(range(n_orb2), n_a2)):
        rt_a[sum(1 << j for j in cc)] = i
    for i, cc in enumerate(combinations(range(n_orb2), n_b2)):
        rt_b[sum(1 << j for j in cc)] = i
    az_of = np.full(da, -1, dtype=np.int64)
    bz_of = np.full(db, -1, dtype=np.int64)
    for i, cc in enumerate(combinations(range(n_orb2), n_a2)):
        az_of[i] = sum(1 << j for j in cc)
    for i, cc in enumerate(combinations(range(n_orb2), n_b2)):
        bz_of[i] = sum(1 << j for j in cc)
    c_map = np.zeros(da * db, dtype=complex)
    for ia_p in range(len(strs)):
        ra = rt_a[int(strs[ia_p])]
        for ib_p in range(len(strs)):
            rb = rt_b[int(strs[ib_p])]
            c_map[ra * db + rb] = C[ia_p, ib_p]
    hf = (1 << na) - 1
    ra_hf = rt_a[hf]
    hd_hf = hd[ra_hf * db + ra_hf]
    t_az, t_bz, t_v = apply(az_of[ra_hf:ra_hf+1], bz_of[ra_hf:ra_hf+1], np.array([1.0 + 0j]))
    ti = rt_a[t_az] * db + rt_b[t_bz]
    H_off = np.zeros(da * db, dtype=complex)
    np.add.at(H_off, ti, t_v)
    dE = hd - hd_hf
    cpl = (np.abs(H_off) > 1e-8) & (np.abs(dE) > 1e-6)
    obs = np.abs(c_map[cpl])
    pred = np.abs(H_off[cpl]) / np.abs(dE[cpl])
    corrB = np.corrcoef(obs, pred)[0, 1]
    slopeB = np.polyfit(np.log(pred[obs > 1e-10]), np.log(obs[obs > 1e-10]), 1)[0]
    hf_w = abs(c_map[ra_hf * db + ra_hf]) ** 2
    # concentration
    Wv = np.abs(c_map) ** 2
    flat = np.sort(Wv)[::-1]
    cum = np.cumsum(flat)
    k999 = int(np.sum(cum <= 0.999 * Wv.sum())) + 1
    print(f'  NO-basis: HF% = {100*hf_w:.2f}   RS-1 corr = {corrB:+.4f}  slope = {slopeB:.3f}  eff-dim(99.9%) = {k999} ({100*k999/len(Wv):.3f}% of dim)')
    # excitation-order composition
    pc = np.array([bin(int(x)).count('1') for x in strs ^ hf])
    K = (pc[:, None] + pc[None, :]) // 2
    Kv = K.ravel()
    print('  k-composition:', ' '.join(f'k{k}:{100*Wv[Kv==k].sum()/Wv.sum():.2f}%' for k in range(0, 6) if (Kv == k).sum() > 0))
    return dict(E1=E1, E2=E2, noon=noon, hf_w=hf_w, corrB=corrB, slopeB=slopeB, k999=k999)


if __name__ == '__main__':
    run()
