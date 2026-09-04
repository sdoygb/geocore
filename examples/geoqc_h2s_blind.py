#!/usr/bin/env python3
"""BLIND TEST: WCI on H2S (never used in development) vs pyscf FCI.

普适性盲测第一弹（validation-matrix.md D.2）：H2S/STO-3G。
纪律：结构与参数与 examples/geoqc_benchmark.py 完全一致（max_wp=30, tol=1e-8,
FCI conv_tol=1e-10），H2S 此前未参与任何开发/调参——首次盲测。
几何：实验值 r(S-H)=1.336 A, 键角 92.1 deg（对称放置，键角平分线沿 z）。
"""
import sys, time, math
import numpy as np
from pyscf import gto, scf, ao2mo, fci

NAME = 'H2S/STO-3G'
r = 1.336          # Angstrom
half = math.radians(92.1) / 2.0
x = r * math.sin(half)
z = r * math.cos(half)
ATOM = f'S 0 0 0; H {x:.6f} 0 {z:.6f}; H {-x:.6f} 0 {z:.6f}'
BASIS = 'sto-3g'
NELEC = (9, 9)     # 18 electrons, full space (minimal basis, like benchmark)

def run_pyscf_fci(mol, mo_coeff, h1e, eri, nelec, n_orb):
    cis = fci.direct_spin1.FCISolver(mol)
    cis.verbose = 0
    cis.conv_tol = 1e-10
    cis.nroots = 1
    t0 = time.time()
    e, c = cis.kernel(h1e, eri, n_orb, nelec)
    # pyscf FCI kernel excludes nuclear repulsion; WCI exterior carries it.
    return float(e) + mol.energy_nuc(), time.time() - t0

def run_wci(mol, mo_coeff, h1e, eri, nelec, n_orb, max_wp=30, tol=1e-8):
    from geoqc import exterior, integrals
    from geoqc.wci import build_rank_tables
    o_s, t_s = integrals.spin_orbital_integrals(h1e, eri)
    nuc = mol.energy_nuc()
    n_spin = 2 * n_orb
    n_a, n_b = nelec
    apply_fn, _, _, _, da, db = exterior.sparse_action_sz(
        n_spin, nelec[0]+nelec[1], 0, o_s, t_s, nuc, 1e-4)
    dim = da * db
    rt_a, rt_b, az_of, bz_of = build_rank_tables(n_orb, n_a, n_b, da, db)
    DIAG_FULL_LIMIT = 2_000_000
    if dim <= DIAG_FULL_LIMIT:
        hd, *_ = exterior.sector_diagonal_sz(
            n_spin, nelec[0]+nelec[1], 0, o_s, t_s, nuc, 1e-4, two_body=False)
        _hd_arr = hd
        def hd_fn(idxs): return _hd_arr[np.asarray(idxs, dtype=np.int64)]
    else:
        def hd_fn(idxs):
            return exterior.sector_diagonal_at(
                n_spin, nelec[0]+nelec[1], 0, o_s, t_s, nuc, 1e-4,
                idxs=np.asarray(idxs, dtype=np.int64))
    e_a = np.array([o_s[2*k, 2*k].real for k in range(n_orb)])
    e_b = np.array([o_s[2*k+1, 2*k+1].real for k in range(n_orb)])
    hf_a = np.sort(np.argsort(e_a)[:n_a])
    hf_b = np.sort(np.argsort(e_b)[:n_b])
    hf_az = int(np.sum(1 << hf_a))
    hf_bz = int(np.sum(1 << hf_b))
    seed = int(rt_a[hf_az]) * db + int(rt_b[hf_bz])
    t0 = time.time()
    from geoqc import wci as wci_mod
    E, unique_idx, coeffs, wavepackets, history = wci_mod.wci(
        apply_fn, hd_fn, seed, db, az_of, bz_of, rt_a, rt_b,
        max_wavepackets=max_wp, tol=tol, verbose=False)
    return float(E), time.time() - t0, len(wavepackets), len(unique_idx)

def main():
    print('=== BLIND TEST: WCI vs pyscf FCI on H2S (never in development) ===')
    print(f'atom: {ATOM}')
    mol = gto.M(atom=ATOM, basis=BASIS, symmetry=False, verbose=0, spin=0)
    mf = scf.RHF(mol); mf.verbose = 0; mf.kernel()
    mo_coeff = mf.mo_coeff
    n_orb = mo_coeff.shape[1]
    h1e = mo_coeff.T @ (mol.intor('int1e_kin') + mol.intor('int1e_nuc')) @ mo_coeff
    eri = ao2mo.kernel(mol, mo_coeff, compact=False).reshape(n_orb, n_orb, n_orb, n_orb)
    actual_dim = math.comb(n_orb, NELEC[0]) * math.comb(n_orb, NELEC[1])
    print(f'n_orb={n_orb}, nelec={NELEC}, dim={actual_dim}')
    e_fci, t_fci = run_pyscf_fci(mol, mo_coeff, h1e, eri, NELEC, n_orb)
    print(f'FCI: E={e_fci:.10f} Ha  t={t_fci:.2f}s')
    e_wci, t_wci, n_wp, v_dim = run_wci(mol, mo_coeff, h1e, eri, NELEC, n_orb)
    print(f'WCI: E={e_wci:.10f} Ha  t={t_wci:.2f}s  n_wp={n_wp}  V={v_dim}')
    err = abs(e_wci - e_fci)
    print(f'|E_WCI - E_FCI| = {err:.3e} Ha')
    print(f'speedup = {t_fci/t_wci:.2f}x')
    ok = err < 1e-6
    print(f'判定: {"PASS (收敛到 FCI, <1e-6 Ha)" if ok else "FAIL (未收敛到 FCI)"}')

if __name__ == '__main__':
    main()
