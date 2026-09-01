#!/usr/bin/env python3
"""N₂ 6-31G frozen-core in the FCI natural-orbital basis — build the
kscan npz (n, o_s, t_s, nuc) for the "NO basis + top-k" engineering
payoff (candidate ① of the roadmap: verify the ~300x k reduction).

Pipeline (memory-safe, one FCI at a time, max_space=4):
  RHF -> FCI#1 (canonical RHF MOs) -> 1-RDM -> natural orbitals (NOON
  sorted) -> transform h1e/eri -> spin_orbital_integrals (the ONLY
  correct spin layout — see geoqc/integrals.py docstring) -> npz.

Machine checks printed:
  - FCI#2 (NO basis) must equal FCI#1 (basis covariance, ~1e-9);
  - NOON top / mid (multiref measure).
Usage: PYTHONPATH=src:. python3 examples/geoqc_no_mkint.py <R> <out.npz>
  e.g. PYTHONPATH=src:. python3 examples/geoqc_no_mkint.py 1.1 /tmp/_n2_no_1.1.npz
"""
import sys
import numpy as np
from pyscf import gto, scf, ao2mo
from pyscf.fci import direct_spin1
from geoqc.integrals import spin_orbital_integrals


def build(R, out):
    mol = gto.M(atom=f'N 0 0 0; N 0 0 {R}', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run()
    n_orb = mol.nao_nr()
    fc = 2
    n_act = n_orb - fc
    nelec = 10
    na = nb = 5
    C = mf.mo_coeff[:, fc:]
    h1e_can = C.T @ mf.get_hcore() @ C
    eri_can = ao2mo.kernel(mol, C, compact=False).reshape(n_act, n_act, n_act, n_act)
    nuc = float(mol.energy_nuc())

    def fci_energy(h1e, eri):
        cis = direct_spin1.FCISolver(mol)
        cis.verbose = 0
        cis.max_space = 4
        cis.conv_tol = 1e-8
        return cis, cis.kernel(h1e, eri, n_act, (na, nb))

    # --- FCI #1 in canonical RHF MOs ---
    cis1, (E1, c1) = fci_energy(h1e_can, eri_can)
    dm1a, dm1b = direct_spin1.make_rdm1s(c1, n_act, (na, nb))
    dm1 = np.asarray(dm1a) + np.asarray(dm1b)
    ev, U = np.linalg.eigh(dm1)
    order = np.argsort(ev)[::-1]
    noon = ev[order]
    U = U[:, order]
    print(f'R={R}: FCI#1 E={E1:.8f}  NOON top-8 = {np.round(noon[:8], 4)}', flush=True)
    print(f'  NOON mid = {np.round(noon[na-1:na+3], 4)}', flush=True)

    # --- natural-orbital basis ---
    h1e_NO = U.T @ h1e_can @ U
    eri_NO = np.einsum('pqrs,pi,qj,rk,sl->ijkl', eri_can, U, U, U, U)

    # --- FCI #2 in NO basis (basis covariance check) ---
    cis2, (E2, c2) = fci_energy(h1e_NO, eri_NO)
    print(f'  FCI#2 (NO) E={E2:.8f}  |E2-E1|={abs(E2-E1):.2e}  (must be ~1e-9)', flush=True)

    # --- spin-orbital integrals (library converter, the only entry) ---
    o_s, t_s = spin_orbital_integrals(h1e_NO, eri_NO)
    np.savez(out, n=n_act, o_s=o_s, t_s=t_s, nuc=nuc)
    print(f'  saved {out}  (n_act={n_act}, FCI-total+nuc = {E2 + nuc:.8f})', flush=True)
    np.savez('_no_cache.npz', c1=c1, c2=c2, noon=noon, U=U)
    print('  cached _no_cache.npz', flush=True)
    return dict(E1=E1, E2=E2, noon=noon)


if __name__ == '__main__':
    R = float(sys.argv[1])
    out = sys.argv[2]
    build(R, out)
