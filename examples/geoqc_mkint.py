#!/usr/bin/env python3
"""Integral file builder for geoqc_kscan.py — the npz (n, o_s, t_s, nuc).

Uses the LIBRARY converter geoqc.integrals.spin_orbital_integrals
(openfermion two-body layout, element-wise verified) — do NOT hand-roll
the spin-mode loop here: a previous hand-rolled abab-mode version
produced a wrong spin layout that silently broke every energy (LiH
6-31G E0 came out −19.25 instead of −7.9984).

用法: PYTHONPATH=src:. python3 examples/geoqc_mkint.py <name> <geometry> <basis> <fc> <nelec> <out.npz>
  例:  PYTHONPATH=src:. python3 examples/geoqc_mkint.py lih "Li 0 0 0; H 0 0 1.6" 6-31g 0 4 /tmp/_lih.npz
"""
import sys
import numpy as np
from pyscf import gto, scf, ao2mo
from geoqc.integrals import spin_orbital_integrals


def main():
    name, geometry, basis, fc, nelec, out = (
        sys.argv[1], sys.argv[2], sys.argv[3],
        int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
    mol = gto.M(atom=geometry, basis=basis, verbose=0)
    mf = scf.RHF(mol).run()
    n_orb = mol.nao_nr()
    n_act = n_orb - fc
    C = mf.mo_coeff[:, fc:]
    h1e = C.T @ mf.get_hcore() @ C
    eri = ao2mo.kernel(mol, C, compact=False).reshape(n_act, n_act, n_act, n_act)
    o_s, t_s = spin_orbital_integrals(h1e, eri)
    np.savez(out, n=n_act, o_s=o_s, t_s=t_s, nuc=float(mol.energy_nuc()))
    print(f'{name}: n_act={n_act} nelec={nelec} E_RHF={mf.e_tot:.8f} '
          f'-> {out}  (o_s {o_s.shape}, t_s {t_s.shape})')


if __name__ == '__main__':
    main()
