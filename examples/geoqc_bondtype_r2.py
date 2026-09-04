#!/usr/bin/env python3
"""Bond-type -> spectral clustering mapping (incremental save version)."""
import sys, time, os
import numpy as np
from pyscf import gto, scf
from geoqc import spectral

OUT = '/tmp/bondtype_r.npz'

def h2o(s):
    r=0.957*s; th=104.5*np.pi/180; x=r*np.sin(th/2); z=r*np.cos(th/2)
    return gto.M(atom=f'O 0 0 0; H 0 {x:.6f} {z:.6f}; H 0 {-x:.6f} {z:.6f}', basis='sto-3g', symmetry=True, verbose=0, spin=0)

def ch4(s):
    r=1.087*s; a=r/np.sqrt(3)
    return gto.M(atom=f'C 0 0 0; H {a} {a} {a}; H {a} {-a} {-a}; H {-a} {a} {-a}; H {-a} {-a} {a}', basis='sto-3g', symmetry=True, verbose=0, spin=0)

def nh3(s):
    r=1.012*s
    return gto.M(atom=f'N 0 0 0; H {r} 0 0; H {-r/2} {r*np.sqrt(3)/2:.6f} 0; H {-r/2} {-r*np.sqrt(3)/2:.6f} 0', basis='sto-3g', symmetry=True, verbose=0, spin=0)

def bh3(s):
    r=1.190*s
    return gto.M(atom=f'B 0 0 0; H {r} 0 0; H {-r/2} {r*np.sqrt(3)/2:.6f} 0; H {-r/2} {-r*np.sqrt(3)/2:.6f} 0', basis='sto-3g', symmetry=True, verbose=0, spin=0)

def n2(s):
    r=1.098*s
    return gto.M(atom=f'N 0 0 0; N 0 0 {r:.6f}', basis='sto-3g', symmetry=True, verbose=0, spin=0)

def o2(s):
    r=1.208*s
    return gto.M(atom=f'O 0 0 0; O 0 0 {r:.6f}', basis='sto-3g', symmetry=True, verbose=0, spin=2)

def c2h2(s):
    rc=1.203*s; rh=1.060*s
    return gto.M(atom=f'C 0 0 0; C 0 0 {rc:.6f}; H 0 0 {-rh:.6f}; H 0 0 {rc+rh:.6f}', basis='sto-3g', symmetry=True, verbose=0, spin=0)

MOLECULES = [
    ('H2O',  h2o,  100, 'O-center 2 bonds'),
    ('CH4',  ch4,  100, 'C-center 4 bonds'),
    ('NH3',  nh3,  100, 'N-center 3 bonds+LP'),
    ('BH3',  bh3,  100, 'B-center 3 bonds e-def'),
    ('N2',   n2,   100, 'homonuclear triple'),
    ('O2',   o2,   100, 'homonuclear double triplet'),
    ('C2H2', c2h2, 30,  'homonuclear triple+C-H (dim=627k)'),
]
SCALES = [1.0, 1.5, 2.0, 2.5, 3.0]

def load():
    if os.path.exists(OUT):
        d = np.load(OUT, allow_pickle=True)
        return {k.replace('_r',''): d[k] for k in d.files if k.endswith('_r')}
    return {}

def save(results):
    data = {}
    for name, rvals in results.items():
        data[f'{name}_scales'] = np.array(SCALES)
        data[f'{name}_r'] = np.array(rvals)
    np.savez(OUT, **data)

def main():
    results = load()
    print(f'Loaded {len(results)} molecules from cache')
    for name, build_fn, nroots, desc in MOLECULES:
        if name in results and not np.any(np.isnan(results[name])):
            print(f'--- {name}: {desc} (cached) ---')
            for s, r in zip(SCALES, results[name]):
                print(f'    scale={s:.1f}  <r>={r:.4f}')
            continue
        print(f'--- {name}: {desc} (nroots={nroots}) ---')
        rvals = []
        for s in SCALES:
            t0 = time.time()
            try:
                mol = build_fn(s)
                mf = scf.UHF(mol) if mol.spin > 0 else scf.RHF(mol)
                mf.verbose = 0; mf.kernel()
                res = spectral.molecule_spectral_clustering(
                    mol, mo_coeff=mf.mo_coeff, nelec=mol.nelec,
                    nroots=nroots, min_states=5, verbose=0)
                r = res['r_mean']
            except Exception as e:
                r = None; print(f'    ERROR scale={s}: {e}')
            dt = time.time() - t0
            rvals.append(r if r is not None else np.nan)
            print(f'    scale={s:.1f}  <r>={r:.4f}' if r is not None else f'    scale={s:.1f}  FAILED', end='')
            print(f'  ({dt:.1f}s)')
        results[name] = rvals
        save(results)
        print(f'    saved -> {OUT}')

    # summary
    print('\n=== Summary ===')
    print(f'{"Mol":<6}', end='')
    for s in SCALES: print(f' {s:>5.1f}x', end='')
    print('  min')
    for name, _, _, desc in MOLECULES:
        if name in results:
            print(f'{name:<6}', end='')
            for r in results[name]:
                print(f' {r:5.3f}' if not np.isnan(r) else f' {"nan":>5}', end='')
            mn = np.nanmin(results[name])
            print(f'  {mn:.3f}')

if __name__ == '__main__':
    main()
