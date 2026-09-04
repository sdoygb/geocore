#!/usr/bin/env python3
"""Systematic bond-type -> spectral clustering (<r>) mapping.

Runs FCI dissociation curves for 7 molecules:
  - heteronuclear hydrides (different central atoms): H2O, CH4, NH3, BH3
  - homonuclear bonds: N2, O2 (triplet), C2H2

Computes the mean gap ratio <r> at each bond length, establishing the
"bond type -> deep clustering" mapping (article 10.89 §7.5, candidate ①).

Usage:
  PYTHONPATH=. python3 examples/geoqc_bondtype_r.py [--nroots 100] [--out /tmp/bondtype_r.npz]
"""
import sys, time
import numpy as np
from pyscf import gto, scf
from geoqc import spectral

# ---------------------------------------------------------------------------
# Molecule definitions: (name, build_fn(bond_scale), nelec, spin)
# build_fn returns a pyscf Mole at the given bond-length scale factor.
# ---------------------------------------------------------------------------

def h2o(scale):
    r = 0.957 * scale
    theta = 104.5 * np.pi / 180
    x = r * np.sin(theta/2)
    z = r * np.cos(theta/2)
    return gto.M(atom=f'O 0 0 0; H 0 {x:.6f} {z:.6f}; H 0 {-x:.6f} {z:.6f}',
                 basis='sto-3g', symmetry=True, verbose=0, spin=0)

def ch4(scale):
    r = 1.087 * scale
    # tetrahedral geometry
    a = r / np.sqrt(3)
    return gto.M(atom=f'C 0 0 0; H {a} {a} {a}; H {a} {-a} {-a}; H {-a} {a} {-a}; H {-a} {-a} {a}',
                 basis='sto-3g', symmetry=True, verbose=0, spin=0)

def nh3(scale):
    r = 1.012 * scale
    # pyramidal (C3v), approximate with pyscf symmetry
    return gto.M(atom=f'N 0 0 0; H {r} 0 0; H {-r/2} {r*np.sqrt(3)/2:.6f} 0; H {-r/2} {-r*np.sqrt(3)/2:.6f} 0',
                 basis='sto-3g', symmetry=True, verbose=0, spin=0)

def bh3(scale):
    r = 1.190 * scale
    # planar D3h
    return gto.M(atom=f'B 0 0 0; H {r} 0 0; H {-r/2} {r*np.sqrt(3)/2:.6f} 0; H {-r/2} {-r*np.sqrt(3)/2:.6f} 0',
                 basis='sto-3g', symmetry=True, verbose=0, spin=0)

def n2(scale):
    r = 1.098 * scale
    return gto.M(atom=f'N 0 0 0; N 0 0 {r:.6f}',
                 basis='sto-3g', symmetry=True, verbose=0, spin=0)

def o2(scale):
    r = 1.208 * scale
    # O2 triplet ground state
    return gto.M(atom=f'O 0 0 0; O 0 0 {r:.6f}',
                 basis='sto-3g', symmetry=True, verbose=0, spin=2)

def c2h2(scale):
    r_cc = 1.203 * scale
    r_ch = 1.060 * scale
    return gto.M(atom=f'C 0 0 0; C 0 0 {r_cc:.6f}; H 0 0 {-r_ch:.6f}; H 0 0 {r_cc + r_ch:.6f}',
                 basis='sto-3g', symmetry=True, verbose=0, spin=0)

MOLECULES = {
    'H2O':  (h2o,  10, 0, 'O-center, 2 polar bonds'),
    'CH4':  (ch4,  10, 0, 'C-center, 4 polar bonds'),
    'NH3':  (nh3,  10, 0, 'N-center, 3 polar bonds + lone pair'),
    'BH3':  (bh3,   8, 0, 'B-center, 3 polar bonds, electron-deficient'),
    'N2':   (n2,   14, 0, 'homonuclear triple bond'),
    'O2':   (o2,   16, 2, 'homonuclear double bond, triplet'),
    'C2H2': (c2h2, 14, 0, 'homonuclear triple bond + C-H'),
}

# Bond length scales: equilibrium, 1.5x, 2.0x, 2.5x, 3.0x
SCALES = [1.0, 1.5, 2.0, 2.5, 3.0]


def run_molecule(name, build_fn, nelec, spin, nroots=100, min_states=5):
    """Run dissociation curve for one molecule."""
    results = []
    for scale in SCALES:
        t0 = time.time()
        try:
            mol = build_fn(scale)
            # RHF for singlets, UHF for triplets
            if spin == 0:
                mf = scf.RHF(mol)
            else:
                mf = scf.UHF(mol)
            mf.verbose = 0
            mf.kernel()
            mo_coeff = mf.mo_coeff

            res = spectral.molecule_spectral_clustering(
                mol, mo_coeff=mo_coeff, nelec=mol.nelec,
                nroots=nroots, min_states=min_states, verbose=0)
            r_mean = res['r_mean']
            n_blocks = sum(1 for v in res['blocks'].values() if v[0] is not None)
            n_total = res['n_total']
        except Exception as e:
            r_mean = None
            n_blocks = 0
            n_total = 0
            print(f'    ERROR at scale={scale}: {e}')

        dt = time.time() - t0
        results.append({
            'scale': scale,
            'r_mean': r_mean,
            'n_blocks': n_blocks,
            'n_total': n_total,
            'time': dt,
        })
        r_str = f'{r_mean:.4f}' if r_mean is not None else 'FAILED'
        print(f'    scale={scale:.1f}  <r>={r_str}  blocks={n_blocks}  states={n_total}  ({dt:.1f}s)')
    return results


def main():
    args = sys.argv[1:]
    nroots = 100
    out_path = '/tmp/bondtype_r.npz'
    if '--nroots' in args:
        nroots = int(args[args.index('--nroots') + 1])
    if '--out' in args:
        out_path = args[args.index('--out') + 1]

    print(f'=== Bond-type -> Spectral Clustering Mapping ===')
    print(f'nroots={nroots}, scales={SCALES}')
    print()

    all_results = {}
    for name, (build_fn, nelec, spin, desc) in MOLECULES.items():
        print(f'--- {name}: {desc} ---')
        results = run_molecule(name, build_fn, nelec, spin, nroots=nroots)
        all_results[name] = results
        print()

    # Summary table
    print('=== Summary: <r> vs bond scale ===')
    print(f'{"Molecule":<8} {"desc":<45}', end='')
    for s in SCALES:
        print(f' {s:>5.1f}x', end='')
    print()
    print('-' * 90)
    for name, (build_fn, nelec, spin, desc) in MOLECULES.items():
        print(f'{name:<8} {desc:<45}', end='')
        for r in all_results[name]:
            if r['r_mean'] is not None:
                print(f' {r["r_mean"]:5.3f}', end='')
            else:
                print(f' {"FAIL":>5}', end='')
        print()

    # Save
    save_data = {}
    for name, results in all_results.items():
        save_data[f'{name}_scales'] = np.array([r['scale'] for r in results])
        save_data[f'{name}_r'] = np.array([r['r_mean'] if r['r_mean'] is not None else np.nan
                                             for r in results])
    np.savez(out_path, **save_data)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
