#!/usr/bin/env python3
"""H2O/cc-pVDZ WCI test — step-by-step timing."""
import sys, time
import numpy as np
from geoqc import exterior, integrals
from geoqc.wci import build_rank_tables, wci
from pyscf import gto, scf, ao2mo

def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

t0 = time.time()
log("=== H2O/cc-pVDZ WCI ===")

# 1. Molecule + SCF
mol = gto.M(atom='O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0',
            basis='cc-pvdz', symmetry=False, verbose=0)
mf = scf.RHF(mol); mf.verbose=0; mf.kernel()
mo = mf.mo_coeff; n_orb = mo.shape[1]
log(f"SCF done: n_orb={n_orb}, E_HF={mf.e_tot:.6f} Ha")

# 2. Integrals
h1e = mo.T @ (mol.intor('int1e_kin')+mol.intor('int1e_nuc')) @ mo
eri = ao2mo.kernel(mol, mo, compact=False).reshape(n_orb,n_orb,n_orb,n_orb)
o_s, t_s = integrals.spin_orbital_integrals(h1e, eri)
nuc = mol.energy_nuc()
n_a = n_b = 5; nelec = 10; n_spin = 2*n_orb
log(f"Integrals done: n_spin={n_spin}")

# 3. apply_fn
log("Building apply_fn (sparse_action_sz_vec)...")
t1 = time.time()
apply_fn, _, _, _, da, db = exterior.sparse_action_sz_vec(
    n_spin, nelec, 0, o_s, t_s, nuc, 1e-4)
dim = da * db
log(f"apply_fn done in {time.time()-t1:.1f}s: dim={dim:,} ({dim:.2e})")

# 4. rank tables
log("Building rank tables (2^24 entries)...")
t1 = time.time()
rt_a, rt_b, az_of, bz_of = build_rank_tables(n_orb, n_a, n_b, da, db)
log(f"rank tables done in {time.time()-t1:.1f}s")

# 5. on-demand diagonal
def hd_fn(idxs):
    return exterior.sector_diagonal_at(
        n_spin, nelec, 0, o_s, t_s, nuc, 1e-4,
        np.asarray(idxs, dtype=np.int64))

# 6. HF seed
hf_az = int(np.sum(1 << np.arange(n_a)))
hf_bz = int(np.sum(1 << np.arange(n_b)))
seed = int(rt_a[hf_az]) * db + int(rt_b[hf_bz])
log(f"HF seed: idx={seed}, diag={hd_fn(np.array([seed]))[0]:.6f} Ha")

# 7. Test first wavepacket
log("Building first wavepacket...")
t1 = time.time()
from geoqc.wci import Wavepacket
wp0 = Wavepacket(seed, apply_fn, hd_fn, db, az_of, bz_of, rt_a, rt_b)
log(f"First wavepacket done in {time.time()-t1:.1f}s: support={len(wp0.couplings)} states")

# 8. Run WCI (energy-tolerance convergence: 0.1 mHa = 16x chemical precision)
log("Running WCI (max_wp=50, energy_tol=1e-4 Ha)...")
t1 = time.time()
E, uid, coeffs, wps, history = wci(
    apply_fn, hd_fn, seed, db, az_of, bz_of, rt_a, rt_b,
    max_wavepackets=50, tol=1e-7, verbose=True, energy_tol=1e-4)
log(f"WCI done in {time.time()-t1:.1f}s")

print()
print(f"=== RESULTS ===", flush=True)
print(f"  WCI energy = {E:.8f} Ha", flush=True)
print(f"  HF energy  = {mf.e_tot:.8f} Ha", flush=True)
print(f"  Correlation = {E - mf.e_tot:.8f} Ha", flush=True)
print(f"  Wavepackets = {len(wps)}", flush=True)
print(f"  Variational space = {len(uid)} ({len(uid)/dim*100:.4f}% of FCI)", flush=True)
print(f"  Total time = {time.time()-t0:.1f}s", flush=True)
