#!/usr/bin/env python3
"""精准定位 GPU kernel 崩溃的态。
对已知激发到高能级轨道的态逐个调用 apply_fn，看哪个崩溃。
"""
import sys, time
import numpy as np
sys.path.insert(0, '.')
from geoqc import exterior
from geoqc.integrals import spin_orbital_integrals
from examples.gpu_occ_aware_doubles_sp import GPUApplyOccAwareSP

# Load integrals
d = np.load('/tmp/_h2o_ccpvtz.npz')
n_orb, h, t, nuc = int(d['n']), d['h'], d['t'], float(d['nuc'])
ns = 2 * n_orb
nelec = 10
n_occ = nelec // 2  # 5

print(f'H2O/cc-pVTZ: n_orb={n_orb}, n_occ={n_occ}')

# Convert to spin-orbital integrals
o_s, t_s = spin_orbital_integrals(h, t)
print(f'Spin-orbital integrals: o_s={o_s.shape}, t_s={t_s.shape}')

# GPU apply
gpu = GPUApplyOccAwareSP(t, n_orb, n_occ, chunk_size=4)
print(f'GPU initialized: chunk_size=4, max_out={gpu.max_out}')

def make_state(occ_orbs):
    """Create bitstring from occupied orbital indices (spatial, 0-indexed)."""
    az = 0
    bz = 0
    for o in occ_orbs[:n_occ]:
        az |= (1 << o)
        bz |= (1 << o)
    return az, bz

def test_state(name, az, bz):
    """Test apply_fn on a single state."""
    print(f'\n--- Testing: {name} ---')
    print(f'  az={az:0{n_orb}b}')
    print(f'  bz={bz:0{n_orb}b}')
    print(f'  alpha occ: {[i for i in range(n_orb) if (az>>i)&1]}')
    print(f'  beta occ:  {[i for i in range(n_orb) if (bz>>i)&1]}')
    try:
        t0 = time.time()
        azs = np.array([az], dtype=np.int64)
        bzs = np.array([bz], dtype=np.int64)
        vals = np.array([1.0], dtype=np.float64)
        result = gpu.doubles(azs, bzs, vals)
        t1 = time.time()
        n_out = len(result[0])
        print(f'  OK: n_out={n_out}, time={t1-t0:.3f}s')
        if n_out > 0:
            print(f'  val range: [{result[2].min():.6e}, {result[2].max():.6e}]')
        return True
    except Exception as e:
        print(f'  FAILED: {type(e).__name__}: {e}')
        return False

# Test states
results = {}

# 1. HF state
az_hf, bz_hf = make_state([0,1,2,3,4])
results['HF'] = test_state('HF (occ 0-4)', az_hf, bz_hf)

# 2. Single excitation to high orbital
for high_orb in [20, 30, 40, 50, 55, 57]:
    occ = [0,1,2,3, high_orb]
    az, bz = make_state(occ)
    results[f'single_{high_orb}'] = test_state(f'Single exc to orb {high_orb}', az, bz)

# 3. Double excitation to high orbitals
for (h1, h2) in [(50,51), (55,56), (56,57), (40,57)]:
    occ = [0,1,2, h1, h2]
    az, bz = make_state(occ)
    results[f'double_{h1}_{h2}'] = test_state(f'Double exc to orb {h1},{h2}', az, bz)

# 4. Mixed: one low, one very high
for (low, high) in [(0, 57), (1, 57), (2, 57), (3, 57)]:
    occ = [0,1, low, 4, high] if low != 4 else [0,1,2,3,high]
    # avoid duplicates
    if len(set(occ)) < 5:
        continue
    az, bz = make_state(occ)
    results[f'mixed_{low}_{high}'] = test_state(f'Mixed exc orb {low},{high}', az, bz)

print('\n' + '='*60)
print('SUMMARY:')
for name, ok in results.items():
    print(f'  {name}: {"OK" if ok else "FAILED"}')
print(f'Total: {sum(results.values())}/{len(results)} passed')
