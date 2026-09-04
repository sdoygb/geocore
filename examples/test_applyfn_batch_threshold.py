#!/usr/bin/env python3
"""大批量 apply_fn 崩溃阈值定位。
用 HF 的 Bruhat 2-邻域态（和真实 WCI 一致），逐步增加态数。
子进程隔离 + 超时保护，不会导致电脑重启。
"""
import sys, os, time, subprocess, json
import numpy as np

GEOCore = '/Users/oygb/Downloads/GeometryAI-Mac-Build/geocore'
INTEGRAL_FILE = os.path.join(GEOCore, 'data', 'h2o_ccpvtz_integrals.npz')
STATES_FILE = os.path.join(GEOCore, 'data', 'hf_bruhat2_states.npz')
PYTHON = sys.executable

# 子进程脚本：对 N 个态调用 apply_fn
CHILD_SCRIPT = '''
import sys, os, time, resource
import numpy as np
sys.path.insert(0, GEOCore)
from geoqc import exterior
from geoqc.integrals import spin_orbital_integrals
from examples.gpu_occ_aware_doubles_sp import GPUApplyOccAwareSP

n_states = int(sys.argv[1])

d = np.load(INTEGRAL_FILE)
n_orb, h, t, nuc = int(d['n']), d['h'], d['t'], float(d['nuc'])
n_occ = 5

s = np.load(STATES_FILE)
azs_all = s['azs'][:n_states]
bzs_all = s['bzs'][:n_states]
vals_all = np.ones(n_states, dtype=np.float64)

o_s, t_s = spin_orbital_integrals(h, t)
gpu = GPUApplyOccAwareSP(n_orb, n_occ, t, chunk_size=32)

# Mixed apply: singles CPU + doubles GPU (same as real WCI)
apply_fn, _, _, _, _, _ = exterior.sparse_action_sz_vec(
    n=2*n_orb, N=2*n_occ, sz=0, o=o_s, t=t_s, const=nuc, gpu_apply=gpu)

# Warmup
_ = apply_fn(azs_all[:10], bzs_all[:10], vals_all[:10])

# Actual test
t0 = time.time()
result = apply_fn(azs_all, bzs_all, vals_all)
t1 = time.time()

n_out = len(result[0])
mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f"RESULT n_states={n_states} n_out={n_out} time={t1-t0:.2f}s mem={mem_mb:.0f}MB")
gpu.release()
'''

def generate_states():
    """生成 HF 的 Bruhat 2-邻域态，保存到文件。"""
    if os.path.exists(STATES_FILE):
        print(f"States file exists: {STATES_FILE}")
        s = np.load(STATES_FILE)
        print(f"  {len(s['azs'])} states")
        return

    print("Generating HF Bruhat 2-neighborhood states...")
    sys.path.insert(0, GEOCore)
    from geoqc import exterior
    from geoqc.integrals import spin_orbital_integrals

    d = np.load(INTEGRAL_FILE)
    n_orb, h, t, nuc = int(d['n']), d['h'], d['t'], float(d['nuc'])
    n_occ = 5

    o_s, t_s = spin_orbital_integrals(h, t)

    # HF state
    az_hf = sum(1 << i for i in range(n_occ))
    bz_hf = az_hf

    # Get HF's Bruhat 2-neighborhood
    azs = np.array([az_hf], dtype=np.int64)
    bzs = np.array([bz_hf], dtype=np.int64)
    vals = np.array([1.0], dtype=np.float64)

    apply_fn, _, _, _, _, _ = exterior.sparse_action_sz_vec(
        n=2*n_orb, N=2*n_occ, sz=0, o=o_s, t=t_s, const=nuc)

    t0 = time.time()
    out_az, out_bz, out_val, out_src = apply_fn(azs, bzs, vals)
    t1 = time.time()
    print(f"  apply_fn(HF) done: {len(out_az)} outputs, {t1-t0:.2f}s")

    # Deduplicate and sort by |val|
    # Combine az,bz into a single key for deduplication
    keys = out_az.astype(np.int64) * (1 << 30) + out_bz.astype(np.int64)  # rough, may collide
    # Better: use structured array
    dtype = [('az', np.int64), ('bz', np.int64), ('val', np.float64)]
    arr = np.zeros(len(out_az), dtype=dtype)
    arr['az'] = out_az
    arr['bz'] = out_bz
    arr['val'] = out_val

    # Deduplicate by (az,bz), sum vals
    arr.sort(order=['az', 'bz'])
    unique_keys = np.unique(arr[['az', 'bz']])
    print(f"  Unique states: {len(unique_keys)}")

    # Sum vals for duplicate keys
    # (simplified: just take first occurrence, since duplicates should be rare)
    azs_unique = unique_keys['az']
    bzs_unique = unique_keys['bz']

    # Sort by |val| descending (approximate: use first occurrence's val)
    # Build a mapping from (az,bz) to val
    val_dict = {}
    for i in range(len(arr)):
        key = (arr['az'][i], arr['bz'][i])
        if key not in val_dict:
            val_dict[key] = 0
        val_dict[key] += arr['val'][i]

    vals_unique = np.array([val_dict[(azs_unique[i], bzs_unique[i])] for i in range(len(azs_unique))])
    order = np.argsort(-np.abs(vals_unique))
    azs_sorted = azs_unique[order]
    bzs_sorted = bzs_unique[order]
    vals_sorted = vals_unique[order]

    # Take top 10000
    n_take = min(10000, len(azs_sorted))
    np.savez(STATES_FILE, azs=azs_sorted[:n_take], bzs=bzs_sorted[:n_take], vals=vals_sorted[:n_take])
    print(f"  Saved top {n_take} states to {STATES_FILE}")
    print(f"  val range: [{vals_sorted[:n_take].min():.4e}, {vals_sorted[:n_take].max():.4e}]")

def run_batch(n_states, timeout=300):
    """在子进程中对 n_states 个态调用 apply_fn。"""
    script_path = '/tmp/_batch_test_child.py'
    with open(script_path, 'w') as f:
        f.write(CHILD_SCRIPT.replace('GEOCore', repr(GEOCore)).replace('INTEGRAL_FILE', repr(INTEGRAL_FILE)).replace('STATES_FILE', repr(STATES_FILE)))

    cmd = [PYTHON, '-u', script_path, str(n_states)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            return False, f"exit={proc.returncode}", proc.stderr[-300:] if proc.stderr else ""
        for line in proc.stdout.split('\n'):
            if line.startswith('RESULT'):
                return True, line, ""
        return False, "no_result", proc.stdout[-300:]
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT({timeout}s)", ""
    except Exception as e:
        return False, f"error={e}", ""

def main():
    print("=== Apply_fn Batch Crash Threshold Locator ===")
    print(f"In tegrals: {INTEGRAL_FILE}")
    print(f"States: {STATES_FILE}")
    print()

    # Step 1: Generate states
    generate_states()

    # Step 2: Test increasing batch sizes
    batch_sizes = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    # Filter to <= available states
    s = np.load(STATES_FILE)
    max_available = len(s['azs'])
    batch_sizes = [b for b in batch_sizes if b <= max_available]

    print(f"\nTesting batch sizes: {batch_sizes}")
    print(f"Timeout per batch: 300s")
    print()

    results = []
    for n in batch_sizes:
        print(f"Batch {n:5d} states...", end=' ', flush=True)
        ok, info, err = run_batch(n)
        status = "OK" if ok else "FAIL"
        print(f"{status} | {info}")
        if err and not ok:
            print(f"  stderr: {err[:200]}")
        results.append((n, ok, info))
        if not ok:
            print(f"\n  *** First failure at {n} states ***")
            print(f"  Threshold between {batch_sizes[batch_sizes.index(n)-1] if batch_sizes.index(n) > 0 else 0} and {n}")
            # Don't continue to larger sizes
            break

    print()
    print("=" * 60)
    print("SUMMARY:")
    for n, ok, info in results:
        print(f"  {n:5d} states: {'OK' if ok else 'FAIL':4s} {info}")

    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"\nTotal: {n_ok}/{len(results)} passed")
    if n_ok < len(results):
        first_fail = next(n for n, ok, _ in results if not ok)
        print(f"First failure: {first_fail} states")
        print(f"Need to investigate memory or GPU context issues at this scale.")

if __name__ == '__main__':
    main()
