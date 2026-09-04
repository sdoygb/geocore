#!/usr/bin/env python3
"""安全的 GPU 崩溃定位测试。
每个态在独立子进程中测试，超时 30s 自动 kill，不会导致电脑重启。
先用 CPU 版本验证算法，再用 GPU 版本定位崩溃态。
"""
import sys, os, time, signal, subprocess
import numpy as np

# 安全路径（重启不丢失）
INTEGRAL_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'h2o_ccpvtz_integrals.npz')
PYTHON = sys.executable

# 测试态：(名称, alpha占据轨道, beta占据轨道)
TEST_STATES = [
    ('HF', [0,1,2,3,4], [0,1,2,3,4]),
    ('single_20', [0,1,2,3,20], [0,1,2,3,20]),
    ('single_30', [0,1,2,3,30], [0,1,2,3,30]),
    ('single_40', [0,1,2,3,40], [0,1,2,3,40]),
    ('single_50', [0,1,2,3,50], [0,1,2,3,50]),
    ('single_57', [0,1,2,3,57], [0,1,2,3,57]),
    ('double_50_51', [0,1,2,50,51], [0,1,2,50,51]),
    ('double_55_56', [0,1,2,55,56], [0,1,2,55,56]),
    ('double_56_57', [0,1,2,56,57], [0,1,2,56,57]),
    ('double_40_57', [0,1,2,40,57], [0,1,2,40,57]),
    ('mixed_0_57', [0,1,2,3,57], [0,1,2,4,57]),  # alpha和beta不同激发
    ('mixed_high', [0,1,50,55,57], [0,2,51,56,57]),  # 复杂激发
]

# 子进程脚本：测试单个态
CHILD_SCRIPT = '''
import sys, os, time
import numpy as np
sys.path.insert(0, '/Users/oygb/Downloads/GeometryAI-Mac-Build/geocore')
from geoqc import exterior
from geoqc.integrals import spin_orbital_integrals
from examples.gpu_occ_aware_doubles_sp import GPUApplyOccAwareSP

d = np.load(sys.argv[1])
n_orb, h, t, nuc = int(d['n']), d['h'], d['t'], float(d['nuc'])
n_occ = 5

az = int(sys.argv[2])
bz = int(sys.argv[3])

o_s, t_s = spin_orbital_integrals(h, t)
gpu = GPUApplyOccAwareSP(n_orb, n_occ, t, chunk_size=1)

t0 = time.time()
azs = np.array([az], dtype=np.int64)
bzs = np.array([bz], dtype=np.int64)
vals = np.array([1.0], dtype=np.float64)
result = gpu.doubles(azs, bzs, vals)
t1 = time.time()
n_out = len(result[0])
val_sum = float(result[2].sum()) if n_out > 0 else 0.0
print(f"RESULT n_out={n_out} time={t1-t0:.3f}s val_sum={val_sum:.6e}")
gpu.release()
'''

def run_child(name, az, bz, timeout=30):
    """在子进程中测试单个态，超时自动 kill。"""
    script_path = '/tmp/_gpu_test_child.py'
    with open(script_path, 'w') as f:
        f.write(CHILD_SCRIPT)

    cmd = [PYTHON, '-u', script_path, INTEGRAL_FILE, str(az), str(bz)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            return False, f"exit={proc.returncode}", proc.stderr[-200:] if proc.stderr else ""
        for line in proc.stdout.split('\n'):
            if line.startswith('RESULT'):
                return True, line, ""
        return False, "no_result", proc.stdout[-200:]
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT({timeout}s)", ""
    except Exception as e:
        return False, f"error={e}", ""

def main():
    print(f"=== GPU Crash Locator (safe, subprocess-isolated) ===")
    print(f"Integrals: {INTEGRAL_FILE}")
    print(f"States to test: {len(TEST_STATES)}")
    print(f"Timeout per state: 30s")
    print()

    results = []
    for name, occ_a, occ_b in TEST_STATES:
        az = sum(1 << o for o in occ_a)
        bz = sum(1 << o for o in occ_b)
        print(f"Testing {name:20s} (az={az:058b})...", end=' ', flush=True)
        ok, info, err = run_child(name, az, bz)
        status = "OK" if ok else "FAIL"
        print(f"{status} | {info}")
        if err:
            print(f"  stderr: {err[:150]}")
        results.append((name, ok, info))

    print()
    print("=" * 60)
    print("SUMMARY:")
    n_ok = sum(1 for _, ok, _ in results if ok)
    for name, ok, info in results:
        print(f"  {name:20s} {'OK' if ok else 'FAIL':4s} {info}")
    print(f"\nTotal: {n_ok}/{len(results)} passed")
    if n_ok < len(results):
        print("\nFailed states need further investigation.")
        print("Use CPU version to verify algorithm correctness for these states.")

if __name__ == '__main__':
    main()
