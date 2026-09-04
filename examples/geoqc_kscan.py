#!/usr/bin/env python3
"""k 标度验证：sparse matrix-free H·v + top-k 幂迭代（动态演化求解器）。

用法: PYTHONPATH=src:. python3 examples/geoqc_kscan.py <npz> <nelec> [--k 20 50 100 300 ...] [--iters 300]

npz 由 geoqc_mkint.py 生成（n, o_s, t_s, nuc）。nelec = 电子数（n_a = n_b = nelec/2, S_z=0）。
对角基态（HF）出发的幂迭代：v ← H·v（稀疏作用，含对角），top-k 截断，归一化。
参考：dense/sparse eigsh（dim 小）或大 k 收敛值（dim 大时）。
"""
import sys, time
import numpy as np
from itertools import combinations
from geoqc import exterior
from scipy import sparse

def load(npz_path):
    d = np.load(npz_path)
    return int(d['n']), d['o_s'], d['t_s'], float(d['nuc'])

def build_rank_tables(n_orb, n_a, n_b, da, db):
    rt_a = np.full(1 << n_orb, -1, dtype=np.int64)
    rt_b = np.full(1 << n_orb, -1, dtype=np.int64)
    for i, c in enumerate(combinations(range(n_orb), n_a)):
        rt_a[sum(1 << j for j in c)] = i
    for i, c in enumerate(combinations(range(n_orb), n_b)):
        rt_b[sum(1 << j for j in c)] = i
    az_of = np.full(da, -1, dtype=np.int64)
    bz_of = np.full(db, -1, dtype=np.int64)
    for i, c in enumerate(combinations(range(n_orb), n_a)):
        az_of[i] = sum(1 << j for j in c)
    for i, c in enumerate(combinations(range(n_orb), n_b)):
        bz_of[i] = sum(1 << j for j in c)
    return rt_a, rt_b, az_of, bz_of

def power_iter(apply, hd_fn, az_of, bz_of, rt_a, rt_b, db, dim, k, iters, seed_idx):
    """top-k 幂迭代：v ← H·v（对角+离对角一次合并），top-k 截断。返回 (E, final_idx, final_vals)。

    hd_fn: 按需对角函数 hd_fn(idxs) -> 对角值数组（大 dim 时用
    sector_diagonal_at，避免 O(dim) 全数组；小 dim 时是 sector_diagonal_sz
    的 hd 数组的索引包装）。最终 Rayleigh 商同样按需聚合（不建 w_full）。"""
    idx = np.array([seed_idx], dtype=np.int64)
    vals = np.ones(1, dtype=complex)
    for it in range(iters):
        t_az, t_bz, t_v = apply(az_of[idx // db], bz_of[idx % db], vals)
        ti = rt_a[t_az] * db + rt_b[t_bz]
        # 合并对角贡献：输入态自身作为目标（一次 unique）
        all_ti = np.concatenate([ti, idx])
        all_v = np.concatenate([t_v, hd_fn(idx) * vals])
        u, inv = np.unique(all_ti, return_inverse=True)
        w = np.zeros(len(u), dtype=complex)
        np.add.at(w, inv, all_v)
        if k < len(u):
            top = np.argsort(np.abs(w))[::-1][:k]
            idx = u[top]; vals = w[top]
        else:
            idx = u; vals = w
        vals /= np.linalg.norm(vals)
    # 最终 Rayleigh 商（对角 + 离对角，按需聚合——不建 dim 全数组）
    t_az, t_bz, t_v = apply(az_of[idx // db], bz_of[idx % db], vals)
    ti = rt_a[t_az] * db + rt_b[t_bz]
    # 聚合到 idx ∪ 目标 的稀疏集合
    all_ti = np.concatenate([ti, idx])
    u, inv = np.unique(all_ti, return_inverse=True)
    w = np.zeros(len(u), dtype=complex)
    np.add.at(w, inv, np.concatenate([t_v, np.zeros(len(idx), dtype=complex)]))
    # w[u 中属于 idx 的项] += 对角
    mask = np.isin(u, idx)
    w[mask] += hd_fn(u[mask]) * vals[np.searchsorted(np.sort(idx), u[mask])]
    w_idx = w[np.searchsorted(u, idx)]
    E = (np.vdot(vals, w_idx) + np.vdot(vals, hd_fn(idx) * vals)).real
    return E, idx, vals

def main():
    args = sys.argv[1:]
    npz = args[0]; nelec = int(args[1])
    nprocs = 0  # 0 = 串行
    if '--parallel' in args:
        nprocs = int(args[args.index('--parallel') + 1])
    ks = []
    if '--k' in args:
        i = args.index('--k')
        for a in args[i + 1:]:
            if a.startswith('--'):
                break
            ks.append(int(a))
    iters = 300
    if '--iters' in args:
        iters = int(args[args.index('--iters') + 1])
    n, o_s, t_s, nuc = load(npz)
    n_a = n_b = nelec // 2
    ns = 2 * n
    _pclose = None
    use_vec = '--vec' in args
    if nprocs > 0:
        apply, _pclose, n_a2, n_b2, n_orb, da, db = \
            exterior.parallel_apply_factory(ns, nelec, 0, o_s, t_s, nuc,
                                            1e-4, nprocs=nprocs,
                                            vec=use_vec)
        print(f'  并行 apply: {nprocs} 进程{" + 向量化" if use_vec else ""}')
    elif use_vec:
        apply, n_a2, n_b2, n_orb, da, db = exterior.sparse_action_sz_vec(
            ns, nelec, 0, o_s, t_s, nuc, 1e-4)
        print('  向量化 apply (sparse_action_sz_vec)')
    else:
        apply, n_a2, n_b2, n_orb, da, db = exterior.sparse_action_sz(
            ns, nelec, 0, o_s, t_s, nuc, 1e-4)
    assert (n_a2, n_b2) == (n_a, n_b)
    dim = da * db
    rt_a, rt_b, az_of, bz_of = build_rank_tables(n_orb, n_a, n_b, da, db)
    print(f'{npz}: n_orb={n_orb} nelec={nelec} dim={dim}  k in {ks}  iters={iters}')
    # 参考 E0：--ref-e0 显式传入（如 pyscf FCI）；否则 dim 小 → 精确 eig
    ref = None
    if '--ref-e0' in args:
        ref = float(args[args.index('--ref-e0') + 1])
    # 对角策略：小 dim 建全数组（sector_diagonal_sz），大 dim 按需
    # （sector_diagonal_at，避免 O(dim) 内存）。two_body=False——
    # sparse_action_sz 的 apply 已在 row==col 项里携带两体对角，
    # 完整对角会 double count（机器验证）。
    DIAG_FULL_LIMIT = 2_000_000
    if dim <= DIAG_FULL_LIMIT:
        t0 = time.time()
        hd, *_ = exterior.sector_diagonal_sz(ns, nelec, 0, o_s, t_s, nuc, 1e-4,
                                               two_body=False)
        print(f'  one-body diagonal (full {dim} elems) {time.time()-t0:.1f}s')
        _hd_arr = hd
        def hd_fn(idxs):
            return _hd_arr[np.asarray(idxs, dtype=np.int64)]
    else:
        print(f'  on-demand diagonal via sector_diagonal_at (dim={dim})')
        def hd_fn(idxs):
            return exterior.sector_diagonal_at(
                ns, nelec, 0, o_s, t_s, nuc, 1e-4,
                idxs=np.asarray(idxs, dtype=np.int64))
    if ref is not None:
        E0 = ref
        print(f'  ref E0 (external) = {E0:.10f}')
    elif dim <= 40000:
        hd2, H2 = exterior.exterior_hamiltonian_sz(ns, nelec, 0, o_s, t_s, nuc, 1e-4)
        H = sparse.diags(hd2) + H2
        E0 = np.linalg.eigvalsh(H.toarray())[0]
        print(f'  exact E0 = {E0:.10f}')
    else:
        E0 = None  # 大 dim 时由最大 k 的收敛值充当参考（相对收敛）
        print('  E0: 大 dim 无外部参考 → 报告相对收敛 E(k)')
    # 种子：HF 态（占据最低 n_a 个 alpha / n_b 个 beta 单电子能级）。
    # argmin(hd) 需要全对角数组，大 dim 不可行；HF 态对两种 dim 均适用。
    e_a = np.array([o_s[2 * k, 2 * k].real for k in range(n_orb)])
    e_b = np.array([o_s[2 * k + 1, 2 * k + 1].real for k in range(n_orb)])
    hf_a = np.sort(np.argsort(e_a)[:n_a])
    hf_b = np.sort(np.argsort(e_b)[:n_b])
    hf_az = int(np.sum(1 << hf_a))
    hf_bz = int(np.sum(1 << hf_b))
    seed = int(rt_a[hf_az]) * db + int(rt_b[hf_bz])
    print(f'  HF seed idx={seed} (a occ={hf_a.tolist()}, b occ={hf_b.tolist()})')
    results = {}
    for k in ks:
        t0 = time.time()
        E, idx, vals = power_iter(apply, hd_fn, az_of, bz_of, rt_a, rt_b, db, dim, k, iters, seed)
        results[k] = E
        err = '' if E0 is None else f'  err={abs(E-E0):.2e}  {"CHEM" if abs(E-E0) < 1.6e-3 else ""}'
        print(f'  k={k:6d}: E={E:.10f}{err}  ({time.time()-t0:.1f}s)')
    if E0 is None and ks:
        # 相对收敛：报告相邻 k 之差（参考 = 最大 k）
        kmax = max(ks)
        for k in sorted(ks)[:-1]:
            print(f'  delta E({k}) - E({kmax}) = {results[k]-results[kmax]:+.2e}')
    if _pclose is not None:
        _pclose()
    print('DONE')

if __name__ == '__main__':
    main()
