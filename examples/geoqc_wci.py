#!/usr/bin/env python3
"""Wavepacket Configuration Interaction (WCI) — 方法 C：波包参数化迭代。

基于 10.87 §6.03-05 的机器验证结果：
  - 基态是 Grassmann 流形上少数局部波包的叠加（单参考 1 个，多参考 3-5 个）
  - 波包 = 中心行列式 + 其所有 1-2 激发（Bruhat 2-邻域）
  - NO 基使波包坍缩 289 倍（eff-dim 160423 → 555）

核心算法（selected CI 的波包版本）：
  1. 初始化：HF 波包（中心 + 1-2 激发支撑集）
  2. 变分空间 = 所有波包支撑集的并集（累积、Bruhat 连通）
  3. Rayleigh-Ritz：在变分空间内构造 H 矩阵并对角化
  4. 残差 r = H|ψ> - E|ψ>
  5. 选择新波包：变分空间外残差最大的行列式作为新中心
  6. 重复直到收敛

与 top-k 幂迭代的本质区别：
  - top-k：每次迭代重新选 k 个最大分量，支撑集无结构，可能割裂波包
  - WCI：变分空间累积增长（只增不减），每个添加单元是 Bruhat 连通的波包

用法:
  PYTHONPATH=. python3 examples/geoqc_wci.py <npz> <nelec> [--max-wp 30] [--tol 1e-6] [--ref-e0 X.XX]
"""
import sys, time
import numpy as np
from itertools import combinations
from math import comb
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


# ---------------------------------------------------------------------------
# 完整 H|ψ>  =  apply（激发项 + 两体对角） + hd_fn（单电子对角）
# ---------------------------------------------------------------------------

def apply_H(vals, idx, apply_fn, hd_fn, db, az_of, bz_of, rt_a, rt_b):
    """完整 H|ψ>，返回 (unique_target_idx, merged_vals)。"""
    azs = az_of[idx // db]
    bzs = bz_of[idx % db]
    result = apply_fn(azs, bzs, vals)
    if len(result) == 4:
        t_az, t_bz, t_v, _ = result
    else:
        t_az, t_bz, t_v = result
    t_idx = rt_a[t_az] * db + rt_b[t_bz]
    all_idx = np.concatenate([t_idx, idx])
    # 分子积分在实基组下为实数，用 float64 存储（eigh 对实矩阵快 ~2x）
    all_v = np.concatenate([np.asarray(t_v).real, np.asarray(hd_fn(idx) * vals).real])
    u, inv = np.unique(all_idx, return_inverse=True)
    w = np.zeros(len(u), dtype=float)
    np.add.at(w, inv, all_v)
    return u, w


# ---------------------------------------------------------------------------
# 波包
# ---------------------------------------------------------------------------

class Wavepacket:
    """以 center_idx 为中心的 Bruhat 2-邻域波包。

    支撑集 = {中心} ∪ {apply(中心) 的所有目标}（1-2 激发 + 两体对角目标）。
    因为 H 是 1+2 体算符，apply(中心) 的目标恰为 Bruhat 距离 ≤ 2 的态。
    """

    def __init__(self, center_idx, apply_fn, hd_fn, db, az_of, bz_of, rt_a, rt_b):
        self.center = int(center_idx)
        u, w = apply_H(np.array([1.0 + 0j]), np.array([self.center]),
                        apply_fn, hd_fn, db, az_of, bz_of, rt_a, rt_b)
        # 支撑集 = H|center> 的所有非零目标（含中心自身的对角项）
        self.support = u
        self.size = len(u)
        # 中心到各支撑态的耦合 H_{D, center}（含中心自身的完整对角）
        self.couplings = dict(zip(u.tolist(), w.tolist()))

    def full_diagonal(self, state_idx):
        """中心或支撑态的完整对角 E_D = H_{D,D}。"""
        return self.couplings.get(int(state_idx), 0.0)


# ---------------------------------------------------------------------------
# 变分空间与 H 矩阵
# ---------------------------------------------------------------------------

def build_variational_space(wavepackets):
    all_idx = np.concatenate([wp.support for wp in wavepackets])
    return np.unique(all_idx)


def build_H_matrix(unique_idx, apply_fn, hd_fn, db, az_of, bz_of, rt_a, rt_b,
                   H_cols=None):
    """向量化 H 矩阵构造：预分配 COO + np.searchsorted，避免字典查找和动态列表。

    对每个源态 j 做 apply_H 得到 H|D_j>，用 searchsorted 向量化定位目标在
    unique_idx 中的位置，预分配数组收集 (row, col, val)，最后 COO→稠密组装。
    若提供 H_cols（dict），同时缓存每列完整 H|D_j>（含空间外目标），供增量残差复用。"""
    from scipy.sparse import coo_matrix
    n = len(unique_idx)

    # 用第一个态估计每列非零数，预分配（留 2x 余量）
    u0, w0 = apply_H(np.array([1.0 + 0j]), np.array([unique_idx[0]]),
                     apply_fn, hd_fn, db, az_of, bz_of, rt_a, rt_b)
    if H_cols is not None:
        H_cols[int(unique_idx[0])] = (u0, w0)
    nnz_per_col = max(len(u0), 1)
    max_nnz = nnz_per_col * n * 2

    rows = np.empty(max_nnz, dtype=np.int64)
    cols = np.empty(max_nnz, dtype=np.int64)
    data = np.empty(max_nnz, dtype=float)
    cursor = 0

    # 辅助：把一列的 apply 结果填入 COO 数组
    def _fill_col(j, u, w):
        nonlocal cursor, max_nnz, rows, cols, data
        pos = np.searchsorted(unique_idx, u)
        valid = pos < n
        if valid.any():
            valid[valid] &= unique_idx[pos[valid]] == u[valid]
        m = int(valid.sum())
        if m == 0:
            return
        if cursor + m > max_nnz:
            new_max = max(max_nnz * 2, cursor + m)
            rows = np.resize(rows, new_max)
            cols = np.resize(cols, new_max)
            data = np.resize(data, new_max)
            max_nnz = new_max
        rows[cursor:cursor + m] = pos[valid]
        cols[cursor:cursor + m] = j
        data[cursor:cursor + m] = w[valid]
        cursor += m

    # 第 0 列（已在估计 nnz 时计算）
    _fill_col(0, u0, w0)

    for j in range(1, n):
        u, w = apply_H(np.array([1.0 + 0j]), np.array([unique_idx[j]]),
                        apply_fn, hd_fn, db, az_of, bz_of, rt_a, rt_b)
        if H_cols is not None:
            H_cols[int(unique_idx[j])] = (u, w)
        _fill_col(j, u, w)

    rows = rows[:cursor]
    cols = cols[:cursor]
    data = data[:cursor]
    H_sparse = coo_matrix((data, (rows, cols)), shape=(n, n))
    H_mat = H_sparse.toarray()
    H_mat = (H_mat + H_mat.conj().T) / 2.0
    return H_mat


# ---------------------------------------------------------------------------
# 残差
# ---------------------------------------------------------------------------

def compute_residual(coeffs, unique_idx, E, apply_fn, hd_fn, db,
                     az_of, bz_of, rt_a, rt_b):
    """向量化残差 r = H|ψ> - E|ψ>：searchsorted 代替字典查找。"""
    u, w = apply_H(coeffs, unique_idx, apply_fn, hd_fn, db,
                    az_of, bz_of, rt_a, rt_b)
    n = len(unique_idx)
    pos = np.searchsorted(unique_idx, u)
    in_space = pos < n
    if in_space.any():
        in_space[in_space] &= unique_idx[pos[in_space]] == u[in_space]

    r_in_norm_sq = 0.0
    if in_space.any():
        r_vals = w[in_space] - E * coeffs[pos[in_space]]
        r_in_norm_sq = float(np.sum(np.abs(r_vals) ** 2))

    out_mask = ~in_space
    r_out_idx = u[out_mask]
    r_out_vals = w[out_mask]
    return np.sqrt(r_in_norm_sq), r_out_idx, r_out_vals


def compute_residual_incremental(coeffs, unique_idx, H_cols, E):
    """增量残差：用保存的 H 列（H_cols[D] = (target_idx, value)）线性组合，
    不调用 apply_fn。旧态列已缓存，新态列在主循环中补充。

    H|ψ> = Σ_D c_D · H|D>，用 numpy 向量化 concatenate + unique + add.at 合并，
    无 apply_fn 内部的单激发 Python 循环——残差成本从 O(n_var×行度 Python)
    降到 O(n_var×行度 numpy)。"""
    n = len(unique_idx)
    all_targets = []
    all_values = []
    for j in range(n):
        c = coeffs[j]
        if c == 0.0:
            continue
        t_idx, t_val = H_cols[int(unique_idx[j])]
        all_targets.append(t_idx)
        all_values.append(c * t_val)

    all_targets = np.concatenate(all_targets)
    all_values = np.concatenate(all_values)
    u, inv = np.unique(all_targets, return_inverse=True)
    w = np.zeros(len(u), dtype=float)
    np.add.at(w, inv, all_values)

    pos = np.searchsorted(unique_idx, u)
    in_space = pos < n
    if in_space.any():
        in_space[in_space] &= unique_idx[pos[in_space]] == u[in_space]

    r_in_norm_sq = 0.0
    if in_space.any():
        r_vals = w[in_space] - E * coeffs[pos[in_space]]
        r_in_norm_sq = float(np.sum(r_vals ** 2))

    out_mask = ~in_space
    r_out_idx = u[out_mask]
    r_out_vals = w[out_mask]
    return np.sqrt(r_in_norm_sq), r_out_idx, r_out_vals


def _build_H_incremental(unique_idx, unique_idx_prev, H_mat_prev,
                          apply_fn, hd_fn, db, az_of, bz_of, rt_a, rt_b,
                          H_cols=None):
    """增量 H 矩阵构造：复用旧块，只对新增态做 apply 填充新列。

    每轮变分空间只新增 ~150-200 个态，旧块 H_{old,old} 完全复用，
    新列 H_{:,new} 只需 m 次 apply（而非 n_var 次），新行由对称性补全。
    若提供 H_cols，同时缓存新列完整 H|D>（含空间外目标）。"""
    n = len(unique_idx)
    new_mask = ~np.isin(unique_idx, unique_idx_prev)
    new_positions = np.where(new_mask)[0]
    m = len(new_positions)
    if m == 0:
        return H_mat_prev

    H_mat = np.zeros((n, n), dtype=float)
    old_positions = np.where(~new_mask)[0]
    H_mat[np.ix_(old_positions, old_positions)] = H_mat_prev

    for j in new_positions:
        u, w = apply_H(np.array([1.0 + 0j]), np.array([unique_idx[j]]),
                        apply_fn, hd_fn, db, az_of, bz_of, rt_a, rt_b)
        if H_cols is not None:
            H_cols[int(unique_idx[j])] = (u, w)
        pos = np.searchsorted(unique_idx, u)
        valid = pos < n
        if valid.any():
            valid[valid] &= unique_idx[pos[valid]] == u[valid]
        H_mat[pos[valid], j] = w[valid]

    # 由 Hermiticity 填充新行：H[new, :] = H[:, new].conj().T
    # （旧块已正确，新列已填充，新行由对称性补全；不再做 (H+H†)/2 对称化，
    #  否则会把未填充的新行 0 平均进新列，导致新列值被除以 2）
    H_mat[new_positions, :] = H_mat[:, new_positions].conj().T
    return H_mat


# ---------------------------------------------------------------------------
# WCI 主迭代
# ---------------------------------------------------------------------------

def wci(apply_fn, hd_fn, seed_idx, db, az_of, bz_of, rt_a, rt_b,
        max_wavepackets=30, tol=1e-6, verbose=True,
        select_weighted=False, hd_full_fn=None):
    """Wavepacket CI。返回 (E, unique_idx, coeffs, wavepackets, history).

    select_weighted=True 时用能量最优（分母加权）判据选新波包中心：
    argmax |<D|r>|^2 / |<D|H|D> - E|（命题 10.88.3.03/3.04，CIPSI PT2 风格），
    需 hd_full_fn 提供完整对角 <D|H|D>（含两体与核排斥）。"""
    t_start = time.time()
    wavepackets = [Wavepacket(seed_idx, apply_fn, hd_fn, db, az_of, bz_of, rt_a, rt_b)]
    if verbose:
        print(f'  WP  1: center={wavepackets[0].center}, size={wavepackets[0].size}')

    E_prev = None
    history = []
    H_mat_prev = None
    unique_idx_prev = None
    H_cols = {}  # 缓存每列完整 H|D> = (target_idx, value)，供增量残差复用

    for iteration in range(max_wavepackets):
        # 1. 变分空间
        unique_idx = build_variational_space(wavepackets)
        n_var = len(unique_idx)

        # 2. Rayleigh-Ritz（增量 H 矩阵构造 + H_cols 缓存）
        t0 = time.time()
        if H_mat_prev is not None and unique_idx_prev is not None:
            H_mat = _build_H_incremental(
                unique_idx, unique_idx_prev, H_mat_prev,
                apply_fn, hd_fn, db, az_of, bz_of, rt_a, rt_b, H_cols)
        else:
            H_mat = build_H_matrix(unique_idx, apply_fn, hd_fn, db, az_of, bz_of, rt_a, rt_b, H_cols)
        E_all, vecs = np.linalg.eigh(H_mat)
        E = float(E_all[0])
        coeffs = vecs[:, 0]
        t_h = time.time() - t0

        # 3. 残差（增量：用 H_cols 线性组合，不调用 apply_fn）
        t0 = time.time()
        r_in_norm, r_out_idx, r_out_vals = compute_residual_incremental(
            coeffs, unique_idx, H_cols, E)
        r_out_norm = np.linalg.norm(r_out_vals) if len(r_out_vals) > 0 else 0.0
        t_r = time.time() - t0

        history.append((n_var, E, r_in_norm, r_out_norm))
        if verbose:
            print(f'  iter {iteration+1:2d}: n_var={n_var:7d}  E={E:.10f}  '
                  f'||r_in||={r_in_norm:.2e}  ||r_out||={r_out_norm:.2e}  '
                  f'n_wp={len(wavepackets)}  H={t_h:.1f}s  r={t_r:.1f}s')

        # 4. 收敛
        if E_prev is not None and abs(E - E_prev) < tol and r_out_norm < tol:
            if verbose:
                print(f'  Converged at iter {iteration+1}')
            break
        E_prev = E
        H_mat_prev = H_mat
        unique_idx_prev = unique_idx
        if iteration == max_wavepackets - 1:
            break

        # 5. 新波包中心：简化判据 argmax|<D|r>| 或能量最优分母加权判据
        if len(r_out_vals) > 0:
            if select_weighted and hd_full_fn is not None:
                denom = hd_full_fn(r_out_idx) - E
                score = np.abs(r_out_vals) ** 2 / np.maximum(np.abs(denom), 1e-10)
                best = int(np.argmax(score))
                new_center = int(r_out_idx[best])
            else:
                best = int(np.argmax(np.abs(r_out_vals)))
                new_center = int(r_out_idx[best])
            new_wp = Wavepacket(new_center, apply_fn, hd_fn, db, az_of, bz_of, rt_a, rt_b)
            wavepackets.append(new_wp)
            if verbose:
                print(f'  WP {len(wavepackets):2d}: center={new_center}, '
                      f'size={new_wp.size}, |r|={abs(r_out_vals[best]):.2e}')
        else:
            if verbose:
                print('  No residual outside space — fully converged')
            break

    if verbose:
        print(f'  WCI done in {time.time()-t_start:.1f}s: '
              f'n_wp={len(wavepackets)}, n_var={len(unique_idx)}, E={E:.10f}')
    return E, unique_idx, coeffs, wavepackets, history


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    npz = args[0]
    nelec = int(args[1])
    n, o_s, t_s, nuc = load(npz)
    n_a = n_b = nelec // 2
    ns = 2 * n
    n_orb = n

    apply, n_a2, n_b2, n_orb2, da, db = exterior.sparse_action_sz_vec(
        ns, nelec, 0, o_s, t_s, nuc, 1e-4)
    assert (n_a2, n_b2) == (n_a, n_b)
    dim = da * db
    rt_a, rt_b, az_of, bz_of = build_rank_tables(n_orb, n_a, n_b, da, db)

    # 对角策略（同 kscan）
    DIAG_FULL_LIMIT = 2_000_000
    if dim <= DIAG_FULL_LIMIT:
        hd, *_ = exterior.sector_diagonal_sz(ns, nelec, 0, o_s, t_s, nuc, 1e-4, two_body=False)
        _hd_arr = hd
        def hd_fn(idxs):
            return _hd_arr[np.asarray(idxs, dtype=np.int64)]
    else:
        def hd_fn(idxs):
            return exterior.sector_diagonal_at(ns, nelec, 0, o_s, t_s, nuc, 1e-4,
                                                idxs=np.asarray(idxs, dtype=np.int64))

    # HF 种子
    e_a = np.array([o_s[2 * k, 2 * k].real for k in range(n_orb)])
    e_b = np.array([o_s[2 * k + 1, 2 * k + 1].real for k in range(n_orb)])
    hf_a = np.sort(np.argsort(e_a)[:n_a])
    hf_b = np.sort(np.argsort(e_b)[:n_b])
    hf_az = int(np.sum(1 << hf_a))
    hf_bz = int(np.sum(1 << hf_b))
    seed = int(rt_a[hf_az]) * db + int(rt_b[hf_bz])

    print(f'{npz}: n_orb={n_orb} nelec={nelec} dim={dim} HF seed={seed}')

    # 参考 E0
    ref = None
    if '--ref-e0' in args:
        ref = float(args[args.index('--ref-e0') + 1])
    if ref is not None:
        E0 = ref
        print(f'  ref E0 (external) = {E0:.10f}')
    elif dim <= 40000:
        hd2, H2 = exterior.exterior_hamiltonian_sz(ns, nelec, 0, o_s, t_s, nuc, 1e-4)
        H_full = sparse.diags(hd2) + H2
        E0 = float(np.linalg.eigvalsh(H_full.toarray())[0])
        print(f'  exact E0 = {E0:.10f}')
    else:
        E0 = None
        print('  E0: large dim, no external ref')

    max_wp = 30
    if '--max-wp' in args:
        max_wp = int(args[args.index('--max-wp') + 1])
    tol = 1e-6
    if '--tol' in args:
        tol = float(args[args.index('--tol') + 1])

    # 能量最优（分母加权）选择判据：argmax |<D|r>|^2 / |<D|H|D> - E|
    # （命题 10.88.3.03/3.04，CIPSI PT2 风格）——与简化判据 argmax|<D|r>| 对比
    select_weighted = '--weighted' in args
    hd_full_fn = None
    if select_weighted:
        if dim <= DIAG_FULL_LIMIT:
            hd_full, *_ = exterior.sector_diagonal_sz(
                ns, nelec, 0, o_s, t_s, nuc, 1e-4, two_body=True)
            _hd_full_arr = hd_full
            def hd_full_fn(idxs):
                return _hd_full_arr[np.asarray(idxs, dtype=np.int64)]
        else:
            def hd_full_fn(idxs):
                return exterior.sector_diagonal_at(
                    ns, nelec, 0, o_s, t_s, nuc, 1e-4,
                    idxs=np.asarray(idxs, dtype=np.int64))
        print('  weighted selection: argmax |<D|r>|^2 / |<D|H|D> - E|')

    E, unique_idx, coeffs, wavepackets, history = wci(
        apply, hd_fn, seed, db, az_of, bz_of, rt_a, rt_b,
        max_wavepackets=max_wp, tol=tol, verbose=True,
        select_weighted=select_weighted, hd_full_fn=hd_full_fn)

    print(f'\nFinal E = {E:.10f}')
    print(f'Wavepackets: {len(wavepackets)}, variational space: {len(unique_idx)}')
    if E0 is not None:
        err = E - E0
        print(f'FCI E = {E0:.10f}')
        print(f'Error = {err:+.2e} Ha')
        if abs(err) < 1.6e-3:
            print('CHEMICAL ACCURACY (1.6 mHa)')

    print('\nConvergence history:')
    print(f'  {"iter":>4} {"n_var":>8} {"E":>16} {"||r_in||":>12} {"||r_out||":>12}')
    for i, (nv, e, ri, ro) in enumerate(history):
        print(f'  {i+1:4d} {nv:8d} {e:16.10f} {ri:12.2e} {ro:12.2e}')


if __name__ == '__main__':
    main()
