"""GPU acceleration for WCI (OpenCL on AMD RX 570).

Provides GPUMatrix class that stores H in GPU memory and performs
fast matrix-vector products (2.5-3.0x speedup for V > 5000).

Usage:
    from geoqc.gpu import GPUMatrix
    gpu_H = GPUMatrix(H_mat)
    E, vecs = eigsh(gpu_H.as_linear_operator(), k=3, ...)
    gpu_H.release()
"""

import numpy as np

try:
    import pyopencl as cl
    _HAS_PYOPENCL = True
except ImportError:
    _HAS_PYOPENCL = False

# OpenCL GEMV kernel (naive, proven correct at 1e-13)
_GEMV_KERNEL = """
__kernel void gemv(
    __global const double *A,
    __global const double *x,
    __global double *y,
    int M, int N)
{
    int row = get_global_id(0);
    if (row < M) {
        double sum = 0.0;
        for (int col = 0; col < N; col++) {
            sum += A[row * N + col] * x[col];
        }
        y[row] = sum;
    }
}
"""


def find_gpu_device():
    """Find AMD Radeon GPU device. Returns (platform, device) or None."""
    if not _HAS_PYOPENCL:
        return None
    for plat in cl.get_platforms():
        for dev in plat.get_devices():
            if 'Radeon' in dev.name or 'RX' in dev.name:
                return plat, dev
    return None


# Global singleton OpenCL context/queue/kernel (avoid frequent create/destroy)
_global_ctx = None
_global_queue = None
_global_kernel = None


def _get_global_gpu():
    """Get or create global OpenCL context, queue, and compiled kernel."""
    global _global_ctx, _global_queue, _global_kernel
    if _global_ctx is None:
        result = find_gpu_device()
        if result is None:
            raise RuntimeError("No AMD GPU found")
        _, dev = result
        _global_ctx = cl.Context([dev])
        _global_queue = cl.CommandQueue(_global_ctx)
        prg = cl.Program(_global_ctx, _GEMV_KERNEL).build()
        _global_kernel = prg.gemv
        print(f'  [GPU] Initialized global context on {dev.name}')
    return _global_ctx, _global_queue, _global_kernel


class GPUMatrix:
    """Matrix stored in GPU memory with fast matvec.

    H matrix stays resident in GPU memory between matvec calls.
    Only the vector x is transferred per matvec (small data).
    Uses global singleton OpenCL context/queue to avoid resource leaks.
    """

    def __init__(self, H, ctx=None, queue=None):
        if not _HAS_PYOPENCL:
            raise RuntimeError("PyOpenCL not installed")

        self.M, self.N = H.shape

        # Use global singleton context/queue/kernel by default
        if ctx is None:
            self.ctx, self.queue, self.kernel = _get_global_gpu()
        else:
            self.ctx = ctx
            self.queue = queue
            prg = cl.Program(self.ctx, _GEMV_KERNEL).build()
            self.kernel = prg.gemv

        # Transfer H to GPU (one-time)
        mf = cl.mem_flags
        self.H_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=H)
        self.x_buf = cl.Buffer(self.ctx, mf.READ_ONLY, self.N * 8)
        self.y_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.M * 8)
        self.queue.finish()

        # Work group setup
        self.local_size = (min(256, self.M),)
        self.global_size = (((self.M + 255) // 256) * 256,)

        self.y_cpu = np.empty(self.M, dtype=np.float64)
        self._released = False

    def matvec(self, x):
        """Compute H @ x on GPU. x is CPU numpy array of shape (N,)."""
        if self._released:
            raise RuntimeError("GPUMatrix has been released")
        cl.enqueue_copy(self.queue, self.x_buf, x)
        self.kernel(self.queue, self.global_size, self.local_size,
                    self.H_buf, self.x_buf, self.y_buf,
                    np.int32(self.M), np.int32(self.N))
        cl.enqueue_copy(self.queue, self.y_cpu, self.y_buf)
        self.queue.finish()
        return self.y_cpu.copy()

    def update_rows(self, row_indices, new_rows):
        """Update specific rows of H on GPU (for incremental WCI).

        row_indices: array of row indices to update
        new_rows: array of shape (len(row_indices), N)
        """
        if self._released:
            raise RuntimeError("GPUMatrix has been released")
        mf = cl.mem_flags
        for i, row_idx in enumerate(row_indices):
            offset = int(row_idx) * self.N * 8
            row_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=np.ascontiguousarray(new_rows[i]))
            cl.enqueue_copy(self.queue, self.H_buf, row_buf,
                          device_offset=offset, size=self.N * 8)
            row_buf.release()
        self.queue.finish()

    def as_linear_operator(self):
        """Return scipy LinearOperator wrapping this GPU matrix."""
        from scipy.sparse.linalg import LinearOperator
        return LinearOperator((self.M, self.N), matvec=self.matvec, dtype=float)

    def release(self):
        """Free GPU buffer memory (global context/queue are kept)."""
        if not self._released:
            self.H_buf.release()
            self.x_buf.release()
            self.y_buf.release()
            self._released = True

    def __del__(self):
        self.release()


# ---------------------------------------------------------------------------
# GPU-accelerated double excitation apply (WCI H-build bottleneck)
# ---------------------------------------------------------------------------

_DOUBLES_KERNEL = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// [文章标注] 10.88 §7.12.2 符号计算
// alpha_k: popcount(az & ((1<<k)-1)) + popcount(bz & ((1<<k)-1))
// beta_k:  popcount(az & ((1<<(k+1))-1)) + popcount(bz & ((1<<k)-1))
// 与 exterior._spin_sign 完全等价，用 OpenCL 内置 popcount(long)
inline double spin_sign_alpha(long az, long bz, int k) {
    long m = (1L << k) - 1;
    int cnt = popcount(az & m) + popcount(bz & m);
    return (cnt & 1) ? -1.0 : 1.0;
}

inline double spin_sign_beta(long az, long bz, int k) {
    long m_ba = (1L << (k + 1)) - 1;
    long m_bb = (1L << k) - 1;
    int cnt = popcount(az & m_ba) + popcount(bz & m_bb);
    return (cnt & 1) ? -1.0 : 1.0;
}

inline double spin_sign(long az, long bz, int k, int is_beta) {
    return is_beta ? spin_sign_beta(az, bz, k) : spin_sign_alpha(az, bz, k);
}

__kernel void doubles_kernel(
    __global const long *azs,
    __global const long *bzs,
    __global const double *vals,
    __global const int *term_kp,
    __global const int *term_kq,
    __global const int *term_kr,
    __global const int *term_ks,
    __global const int *term_sp,
    __global const int *term_sq,
    __global const int *term_sr,
    __global const int *term_ss,
    __global const double *term_c,
    int S, int T,
    __global long *out_az,
    __global long *out_bz,
    __global double *out_val,
    __global int *out_src,
    __global int *counter)
{
    int idx = get_global_id(0);
    int s = idx / T;
    int t = idx % T;
    if (s >= S || t >= T) return;

    long az = azs[s];
    long bz = bzs[s];
    double val = vals[s];

    int kp = term_kp[t], kq = term_kq[t], kr = term_kr[t], ks = term_ks[t];
    int sp = term_sp[t], sq = term_sq[t], sr = term_sr[t], ss = term_ss[t];
    double c = term_c[t];

    // Check r,s occupied
    int r_occ = (sr == 0) ? ((az >> kr) & 1) : ((bz >> kr) & 1);
    if (!r_occ) return;
    int s_occ = (ss == 0) ? ((az >> ks) & 1) : ((bz >> ks) & 1);
    if (!s_occ) return;

    double sign = 1.0;

    // Annihilate s
    sign *= spin_sign(az, bz, ks, ss);
    if (ss == 0) az ^= (1L << ks); else bz ^= (1L << ks);

    // Annihilate r
    sign *= spin_sign(az, bz, kr, sr);
    if (sr == 0) az ^= (1L << kr); else bz ^= (1L << kr);

    // Check q empty
    int q_empty = (sq == 0) ? (((az >> kq) & 1) == 0) : (((bz >> kq) & 1) == 0);
    if (!q_empty) return;

    // [关键正确性] 产生算符符号用产生后位串（与 exterior.py 一致）
    // 最初误用产生前位串导致单扇区双激发(alpha-alpha/beta-beta)符号错误
    // Create q (sign uses POST-creation bitstring, matching exterior.py)
    if (sq == 0) az ^= (1L << kq); else bz ^= (1L << kq);
    sign *= spin_sign(az, bz, kq, sq);

    // Check p empty
    int p_empty = (sp == 0) ? (((az >> kp) & 1) == 0) : (((bz >> kp) & 1) == 0);
    if (!p_empty) return;

    // Create p (sign uses POST-creation bitstring)
    if (sp == 0) az ^= (1L << kp); else bz ^= (1L << kp);
    sign *= spin_sign(az, bz, kp, sp);

    // Write output
    int pos = atomic_inc(counter);
    out_az[pos] = az;
    out_bz[pos] = bz;
    out_val[pos] = c * sign * val;
    out_src[pos] = s;
}
"""


class GPUApply:
    """GPU-accelerated double-excitation apply for WCI H-build.

    [文章标注] 10.88 §7.12 优化九：GPU 加速双激发 apply
    - 大系统 H₂O/cc-pVDZ (dim=1.8e9) 上 H_build 占 80%+，双激发占 apply 85%+
    - 双激发单独加速 22x，H_build 加速 2.6-3.7x，端到端 WCI 加速 2.2x
    - 正确性：H₂O/cc-pVDZ 2波包 dE=7.11e-14 Ha，V=6752 完全一致

    Pre-computes double-excitation terms on GPU, then for each batch of
    source determinants, computes all valid double excitations in parallel.
    Uses atomic counter for sparse output collection.

    Key correctness details:
    - Annihilation sign uses PRE-annihilation bitstring
    - Creation sign uses POST-creation bitstring (matching exterior._spin_sign)
    - Each chunk uses its own source buffer (kernel indexes from 0)

    Usage:
        gpu_apply = GPUApply(n_orb, o_s, t_s, eps=1e-4)
        az_out, bz_out, v_out, src_out = gpu_apply.doubles(azs, bzs, vals)
        gpu_apply.release()
    """

    def __init__(self, n_orb, o_s, t_s, eps=1e-4, chunk_size=64):
        if not _HAS_PYOPENCL:
            raise RuntimeError("PyOpenCL not installed")

        self.n_orb = n_orb
        self.eps = eps
        self.chunk_size = chunk_size

        # Get global context/queue/kernel
        self.ctx, self.queue, _ = _get_global_gpu()

        # Compile doubles kernel
        prg = cl.Program(self.ctx, _DOUBLES_KERNEL).build()
        self.kernel = prg.doubles_kernel

        # Pre-compute double-excitation terms
        self._compute_terms(o_s, t_s)

        # Transfer terms to GPU
        mf = cl.mem_flags
        self.term_kp_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.term_kp)
        self.term_kq_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.term_kq)
        self.term_kr_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.term_kr)
        self.term_ks_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.term_ks)
        self.term_sp_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.term_sp)
        self.term_sq_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.term_sq)
        self.term_sr_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.term_sr)
        self.term_ss_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.term_ss)
        self.term_c_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.term_c)
        self.T = len(self.term_kp)

        # Pre-allocate output buffers (chunk_size * T, worst case)
        out_size = self.chunk_size * self.T
        self.out_az_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, out_size * 8)
        self.out_bz_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, out_size * 8)
        self.out_val_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, out_size * 8)
        self.out_src_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, out_size * 4)
        self.counter_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                                     hostbuf=np.zeros(1, dtype=np.int32))

        self._released = False
        print(f'  [GPUApply] Initialized: T={self.T:,} terms, chunk={self.chunk_size}')

    def _compute_terms(self, o_s, t_s):
        """Pre-compute double-excitation terms (same logic as exterior.py)."""
        n = 2 * self.n_orb  # spin orbitals
        tt = []
        for p in range(n):
            for q in range(p + 1, n):
                for r in range(n):
                    for s in range(r + 1, n):
                        c2 = 2.0 * (t_s[p, q, r, s] - t_s[p, q, s, r])
                        if abs(c2) <= self.eps:
                            continue
                        if (1 - p % 2) + (1 - q % 2) != (1 - r % 2) + (1 - s % 2):
                            continue
                        tt.append((p, q, r, s, float(c2.real)))

        self.term_kp = np.array([x[0] // 2 for x in tt], dtype=np.int32)
        self.term_kq = np.array([x[1] // 2 for x in tt], dtype=np.int32)
        self.term_kr = np.array([x[2] // 2 for x in tt], dtype=np.int32)
        self.term_ks = np.array([x[3] // 2 for x in tt], dtype=np.int32)
        self.term_sp = np.array([x[0] % 2 for x in tt], dtype=np.int32)
        self.term_sq = np.array([x[1] % 2 for x in tt], dtype=np.int32)
        self.term_sr = np.array([x[2] % 2 for x in tt], dtype=np.int32)
        self.term_ss = np.array([x[3] % 2 for x in tt], dtype=np.int32)
        self.term_c = np.array([x[4] for x in tt], dtype=np.float64)

    def doubles(self, azs, bzs, vals):
        """Compute all double excitations for a batch of source determinants.

        Parameters
        ----------
        azs, bzs : ndarray (S,)
            Alpha/beta bitstrings of source determinants.
        vals : ndarray (S,)
            Coefficients of source determinants (real).

        Returns
        -------
        out_az, out_bz, out_val, out_src : ndarrays
            Target bitstrings, values, and source indices for all valid
            double excitations.
        """
        if self._released:
            raise RuntimeError("GPUApply has been released")

        S = len(azs)
        azs = np.asarray(azs, dtype=np.int64)
        bzs = np.asarray(bzs, dtype=np.int64)
        vals = np.asarray(vals).real.astype(np.float64)

        mf = cl.mem_flags
        all_az = []
        all_bz = []
        all_val = []
        all_src = []

        # [文章标注] 10.88 §7.12.2 分块处理
        # chunk_size=64 源态/块，避免 S×T 全网格(S=5000,T=126484→6.3亿)
        # [关键 bug 修复] 每块必须用自己的源态 buffer——最初误用全部 S 的 buffer，
        #   导致 kernel 内 s=idx/T 从 0 开始时访问的是 azs[0:chunk_S] 而非
        #   azs[c0:c0+chunk_S]，S>64 时结果完全错误（dE=167 Ha）
        for c0 in range(0, S, self.chunk_size):
            c1 = min(c0 + self.chunk_size, S)
            chunk_S = c1 - c0

            # Create per-chunk source buffers (CRITICAL: kernel indexes
            # from 0, so each chunk must see only its own sources)
            chunk_azs = azs[c0:c1]
            chunk_bzs = bzs[c0:c1]
            chunk_vals = vals[c0:c1]
            azs_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=chunk_azs)
            bzs_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=chunk_bzs)
            vals_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=chunk_vals)

            # Reset counter
            cl.enqueue_copy(self.queue, self.counter_buf, np.zeros(1, dtype=np.int32))

            # Launch kernel: chunk_S * T work items
            global_size = (chunk_S * self.T,)
            local_size = (256,)
            # Pad to multiple of 256
            global_size = (((global_size[0] + 255) // 256) * 256,)

            self.kernel(self.queue, global_size, local_size,
                        azs_buf, bzs_buf, vals_buf,
                        self.term_kp_buf, self.term_kq_buf,
                        self.term_kr_buf, self.term_ks_buf,
                        self.term_sp_buf, self.term_sq_buf,
                        self.term_sr_buf, self.term_ss_buf,
                        self.term_c_buf,
                        np.int32(chunk_S), np.int32(self.T),
                        self.out_az_buf, self.out_bz_buf,
                        self.out_val_buf, self.out_src_buf,
                        self.counter_buf)

            # Read counter
            counter = np.zeros(1, dtype=np.int32)
            cl.enqueue_copy(self.queue, counter, self.counter_buf)
            self.queue.finish()
            n_out = int(counter[0])

            if n_out > 0:
                # PyOpenCL reads exactly len(dest) elements from buffer start
                out_az = np.empty(n_out, dtype=np.int64)
                out_bz = np.empty(n_out, dtype=np.int64)
                out_val = np.empty(n_out, dtype=np.float64)
                out_src = np.empty(n_out, dtype=np.int32)
                cl.enqueue_copy(self.queue, out_az, self.out_az_buf, is_blocking=True)
                cl.enqueue_copy(self.queue, out_bz, self.out_bz_buf, is_blocking=True)
                cl.enqueue_copy(self.queue, out_val, self.out_val_buf, is_blocking=True)
                cl.enqueue_copy(self.queue, out_src, self.out_src_buf, is_blocking=True)
                self.queue.finish()
                # Adjust source indices for chunk offset
                out_src = out_src + c0
                all_az.append(out_az)
                all_bz.append(out_bz)
                all_val.append(out_val)
                all_src.append(out_src)

            # Release per-chunk buffers
            azs_buf.release(); bzs_buf.release(); vals_buf.release()

        if all_az:
            return (np.concatenate(all_az), np.concatenate(all_bz),
                    np.concatenate(all_val), np.concatenate(all_src))
        else:
            return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int32))

    def release(self):
        """Free GPU resources."""
        if not self._released:
            for buf in [self.term_kp_buf, self.term_kq_buf, self.term_kr_buf,
                        self.term_ks_buf, self.term_sp_buf, self.term_sq_buf,
                        self.term_sr_buf, self.term_ss_buf, self.term_c_buf,
                        self.out_az_buf, self.out_bz_buf, self.out_val_buf,
                        self.out_src_buf, self.counter_buf]:
                buf.release()
            self._released = True

    def __del__(self):
        self.release()


# ---------------------------------------------------------------------------
# GPUArgsort: stable radix sort carrying index (for residual computation)
# ---------------------------------------------------------------------------

_ARGSORT_BIT_HIST_KERNEL = """
__kernel void argsort_bit_and_hist(
    __global const int* data,
    __global int* bit,
    __global int* h1_buf,
    int bit_idx, int n) {
    int gid = get_global_id(0);
    int local_count = 0;
    for (int i = gid; i < n; i += get_global_size(0)) {
        unsigned int key = (unsigned int)data[i] ^ 0x80000000u;
        int b = (key >> bit_idx) & 1;
        bit[i] = b;
        local_count += b;
    }
    atomic_add(h1_buf, local_count);
}
"""

_ARGSORT_SCATTER_KERNEL = """
__kernel void argsort_scatter(
    __global const int* input_key,
    __global int* output_key,
    __global const int* input_idx,
    __global int* output_idx,
    __global const int* prefix1,
    __global const int* bit,
    __global const int* h1_buf,
    int n) {
    int gid = get_global_id(0);
    int h1 = h1_buf[0];
    int h0 = n - h1;
    for (int i = gid; i < n; i += get_global_size(0)) {
        int b = bit[i];
        int p1 = prefix1[i];
        if (b == 0) {
            int pos = i - p1;
            output_key[pos] = input_key[i];
            output_idx[pos] = input_idx[i];
        } else {
            int pos = h0 + p1 - 1;
            output_key[pos] = input_key[i];
            output_idx[pos] = input_idx[i];
        }
    }
}
"""


class GPUArgsort:
    """GPU stable argsort for int32 arrays. Returns (sorted_keys, sorted_indices).

    [文章 10.88 §7.13] GPU 加速残差计算的核心：稳定 radix sort 同时
    携带 index，返回排序后的 key 和 argsort 置换。用于残差计算中
    重排 value，然后 CPU 做 reduceat 分段求和。

    性能：1340 万元素，CPU argsort 1.0s → GPU 0.25s，残差整体
    1.8s→0.59s，加速 3x。仅支持 int32（max<2^31）。

    算法：32 轮 radix sort，每轮：
    1. bit+hist kernel：计算每个元素的当前 bit，atomic_add 统计 1-bit 总数
    2. GenericScanKernel 前缀和：对 bit 数组做 inclusive prefix sum
    3. scatter kernel：根据前缀和计算输出位置，同时交换 key 和 index
    """

    def __init__(self, ctx, queue, max_n):
        self.ctx = ctx
        self.queue = queue
        self.max_n = max_n
        self._released = False

        import pyopencl.array as cl_array
        from pyopencl.scan import GenericScanKernel

        prg = cl.Program(ctx, _ARGSORT_BIT_HIST_KERNEL + _ARGSORT_SCATTER_KERNEL).build()
        self.bit_hist_kernel = prg.argsort_bit_and_hist
        self.scatter_kernel = prg.argsort_scatter

        self.scan = GenericScanKernel(ctx, np.int32,
            arguments='__global int* x, __global int* y',
            input_expr='x[i]',
            scan_expr='a+b',
            output_statement='y[i] = item',
            neutral='0')

        self.bit_gpu = cl_array.empty(queue, max_n, np.int32)
        self.prefix_gpu = cl_array.empty(queue, max_n, np.int32)
        self.tmp_key_gpu = cl_array.empty(queue, max_n, np.int32)
        self.tmp_idx_gpu = cl_array.empty(queue, max_n, np.int32)
        self.h1_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, 4)

    def argsort(self, keys):
        """Stable argsort. Returns (sorted_keys, sorted_indices)."""
        import pyopencl.array as cl_array
        n = len(keys)
        key_gpu = cl_array.to_device(self.queue, keys.astype(np.int32))
        idx_gpu = cl_array.to_device(self.queue, np.arange(n, dtype=np.int32))
        global_size = (min(n, 65536),)

        for bit_idx in range(32):
            cl.enqueue_copy(self.queue, self.h1_buf, np.zeros(1, dtype=np.int32))
            self.bit_hist_kernel(self.queue, global_size, None,
                                key_gpu.data, self.bit_gpu.data, self.h1_buf,
                                np.int32(bit_idx), np.int32(n))
            self.scan(self.bit_gpu, self.prefix_gpu, queue=self.queue, size=n)
            self.scatter_kernel(self.queue, global_size, None,
                               key_gpu.data, self.tmp_key_gpu.data,
                               idx_gpu.data, self.tmp_idx_gpu.data,
                               self.prefix_gpu.data, self.bit_gpu.data,
                               self.h1_buf, np.int32(n))
            self.queue.finish()
            key_gpu, self.tmp_key_gpu = self.tmp_key_gpu, key_gpu
            idx_gpu, self.tmp_idx_gpu = self.tmp_idx_gpu, idx_gpu

        sorted_keys = key_gpu.get()
        sorted_indices = idx_gpu.get()
        return sorted_keys, sorted_indices

    def release(self):
        """Free GPU resources."""
        if not self._released:
            for buf in [self.h1_buf]:
                buf.release()
            for arr in [self.bit_gpu, self.prefix_gpu, self.tmp_key_gpu, self.tmp_idx_gpu]:
                arr.data.release()
            self._released = True

    def __del__(self):
        self.release()


# ---------------------------------------------------------------------------
# GPUArgsort64: stable radix sort for int64 (for dim > 2^31 systems)
# ---------------------------------------------------------------------------

_ARGSORT64_BIT_HIST_KERNEL = """
__kernel void argsort64_bit_and_hist(
    __global const long* data,
    __global int* bit,
    __global int* h1_buf,
    int bit_idx, int n) {
    int gid = get_global_id(0);
    int local_count = 0;
    for (int i = gid; i < n; i += get_global_size(0)) {
        unsigned long key = (unsigned long)data[i] ^ 0x8000000000000000UL;
        int b = (int)((key >> bit_idx) & 1UL);
        bit[i] = b;
        local_count += b;
    }
    atomic_add(h1_buf, local_count);
}
"""

_ARGSORT64_SCATTER_KERNEL = """
__kernel void argsort64_scatter(
    __global const long* input_key,
    __global long* output_key,
    __global const int* input_idx,
    __global int* output_idx,
    __global const int* prefix1,
    __global const int* bit,
    __global const int* h1_buf,
    int n) {
    int gid = get_global_id(0);
    int h1 = h1_buf[0];
    int h0 = n - h1;
    for (int i = gid; i < n; i += get_global_size(0)) {
        int b = bit[i];
        int p1 = prefix1[i];
        if (b == 0) {
            int pos = i - p1;
            output_key[pos] = input_key[i];
            output_idx[pos] = input_idx[i];
        } else {
            int pos = h0 + p1 - 1;
            output_key[pos] = input_key[i];
            output_idx[pos] = input_idx[i];
        }
    }
}
"""


class GPUArgsort64:
    """GPU stable argsort for int64 arrays. Returns (sorted_keys, sorted_indices).

    [文章 10.88 §7.13] 用于 dim > 2^31 的大系统（如 NH₃/cc-pVDZ, dim=1.41×10^10）。
    64 轮 radix sort，kernel 用 long 类型。index 仍用 int32（N_elem < 2^31）。

    性能：NH₃/cc-pVDZ 第2轮残差 5000 万元素，CPU argsort ~10s → GPU ~3s。
    """

    def __init__(self, ctx, queue, max_n):
        self.ctx = ctx
        self.queue = queue
        self.max_n = max_n
        self._released = False

        import pyopencl.array as cl_array
        from pyopencl.scan import GenericScanKernel

        prg = cl.Program(ctx, _ARGSORT64_BIT_HIST_KERNEL + _ARGSORT64_SCATTER_KERNEL).build()
        self.bit_hist_kernel = prg.argsort64_bit_and_hist
        self.scatter_kernel = prg.argsort64_scatter

        self.scan = GenericScanKernel(ctx, np.int32,
            arguments='__global int* x, __global int* y',
            input_expr='x[i]',
            scan_expr='a+b',
            output_statement='y[i] = item',
            neutral='0')

        self.bit_gpu = cl_array.empty(queue, max_n, np.int32)
        self.prefix_gpu = cl_array.empty(queue, max_n, np.int32)
        self.key_gpu = cl_array.empty(queue, max_n, np.int64)
        self.idx_gpu = cl_array.empty(queue, max_n, np.int32)
        self.tmp_key_gpu = cl_array.empty(queue, max_n, np.int64)
        self.tmp_idx_gpu = cl_array.empty(queue, max_n, np.int32)
        self.h1_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, 4)

    def argsort(self, keys):
        """Stable argsort for int64. Returns (sorted_keys, sorted_indices)."""
        import pyopencl.array as cl_array
        n = len(keys)
        # Copy data into pre-allocated buffers (avoid runtime allocation)
        self.key_gpu[:n].set(keys.astype(np.int64))
        self.idx_gpu[:n].set(np.arange(n, dtype=np.int32))
        key_gpu = self.key_gpu
        idx_gpu = self.idx_gpu
        global_size = (min(n, 65536),)

        for bit_idx in range(64):
            cl.enqueue_copy(self.queue, self.h1_buf, np.zeros(1, dtype=np.int32))
            self.bit_hist_kernel(self.queue, global_size, None,
                                key_gpu.data, self.bit_gpu.data, self.h1_buf,
                                np.int32(bit_idx), np.int32(n))
            self.scan(self.bit_gpu, self.prefix_gpu, queue=self.queue, size=n)
            self.scatter_kernel(self.queue, global_size, None,
                               key_gpu.data, self.tmp_key_gpu.data,
                               idx_gpu.data, self.tmp_idx_gpu.data,
                               self.prefix_gpu.data, self.bit_gpu.data,
                               self.h1_buf, np.int32(n))
            self.queue.finish()
            key_gpu, self.tmp_key_gpu = self.tmp_key_gpu, key_gpu
            idx_gpu, self.tmp_idx_gpu = self.tmp_idx_gpu, idx_gpu

        sorted_keys = key_gpu[:n].get()
        sorted_indices = idx_gpu[:n].get()
        return sorted_keys, sorted_indices

    def release(self):
        """Free GPU resources."""
        if not self._released:
            for buf in [self.h1_buf]:
                buf.release()
            for arr in [self.bit_gpu, self.prefix_gpu, self.key_gpu, self.idx_gpu,
                        self.tmp_key_gpu, self.tmp_idx_gpu]:
                arr.data.release()
            self._released = True

    def __del__(self):
        self.release()


__all__ = ["GPUMatrix", "GPUApply", "GPUArgsort", "GPUArgsort64", "find_gpu_device", "_HAS_PYOPENCL"]
