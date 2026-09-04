#!/usr/bin/env python3
"""GPU argsort for int32: stable radix sort carrying index alongside key.

Returns (sorted_keys, sorted_indices) where sorted_indices is the argsort
permutation. Used in residual computation to reorder values.

Optimization: sort (key, index) pairs on GPU, then CPU does gather+reduceat.
"""
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.scan import GenericScanKernel
from geoqc.gpu import _get_global_gpu

_BIT_HIST_KERNEL = """
__kernel void bit_and_hist(
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

_SCATTER_KERNEL = """
__kernel void scatter_with_idx(
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
    """GPU stable argsort for int32 arrays. Returns (sorted_keys, sorted_indices)."""

    def __init__(self, ctx, queue, max_n):
        self.ctx = ctx
        self.queue = queue
        self.max_n = max_n

        prg = cl.Program(ctx, _BIT_HIST_KERNEL + _SCATTER_KERNEL).build()
        self.bit_hist_kernel = prg.bit_and_hist
        self.scatter_kernel = prg.scatter_with_idx

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


if __name__ == '__main__':
    import time
    ctx, queue, _ = _get_global_gpu()

    print("=== Correctness ===")
    for N in [100, 10000, 1000000]:
        rng = np.random.default_rng(42)
        keys = rng.integers(-1000000, 1000000, size=N, dtype=np.int32)
        sorter = GPUArgsort(ctx, queue, N)
        sorted_keys, sorted_idx = sorter.argsort(keys.copy())
        # Check sorted_keys is sorted
        keys_sorted = np.sort(keys)
        key_match = np.array_equal(sorted_keys, keys_sorted)
        # Check sorted_idx is valid argsort
        idx_match = np.array_equal(keys[sorted_idx], keys_sorted)
        print(f'N={N:>10}: key_match={key_match}, idx_match={idx_match}')

    print("\n=== Performance: GPU argsort vs CPU argsort ===")
    N = 13_412_790
    rng = np.random.default_rng(42)
    keys = rng.integers(0, 2_000_000, size=N, dtype=np.int32)
    values = rng.random(N).astype(np.float64)

    # CPU full residual
    t0 = time.time()
    for _ in range(3):
        order = np.argsort(keys)
        t_sorted = keys[order]
        v_sorted = values[order]
        diff = np.empty(N, dtype=bool)
        diff[0] = True
        diff[1:] = t_sorted[1:] != t_sorted[:-1]
        u = t_sorted[diff]
        bounds = np.nonzero(diff)[0]
        w = np.add.reduceat(v_sorted, bounds)
    t_cpu = (time.time() - t0) / 3
    print(f'CPU full residual: {t_cpu:.3f}s')

    # GPU argsort + CPU gather+reduceat
    sorter = GPUArgsort(ctx, queue, N)
    sorter.argsort(keys.copy())  # warmup

    t0 = time.time()
    for _ in range(3):
        sorted_keys, sorted_idx = sorter.argsort(keys.copy())
        v_sorted = values[sorted_idx]
        diff = np.empty(N, dtype=bool)
        diff[0] = True
        diff[1:] = sorted_keys[1:] != sorted_keys[:-1]
        u = sorted_keys[diff]
        bounds = np.nonzero(diff)[0]
        w = np.add.reduceat(v_sorted, bounds)
    t_gpu = (time.time() - t0) / 3
    print(f'GPU argsort + CPU rest: {t_gpu:.3f}s')
    print(f'Speedup: {t_cpu/t_gpu:.2f}x')

    # Verify result matches
    order_cpu = np.argsort(keys)
    t_cpu_s = keys[order_cpu]
    v_cpu_s = values[order_cpu]
    diff_cpu = np.empty(N, dtype=bool)
    diff_cpu[0] = True
    diff_cpu[1:] = t_cpu_s[1:] != t_cpu_s[:-1]
    u_cpu = t_cpu_s[diff_cpu]
    b_cpu = np.nonzero(diff_cpu)[0]
    w_cpu = np.add.reduceat(v_cpu_s, b_cpu)
    print(f'Result match: u={np.array_equal(u, u_cpu)}, w={np.allclose(w, w_cpu)}')
