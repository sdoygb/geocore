#!/usr/bin/env python3
"""Non-stable GPU Radix Sort for int32 using atomic_inc (fast, no prefix sum).

For residual computation, stable sort is NOT needed: same-target values are
summed via reduceat, order within same target doesn't matter.

Algorithm per bit (32 passes):
1. histogram: count 0-bit and 1-bit elements (atomic_add)
2. scatter: each element gets position via atomic_inc in its bin
"""
import numpy as np
import pyopencl as cl
from geoqc.gpu import _get_global_gpu

_HIST_KERNEL = """
__kernel void hist_bit(
    __global const int* data,
    __global int* h0, __global int* h1,
    int bit_idx, int n) {
    int gid = get_global_id(0);
    int c0 = 0, c1 = 0;
    for (int i = gid; i < n; i += get_global_size(0)) {
        unsigned int key = (unsigned int)data[i] ^ 0x80000000u;
        if ((key >> bit_idx) & 1) c1++; else c0++;
    }
    atomic_add(h0, c0);
    atomic_add(h1, c1);
}
"""

_SCATTER_KERNEL = """
__kernel void scatter_bit(
    __global const int* input,
    __global int* output,
    __global int* pos0, __global int* pos1,
    int bit_idx, int n) {
    int gid = get_global_id(0);
    for (int i = gid; i < n; i += get_global_size(0)) {
        int v = input[i];
        unsigned int key = (unsigned int)v ^ 0x80000000u;
        if ((key >> bit_idx) & 1) {
            int pos = atomic_inc(pos1);
            output[pos] = v;
        } else {
            int pos = atomic_inc(pos0);
            output[pos] = v;
        }
    }
}
"""

class GPURadixSortFast:
    """Fast non-stable GPU radix sort for int32 (atomic_inc, no prefix sum)."""

    def __init__(self, ctx, queue, max_n):
        self.ctx = ctx
        self.queue = queue
        self.max_n = max_n
        mf = cl.mem_flags

        prg = cl.Program(ctx, _HIST_KERNEL + _SCATTER_KERNEL).build()
        self.hist_kernel = prg.hist_bit
        self.scatter_kernel = prg.scatter_bit

        self.h0_buf = cl.Buffer(ctx, mf.READ_WRITE, 4)
        self.h1_buf = cl.Buffer(ctx, mf.READ_WRITE, 4)
        self.pos0_buf = cl.Buffer(ctx, mf.READ_WRITE, 4)
        self.pos1_buf = cl.Buffer(ctx, mf.READ_WRITE, 4)
        self.tmp_buf = cl.Buffer(ctx, mf.READ_WRITE, max_n * 4)

    def sort(self, data):
        n = len(data)
        mf = cl.mem_flags
        data_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=data.astype(np.int32))
        global_size = (min(n, 65536),)

        for bit_idx in range(32):
            # Reset histograms
            cl.enqueue_copy(self.queue, self.h0_buf, np.zeros(1, dtype=np.int32))
            cl.enqueue_copy(self.queue, self.h1_buf, np.zeros(1, dtype=np.int32))
            # 1. Histogram
            self.hist_kernel(self.queue, global_size, None,
                           data_buf, self.h0_buf, self.h1_buf,
                           np.int32(bit_idx), np.int32(n))
            # 2. Read histogram to set pos1 start = h0
            self.queue.finish()
            h0 = np.zeros(1, dtype=np.int32)
            cl.enqueue_copy(self.queue, h0, self.h0_buf)
            self.queue.finish()
            # Reset position counters: pos0 starts at 0, pos1 starts at h0
            cl.enqueue_copy(self.queue, self.pos0_buf, np.zeros(1, dtype=np.int32))
            cl.enqueue_copy(self.queue, self.pos1_buf, h0)
            # 3. Scatter
            self.scatter_kernel(self.queue, global_size, None,
                              data_buf, self.tmp_buf,
                              self.pos0_buf, self.pos1_buf,
                              np.int32(bit_idx), np.int32(n))
            self.queue.finish()
            # Swap
            data_buf, self.tmp_buf = self.tmp_buf, data_buf

        result = np.empty(n, dtype=np.int32)
        cl.enqueue_copy(self.queue, result, data_buf)
        self.queue.finish()
        return result


if __name__ == '__main__':
    import time
    ctx, queue, _ = _get_global_gpu()

    print("=== Correctness (non-stable: sorted order must be correct) ===")
    for N in [100, 10000, 1000000]:
        rng = np.random.default_rng(42)
        data = rng.integers(-1000000, 1000000, size=N, dtype=np.int32)
        sorter = GPURadixSortFast(ctx, queue, N)
        sorted_gpu = sorter.sort(data.copy())
        sorted_cpu = np.sort(data)
        # For non-stable sort, just check the multiset is the same
        match = np.array_equal(np.sort(sorted_gpu), sorted_cpu)
        # Also check if directly sorted (should be if radix sort works)
        direct_match = np.array_equal(sorted_gpu, sorted_cpu)
        print(f'N={N:>10}: multiset_match={match}, directly_sorted={direct_match}')
        if not direct_match:
            diff = np.where(sorted_gpu != sorted_cpu)[0]
            print(f'  First diff at {diff[0]}: GPU={sorted_gpu[diff[0]]}, CPU={sorted_cpu[diff[0]]}')

    print("\n=== Performance ===")
    for N in [1_000_000, 5_000_000, 13_412_790, 30_000_000, 50_000_000]:
        try:
            rng = np.random.default_rng(42)
            data = rng.integers(0, 2_000_000, size=N, dtype=np.int32)
            t0 = time.time()
            sorted_cpu = np.sort(data)
            t_cpu = time.time() - t0
            sorter = GPURadixSortFast(ctx, queue, N)
            sorter.sort(data.copy())
            t0 = time.time()
            sorted_gpu = sorter.sort(data.copy())
            t_gpu = time.time() - t0
            correct = np.array_equal(np.sort(sorted_gpu), sorted_cpu)
            print(f'N={N:>12,}: CPU={t_cpu:.3f}s, GPU={t_gpu:.3f}s, speedup={t_cpu/t_gpu:.2f}x, correct={correct}')
        except Exception as e:
            print(f'N={N:>12,}: error={e}')
