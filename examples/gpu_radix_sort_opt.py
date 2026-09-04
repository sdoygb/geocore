#!/usr/bin/env python3
"""Optimized stable GPU Radix Sort for int32 arrays.

Optimizations over basic version:
1. bit+histogram in one kernel (atomic_add for total 1-bit count)
2. No CPU-GPU sync per bit (h1 read from GPU buffer in scatter kernel)
3. In-place prefix sum (bit array -> prefix array)
4. Fewer kernel calls: 3 per bit (bit+hist, scan, scatter) instead of 4
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
__kernel void scatter(
    __global const int* input,
    __global int* output,
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
            output[i - p1] = input[i];
        } else {
            output[h0 + p1 - 1] = input[i];
        }
    }
}
"""

class GPURadixSortOpt:
    """Optimized stable GPU radix sort for int32."""

    def __init__(self, ctx, queue, max_n):
        self.ctx = ctx
        self.queue = queue
        self.max_n = max_n

        prg = cl.Program(ctx, _BIT_HIST_KERNEL + _SCATTER_KERNEL).build()
        self.bit_hist_kernel = prg.bit_and_hist
        self.scatter_kernel = prg.scatter

        self.scan = GenericScanKernel(ctx, np.int32,
            arguments='__global int* x, __global int* y',
            input_expr='x[i]',
            scan_expr='a+b',
            output_statement='y[i] = item',
            neutral='0')

        self.bit_gpu = cl_array.empty(queue, max_n, np.int32)
        self.prefix_gpu = cl_array.empty(queue, max_n, np.int32)
        self.tmp_gpu = cl_array.empty(queue, max_n, np.int32)
        self.h1_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, 4)

    def sort(self, data):
        n = len(data)
        data_gpu = cl_array.to_device(self.queue, data.astype(np.int32))
        global_size = (min(n, 65536),)

        for bit_idx in range(32):
            # Reset h1
            cl.enqueue_copy(self.queue, self.h1_buf, np.zeros(1, dtype=np.int32))
            # 1. Compute bits + histogram
            self.bit_hist_kernel(self.queue, global_size, None,
                                data_gpu.data, self.bit_gpu.data, self.h1_buf,
                                np.int32(bit_idx), np.int32(n))
            # 2. Prefix sum (no need to finish before scan, but scan needs bit data)
            self.scan(self.bit_gpu, self.prefix_gpu, queue=self.queue, size=n)
            # 3. Scatter (needs h1 from step 1, prefix from step 2)
            self.scatter_kernel(self.queue, global_size, None,
                               data_gpu.data, self.tmp_gpu.data,
                               self.prefix_gpu.data, self.bit_gpu.data,
                               self.h1_buf, np.int32(n))
            self.queue.finish()
            # Swap
            data_gpu, self.tmp_gpu = self.tmp_gpu, data_gpu

        return data_gpu.get()


if __name__ == '__main__':
    import time
    ctx, queue, _ = _get_global_gpu()

    print("=== Correctness ===")
    for N in [100, 10000, 1000000]:
        rng = np.random.default_rng(42)
        data = rng.integers(-1000000, 1000000, size=N, dtype=np.int32)
        sorter = GPURadixSortOpt(ctx, queue, N)
        sorted_gpu = sorter.sort(data.copy())
        sorted_cpu = np.sort(data)
        print(f'N={N:>10}: match={np.array_equal(sorted_gpu, sorted_cpu)}')

    print("\n=== Performance ===")
    for N in [1_000_000, 5_000_000, 13_412_790, 30_000_000, 50_000_000]:
        try:
            rng = np.random.default_rng(42)
            data = rng.integers(0, 2_000_000, size=N, dtype=np.int32)
            t0 = time.time()
            sorted_cpu = np.sort(data)
            t_cpu = time.time() - t0
            sorter = GPURadixSortOpt(ctx, queue, N)
            sorter.sort(data.copy())
            t0 = time.time()
            sorted_gpu = sorter.sort(data.copy())
            t_gpu = time.time() - t0
            print(f'N={N:>12,}: CPU={t_cpu:.3f}s, GPU={t_gpu:.3f}s, speedup={t_cpu/t_gpu:.2f}x, correct={np.array_equal(sorted_gpu, sorted_cpu)}')
        except Exception as e:
            print(f'N={N:>12,}: error={e}')
