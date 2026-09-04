#!/usr/bin/env python3
"""Stable GPU Radix Sort for int32 arrays using prefix sum (no atomic_inc).

Algorithm per bit (32 passes for int32):
1. bit_kernel: compute each element's current bit (sign bit flipped for negatives)
2. prefix_sum: GenericScanKernel inclusive prefix sum of bit array
3. scatter_kernel: each element goes to position = i - prefix1[i] (bit=0)
   or h0 + prefix1[i] - 1 (bit=1), preserving input order within each group.
"""
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.scan import GenericScanKernel
from geoqc.gpu import _get_global_gpu

_BIT_KERNEL = """
__kernel void compute_bit(
    __global const int* data,
    __global int* bit,
    int bit_idx, int n) {
    int gid = get_global_id(0);
    for (int i = gid; i < n; i += get_global_size(0)) {
        // Flip sign bit so negative ints sort before positive
        unsigned int key = (unsigned int)data[i] ^ 0x80000000u;
        bit[i] = (key >> bit_idx) & 1;
    }
}
"""

_SCATTER_KERNEL = """
__kernel void scatter(
    __global const int* input,
    __global int* output,
    __global const int* prefix1,
    __global const int* bit,
    int h0, int n) {
    int gid = get_global_id(0);
    for (int i = gid; i < n; i += get_global_size(0)) {
        int b = bit[i];
        int p1 = prefix1[i];  // inclusive prefix sum: sum(bit[0..i])
        if (b == 0) {
            // 0-bit elements: position = i - (number of 1-bits before i)
            // number of 1-bits before i = p1 - b = p1 (since b=0)
            int pos = i - p1;
            output[pos] = input[i];
        } else {
            // 1-bit elements: position = h0 + (number of 1-bits before and including i) - 1
            // number of 1-bits before and including i = p1
            int pos = h0 + p1 - 1;
            output[pos] = input[i];
        }
    }
}
"""

class GPURadixSortStable:
    """Stable GPU radix sort for int32 arrays using prefix sum."""

    def __init__(self, ctx, queue, max_n):
        self.ctx = ctx
        self.queue = queue
        self.max_n = max_n

        # Compile kernels
        prg_bit = cl.Program(ctx, _BIT_KERNEL).build()
        prg_scatter = cl.Program(ctx, _SCATTER_KERNEL).build()
        self.bit_kernel = prg_bit.compute_bit
        self.scatter_kernel = prg_scatter.scatter

        # Prefix sum kernel (inclusive)
        self.scan = GenericScanKernel(ctx, np.int32,
            arguments='__global int* x, __global int* y',
            input_expr='x[i]',
            scan_expr='a+b',
            output_statement='y[i] = item',
            neutral='0')

        # Pre-allocate GPU arrays
        self.bit_gpu = cl_array.empty(queue, max_n, np.int32)
        self.prefix_gpu = cl_array.empty(queue, max_n, np.int32)
        self.tmp_gpu = cl_array.empty(queue, max_n, np.int32)

    def sort(self, data):
        """Sort int32 array on GPU. Returns sorted array."""
        n = len(data)
        data_gpu = cl_array.to_device(self.queue, data.astype(np.int32))

        global_size = (min(n, 65536),)

        for bit_idx in range(32):
            # 1. Compute bit values
            self.bit_kernel(self.queue, global_size, None,
                           data_gpu.data, self.bit_gpu.data,
                           np.int32(bit_idx), np.int32(n))
            self.queue.finish()

            # 2. Inclusive prefix sum
            self.scan(self.bit_gpu, self.prefix_gpu, queue=self.queue, size=n)
            self.queue.finish()

            # 3. Read total 1-bit count (last element of inclusive prefix)
            h1 = self.prefix_gpu[n-1:n].get()[0]
            h0 = n - h1

            # 4. Scatter to tmp
            self.scatter_kernel(self.queue, global_size, None,
                               data_gpu.data, self.tmp_gpu.data,
                               self.prefix_gpu.data, self.bit_gpu.data,
                               np.int32(h0), np.int32(n))
            self.queue.finish()

            # 5. Swap
            data_gpu, self.tmp_gpu = self.tmp_gpu, data_gpu

        return data_gpu.get()


if __name__ == '__main__':
    import time
    ctx, queue, _ = _get_global_gpu()

    # Correctness tests
    print("=== Correctness Tests ===")
    for N in [100, 10000, 1000000]:
        rng = np.random.default_rng(42)
        # Test with both positive and negative numbers
        data = rng.integers(-1000000, 1000000, size=N, dtype=np.int32)
        sorter = GPURadixSortStable(ctx, queue, N)
        sorted_gpu = sorter.sort(data.copy())
        sorted_cpu = np.sort(data)
        match = np.array_equal(sorted_gpu, sorted_cpu)
        print(f'N={N:>10}: match={match}')
        if not match:
            diff = np.where(sorted_gpu != sorted_cpu)[0]
            print(f'  First diff at {diff[0]}: GPU={sorted_gpu[diff[0]]}, CPU={sorted_cpu[diff[0]]}')
            print(f'  GPU first 10: {sorted_gpu[:10]}')
            print(f'  CPU first 10: {sorted_cpu[:10]}')

    # Performance test
    print("\n=== Performance Test ===")
    N = 13_412_790
    rng = np.random.default_rng(42)
    data = rng.integers(0, 2_000_000, size=N, dtype=np.int32)

    t0 = time.time()
    sorted_cpu = np.sort(data)
    t_cpu = time.time() - t0
    print(f'CPU np.sort: {t_cpu:.3f}s')

    sorter = GPURadixSortStable(ctx, queue, N)
    # Warmup
    sorter.sort(data.copy())

    t0 = time.time()
    for _ in range(3):
        sorted_gpu = sorter.sort(data.copy())
    t_gpu = (time.time() - t0) / 3
    print(f'GPU radix sort: {t_gpu:.3f}s, speedup: {t_cpu/t_gpu:.2f}x')
    print(f'Correct: {np.array_equal(sorted_gpu, sorted_cpu)}')
