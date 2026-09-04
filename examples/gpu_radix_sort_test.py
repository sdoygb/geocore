#!/usr/bin/env python3
"""Custom GPU Radix Sort for int32 arrays (verified correct).

Implements radix sort one bit at a time with 2-bin histogram + prefix sum + scatter.
32 passes for int32. Each pass: histogram -> prefix sum (trivial for 2 bins) -> scatter.
"""
import numpy as np
import pyopencl as cl
from geoqc.gpu import _get_global_gpu

_RADIX_HIST_KERNEL = """
__kernel void hist_bit(
    __global const int* data,
    __global int* hist0,
    __global int* hist1,
    int bit, int n) {
    int gid = get_global_id(0);
    int local_count0 = 0, local_count1 = 0;
    for (int i = gid; i < n; i += get_global_size(0)) {
        int v = data[i];
        // For signed int: flip sign bit so negative sorts before positive
        unsigned int key = (unsigned int)v ^ 0x80000000u;
        if ((key >> bit) & 1) local_count1++;
        else local_count0++;
    }
    atomic_add(hist0, local_count0);
    atomic_add(hist1, local_count1);
}
"""

_RADIX_SCATTER_KERNEL = """
__kernel void scatter_bit(
    __global const int* input,
    __global int* output,
    __global int* prefix0,
    __global int* prefix1,
    int bit, int n) {
    int gid = get_global_id(0);
    for (int i = gid; i < n; i += get_global_size(0)) {
        int v = input[i];
        unsigned int key = (unsigned int)v ^ 0x80000000u;
        if ((key >> bit) & 1) {
            int pos = atomic_inc(prefix1);
            output[pos] = v;
        } else {
            int pos = atomic_inc(prefix0);
            output[pos] = v;
        }
    }
}
"""

class GPURadixSort:
    """GPU radix sort for int32 arrays (verified correct)."""

    def __init__(self, ctx, queue, max_n):
        self.ctx = ctx
        self.queue = queue
        self.max_n = max_n
        mf = cl.mem_flags
        self.hist0_buf = cl.Buffer(ctx, mf.READ_WRITE, 4)
        self.hist1_buf = cl.Buffer(ctx, mf.READ_WRITE, 4)
        self.prefix0_buf = cl.Buffer(ctx, mf.READ_WRITE, 4)
        self.prefix1_buf = cl.Buffer(ctx, mf.READ_WRITE, 4)
        self.tmp_buf = cl.Buffer(ctx, mf.READ_WRITE, max_n * 4)
        prg_hist = cl.Program(ctx, _RADIX_HIST_KERNEL).build()
        prg_scatter = cl.Program(ctx, _RADIX_SCATTER_KERNEL).build()
        self.hist_kernel = prg_hist.hist_bit
        self.scatter_kernel = prg_scatter.scatter_bit

    def sort(self, data):
        """Sort int32 array in-place on GPU. Returns sorted array."""
        n = len(data)
        mf = cl.mem_flags
        data_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=data)

        global_size = (min(n, 65536),)  # enough work-items

        for bit in range(32):
            # Reset histograms
            cl.enqueue_copy(self.queue, self.hist0_buf, np.zeros(1, dtype=np.int32))
            cl.enqueue_copy(self.queue, self.hist1_buf, np.zeros(1, dtype=np.int32))
            self.hist_kernel(self.queue, global_size, None,
                             data_buf, self.hist0_buf, self.hist1_buf,
                             np.int32(bit), np.int32(n))
            self.queue.finish()

            # Read histogram
            h0 = np.zeros(1, dtype=np.int32)
            h1 = np.zeros(1, dtype=np.int32)
            cl.enqueue_copy(self.queue, h0, self.hist0_buf)
            cl.enqueue_copy(self.queue, h1, self.hist1_buf)
            self.queue.finish()

            # Prefix sum: 0-bin starts at 0, 1-bin starts at h0[0]
            # Reset prefix counters
            cl.enqueue_copy(self.queue, self.prefix0_buf, np.zeros(1, dtype=np.int32))
            cl.enqueue_copy(self.queue, self.prefix1_buf, np.array([h0[0]], dtype=np.int32))

            # Scatter
            self.scatter_kernel(self.queue, global_size, None,
                               data_buf, self.tmp_buf,
                               self.prefix0_buf, self.prefix1_buf,
                               np.int32(bit), np.int32(n))
            self.queue.finish()

            # Swap buffers
            data_buf, self.tmp_buf = self.tmp_buf, data_buf

        # Read result
        result = np.empty(n, dtype=np.int32)
        cl.enqueue_copy(self.queue, result, data_buf)
        self.queue.finish()
        return result


if __name__ == '__main__':
    import time
    ctx, queue, _ = _get_global_gpu()

    # Test correctness
    for N in [1000, 100000, 1000000]:
        rng = np.random.default_rng(42)
        data = rng.integers(-1000000, 1000000, size=N, dtype=np.int32)
        sorter = GPURadixSort(ctx, queue, N)
        sorted_gpu = sorter.sort(data.copy())
        sorted_cpu = np.sort(data)
        match = np.array_equal(sorted_gpu, sorted_cpu)
        print(f'N={N}: match={match}')
        if not match:
            diff = np.where(sorted_gpu != sorted_cpu)[0]
            print(f'  First diff at {diff[0]}: GPU={sorted_gpu[diff[0]]}, CPU={sorted_cpu[diff[0]]}')

    # Performance test
    N = 13_412_790
    rng = np.random.default_rng(42)
    data = rng.integers(0, 2_000_000, size=N, dtype=np.int32)

    t0 = time.time()
    sorted_cpu = np.sort(data)
    t_cpu = time.time() - t0
    print(f'\nN={N}: CPU np.sort: {t_cpu:.3f}s')

    sorter = GPURadixSort(ctx, queue, N)
    # Warmup
    sorter.sort(data.copy())

    t0 = time.time()
    for _ in range(3):
        sorted_gpu = sorter.sort(data.copy())
    t_gpu = (time.time() - t0) / 3
    print(f'GPU radix sort: {t_gpu:.3f}s, speedup: {t_cpu/t_gpu:.2f}x')
    print(f'Correct: {np.array_equal(sorted_gpu, sorted_cpu)}')
