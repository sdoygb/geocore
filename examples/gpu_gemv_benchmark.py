"""OpenCL GEMV benchmark for WCI matvec acceleration (SAFE VERSION).

Fixes from previous version that caused thermal runaway:
- Max V limited to 15000 (1.8GB matrix, safe for 16GB system)
- Reduced iterations (n_warmup=1, n_iter=3)
- 3-second cooldown between each benchmark
- Matrix generated with simple pattern instead of 9e8 random numbers
- Memory usage checked before each test
"""

import numpy as np
import time
import sys
import pyopencl as cl

# ---------------------------------------------------------------------------
# Safety limits
# ---------------------------------------------------------------------------
MAX_V = 15000          # 15000x15000 float64 = 1.8GB (safe)
N_WARMUP = 1
N_ITER = 3
COOLDOWN = 3.0         # seconds between benchmarks

# ---------------------------------------------------------------------------
# OpenCL GEMV kernels
# ---------------------------------------------------------------------------

GEMV_NAIVE = """
__kernel void gemv_naive(
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

GEMV_LOCAL = """
__kernel void gemv_local(
    __global const double *A,
    __global const double *x,
    __global double *y,
    int M, int N,
    __local double *x_local)
{
    int row = get_global_id(0);
    int lid = get_local_id(0);
    int lsize = get_local_size(0);

    for (int i = lid; i < N; i += lsize) {
        x_local[i] = x[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (row < M) {
        double sum = 0.0;
        for (int col = 0; col < N; col++) {
            sum += A[row * N + col] * x_local[col];
        }
        y[row] = sum;
    }
}
"""


def setup_opencl():
    for plat in cl.get_platforms():
        for dev in plat.get_devices():
            if 'RX 570' in dev.name or 'Radeon' in dev.name:
                ctx = cl.Context([dev])
                queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
                print(f'Using GPU: {dev.name}')
                print(f'  Compute units: {dev.max_compute_units}')
                print(f'  Global memory: {dev.global_mem_size / 1e9:.1f} GB')
                return ctx, queue, dev
    raise RuntimeError('No AMD GPU found')


def check_memory(V):
    """Check if matrix fits comfortably in system memory."""
    matrix_gb = V * V * 8 / 1e9
    # System has 16GB; use 60% as safe limit (Mac purgeable memory can be freed)
    total_gb = 16.0
    safe_limit = total_gb * 0.6
    print(f'  Memory check: matrix={matrix_gb:.2f}GB, safe_limit={safe_limit:.1f}GB')
    return matrix_gb < safe_limit


def generate_test_matrix(M, N):
    """Generate test matrix with simple pattern (avoids 9e8 random numbers)."""
    print(f'  Generating {M}x{N} test matrix...')
    t0 = time.time()
    # Use a simple banded-ish pattern: A[i,j] = 1.0/(1+|i-j|)
    # This is symmetric, diagonally dominant, and fast to generate
    A = np.zeros((M, N), dtype=np.float64)
    # Fill in chunks to avoid memory pressure
    chunk = 1000
    for i in range(0, M, chunk):
        i_end = min(i + chunk, M)
        rows = np.arange(i, i_end)[:, None]
        cols = np.arange(N)[None, :]
        A[i:i_end, :] = 1.0 / (1.0 + np.abs(rows - cols))
    x = np.ones(N, dtype=np.float64)
    print(f'  Generated in {time.time()-t0:.1f}s')
    return A, x


def benchmark_gemv(ctx, queue, dev, M, N):
    print(f'\n=== GEMV Benchmark: M={M}, N={N} ===')
    print(f'  Matrix size: {M*N*8/1e9:.2f} GB (float64)')

    if not check_memory(M):
        print('  SKIPPED: insufficient memory')
        return None

    A, x = generate_test_matrix(M, N)
    y_ref = A @ x

    results = {}

    # --- CPU (numpy) ---
    print(f'  CPU benchmark ({N_ITER} iterations)...')
    for _ in range(N_WARMUP):
        _ = A @ x
    time.sleep(1)  # brief cooldown
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        y_cpu = A @ x
    t_cpu = (time.perf_counter() - t0) / N_ITER
    err_cpu = np.max(np.abs(y_cpu - y_ref))
    print(f'  CPU: {t_cpu*1000:.2f} ms, err={err_cpu:.2e}')
    results['cpu'] = {'time': t_cpu, 'err': err_cpu}

    time.sleep(COOLDOWN)  # let CPU cool down

    # --- GPU (OpenCL) ---
    print(f'  GPU benchmark...')
    mf = cl.mem_flags
    print(f'  Transferring {M*N*8/1e9:.2f}GB to GPU...')
    t0 = time.time()
    A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    y_buf = cl.Buffer(ctx, mf.WRITE_ONLY, y_ref.nbytes)
    transfer_time = time.time() - t0
    print(f'  Transfer took {transfer_time:.2f}s')
    y_gpu = np.empty_like(y_ref)

    global_size = (M,)
    local_size = (min(256, M),)
    # Pad global_size to multiple of local_size (OpenCL requirement)
    padded_global = (((M + local_size[0] - 1) // local_size[0]) * local_size[0],)

    for kernel_name, kernel_source, use_local in [
        ('naive', GEMV_NAIVE, False),
        ('local', GEMV_LOCAL, True),
    ]:
        try:
            print(f'  Compiling {kernel_name} kernel...')
            prg = cl.Program(ctx, kernel_source).build()
            kernel = getattr(prg, f'gemv_{kernel_name}')

            # Warmup
            for _ in range(N_WARMUP):
                if use_local:
                    local_mem = cl.LocalMemory(N * 8)
                    kernel(queue, padded_global, local_size, A_buf, x_buf, y_buf,
                           np.int32(M), np.int32(N), local_mem)
                else:
                    kernel(queue, padded_global, local_size, A_buf, x_buf, y_buf,
                           np.int32(M), np.int32(N))
                cl.enqueue_copy(queue, y_gpu, y_buf)
            queue.finish()

            # Compute only (data on GPU)
            t0 = time.perf_counter()
            for _ in range(N_ITER):
                if use_local:
                    local_mem = cl.LocalMemory(N * 8)
                    kernel(queue, padded_global, local_size, A_buf, x_buf, y_buf,
                           np.int32(M), np.int32(N), local_mem)
                else:
                    kernel(queue, padded_global, local_size, A_buf, x_buf, y_buf,
                           np.int32(M), np.int32(N))
            queue.finish()
            t_gpu_compute = (time.perf_counter() - t0) / N_ITER

            # Compute + transfer back
            t0 = time.perf_counter()
            for _ in range(N_ITER):
                if use_local:
                    local_mem = cl.LocalMemory(N * 8)
                    kernel(queue, padded_global, local_size, A_buf, x_buf, y_buf,
                           np.int32(M), np.int32(N), local_mem)
                else:
                    kernel(queue, padded_global, local_size, A_buf, x_buf, y_buf,
                           np.int32(M), np.int32(N))
                cl.enqueue_copy(queue, y_gpu, y_buf)
            queue.finish()
            t_gpu_total = (time.perf_counter() - t0) / N_ITER

            cl.enqueue_copy(queue, y_gpu, y_buf)
            queue.finish()
            err_gpu = np.max(np.abs(y_gpu - y_ref))

            speedup_compute = t_cpu / t_gpu_compute
            speedup_total = t_cpu / t_gpu_total
            print(f'  GPU ({kernel_name}): compute={t_gpu_compute*1000:.2f}ms, '
                  f'total={t_gpu_total*1000:.2f}ms, err={err_gpu:.2e}, '
                  f'speedup(compute)={speedup_compute:.2f}x, speedup(total)={speedup_total:.2f}x')
            results[f'gpu_{kernel_name}'] = {
                'compute': t_gpu_compute, 'total': t_gpu_total,
                'err': err_gpu, 'speedup_compute': speedup_compute,
                'speedup_total': speedup_total
            }
        except Exception as e:
            print(f'  GPU ({kernel_name}): FAILED - {e}')

        time.sleep(COOLDOWN)  # cooldown between kernels

    # Cleanup
    A_buf.release(); x_buf.release(); y_buf.release()
    del A, x, y_ref, y_cpu, y_gpu
    import gc; gc.collect()
    time.sleep(COOLDOWN)

    return results


def main():
    print('='*60)
    print('OpenCL GEMV Benchmark (SAFE VERSION)')
    print(f'  Max V={MAX_V}, warmup={N_WARMUP}, iter={N_ITER}, cooldown={COOLDOWN}s')
    print('='*60)

    ctx, queue, dev = setup_opencl()

    all_results = {}
    for V in [1000, 5000, 10000, 15000]:
        if V > MAX_V:
            print(f'\nSkipping V={V}: exceeds MAX_V={MAX_V} safety limit')
            continue
        results = benchmark_gemv(ctx, queue, dev, V, V)
        if results:
            all_results[V] = results

    # Summary
    print('\n' + '='*60)
    print('SUMMARY: CPU vs GPU GEMV speedup')
    print('='*60)
    print(f'{"V":>8} {"CPU(ms)":>10} {"GPU(ms)":>10} {"Speedup":>10}')
    print('-'*45)
    for V, res in all_results.items():
        cpu_ms = res['cpu']['time'] * 1000
        if 'gpu_local' in res:
            gpu_ms = res['gpu_local']['total'] * 1000
            speedup = res['gpu_local']['speedup_total']
            print(f'{V:>8} {cpu_ms:>10.2f} {gpu_ms:>10.2f} {speedup:>10.2f}x')
        else:
            print(f'{V:>8} {cpu_ms:>10.2f} {"N/A":>10} {"N/A":>10}')

    print('\nNote: GPU speedup includes data transfer back to CPU.')
    print('In WCI, H matrix stays on GPU between matvecs (compute-only speedup).')


if __name__ == '__main__':
    main()
