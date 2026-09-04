"""GPU-accelerated diagonalization benchmark (LIGHTWEIGHT VERSION).

Only compares scipy eigsh (Lanczos) with CPU matvec vs GPU matvec.
No dense eigh (O(V^3) too slow). Max V=10000, n_iter=1 for safety.
"""

import numpy as np
import time
import pyopencl as cl
from scipy.sparse.linalg import eigsh, LinearOperator

GEMV_KERNEL = """
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


class GPUMatrix:
    def __init__(self, H, ctx=None, queue=None):
        self.M, self.N = H.shape
        self.ctx = ctx
        self.queue = queue
        if ctx is None:
            for plat in cl.get_platforms():
                for dev in plat.get_devices():
                    if 'RX 570' in dev.name or 'Radeon' in dev.name:
                        self.ctx = cl.Context([dev])
                        self.queue = cl.CommandQueue(self.ctx)
                        print(f'  GPU: {dev.name}')
        prg = cl.Program(self.ctx, GEMV_KERNEL).build()
        self.kernel = prg.gemv
        mf = cl.mem_flags
        print(f'  Transferring {self.M*self.N*8/1e9:.2f}GB to GPU...')
        t0 = time.time()
        self.H_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=H)
        self.x_buf = cl.Buffer(self.ctx, mf.READ_ONLY, self.N * 8)
        self.y_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.M * 8)
        self.queue.finish()
        print(f'  Transfer: {time.time()-t0:.2f}s')
        self.local_size = (min(256, self.M),)
        self.global_size = (((self.M + 255) // 256) * 256,)
        self.y_cpu = np.empty(self.M, dtype=np.float64)

    def matvec(self, x):
        cl.enqueue_copy(self.queue, self.x_buf, x)
        self.kernel(self.queue, self.global_size, self.local_size,
                    self.H_buf, self.x_buf, self.y_buf,
                    np.int32(self.M), np.int32(self.N))
        cl.enqueue_copy(self.queue, self.y_cpu, self.y_buf)
        self.queue.finish()
        return self.y_cpu.copy()

    def release(self):
        self.H_buf.release(); self.x_buf.release(); self.y_buf.release()


def generate_matrix(V):
    print(f'  Generating {V}x{V} matrix...')
    t0 = time.time()
    A = np.zeros((V, V), dtype=np.float64)
    chunk = 2000
    for i in range(0, V, chunk):
        i_end = min(i + chunk, V)
        rows = np.arange(i, i_end)[:, None]
        cols = np.arange(V)[None, :]
        A[i:i_end, :] = 1.0 / (1.0 + np.abs(rows - cols))
    A = (A + A.T) / 2
    A += np.eye(V) * V
    print(f'  Generated in {time.time()-t0:.1f}s')
    return A


def benchmark(V):
    print(f'\n{"="*50}')
    print(f'V={V} (matrix={V*V*8/1e9:.2f}GB)')
    print(f'{"="*50}')

    H = generate_matrix(V)

    # CPU Lanczos
    print(f'\n--- CPU eigsh (Lanczos, k=3, ncv=20) ---')
    H_op = LinearOperator((V, V), matvec=lambda x: H @ x, dtype=float)
    # warmup
    E_cpu, _ = eigsh(H_op, k=3, which='SA', tol=1e-8, ncv=20)
    t0 = time.time()
    E_cpu, _ = eigsh(H_op, k=3, which='SA', tol=1e-8, ncv=20)
    t_cpu = time.time() - t0
    e0_cpu = np.min(E_cpu)
    print(f'  CPU: {t_cpu*1000:.1f}ms, E0={e0_cpu:.6f}')

    time.sleep(3)  # cooldown

    # GPU Lanczos
    print(f'\n--- GPU eigsh (Lanczos, GPU matvec) ---')
    gpu_H = GPUMatrix(H)
    H_op_gpu = LinearOperator((V, V), matvec=gpu_H.matvec, dtype=float)
    # warmup
    E_gpu, _ = eigsh(H_op_gpu, k=3, which='SA', tol=1e-8, ncv=20)
    t0 = time.time()
    E_gpu, _ = eigsh(H_op_gpu, k=3, which='SA', tol=1e-8, ncv=20)
    t_gpu = time.time() - t0
    e0_gpu = np.min(E_gpu)
    err = abs(e0_cpu - e0_gpu)
    speedup = t_cpu / t_gpu
    print(f'  GPU: {t_gpu*1000:.1f}ms, E0={e0_gpu:.6f}, err={err:.2e}')
    print(f'  Speedup: {speedup:.2f}x')

    gpu_H.release()
    del H
    import gc; gc.collect()
    time.sleep(3)

    return {'V': V, 'cpu': t_cpu, 'gpu': t_gpu, 'speedup': speedup, 'err': err}


def main():
    print('='*50)
    print('GPU Diagonalization Benchmark (Lanczos only)')
    print('='*50)

    results = []
    for V in [5000, 10000]:
        results.append(benchmark(V))

    print(f'\n{"="*50}')
    print('SUMMARY')
    print(f'{"="*50}')
    print(f'{"V":>8} {"CPU(ms)":>10} {"GPU(ms)":>10} {"Speedup":>10} {"Err":>10}')
    print('-'*50)
    for r in results:
        print(f'{r["V"]:>8} {r["cpu"]*1000:>9.1f} {r["gpu"]*1000:>9.1f} '
              f'{r["speedup"]:>9.2f}x {r["err"]:>10.2e}')


if __name__ == '__main__':
    main()
