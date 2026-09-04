#!/usr/bin/env python3
"""GPU occupancy-aware double excitation kernel (spatial orbital integrals).

Uses spatial orbital integrals (n_orb^4, ~418MB for n_orb=85) instead of
spin-orbital integrals (n_spin^4, ~6.7GB for n_orb=85), fitting in 8GB VRAM.

Coefficient formulas:
  aa/bb: c2 = 2 * (t[p,q,r,s] - t[p,q,s,r])
  ab:    c2 = 2 * t[p,q,r,s]  (exchange term has mismatched spin -> zero)
"""
import numpy as np

try:
    import pyopencl as cl
    _HAS_PYOPENCL = True
except ImportError:
    _HAS_PYOPENCL = False

from geoqc.gpu import _get_global_gpu

_KERNEL = """
__kernel void doubles_occ_aware_sp(
    __global const long* azs,
    __global const long* bzs,
    __global const double* t_sp,       // spatial integrals (n_orb^4)
    __global long* out_az,
    __global long* out_bz,
    __global double* out_val,
    __global int* out_src,
    __global int* counter,
    int n_orb, int n_occ, int S,
    long max_out) {

    int sid = get_group_id(0);
    int tid = get_local_id(0);
    int lsize = get_local_size(0);
    if (sid >= S) return;

    long az = azs[sid];
    long bz = bzs[sid];

    __local int occ_a[80], occ_b[80];
    __local int vir_a[80], vir_b[80];
    __local int n_va, n_vb;

    if (tid == 0) {
        int ia=0, ib=0, va=0, vb=0;
        for (int i = 0; i < n_orb; i++) {
            if (az & (1L << i)) occ_a[ia++] = i; else vir_a[va++] = i;
            if (bz & (1L << i)) occ_b[ib++] = i; else vir_b[vb++] = i;
        }
        n_va = va; n_vb = vb;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int va = n_va, vb = n_vb;
    // Extended virtual lists include r,s (after annihilation they become empty).
    // This automatically covers "quasi-single" terms where p or q equals r or s.
    int nvp_a_ext = (va + 2) * (va + 1) / 2;  // C(va+2, 2)
    int nvp_b_ext = (vb + 2) * (vb + 1) / 2;
    int n_occ_pair = n_occ * (n_occ - 1) / 2;
    int naa = n_occ_pair * nvp_a_ext;
    int nbb = n_occ_pair * nvp_b_ext;
    int nab = n_occ * n_occ * (va + 1) * (vb + 1);
    int total = naa + nbb + nab;
    int n4 = n_orb * n_orb * n_orb * n_orb;
    int n3 = n_orb * n_orb * n_orb;
    int n2 = n_orb * n_orb;

    // Helper: decode pair index k (0..C(n,2)-1) to (i,j), i<j
    // Use for-loop with explicit upper bound (avoids OpenCL while-loop GPU hangs)
    #define DECODE_PAIR(k, n, i, j) do { \\
        i = 0; int _k = (k); \\
        for (int _i = 0; _i < (n) - 1; _i++) { \\
            int _span = (n) - 1 - _i; \\
            if (_k >= _span) { _k -= _span; } \\
            else { i = _i; break; } \\
        } \\
        j = i + 1 + _k; \\
    } while(0)

    for (int idx = tid; idx < total; idx += lsize) {
        int r, s, p, q;
        int sr, ss, sp, sq;  // spin 0=alpha, 1=beta
        long taz = az, tbz = bz;
        double c2;

        if (idx < naa) {
            int k = idx;
            int occ_pair = k / nvp_a_ext;
            int vir_pair = k % nvp_a_ext;
            int ri, rj; DECODE_PAIR(occ_pair, n_occ, ri, rj);
            r = occ_a[ri]; s = occ_a[rj];
            // Extended virtual list: vir_a[0..va-1] + r + s
            int va_ext = va + 2;
            int vi, vj; DECODE_PAIR(vir_pair, va_ext, vi, vj);
            p = (vi < va) ? vir_a[vi] : (vi == va) ? r : s;
            q = (vj < va) ? vir_a[vj] : (vj == va) ? r : s;
            if (p == q) continue;  // a†_p a†_p = 0
            sr = 0; ss = 0; sp = 0; sq = 0;
            c2 = t_sp[p*n3 + s*n2 + q*n_orb + r]
               - t_sp[p*n3 + r*n2 + q*n_orb + s];
            taz = (az & ~(1L << r) & ~(1L << s)) | (1L << p) | (1L << q);
        } else if (idx < naa + nbb) {
            int k = idx - naa;
            int occ_pair = k / nvp_b_ext;
            int vir_pair = k % nvp_b_ext;
            int ri, rj; DECODE_PAIR(occ_pair, n_occ, ri, rj);
            r = occ_b[ri]; s = occ_b[rj];
            int vb_ext = vb + 2;
            int vi, vj; DECODE_PAIR(vir_pair, vb_ext, vi, vj);
            p = (vi < vb) ? vir_b[vi] : (vi == vb) ? r : s;
            q = (vj < vb) ? vir_b[vj] : (vj == vb) ? r : s;
            if (p == q) continue;
            sr = 1; ss = 1; sp = 1; sq = 1;
            c2 = t_sp[p*n3 + s*n2 + q*n_orb + r]
               - t_sp[p*n3 + r*n2 + q*n_orb + s];
            tbz = (bz & ~(1L << r) & ~(1L << s)) | (1L << p) | (1L << q);
        } else {
            int k = idx - naa - nbb;
            int va_ext = va + 1, vb_ext = vb + 1;
            int ri = k / (n_occ * va_ext * vb_ext);
            int rem = k % (n_occ * va_ext * vb_ext);
            int si = rem / (va_ext * vb_ext);
            int rem2 = rem % (va_ext * vb_ext);
            int pi = rem2 / vb_ext;
            int qi = rem2 % vb_ext;
            r = occ_a[ri]; s = occ_b[si];
            p = (pi < va) ? vir_a[pi] : r;
            q = (qi < vb) ? vir_b[qi] : s;
            sr = 0; ss = 1; sp = 0; sq = 1;
            c2 = -t_sp[p*n3 + r*n2 + q*n_orb + s];
            taz = (az & ~(1L << r)) | (1L << p);
            tbz = (bz & ~(1L << s)) | (1L << q);
        }

        if (fabs(c2) < 1e-10) continue;

        // Sign: annihilate s -> annihilate r -> create q -> create p
        long saz = az, sbz = bz;
        double sgn = 1.0;

        // annihilate s
        int ks = s;
        long mask_s = (1L << ks) - 1;
        int cnt_s = popcount((unsigned long)(saz & mask_s)) + popcount((unsigned long)(sbz & mask_s));
        if (ss == 1) cnt_s += (saz >> ks) & 1;  // beta: include alpha bit k
        sgn *= (cnt_s & 1) ? -1.0 : 1.0;
        if (ss == 0) saz ^= (1L << s); else sbz ^= (1L << s);

        // annihilate r
        int kr = r;
        long mask_r = (1L << kr) - 1;
        int cnt_r = popcount((unsigned long)(saz & mask_r)) + popcount((unsigned long)(sbz & mask_r));
        if (sr == 1) cnt_r += (saz >> kr) & 1;
        sgn *= (cnt_r & 1) ? -1.0 : 1.0;
        if (sr == 0) saz ^= (1L << r); else sbz ^= (1L << r);

        // create q
        int kq = q;
        long mask_q = (1L << kq) - 1;
        int cnt_q = popcount((unsigned long)(saz & mask_q)) + popcount((unsigned long)(sbz & mask_q));
        if (sq == 1) cnt_q += (saz >> kq) & 1;
        sgn *= (cnt_q & 1) ? -1.0 : 1.0;
        if (sq == 0) saz ^= (1L << q); else sbz ^= (1L << q);

        // create p
        int kp = p;
        long mask_p = (1L << kp) - 1;
        int cnt_p = popcount((unsigned long)(saz & mask_p)) + popcount((unsigned long)(sbz & mask_p));
        if (sp == 1) cnt_p += (saz >> kp) & 1;
        sgn *= (cnt_p & 1) ? -1.0 : 1.0;

        double val = c2 * sgn;

        int pos = atomic_inc(counter);
        if (pos < max_out) {
            out_az[pos] = taz;
            out_bz[pos] = tbz;
            out_val[pos] = val;
            out_src[pos] = sid;
        }
    }
}
"""


class GPUApplyOccAwareSP:
    """GPU occupancy-aware doubles using spatial orbital integrals."""

    def __init__(self, n_orb, n_occ, t_spatial, eps=1e-10, chunk_size=32):
        if not _HAS_PYOPENCL:
            raise RuntimeError("PyOpenCL not installed (use CPU path: gpu_apply=None)")
        self.n_orb = n_orb
        self.n_occ = n_occ
        self.eps = eps
        self.chunk_size = chunk_size
        self.ctx, self.queue, _ = _get_global_gpu()

        self.prg = cl.Program(self.ctx, _KERNEL).build()
        self.kernel = self.prg.doubles_occ_aware_sp

        mf = cl.mem_flags
        t_flat = t_spatial.real.astype(np.float64).ravel()
        print(f'  Spatial integrals: {len(t_flat):,} elements, {len(t_flat)*8/1e6:.1f} MB')
        self.t_sp_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=t_flat)

        self.max_out = chunk_size * 250000
        self.out_az = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.max_out * 8)
        self.out_bz = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.max_out * 8)
        self.out_val = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.max_out * 8)
        self.out_src = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.max_out * 4)
        self.counter = cl.Buffer(self.ctx, mf.READ_WRITE, 4)

    def doubles(self, azs, bzs, vals):
        S = len(azs)
        all_az, all_bz, all_val, all_src = [], [], [], []

        for c0 in range(0, S, self.chunk_size):
            c1 = min(c0 + self.chunk_size, S)
            chunk_S = c1 - c0

            cl.enqueue_copy(self.queue, self.counter, np.zeros(1, dtype=np.int32))

            mf = cl.mem_flags
            azs_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                 hostbuf=azs[c0:c1].astype(np.int64))
            bzs_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                 hostbuf=bzs[c0:c1].astype(np.int64))

            global_size = (chunk_S * 256,)
            local_size = (256,)
            self.kernel(self.queue, global_size, local_size,
                       azs_buf, bzs_buf, self.t_sp_buf,
                       self.out_az, self.out_bz, self.out_val, self.out_src,
                       self.counter,
                       np.int32(self.n_orb), np.int32(self.n_occ), np.int32(chunk_S),
                       np.int64(self.max_out))
            self.queue.finish()

            count = np.zeros(1, dtype=np.int32)
            cl.enqueue_copy(self.queue, count, self.counter)
            self.queue.finish()
            n_out = min(count[0], self.max_out)

            if n_out > 0:
                out_az = np.empty(n_out, dtype=np.int64)
                out_bz = np.empty(n_out, dtype=np.int64)
                out_val = np.empty(n_out, dtype=np.float64)
                out_src = np.empty(n_out, dtype=np.int32)
                cl.enqueue_copy(self.queue, out_az, self.out_az)
                cl.enqueue_copy(self.queue, out_bz, self.out_bz)
                cl.enqueue_copy(self.queue, out_val, self.out_val)
                cl.enqueue_copy(self.queue, out_src, self.out_src)
                self.queue.finish()

                out_val = out_val * vals[c0 + out_src]
                out_src = out_src + c0

                all_az.append(out_az); all_bz.append(out_bz)
                all_val.append(out_val); all_src.append(out_src)

            azs_buf.release(); bzs_buf.release()

        if all_az:
            return (np.concatenate(all_az), np.concatenate(all_bz),
                    np.concatenate(all_val), np.concatenate(all_src))
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int32))

    def release(self):
        for buf in [self.t_sp_buf, self.out_az, self.out_bz, self.out_val,
                    self.out_src, self.counter]:
            buf.release()


if __name__ == '__main__':
    import time
    from geoqc import exterior
    from geoqc.integrals import spin_orbital_integrals
    from pyscf import gto, scf

    # Test on H2O/cc-pVDZ
    d = np.load('/tmp/_h2o_ccpvdz.npz')
    n_orb, o_s, t_s, nuc = int(d['n']), d['o_s'], d['t_s'], float(d['nuc'])
    ns = 2 * n_orb; nelec = 10

    # Get spatial integrals from pyscf
    mol = gto.M(atom='O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0', basis='cc-pVDZ', verbose=0)
    t_spatial = mol.intor('int2e')

    print(f'H2O/cc-pVDZ: n_orb={n_orb}')
    print(f'Spatial integrals shape: {t_spatial.shape}, size: {t_spatial.nbytes/1e6:.1f} MB')

    rng = np.random.default_rng(42)
    n_test = 100
    azs = np.zeros(n_test, dtype=np.int64)
    bzs = np.zeros(n_test, dtype=np.int64)
    for i in range(n_test):
        occ = rng.choice(n_orb, 5, replace=False); azs[i] = int(np.sum(1 << occ))
        occ = rng.choice(n_orb, 5, replace=False); bzs[i] = int(np.sum(1 << occ))
    vals = np.ones(n_test, dtype=np.float64)

    # Current method (spin-orbital)
    apply_fn, _, _, _, _, _ = exterior.sparse_action_sz_vec(ns, nelec, 0, o_s, t_s, nuc, 1e-10)
    t0 = time.time()
    az_cur, bz_cur, val_cur, src_cur = apply_fn(azs, bzs, vals)
    t_cur = time.time() - t0
    print(f'\\nCurrent (spin-orb, singles+doubles): {len(val_cur):,} outputs, {t_cur:.3f}s')

    # New method (spatial)
    n_occ = nelec // 2
    gpu = GPUApplyOccAwareSP(n_orb, n_occ, t_spatial, eps=1e-10, chunk_size=32)
    gpu.doubles(azs[:10], bzs[:10], vals[:10])  # warmup
    t0 = time.time()
    az_new, bz_new, val_new, src_new = gpu.doubles(azs, bzs, vals)
    t_new = time.time() - t0
    print(f'New (spatial, doubles only): {len(val_new):,} outputs, {t_new:.3f}s')
    print(f'Speedup: {t_cur/t_new:.2f}x')

    # Correctness: compare doubles outputs
    cur_doubles = {}
    for i in range(len(val_cur)):
        az_diff = bin(az_cur[i] ^ azs[src_cur[i]]).count('1')
        bz_diff = bin(bz_cur[i] ^ bzs[src_cur[i]]).count('1')
        if az_diff + bz_diff == 4:
            key = (int(az_cur[i]), int(bz_cur[i]), int(src_cur[i]))
            cur_doubles[key] = cur_doubles.get(key, 0) + val_cur[i].real

    new_doubles = {}
    for i in range(len(val_new)):
        key = (int(az_new[i]), int(bz_new[i]), int(src_new[i]))
        new_doubles[key] = new_doubles.get(key, 0) + val_new[i]

    n_match = 0; max_diff = 0
    for key, val in new_doubles.items():
        if key in cur_doubles:
            diff = abs(val - cur_doubles[key])
            max_diff = max(max_diff, diff)
            if diff < 1e-8: n_match += 1
            else: print(f'  MISMATCH {key}: new={val:.10f} cur={cur_doubles[key]:.10f}')
        else:
            print(f'  EXTRA in new: {key}={val:.10f}')

    print(f'\\nCorrectness: {n_match}/{len(new_doubles)} matched, max_diff={max_diff:.2e}')
    print(f'{"SUCCESS!" if n_match == len(new_doubles) and max_diff < 1e-8 else "FAILED"}')

    gpu.release()
