#!/usr/bin/env python3
"""GPU occupancy-aware double excitation kernel.

Each work-group handles one source determinant:
1. Collect occupied (5 alpha + 5 beta) and virtual orbitals
2. Enumerate only non-zero double excitations (r,s in occ, p,q in vir)
3. Each work-item handles a portion of the excitations
4. Atomic counter collects sparse output

This reduces computation by ~228x for n_orb=85 (50M -> 223K terms).
"""
import numpy as np
import pyopencl as cl
from geoqc.gpu import _get_global_gpu

_KERNEL = """
__kernel void doubles_occ_aware(
    __global const long* azs,
    __global const long* bzs,
    __global const double* t_s,
    __global long* out_az,
    __global long* out_bz,
    __global double* out_val,
    __global int* out_src,
    __global int* counter,
    int n_orb, int S, int n_spin,
    long max_out) {

    int sid = get_group_id(0);
    int tid = get_local_id(0);
    int lsize = get_local_size(0);
    if (sid >= S) return;

    long az = azs[sid];
    long bz = bzs[sid];

    // Collect occupied and virtual orbitals
    __local int occ_a[5], occ_b[5];
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
    int nvp_a = va * (va - 1) / 2;  // C(va,2)
    int nvp_b = vb * (vb - 1) / 2;

    // Total doubles: aa + bb + ab
    int naa = 10 * nvp_a;
    int nbb = 10 * nvp_b;
    int nab = 25 * va * vb;
    int total = naa + nbb + nab;

    // Helper: decode pair index (0..C(n,2)-1) to (i,j), i<j
    // i = n - 2 - floor(sqrt(-8*k + 4*n*(n-1) - 7) / 2 - 0.5)
    // Use simpler iterative decode for small n (n<=80)
    // For occ pairs (n=5), use lookup
    // For vir pairs, use formula

    for (int idx = tid; idx < total; idx += lsize) {
        int r, s, p, q;
        int sr, ss, sp, sq;  // spin: 0=alpha, 1=beta
        long taz = az, tbz = bz;
        int type;

        if (idx < naa) {
            type = 0;
            int k = idx;
            int occ_pair = k / nvp_a;
            int vir_pair = k % nvp_a;
            // decode occ_pair (0..9) to (r,s)
            r = occ_a[occ_pair < 4 ? 0 : occ_pair < 7 ? 1 : occ_pair < 9 ? 2 : 3];
            // simpler: use formula
            int oi = occ_pair;
            int ri = 0;
            while (oi >= (4 - ri)) { oi -= (4 - ri); ri++; }
            r = occ_a[ri];
            s = occ_a[ri + 1 + oi];
            sr = 0; ss = 0;
            // decode vir_pair to (p,q)
            int vi = 0;
            int vk = vir_pair;
            while (vk >= (va - 1 - vi)) { vk -= (va - 1 - vi); vi++; }
            p = vir_a[vi];
            q = vir_a[vi + 1 + vk];
            sp = 0; sq = 0;
            taz = az & ~(1L << r) & ~(1L << s) | (1L << p) | (1L << q);
        } else if (idx < naa + nbb) {
            type = 1;
            int k = idx - naa;
            int occ_pair = k / nvp_b;
            int vir_pair = k % nvp_b;
            int ri = 0; int oi = occ_pair;
            while (oi >= (4 - ri)) { oi -= (4 - ri); ri++; }
            r = occ_b[ri]; s = occ_b[ri + 1 + oi];
            sr = 1; ss = 1;
            int vi = 0; int vk = vir_pair;
            while (vk >= (vb - 1 - vi)) { vk -= (vb - 1 - vi); vi++; }
            p = vir_b[vi]; q = vir_b[vi + 1 + vk];
            sp = 1; sq = 1;
            tbz = bz & ~(1L << r) & ~(1L << s) | (1L << p) | (1L << q);
        } else {
            type = 2;
            int k = idx - naa - nbb;
            // ab: r in occ_a(5), s in occ_b(5), p in vir_a(va), q in vir_b(vb)
            int ri = k / (5 * va * vb);
            int rem = k % (5 * va * vb);
            int si = rem / (va * vb);
            int rem2 = rem % (va * vb);
            int pi = rem2 / vb;
            int qi = rem2 % vb;
            r = occ_a[ri]; s = occ_b[si];
            p = vir_a[pi]; q = vir_b[qi];
            sr = 0; ss = 1; sp = 0; sq = 1;
            taz = az & ~(1L << r) | (1L << p);
            tbz = bz & ~(1L << s) | (1L << q);
        }

        // Spin-orbital indices
        int isp = 2*p + sp;
        int isq = 2*q + sq;
        int isr = 2*r + sr;
        int iss = 2*s + ss;

        // Coefficient: 2 * (<pq|rs> - <pq|sr>)
        double c2 = 2.0 * (t_s[isp*n_spin*n_spin*n_spin + isq*n_spin*n_spin + isr*n_spin + iss]
                          - t_s[isp*n_spin*n_spin*n_spin + isq*n_spin*n_spin + iss*n_spin + isr]);

        if (fabs(c2) < 1e-10) continue;

        // Sign: annihilate s -> annihilate r -> create q -> create p
        // Use popcount-based sign
        long saz = az, sbz = bz;
        double sgn = 1.0;

        // annihilate s
        int cnt_s = popcount((unsigned long)(saz & ((1L << (2*s+ss)) - 1)) + (unsigned long)(sbz & ((1L << (2*s+ss)) - 1)));
        // Actually sign uses spatial orbital index, not spin-orbital
        // _spin_sign: alpha_k = popcount(az & ((1<<k)-1)) + popcount(bz & ((1<<k)-1))
        // beta_k = popcount(az & ((1<<(k+1))-1)) + popcount(bz & ((1<<k)-1))
        int k_s = s;
        int mask_s_alpha = (1 << k_s) - 1;
        int cnt_s_a = popcount((unsigned int)(saz & mask_s_alpha));
        int cnt_s_b = popcount((unsigned int)(sbz & mask_s_alpha));
        if (ss == 1) {
            // beta: include alpha bit k_s
            cnt_s_a += (saz >> k_s) & 1;
        }
        sgn *= ((cnt_s_a + cnt_s_b) & 1) ? -1.0 : 1.0;
        if (ss == 0) saz ^= (1L << s); else sbz ^= (1L << s);

        // annihilate r
        int k_r = r;
        int mask_r = (1 << k_r) - 1;
        int cnt_r_a = popcount((unsigned int)(saz & mask_r));
        int cnt_r_b = popcount((unsigned int)(sbz & mask_r));
        if (sr == 1) cnt_r_a += (saz >> k_r) & 1;
        sgn *= ((cnt_r_a + cnt_r_b) & 1) ? -1.0 : 1.0;
        if (sr == 0) saz ^= (1L << r); else sbz ^= (1L << r);

        // create q
        int k_q = q;
        int mask_q = (1 << k_q) - 1;
        int cnt_q_a = popcount((unsigned int)(saz & mask_q));
        int cnt_q_b = popcount((unsigned int)(sbz & mask_q));
        if (sq == 1) cnt_q_a += (saz >> k_q) & 1;
        sgn *= ((cnt_q_a + cnt_q_b) & 1) ? -1.0 : 1.0;
        if (sq == 0) saz ^= (1L << q); else sbz ^= (1L << q);

        // create p
        int k_p = p;
        int mask_p = (1 << k_p) - 1;
        int cnt_p_a = popcount((unsigned int)(saz & mask_p));
        int cnt_p_b = popcount((unsigned int)(sbz & mask_p));
        if (sp == 1) cnt_p_a += (saz >> k_p) & 1;
        sgn *= ((cnt_p_a + cnt_p_b) & 1) ? -1.0 : 1.0;

        double val = c2 * sgn;

        // Atomic output
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


class GPUApplyOccAware:
    """GPU occupancy-aware double excitation (only non-zero excitations)."""

    def __init__(self, n_orb, t_s, eps=1e-10, chunk_size=32):
        self.n_orb = n_orb
        self.n_spin = 2 * n_orb
        self.eps = eps
        self.chunk_size = chunk_size
        self.ctx, self.queue, _ = _get_global_gpu()

        # Compile kernel
        self.prg = cl.Program(self.ctx, _KERNEL).build()
        self.kernel = self.prg.doubles_occ_aware

        # Transfer integrals to GPU (take real part)
        mf = cl.mem_flags
        self.t_s_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                  hostbuf=t_s.real.astype(np.float64).ravel())

        # Output buffers (allocated per chunk)
        self.max_out = chunk_size * 250000  # upper bound per chunk
        self.out_az = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.max_out * 8)
        self.out_bz = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.max_out * 8)
        self.out_val = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.max_out * 8)
        self.out_src = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.max_out * 4)
        self.counter = cl.Buffer(self.ctx, mf.READ_WRITE, 4)

    def doubles(self, azs, bzs, vals):
        """Apply double excitations to a batch of source determinants."""
        S = len(azs)
        all_az = []
        all_bz = []
        all_val = []
        all_src = []

        for c0 in range(0, S, self.chunk_size):
            c1 = min(c0 + self.chunk_size, S)
            chunk_S = c1 - c0

            # Reset counter
            cl.enqueue_copy(self.queue, self.counter, np.zeros(1, dtype=np.int32))

            # Transfer source states
            mf = cl.mem_flags
            azs_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                 hostbuf=azs[c0:c1].astype(np.int64))
            bzs_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                 hostbuf=bzs[c0:c1].astype(np.int64))

            # Launch kernel: one work-group per source
            global_size = (chunk_S * 256,)
            local_size = (256,)
            self.kernel(self.queue, global_size, local_size,
                       azs_buf, bzs_buf, self.t_s_buf,
                       self.out_az, self.out_bz, self.out_val, self.out_src,
                       self.counter,
                       np.int32(self.n_orb), np.int32(chunk_S), np.int32(self.n_spin),
                       np.int64(self.max_out))
            self.queue.finish()

            # Read counter
            count = np.zeros(1, dtype=np.int32)
            cl.enqueue_copy(self.queue, count, self.counter)
            self.queue.finish()
            n_out = min(count[0], self.max_out)

            if n_out > 0:
                # Allocate exact-size hostbufs (enqueue_copy reads hostbuf-size bytes)
                out_az = np.empty(n_out, dtype=np.int64)
                out_bz = np.empty(n_out, dtype=np.int64)
                out_val = np.empty(n_out, dtype=np.float64)
                out_src = np.empty(n_out, dtype=np.int32)
                cl.enqueue_copy(self.queue, out_az, self.out_az)
                cl.enqueue_copy(self.queue, out_bz, self.out_bz)
                cl.enqueue_copy(self.queue, out_val, self.out_val)
                cl.enqueue_copy(self.queue, out_src, self.out_src)
                self.queue.finish()

                # Apply source values
                out_val = out_val * vals[c0 + out_src]
                # Adjust source indices to global
                out_src = out_src + c0

                all_az.append(out_az)
                all_bz.append(out_bz)
                all_val.append(out_val)
                all_src.append(out_src)

            azs_buf.release(); bzs_buf.release()

        if all_az:
            return (np.concatenate(all_az), np.concatenate(all_bz),
                    np.concatenate(all_val), np.concatenate(all_src))
        else:
            return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int32))

    def release(self):
        for buf in [self.t_s_buf, self.out_az, self.out_bz, self.out_val,
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
    ns = 2 * n_orb
    nelec = 10

    print(f'H2O/cc-pVDZ: n_orb={n_orb}, ns={ns}')

    # Create some random source states
    rng = np.random.default_rng(42)
    n_test = 100
    azs = np.zeros(n_test, dtype=np.int64)
    bzs = np.zeros(n_test, dtype=np.int64)
    for i in range(n_test):
        occ = rng.choice(n_orb, 5, replace=False)
        azs[i] = int(np.sum(1 << occ))
        occ = rng.choice(n_orb, 5, replace=False)
        bzs[i] = int(np.sum(1 << occ))
    vals = np.ones(n_test, dtype=np.float64)

    # Current method
    apply_fn, _, _, _, _, _ = exterior.sparse_action_sz_vec(ns, nelec, 0, o_s, t_s, nuc, 1e-10)
    t0 = time.time()
    az_cur, bz_cur, val_cur, src_cur = apply_fn(azs, bzs, vals)
    t_cur = time.time() - t0
    print(f'Current method: {len(val_cur)} outputs, {t_cur:.3f}s')

    # New method
    gpu = GPUApplyOccAware(n_orb, t_s, eps=1e-10, chunk_size=32)
    t0 = time.time()
    az_new, bz_new, val_new, src_new = gpu.doubles(azs, bzs, vals)
    t_new = time.time() - t0
    print(f'New method (doubles only): {len(val_new)} outputs, {t_new:.3f}s')
    print(f'Speedup: {t_cur/t_new:.2f}x')

    # Compare
    from scipy.sparse import csr_matrix
    # Build comparison dictionaries
    cur_dict = {}
    for i in range(len(val_cur)):
        key = (int(az_cur[i]), int(bz_cur[i]), int(src_cur[i]))
        cur_dict[key] = cur_dict.get(key, 0) + val_cur[i].real

    new_dict = {}
    for i in range(len(val_new)):
        key = (int(az_new[i]), int(bz_new[i]), int(src_new[i]))
        new_dict[key] = new_dict.get(key, 0) + val_new[i]

    # Check if new outputs are subset of current (new only has doubles, current has singles+doubles)
    n_match = 0
    max_diff = 0
    for key, val in new_dict.items():
        if key in cur_dict:
            diff = abs(val - cur_dict[key])
            max_diff = max(max_diff, diff)
            if diff < 1e-8:
                n_match += 1
            else:
                print(f'  MISMATCH at {key}: new={val:.10f}, cur={cur_dict[key]:.10f}')
        else:
            print(f'  EXTRA in new: {key}={val:.10f}')

    print(f'Matched: {n_match}/{len(new_dict)}')
    print(f'Max diff: {max_diff:.2e}')

    gpu.release()
