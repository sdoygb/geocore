# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython-accelerated double-excitation loop for sparse_action_sz_vec.

Replaces the chunked numpy fancy-index loop in exterior.py with a flat
C-level (source, term) double loop.  Eliminates np.nonzero / np.where /
array-creation overhead; _spin_sign is inlined as a scalar bit operation.

Output contract identical to the numpy version:
    (d_az, d_bz, d_v, d_src)  —  all np.ndarray, same dtypes.
"""

import numpy as np
cimport numpy as cnp
cimport cython

cnp.import_array()

# C compiler builtin for hardware popcount (same as exterior._popcount64)
cdef extern from *:
    int __builtin_popcountll(unsigned long long x)


# ---------------------------------------------------------------------------
# Inlined scalar spin-sign — matches exterior._spin_sign EXACTLY.
#
# Interleaved spin-orbital order: alpha_0, beta_0, alpha_1, beta_1, ...
#   alpha_k: left = alpha_0..alpha_{k-1}, beta_0..beta_{k-1}
#   beta_k:  left = alpha_0..alpha_k,       beta_0..beta_{k-1}
# ---------------------------------------------------------------------------
@cython.inline
cdef inline double _spin_sign_scalar(
    cnp.int64_t az, cnp.int64_t bz, int k, bint is_beta
):
    cdef cnp.int64_t cnt
    if is_beta:
        # beta_k: alpha bits 0..k (inclusive) + beta bits 0..k-1
        cnt = __builtin_popcountll(<unsigned long long>(az & ((1 << (k + 1)) - 1))) + \
              __builtin_popcountll(<unsigned long long>(bz & ((1 << k) - 1)))
    else:
        # alpha_k: alpha bits 0..k-1 + beta bits 0..k-1
        cnt = __builtin_popcountll(<unsigned long long>(az & ((1 << k) - 1))) + \
              __builtin_popcountll(<unsigned long long>(bz & ((1 << k) - 1)))
    return 1.0 - 2.0 * (cnt & 1)


# ---------------------------------------------------------------------------
# Flat C-level double-excitation loop
# ---------------------------------------------------------------------------
def double_excitation_cy(
    cnp.int64_t[:] azs,
    cnp.int64_t[:] bzs,
    cnp.complex128_t[:] vals,
    cnp.int64_t[:] KP,
    cnp.int64_t[:] KQ,
    cnp.int64_t[:] KR,
    cnp.int64_t[:] KS,
    cnp.int64_t[:] SP,
    cnp.int64_t[:] SQ,
    cnp.int64_t[:] SR,
    cnp.int64_t[:] SS,
    cnp.complex128_t[:] C,
    int n_orb,
):
    """Compute all double excitations for S source states.

    Parameters
    ----------
    azs, bzs : int64 array (S,)
        Alpha/beta bitstrings of source states.
    vals : complex128 array (S,)
        Coefficients of source states.
    KP, KQ, KR, KS : int64 array (T,)
        Spatial orbital indices for creation (p,q) and annihilation (r,s).
    SP, SQ, SR, SS : int64 array (T,)
        Spin labels: 0=alpha, 1=beta.
    C : complex128 array (T,)
        Two-body integral coefficients  2*(t[p,q,r,s] - t[p,q,s,r]).
    n_orb : int
        Number of spatial orbitals.

    Returns
    -------
    (d_az, d_bz, d_v, d_src) : tuple of np.ndarray
        COO-format excitation terms.
    """
    cdef int S = azs.shape[0]
    cdef int T = KP.shape[0]
    cdef Py_ssize_t s, t
    cdef cnp.int64_t az, bz, az1, bz1, aq, bq, ap, bp
    cdef double sgn
    cdef bint r_occ, s_occ, q_empty, p_empty
    cdef cnp.int64_t kr, ks, kq, kp
    cdef int sr, ss, sq, sp

    # Pre-allocate output buffers (worst case S*T; truncated at end).
    # For typical systems only ~5-15% of pairs survive occupancy checks,
    # but pre-allocation avoids Python list-append overhead in the hot loop.
    cdef cnp.int64_t[:] buf_az = np.empty(S * T, dtype=np.int64)
    cdef cnp.int64_t[:] buf_bz = np.empty(S * T, dtype=np.int64)
    cdef cnp.complex128_t[:] buf_v = np.empty(S * T, dtype=np.complex128)
    cdef cnp.int64_t[:] buf_src = np.empty(S * T, dtype=np.int64)
    cdef Py_ssize_t count = 0

    for s in range(S):
        az = azs[s]
        bz = bzs[s]
        for t in range(T):
            kr = KR[t]; ks = KS[t]; kq = KQ[t]; kp = KP[t]
            sr = SR[t]; ss = SS[t]; sq = SQ[t]; sp = SP[t]

            # --- check both annihilation orbitals occupied ---
            if sr == 0:
                r_occ = ((az >> kr) & 1) == 1
            else:
                r_occ = ((bz >> kr) & 1) == 1
            if ss == 0:
                s_occ = ((az >> ks) & 1) == 1
            else:
                s_occ = ((bz >> ks) & 1) == 1
            if not (r_occ and s_occ):
                continue

            # --- annihilate s then r (order matters for sign) ---
            sgn = _spin_sign_scalar(az, bz, ks, ss == 1)
            if ss == 0:
                az1 = az ^ (1 << ks)
                bz1 = bz
            else:
                az1 = az
                bz1 = bz ^ (1 << ks)

            sgn *= _spin_sign_scalar(az1, bz1, kr, sr == 1)
            if sr == 0:
                az1 = az1 ^ (1 << kr)
                bz1 = bz1
            else:
                az1 = az1
                bz1 = bz1 ^ (1 << kr)

            # --- check q orbital empty ---
            if sq == 0:
                q_empty = ((az1 >> kq) & 1) == 0
            else:
                q_empty = ((bz1 >> kq) & 1) == 0
            if not q_empty:
                continue

            # --- create q ---
            if sq == 0:
                aq = az1 ^ (1 << kq)
                bq = bz1
            else:
                aq = az1
                bq = bz1 ^ (1 << kq)
            sgn *= _spin_sign_scalar(aq, bq, kq, sq == 1)

            # --- check p orbital empty ---
            if sp == 0:
                p_empty = ((aq >> kp) & 1) == 0
            else:
                p_empty = ((bq >> kp) & 1) == 0
            if not p_empty:
                continue

            # --- create p ---
            if sp == 0:
                ap = aq ^ (1 << kp)
                bp = bq
            else:
                ap = aq
                bp = bq ^ (1 << kp)
            sgn *= _spin_sign_scalar(ap, bp, kp, sp == 1)

            # --- store result ---
            buf_az[count] = ap
            buf_bz[count] = bp
            buf_v[count] = C[t] * sgn * vals[s]
            buf_src[count] = s
            count += 1

    # Truncate to actual count
    return (
        np.array(buf_az[:count], copy=True),
        np.array(buf_bz[:count], copy=True),
        np.array(buf_v[:count], copy=True),
        np.array(buf_src[:count], copy=True),
    )
