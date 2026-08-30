"""Sector geometry: particle-number sectors of a molecular Hamiltonian
(features 45-46; article 10.86 §9.04-05).

The N-electron sector is the degree-N piece of the exterior algebra,
dimension C(n, N).  This module provides:

  - sector_states / pauli_action_int : the sector basis and the Pauli
    action in big-endian qubit order (legacy reference build);
  - n_conserving_terms / term_budget : the conservation filter and
    the a-priori matrix-element budget of the sector build;
  - sector_hamiltonian_fast          : the combinatorial build
    (enumerate matrix elements per surviving term);
  - sector_hamiltonian_merged        : the commuting-term merge
    (same flip set = commuting family; exact but does not shrink the
    unique elements — the elements are physical);
  - sector_diagonal                  : the diagonal over the sector
    states only, never materialising the 2^n diagonal;
  - spectral truncation via eps on term coefficients.

Machine-verified (LiH/H2O STO-3G): the combinatorial and merged
builds equal the naive per-state x per-term build to machine
precision.
"""

import numpy as np
from itertools import combinations
from math import comb

__all__ = [
    "sector_states", "pauli_action_int", "sector_hamiltonian",
    "n_conserving_terms", "group_by_flip_set", "term_budget",
    "sector_hamiltonian_fast", "sector_hamiltonian_merged",
    "sector_diagonal", "sector_evolve",
]


def sector_states(n, N):
    """All bitstrings with exactly N bits set (big-endian ints)."""
    return [sum(1 << (n - 1 - q) for q in comb_i)
            for comb_i in combinations(range(n), N)]


def pauli_action_int(z, ax, n):
    """P|z> as (target_int, phase) for a Pauli axis (big-endian)."""
    t = z
    ph = 1.0
    ny = 0
    for q, ch in enumerate(ax):
        bit = (z >> (n - 1 - q)) & 1
        if ch in "XY":
            t ^= (1 << (n - 1 - q))
        if ch in "ZY":
            if bit:
                ph *= -1
        if ch == "Y":
            ny += 1
    return t, ph * (1j ** ny)


def sector_hamiltonian(n, N, diag, off):
    """(hd, H_off) in the N-sector basis: hd[z_idx] = diagonal value;
    H_off sparse = N-conserving Pauli actions (single flips dropped).
    The naive per-state x per-term build (reference)."""
    from scipy import sparse
    states = sector_states(n, N)
    idx = {z: i for i, z in enumerate(states)}
    dim = len(states)
    hd = np.array([diag[z] for z in states])
    rows, cols, vals = [], [], []
    for si, z in enumerate(states):
        for c, ax in off:
            t, ph = pauli_action_int(z, ax, n)
            if t in idx:
                rows.append(idx[t])
                cols.append(si)
                vals.append(c * ph)
    H_off = sparse.coo_matrix((vals, (rows, cols)),
                              shape=(dim, dim)).tocsr()
    return hd, H_off


def n_conserving_terms(n, N, off, eps=0.0):
    """Pre-filter: keep off terms with |coeff| > eps whose flip set can
    connect two N-sector states.  A term flips w qubits (the X/Y
    positions); N is conserved iff exactly w/2 of the flipped qubits
    are occupied, which needs w even and w/2 <= N.  For a molecular
    JW Hamiltonian the flip weight is always 2 or 4 (1/2-body fermion
    operators), so with N >= 2 every off term is N-conserving; the
    long Z strings are JW phases and do NOT flip.
    Returns [(coeff, flip_bits, Z_phase_bits, n_y)] with flip_bits the
    X/Y positions, Z_phase_bits the Z and Y positions (both contribute
    a -1 sign when occupied), big-endian qubit order."""
    keep = []
    for c, ax in off:
        if abs(c) <= eps:
            continue
        F = [q for q, ch in enumerate(ax) if ch in "XY"]
        w = len(F)
        if w % 2 or w // 2 > N:
            continue
        # Pauli action on |z>: X flips, Z AND Y flip a sign when the
        # qubit is occupied, Y additionally brings i.
        Zph = [q for q, ch in enumerate(ax) if ch in "ZY"]
        ny = sum(1 for ch in ax if ch == "Y")
        keep.append((float(c), F, Zph, ny))
    return keep


def group_by_flip_set(n, N, off, eps=0.0):
    """Merge commuting terms: terms that share the same flip set F are
    mutually commuting in the molecular case (measured 3216/3216 on
    LiH STO-3G; X<->Y symmetric patterns give even mismatch parity).
    Returns {F: [(c, Zph, ny)]}."""
    groups = {}
    for c, F, Zph, ny in n_conserving_terms(n, N, off, eps):
        groups.setdefault(tuple(F), []).append((c, Zph, ny))
    return groups


def term_budget(n, N, off, eps=0.0):
    """(kept_terms, matrix_elements): the exact H_N build cost, known
    a priori from the conservation geometry (C(w,w/2) occupied choices
    inside the flip set x C(n-w, N-w/2) outside)."""
    kept, nnz = 0, 0
    detail = {}
    for c, F, Zs, ny in n_conserving_terms(n, N, off, eps):
        w = len(F)
        per = comb(w, w // 2) * comb(n - w, N - w // 2)
        kept += 1
        nnz += per
        detail[w] = detail.get(w, 0) + per
    return kept, nnz, detail


def sector_hamiltonian_fast(n, N, diag, off, eps=0.0):
    """(hd, H_off) in the N-sector basis, built by enumerating the
    matrix elements combinatorially per surviving term — the build
    loop is over the (kept) terms, and inside each term the source
    states are enumerated as (occupied inside flip set) x (occupied
    outside), vectorised over the outer choices.  `diag` is either
    the full 2^n diagonal vector or a pre-built sector diagonal of
    length C(n, N).  `eps`: spectral truncation (|coeff| <= eps)."""
    from scipy import sparse
    states = sector_states(n, N)
    idx = {z: i for i, z in enumerate(states)}
    dim = len(states)
    if np.ndim(diag) == 1 and len(diag) == dim:
        hd = np.asarray(diag, dtype=complex)
    else:
        hd = np.array([diag[z] for z in states])
    rows_l, cols_l, vals_l = [], [], []
    for c, F, Zs, ny in n_conserving_terms(n, N, off, eps):
        w = len(F)
        k = w // 2
        m = N - k
        Fset = set(F)
        rest = [q for q in range(n) if q not in Fset]
        Fmask = sum(1 << (n - 1 - q) for q in F)
        ZF = set(Zs) & Fset          # Z phase from occupied flip bits
        ZR = set(Zs) - Fset          # Z phase from occupied rest bits
        s0 = (1j ** ny) * c
        outer_combos = list(combinations(range(len(rest)), m))
        n_out = len(outer_combos)
        outer_z = np.empty(n_out, dtype=np.int64)
        outer_ph = np.empty(n_out, dtype=complex)
        for j, oc in enumerate(outer_combos):
            z, ph = 0, 1.0
            for i in oc:
                q = rest[i]
                z |= 1 << (n - 1 - q)
                if q in ZR:
                    ph *= -1
            outer_z[j] = z
            outer_ph[j] = ph
        for ic in combinations(range(w), k):
            zF, phF = 0, 1.0
            for i in ic:
                q = F[i]
                zF |= 1 << (n - 1 - q)
                if q in ZF:
                    phF *= -1
            zs = zF | outer_z                 # all source states
            zts = zs ^ Fmask                  # all target states
            phs = s0 * phF * outer_ph
            rows_l.append(np.fromiter((idx[z] for z in zts),
                                      dtype=np.int64, count=n_out))
            cols_l.append(np.fromiter((idx[z] for z in zs),
                                      dtype=np.int64, count=n_out))
            vals_l.append(phs)
    rows = np.concatenate(rows_l)
    cols = np.concatenate(cols_l)
    vals = np.concatenate(vals_l)
    H_off = sparse.coo_matrix((vals, (rows, cols)),
                              shape=(dim, dim)).tocsr()
    return hd, H_off


def sector_hamiltonian_merged(n, N, diag, off, eps=0.0):
    """(hd, H_off) built from the merged commuting groups (terms with
    the same flip set F).  For each F and each N-sector source z the
    matrix element is a single complex number summing the F-group
    terms — the commuting merge in action.  Must equal the naive
    build to machine precision (verified on STO-3G)."""
    from scipy import sparse
    states = sector_states(n, N)
    idx = {z: i for i, z in enumerate(states)}
    dim = len(states)
    if np.ndim(diag) == 1 and len(diag) == dim:
        hd = np.asarray(diag, dtype=complex)
    else:
        hd = np.array([diag[z] for z in states])
    rows_l, cols_l, vals_l = [], [], []
    for F, items in group_by_flip_set(n, N, off, eps).items():
        w = len(F)
        k = w // 2
        m = N - k
        Fset = set(F)
        rest = [q for q in range(n) if q not in Fset]
        Fmask = sum(1 << (n - 1 - q) for q in F)
        outer_combos = list(combinations(range(len(rest)), m))
        outer_z = [0] * len(outer_combos)
        for j, oc in enumerate(outer_combos):
            for i in oc:
                outer_z[j] |= 1 << (n - 1 - rest[i])
        for ic in combinations(range(w), k):
            zF = 0
            for i in ic:
                zF |= 1 << (n - 1 - F[i])
            rows = np.empty(len(outer_z), dtype=np.int64)
            cols = np.empty(len(outer_z), dtype=np.int64)
            vals = np.empty(len(outer_z), dtype=complex)
            for j, oz in enumerate(outer_z):
                z = zF | oz
                ph = 0j
                for c, Zph, ny in items:
                    occ = 0
                    for q in Zph:
                        occ += (z >> (n - 1 - q)) & 1
                    ph += c * (1j ** ny) * (-1.0) ** occ
                rows[j] = idx[z ^ Fmask]
                cols[j] = idx[z]
                vals[j] = ph
            rows_l.append(rows)
            cols_l.append(cols)
            vals_l.append(vals)
    rows = np.concatenate(rows_l)
    cols = np.concatenate(cols_l)
    vals = np.concatenate(vals_l)
    H_off = sparse.coo_matrix((vals, (rows, cols)),
                              shape=(dim, dim)).tocsr()
    return hd, H_off


def sector_diagonal(n, N, constant, z_terms):
    """Diagonal values over the sector states only, built from the
    Z-only terms — never materialises the 2^n diagonal vector."""
    states = sector_states(n, N)
    sz = np.array(states, dtype=np.int64)
    hd = np.full(sz.size, constant, dtype=complex)
    for c, ax in z_terms:
        ph = np.ones(sz.size, dtype=complex)
        for i, ch in enumerate(ax):
            if ch == "Z":
                b = (sz >> (n - 1 - i)) & 1
                ph *= np.where(b, -1.0, 1.0)
        hd += c * ph
    return hd


def sector_evolve(n, N, diag, off, p, T, sparse=False):
    """Zero-gradient discrete adiabatic inside the N sector:
    H(s) = H_diag + s * H_off, init = the sector's diagonal ground
    state."""
    from scipy.linalg import expm
    from scipy.sparse.linalg import expm_multiply
    hd, H_off = sector_hamiltonian(n, N, diag, off)
    dim = hd.size
    i0 = int(np.argmin(hd))
    psi = np.zeros(dim, dtype=complex)
    psi[i0] = 1
    dt = T / p
    for k in range(p):
        s = (k + 0.5) / p
        if sparse:
            Hs = sparse.diags(hd) + s * H_off
            psi = expm_multiply(-1j * dt * Hs, psi)
        else:
            Hs = np.diag(hd) + s * H_off.toarray()
            psi = expm(-1j * dt * Hs) @ psi
    return psi, hd, H_off
