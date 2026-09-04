"""Schubert-cell structure of the sector basis — the Grassmannian
geometry of the sector (feature 50; article 10.86 §9.09).

Two sector conventions are supported:

  1. N-sector (legacy Pauli/JW path): a single big-endian bitstring of
     n qubits with N occupied; states are points of Gr(N, n).

  2. S_z-sector (exterior / WCI path): the state factorises into an
     alpha bitstring and a beta bitstring, each of n_orb orbitals, with
     n_a and n_b occupied respectively; states are points of the product
     Grassmannian Gr(n_a, n_orb) × Gr(n_b, n_orb).  The bit order is
     little-endian (bit k = orbital k), matching exterior.sparse_action_sz
     and geoqc.wci.

For each Grassmannian factor, the Schubert cell decomposition is marked
by partitions λ inside the N × (n_orb-N) box:

    occupied orbitals i_1 < ... < i_N
    <->  λ_j = i_{N+1-j} - (N - j),  j = 1..N

with cell complex dimension |λ| = sum λ_j.  The Bruhat (dominance)
order organises the closures:

    μ ≤ λ  <=>  sum_{j≤k} μ_j ≤ sum_{j≤k} λ_j  for all k.

For the S_z product, the Bruhat order is the product partial order:
(λ_α, λ_β) ≤ (μ_α, μ_β)  <=>  λ_α ≤ μ_α AND λ_β ≤ μ_β.

Excitation level vs Bruhat distance:
  - excitation_level = Hamming distance / 2 = number of single-particle
    excitations needed to reach one determinant from the other.  A 1+2-body
    Hamiltonian only connects states at excitation level ≤ 2, so a WCI
    wavepacket support is exactly the excitation-level-2 neighbourhood.
  - bruhat_distance = | |λ_1| - |λ_2| | (cell-weight difference), a
    coarser measure that correlates with excitation level but is not
    identical (two states at the same excitation level can have different
    Bruhat distances depending on which orbitals are excited).

Machine-verified: the bijection is exact, the order is a partial order,
every H matrix element connects states of even excitation level (S_z
conservation), and the diagonal energy trends with the cell weight |λ|.
"""

import numpy as np

__all__ = [
    # legacy N-sector (big-endian qubit order)
    "partition_of_state", "state_of_partition", "partition_weight",
    "bruhat_le", "excitation_bruhat_distance",
    # S_z-sector (alpha/beta little-endian, exterior/WCI path)
    "partition_of_alpha", "alpha_of_partition",
    "partition_of_state_sz", "state_of_partition_sz",
    "bruhat_le_sz", "excitation_level_sz", "bruhat_distance_sz",
    "wavepacket_support_estimate", "analyze_wavepacket_sz",
]


# ---------------------------------------------------------------------------
# Legacy N-sector (big-endian qubit order)
# ---------------------------------------------------------------------------

def partition_of_state(z, n, N):
    """The Schubert partition of the sector state z (big-endian
    bitstring, N occupied orbitals): lambda_j = i_{N+1-j} - (N-j)."""
    occ = [q for q in range(n) if (z >> (n - 1 - q)) & 1]
    lam = [occ[N - j] - (N - j) for j in range(1, N + 1)]
    return tuple(lam)


def state_of_partition(lam, n):
    """Inverse: the big-endian bitstring with occupied orbitals
    i_j = lambda_{N+1-j} + (j-1)."""
    N = len(lam)
    z = 0
    for j in range(N):
        i = lam[N - 1 - j] + j
        z |= 1 << (n - 1 - i)
    return z


def partition_weight(lam):
    return int(sum(lam))


def bruhat_le(mu, lam):
    """Bruhat order (dominance): mu <= lam iff partial sums of mu do
    not exceed those of lam at every prefix."""
    s_mu = s_la = 0
    for a, b in zip(mu, lam):
        s_mu += a
        s_la += b
        if s_mu > s_la:
            return False
    return True


def excitation_bruhat_distance(z, zt, n, N):
    """Bruhat distance between two sector states: the number of boxes
    by which the cell weights differ."""
    return abs(partition_weight(partition_of_state(z, n, N))
               - partition_weight(partition_of_state(zt, n, N)))


# ---------------------------------------------------------------------------
# S_z-sector (alpha/beta little-endian, exterior / WCI path)
# ---------------------------------------------------------------------------

def partition_of_alpha(az, n_orb, n_a):
    """Schubert partition of an alpha bitstring (little-endian, bit k =
    orbital k, n_a occupied).  λ_j = i_{n_a+1-j} - (n_a - j)."""
    occ = [k for k in range(n_orb) if (int(az) >> k) & 1]
    lam = [occ[n_a - j] - (n_a - j) for j in range(1, n_a + 1)]
    return tuple(lam)


def alpha_of_partition(lam, n_orb):
    """Inverse: alpha bitstring (little-endian) from a Schubert partition.
    i_j = λ_{n_a+1-j} + (j-1)."""
    n_a = len(lam)
    az = 0
    for j in range(n_a):
        i = lam[n_a - 1 - j] + j
        az |= 1 << i
    return az


def partition_of_state_sz(az, bz, n_orb, n_a, n_b):
    """S_z-sector Schubert partitions: (λ_α, λ_β) for the product
    Grassmannian Gr(n_a, n_orb) × Gr(n_b, n_orb)."""
    return (partition_of_alpha(az, n_orb, n_a),
            partition_of_alpha(bz, n_orb, n_b))


def state_of_partition_sz(lam_a, lam_b, n_orb):
    """Inverse: (az, bz) little-endian bitstrings from S_z partitions."""
    return (alpha_of_partition(lam_a, n_orb),
            alpha_of_partition(lam_b, n_orb))


def bruhat_le_sz(lam1_a, lam1_b, lam2_a, lam2_b):
    """S_z Bruhat order (product partial order):
    (λ_α1, λ_β1) ≤ (λ_α2, λ_β2)  <=>  λ_α1 ≤ λ_α2 AND λ_β1 ≤ λ_β2."""
    return bruhat_le(lam1_a, lam2_a) and bruhat_le(lam1_b, lam2_b)


def _popcount64(x):
    """Vectorised popcount for uint64 numpy arrays (Brian Kernighan is
    not vectorised; use the bit-hack popcount)."""
    x = np.asarray(x, dtype=np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def excitation_level_sz(az1, bz1, az2, bz2):
    """Excitation level between two S_z determinants = Hamming distance / 2.

    Each single excitation flips 2 bits (annihilate one occupied, create
    one virtual), so the excitation level is half the total Hamming
    distance across alpha and beta bitstrings.  A 1+2-body Hamiltonian
    only connects states at excitation level ≤ 2.

    Vectorised: az/bz may be scalars or numpy arrays (same shape).
    Returns a scalar if inputs are scalars, otherwise an array."""
    az1 = np.asarray(az1, dtype=np.int64)
    bz1 = np.asarray(bz1, dtype=np.int64)
    az2 = np.asarray(az2, dtype=np.int64)
    bz2 = np.asarray(bz2, dtype=np.int64)
    pop_a = _popcount64(np.bitwise_xor(az1, az2).astype(np.uint64))
    pop_b = _popcount64(np.bitwise_xor(bz1, bz2).astype(np.uint64))
    level = (pop_a + pop_b) // 2
    if level.ndim == 0:
        return int(level)
    return level.astype(np.int64)


def bruhat_distance_sz(az1, bz1, az2, bz2, n_orb, n_a, n_b):
    """Bruhat distance (cell-weight difference) between two S_z
    determinants: ||λ_α1| - |λ_α2|| + ||λ_β1| - |λ_β2||.

    This is a coarser measure than excitation_level_sz: two states at the
    same excitation level can have different Bruhat distances depending on
    which orbitals are involved.  For WCI wavepacket analysis,
    excitation_level_sz is the physically relevant measure (H only
    connects level ≤ 2)."""
    la1, lb1 = partition_of_state_sz(az1, bz1, n_orb, n_a, n_b)
    la2, lb2 = partition_of_state_sz(az2, bz2, n_orb, n_a, n_b)
    return abs(partition_weight(la1) - partition_weight(la2)) + \
           abs(partition_weight(lb1) - partition_weight(lb2))


def wavepacket_support_estimate(n_orb, n_a, n_b, k=2):
    """Estimated size of a Bruhat-k (excitation-level-k) wavepacket
    centred at an arbitrary determinant in the S_z sector.

    For k=2 (the WCI wavepacket):
      - 1 centre determinant
      - single excitations: n_a(n_orb-n_a) [alpha] + n_b(n_orb-n_b) [beta]
      - double excitations:
          same-spin alpha: C(n_a,2) C(n_orb-n_a,2)
          same-spin beta:  C(n_b,2) C(n_orb-n_b,2)
          mixed-spin:      n_a n_b (n_orb-n_a)(n_orb-n_b)

    The actual support may be smaller due to integral truncation (eps)
    and zero matrix elements.

    Returns dict with breakdown by excitation level."""
    from math import comb
    va = n_orb - n_a  # alpha virtuals
    vb = n_orb - n_b  # beta virtuals

    single = n_a * va + n_b * vb
    double_alpha = comb(n_a, 2) * comb(va, 2)
    double_beta = comb(n_b, 2) * comb(vb, 2)
    double_mixed = n_a * n_b * va * vb
    double = double_alpha + double_beta + double_mixed

    if k == 1:
        total = 1 + single
    elif k == 2:
        total = 1 + single + double
    else:
        raise ValueError(f"k={k} not supported (only k=1,2 for 1+2-body H)")

    return {
        "total": total,
        "centre": 1,
        "single": single,
        "double": double,
        "double_alpha": double_alpha,
        "double_beta": double_beta,
        "double_mixed": double_mixed,
        "n_orb": n_orb, "n_a": n_a, "n_b": n_b,
    }


def analyze_wavepacket_sz(center_idx, support_idx, db, az_of, bz_of,
                           n_orb, n_a, n_b):
    """Analyse a WCI wavepacket: compute the excitation level and Bruhat
    distance of every support state relative to the centre.

    Parameters
    ----------
    center_idx : int
        Combined index of the centre determinant (alpha_rank * db + beta_rank).
    support_idx : ndarray of int
        Combined indices of the wavepacket support states.
    db : int
        Beta-sector dimension.
    az_of, bz_of : ndarray
        Rank → bitstring maps from wci.build_rank_tables.
    n_orb, n_a, n_b : int
        Orbital and occupation counts.

    Returns
    -------
    result : dict
        - 'excitation_levels': array of excitation level for each support state
        - 'bruhat_distances': array of Bruhat distance for each support state
        - 'level_histogram': dict {level: count}
        - 'bruhat_histogram': dict {distance: count}
        - 'centre_partition': (λ_α, λ_β) of the centre
        - 'max_excitation_level': maximum excitation level in the support
    """
    support_idx = np.asarray(support_idx, dtype=np.int64)
    c_az = int(az_of[center_idx // db])
    c_bz = int(bz_of[center_idx % db])

    s_az = az_of[support_idx // db]
    s_bz = bz_of[support_idx % db]

    levels = excitation_level_sz(c_az, c_bz, s_az, s_bz)

    # Bruhat distances (scalar loop; support size is O(200) so this is fine)
    c_lam = partition_of_state_sz(c_az, c_bz, n_orb, n_a, n_b)
    c_wt_a = partition_weight(c_lam[0])
    c_wt_b = partition_weight(c_lam[1])
    bruhat_dists = np.zeros(len(support_idx), dtype=np.int64)
    for i in range(len(support_idx)):
        lam = partition_of_state_sz(int(s_az[i]), int(s_bz[i]), n_orb, n_a, n_b)
        bruhat_dists[i] = (abs(partition_weight(lam[0]) - c_wt_a)
                            + abs(partition_weight(lam[1]) - c_wt_b))

    level_hist = {}
    for lv in levels:
        level_hist[int(lv)] = level_hist.get(int(lv), 0) + 1
    bruhat_hist = {}
    for bd in bruhat_dists:
        bruhat_hist[int(bd)] = bruhat_hist.get(int(bd), 0) + 1

    return {
        "excitation_levels": levels,
        "bruhat_distances": bruhat_dists,
        "level_histogram": dict(sorted(level_hist.items())),
        "bruhat_histogram": dict(sorted(bruhat_hist.items())),
        "centre_partition": c_lam,
        "max_excitation_level": int(levels.max()),
        "support_size": len(support_idx),
    }
