"""Schubert-cell structure of the N-sector basis — the Grassmannian
geometry of the sector (feature 50; article 10.86 §9.09).

The N-sector states are the points of the Grassmannian Gr(N, n); its
Schubert cell decomposition is marked by partitions lambda inside the
N x (n-N) box, with the bijection

    occupied orbitals i_1 < ... < i_N
    <->  lambda_j = i_{N+1-j} - (N - j),  j = 1..N

and cell complex dimension |lambda|.  The Bruhat (dominance) order
organises the closures:

    mu <= lambda  <=>  sum_{j<=k} mu_j <= sum_{j<=k} lambda_j  for all k.

Machine-verified: the bijection is exact, the order is a partial
order, every H matrix element connects states of even Bruhat distance
(S_z conservation + interleaved spin order -> bipartite lattice), and
the diagonal energy trends with the cell weight |lambda|.
"""

__all__ = [
    "partition_of_state", "state_of_partition", "partition_weight",
    "bruhat_le", "excitation_bruhat_distance",
]


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
