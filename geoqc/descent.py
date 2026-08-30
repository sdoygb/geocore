"""Descent: Krylov exponentials — the numerical form of the geometry's
"discrete descent" (article 0.13; feature 49, article 10.86 §9.08).

Evolution and ground states are built from matvecs only: the Krylov
subspace {v, Hv, H^2 v, ...} is the orbit of v under the operator,
the small Hessenberg matrix is exponentiated, and the result lifted
back.  Works with any matvec — in particular the matrix-free
exterior action (geoqc.exterior.exterior_action).
"""

import numpy as np

__all__ = ["krylov_expm", "matfree_evolve"]


def krylov_expm(matvec, v, dt, m=30, tol=1e-13):
    """e^{-i*dt*H} v by the Arnoldi (Krylov) method: project onto the
    Krylov subspace spanned by {v, Hv, H^2 v, ...}, exponentiate the
    small Hessenberg matrix, lift back.  This is the numerical form of
    the geometry's "discrete descent": the evolution is built on the
    orbit of v under the operator — no matrix, only matvec."""
    n = v.shape[0]
    beta = np.linalg.norm(v)
    if beta == 0:
        return v.copy()
    V = np.zeros((n, m + 1), dtype=complex)
    H = np.zeros((m + 1, m), dtype=complex)
    V[:, 0] = v / beta
    j = 0
    for j in range(m):
        w = matvec(V[:, j])
        for i in range(j + 1):
            H[i, j] = np.vdot(V[:, i], w)
            w -= H[i, j] * V[:, i]
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] < tol:
            break
        V[:, j + 1] = w / H[j + 1, j]
    k = j + 1
    from scipy.linalg import expm
    e1 = np.zeros(k)
    e1[0] = beta
    coef = expm(-1j * dt * H[:k, :k]) @ e1
    return V[:, :k] @ coef


def matfree_evolve(L, hd, p, T, m=30):
    """Zero-gradient discrete adiabatic with the matrix-free exterior
    action: every step is a Krylov exponential on the operator, so no
    sector matrix is ever built.  H(s) = (1-s) hd + s L (L carries the
    full H, diagonal included)."""
    dim = L.shape[0]
    i0 = int(np.argmin(hd))
    psi = np.zeros(dim, dtype=complex)
    psi[i0] = 1
    dt = T / p
    for k in range(p):
        s = (k + 0.5) / p
        psi = krylov_expm(
            lambda v, s=s: (1 - s) * hd * v + s * (L @ v), psi, dt, m=m)
        psi /= np.linalg.norm(psi)
    return psi
