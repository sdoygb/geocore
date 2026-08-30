"""Fubini-Study geometry of the state space: path metrics, adiabatic
path geometry, adiabatic-quality measures (feature 48; article 10.86
§9.07).

The zero-gradient path H(s) = H_diag + s H_off draws a curve in the
N-sector state space (projective space CP^{C(n,N)-1}); its arc length
under the Fubini-Study metric g = <dpsi|dpsi> - |<psi|dpsi>|^2 obeys
the geodesic inequality L >= d_FS, the final fidelity equals
cos^2(d_FS), and the deviation of the path from the instantaneous-GS
manifold is the geometric measure of adiabatic quality (decreases
with T, ~1/T).
"""

import numpy as np

__all__ = [
    "fs_metric", "fs_distance", "path_geometry", "adiabatic_path",
    "instantaneous_path", "manifold_deviation",
]


def fs_metric(psi, dpsi):
    """Fubini-Study metric element g = <dpsi|dpsi> - |<psi|dpsi>|^2
    for a normalized state psi and tangent vector dpsi."""
    g = np.vdot(dpsi, dpsi) - abs(np.vdot(psi, dpsi)) ** 2
    return float(np.real(g))


def fs_distance(a, b):
    """Fubini-Study distance d_FS(a,b) = arccos |<a|b>| (normalized)."""
    ov = abs(np.vdot(a, b))
    ov = min(1.0, max(0.0, ov))
    return float(np.arccos(ov))


def path_geometry(states_path, s_grid=None):
    """Geometric quantities of a discrete path psi_0..psi_p:
    (length L, per-step metric g_k, Berry phase gamma, FS distance
    d_FS(psi_0, psi_p), continuum-limit metric integral L_int).
    The arc length is the chord sum of the adjacent Fubini-Study
    distances (the geodesic inequality L >= d_FS then holds by the
    triangle inequality at machine precision)."""
    p = len(states_path) - 1
    ds = 1.0 / p
    L = 0.0
    gs_ = np.zeros(p + 1)
    for k in range(p):
        L += fs_distance(states_path[k], states_path[k + 1])
    L_int = 0.0
    for k in range(1, p):
        v = (states_path[k + 1] - states_path[k - 1]) / (2 * ds)
        g = fs_metric(states_path[k], v)
        gs_[k] = g
        L_int += np.sqrt(max(g, 0.0)) * ds
    ph = 1.0 + 0.0j
    for k in range(p):
        ph *= np.vdot(states_path[k], states_path[k + 1])
    gamma = float(np.angle(ph))
    d = fs_distance(states_path[0], states_path[-1])
    return L, gs_, gamma, d, L_int


def adiabatic_path(H_off, hd, p, T):
    """The zero-gradient discrete adiabatic path psi_0..psi_p in the
    sector (H(s) = H_diag + s H_off, init = diagonal ground state),
    using exact sparse exponentials (no Trotter).  H_off may be a
    sparse matrix or a scipy LinearOperator (matrix-free exterior
    action, feature 49); hd may be a zero vector in that case."""
    from scipy import sparse
    from scipy.sparse.linalg import expm_multiply, LinearOperator
    dim = hd.size
    i0 = int(np.argmin(hd))
    psi = np.zeros(dim, dtype=complex)
    psi[i0] = 1
    path = [psi.copy()]
    dt = T / p
    for k in range(p):
        s = (k + 0.5) / p
        if sparse.issparse(H_off):
            Hs = sparse.diags(hd) + s * H_off
            psi = expm_multiply(-1j * dt * Hs, psi)
        else:
            def _matvec(v, s=s):
                shp = np.shape(v)
                v1 = np.asarray(v).reshape(-1)
                r = (-1j * dt) * (hd * v1 + s * (H_off @ v1))
                return r.reshape(shp) if len(shp) > 1 else r
            Hs = LinearOperator((dim, dim), matvec=_matvec, dtype=complex)
            psi = expm_multiply(Hs, psi)
        path.append(psi.copy())
    for i in range(p + 1):
        nrm = np.linalg.norm(path[i])
        if nrm > 0:
            path[i] = path[i] / nrm
    return path


def instantaneous_path(H_off, hd, p):
    """The curve of instantaneous ground states |GS(s_k)> (k=0..p):
    the adiabatic core of the path, T-independent."""
    from scipy import sparse
    import scipy.sparse.linalg as spla
    path = []
    for k in range(p + 1):
        s = k / p
        Hs = sparse.diags(hd) + s * H_off
        _, v = spla.eigsh(Hs, k=1, which="SA")
        path.append(v[:, 0])
    return path


def manifold_deviation(path, H_off, hd, p):
    """Per-point Fubini-Study distance from the path to the
    instantaneous-GS manifold: d_FS(psi_k, GS(s_k)) — the geometric
    measure of adiabatic quality along the trajectory."""
    from scipy import sparse
    import scipy.sparse.linalg as spla
    dev = np.zeros(p + 1)
    for k in range(p + 1):
        s = k / p
        Hs = sparse.diags(hd) + s * H_off
        _, v = spla.eigsh(Hs, k=1, which="SA")
        dev[k] = fs_distance(path[k], v[:, 0])
    return dev
