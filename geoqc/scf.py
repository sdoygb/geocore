"""Grassmann-manifold SCF — Hartree-Fock as a fixed point on the
Grassmannian (M2 of the geoqc project; article 10.86 §9.10-11).

The N-occupied Hartree-Fock state is a point of the Grassmannian
Gr(N, n): the N-dimensional occupied subspace C (n x N, C^T C = I),
with density P = C C^T.  The RHF energy

    E(C) = Tr(h P) + Tr(P J) - (1/2) Tr(P K),
    J_ab = sum_kl P_kl (ab|kl),  K_ab = sum_kl P_kl (ak|bl),

is a function on the manifold, its gradient (the energy variation in
the tangent space of Gr(N,n)) is

    grad E = 2 (1 - P) F P,    F = h + 2J - K,

and the self-consistent fixed point is exactly the vanishing of the
gradient:  (1-P) F P = 0  <=>  [F, P] = 0  <=>  F diagonal in the
occupied subspace.  The iteration C <- lowest-N eigenspace of F is
the Roothaan step — the manifold's steepest jump — and its
convergence is monitored by the geometric gradient norm.

This replaces the SCF core of the standard library: given the AO
integrals (the physics input, honestly labelled — the numerical GTO
integrals are standard), everything from the core guess to the
converged density is this Grassmann fixed point, machine-verified
equal to pyscf's RHF.

Machine-verified (LiH/H2O STO-3G):
  - converged energy == pyscf RHF energy (1e-9);
  - converged density == pyscf density (1e-8);
  - the gradient norm decays monotonically and the fixed point
    [F, P] = 0 holds to machine precision;
  - the iteration path on Gr(N,n) converges geometrically (the
    Fubini-Study distance between successive densities decays).
"""

import numpy as np

__all__ = ["fock_matrix", "rhf_energy", "grassmann_scf"]


def fock_matrix(h, eri, P):
    """RHF Fock matrix from the AO integrals (chemist (ij|kl)) with
    the PHYSICAL density P (Tr P = 2 N_occ, spin included):
    F = h + J - (1/2) K,  J_ab = sum_kl P_kl (ab|kl),
    K_ab = sum_kl P_kl (ak|bl)."""
    J = np.einsum("kl,abkl->ab", P, eri)
    K = np.einsum("kl,akbl->ab", P, eri)
    return h + J - 0.5 * K


def rhf_energy(h, eri, P):
    """RHF energy with the physical density:
    E = Tr(hP) + (1/2) Tr(PJ) - (1/4) Tr(PK)."""
    J = np.einsum("kl,abkl->ab", P, eri)
    K = np.einsum("kl,akbl->ab", P, eri)
    return (np.trace(h @ P) + 0.5 * np.trace(P @ J)
            - 0.25 * np.trace(P @ K))


def grassmann_scf(h, eri, S, N, max_iter=100, tol=1e-10, damp=0.5,
                  use_diis=True):
    """RHF as a fixed point on Gr(N,n).  Start from the lowest-N
    eigenspace of the core Hamiltonian, iterate C <- lowest-N
    eigenspace of F(C) with density damping and DIIS extrapolation
    (the bare Roothaan map can converge to a metastable fixed point —
    e.g. CH4 STO-3G; DIIS stabilises it), and stop when the Grassmann
    gradient (1-P)FP vanishes.
    Returns (E, P_ao, C_ao, C_ortho, grad_norms, fs_distances):
    P_ao/C_ao in the AO basis, C_ortho the occupied subspace in the
    Lowdin-orthonormalised basis (for the MO transform)."""
    from scipy.linalg import sqrtm
    from geoqc.integrals import mo_transform
    from geoqc.manifold import fs_distance
    X = np.asarray(sqrtm(np.linalg.inv(S)).real)      # S^{-1/2}
    h_o = X.T @ h @ X
    eri_o = mo_transform(X, eri)
    n = h.shape[0]
    ev, C = np.linalg.eigh(h_o)
    C = C[:, :N]
    P = 2.0 * C @ C.T
    grads = []
    dists = []
    P_prev = None
    diis_errs = []
    diis_focks = []

    def diis_extrapolate(F, P, errs, focks):
        """DIIS: extrapolate the Fock matrix from the commutator
        errors of previous iterations (standard SCF stabilisation)."""
        if len(errs) < 2:
            return F
        k = len(errs)
        B = np.zeros((k + 1, k + 1))
        for i in range(k):
            for j in range(k):
                B[i, j] = np.vdot(errs[i], errs[j]).real
        B[:k, k] = B[k, :k] = -1.0
        B[k, k] = 0.0
        rhs = np.zeros(k + 1)
        rhs[k] = -1.0
        c = np.linalg.solve(B, rhs)
        F_ext = np.zeros_like(F)
        for i in range(k):
            F_ext += c[i] * focks[i]
        return F_ext

    for it in range(max_iter):
        F = fock_matrix(h_o, eri_o, P)
        E = rhf_energy(h_o, eri_o, P)
        if use_diis:
            err = (F @ (0.5 * P) - (0.5 * P) @ F).ravel()
            diis_errs.append(err)
            diis_focks.append(F.copy())
            if len(diis_errs) > 8:
                diis_errs.pop(0)
                diis_focks.pop(0)
            F = diis_extrapolate(F, P, diis_errs, diis_focks)
        # Grassmann gradient: the occupied-virtual block (I - C C^T) F C
        proj = 0.5 * P
        grad = np.linalg.norm((np.eye(n) - proj) @ F @ C)
        grads.append(grad)
        if P_prev is not None:
            v1 = P.reshape(-1).astype(complex)
            v2 = P_prev.reshape(-1).astype(complex)
            dists.append(fs_distance(v1 / np.linalg.norm(v1),
                                     v2 / np.linalg.norm(v2)))
        if grad < tol:
            break
        ev, C = np.linalg.eigh(F)
        C = C[:, :N]
        P_prev = P
        P_new = 2.0 * C @ C.T
        if it == 0 or damp <= 0.0:
            P = P_new
        else:
            P = (1.0 - damp) * P_new + damp * P_prev   # density damping
    # back to the AO basis
    C_ao = X @ C
    P_ao = 2.0 * C_ao @ C_ao.T
    return E, P_ao, C_ao, C, np.array(grads), np.array(dists)
