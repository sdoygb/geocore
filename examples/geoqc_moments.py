#!/usr/bin/env python3
"""Information-dynamics verification: do the information-field
theorems (cumulant decay 5.7/5.8, molecular chaos 10.20) hold for the
FCI Hamiltonian, and can the ground state be recovered from low-order
spectral moments (bypassing the nnz storage/diagonalisation wall)?

For each small system we compute the spectral moments
mu_k = Tr(H^k)/dim (k = 1..4), the normalised cumulants
kappa_k / sigma^k (Gaussian spectrum -> 0 for k >= 3), and estimate
the ground state from the low-order moments (Gaussian-tail and
3-moment estimates) against the exact FCI ground state.  Also times
the Tr(H^2) accumulation (the O(nnz) incremental cost that would
replace the TB-scale storage) on numpy vs torch/MPS to assess whether
a GPU helps.

Run:  PYTHONPATH=src python3 examples/geoqc_moments.py
"""

import time
import numpy as np
from math import comb

from geoqc.integrals import ao_integrals, mo_transform, spin_orbital_integrals
from geoqc.scf import grassmann_scf, fock_matrix
from geoqc import exterior
from scipy.linalg import sqrtm
from scipy import sparse
import scipy.sparse.linalg as spla


def sector_H(geom, basis, N, eps=1e-4):
    n, h, eri, S, nuc = ao_integrals(geom, basis)
    E, P, C, C_o, _, _ = grassmann_scf(h, eri, S, N // 2)
    X = np.asarray(sqrtm(np.linalg.inv(S)).real)
    h_o = X.T @ h @ X
    eri_o = mo_transform(X, eri)
    F = fock_matrix(h_o, eri_o, 2.0 * C_o @ C_o.T)
    _, C_all = np.linalg.eigh(F)
    o = C_all.T @ h_o @ C_all
    t = mo_transform(C_all, eri_o)
    o_s, t_s = spin_orbital_integrals(o, t)
    hd, H_off = exterior.exterior_hamiltonian_sz(2 * n, N, 0, o_s, t_s,
                                                 float(nuc), eps)
    return sparse.diags(hd) + H_off


def moments(H, kmax=4):
    """Spectral moments mu_k = Tr(H^k)/dim (k=1..kmax)."""
    dim = H.shape[0]
    mu = [H.diagonal().sum().real / dim]
    # Tr(H^2) = sum of |H_ij|^2 (Hermitian)
    coo = H.tocoo()
    tr2 = (np.abs(coo.data) ** 2).sum()
    mu.append(tr2.real / dim)
    # Tr(H^3), Tr(H^4) via sparse matrix products' traces
    H2 = H @ H
    tr3 = (H2 @ H).diagonal().sum().real
    mu.append(tr3 / dim)
    if kmax >= 4:
        tr4 = (H2 @ H2).diagonal().sum().real
        mu.append(tr4 / dim)
    return mu


def cumulants(mu):
    """Central moments m_k and normalised cumulants kappa_k/sigma^k."""
    m1 = mu[0]
    m2 = mu[1] - m1 ** 2
    m3 = mu[2] - 3 * m1 * mu[1] + 2 * m1 ** 3
    m4 = (mu[3] - 4 * m1 * mu[2] + 6 * m1 ** 2 * mu[1]
          - 3 * m1 ** 4)
    sigma = np.sqrt(m2)
    k3 = m3 / sigma ** 3          # skewness (0 for Gaussian)
    k4 = (m4 - 3 * m2 ** 2) / sigma ** 4   # excess kurtosis (0 for Gaussian)
    return m1, m2, m3, m4, sigma, k3, k4


def estimate_gs(mu, dim, exact):
    """Ground-state estimates from low-order moments vs the exact FCI
    energy.  Gaussian-tail estimate uses mu_1, sigma and the extreme-
    value tail of a Gaussian spectrum; the 3-moment estimate adds the
    skewness correction."""
    m1, m2, m3, m4, sigma, k3, k4 = cumulants(mu)
    # Gaussian-tail: lowest of N draws from N(mu1, sigma^2)
    # E[min of N Gaussians] ~ mu1 - sigma * sqrt(2 ln N) (extreme value)
    eg = m1 - sigma * np.sqrt(2 * np.log(dim))
    # skewness-corrected (3rd moment): E_min = mu1 - sigma*a - k3*sigma*b/6
    a = np.sqrt(2 * np.log(dim))
    b = 2 * (a ** 2 - 1) / a if a > 0 else 0
    e3 = m1 - sigma * a - (k3 * sigma * b) / 6.0
    return eg, e3, m1, sigma, k3, k4


def main():
    print("=" * 74)
    print("Information-dynamics verification: FCI spectral moments,")
    print("cumulant decay (molecular chaos?) and low-order-moment")
    print("ground-state estimates vs exact FCI")
    print("=" * 74)

    cases = [
        ("LiH STO-3G", [["Li", [0, 0, 0]], ["H", [0, 0, 1.6]]],
         "sto-3g", 4),
        ("N2 STO-3G", [["N", [0, 0, 0]], ["N", [0, 0, 1.1]]],
         "sto-3g", 14),
        ("H2O STO-3G", [["O", [0, 0, 0]], ["H", [0.757, 0.586, 0]],
                         ["H", [-0.757, 0.586, 0]]], "sto-3g", 10),
    ]
    for name, geom, basis, N in cases:
        H = sector_H(geom, basis, N)
        dim = H.shape[0]
        w, _ = spla.eigsh(H, k=1, which="SA")
        exact = w[0]
        mu = moments(H, kmax=4)
        eg, e3, m1, sigma, k3, k4 = estimate_gs(mu, dim, exact)
        print(f"\n  {name}: dim {dim}")
        print(f"    moments mu1..4: "
              + ", ".join(f"{m:.6g}" for m in mu))
        print(f"    normalised cumulants: k3/s^3 = {k3:.4f}, "
              f"k4/s^4 = {k4:.4f}  (Gaussian -> 0)")
        print(f"    exact GS {exact:.8f} | Gaussian-tail est {eg:.8f} "
              f"(err {abs(eg - exact):.2e})")
        print(f"    3-moment est {e3:.8f} (err {abs(e3 - exact):.2e})")

        # timing: Tr(H^2) accumulation (the O(nnz) incremental cost)
        coo = H.tocoo()
        t0 = time.time()
        tr2 = (np.abs(coo.data) ** 2).sum()
        t_np = time.time() - t0
        print(f"    Tr(H^2) incremental: {t_np:.3f}s for {coo.nnz:.2e} "
              f"elements ({t_np * 1e9 / coo.nnz:.1f} ns/elem, numpy)")
        # GPU check
        try:
            import torch
            if torch.backends.mps.is_available():
                dev = "mps"
            elif torch.cuda.is_available():
                dev = "cuda"
            else:
                dev = None
            if dev == "mps":
                print("    MPS does not support complex128/float64 —")
                print("    FCI needs full precision -> GPU cannot assist")
                print("    (this is itself the answer to the GPU question)")
            elif dev == "cuda":
                d = torch.tensor(coo.data, dtype=torch.complex128,
                                 device=dev)
                t0 = time.time()
                tr2t = (d.abs() ** 2).sum().item()
                t_t = time.time() - t0
                print(f"    Tr(H^2) on {dev}: {t_t:.3f}s "
                      f"({t_t * 1e9 / coo.nnz:.1f} ns/elem)")
            else:
                print("    no GPU/MPS available (torch CPU check)")
        except ImportError:
            print("    torch not installed")


if __name__ == "__main__":
    main()
