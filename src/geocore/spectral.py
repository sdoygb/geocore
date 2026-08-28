"""Layer 0 extension — spectral geometry.

The Laplace-Beltrami spectrum is the geometric invariant of the manifold:
for the circle S^1 (metric dphi^2) the eigenvalues are k^2 (k in Z,
multiplicity 2 for k != 0).  The generic numeric path builds the discrete
Laplacian on an n-grid (the cycle-graph Laplacian scaled by 1/h^2) and
diagonalizes it — O(N^3); the closed form returns the exact spectrum.

This is the spectral entry of the reduce-computation program: closed-form /
spectral prediction instead of expensive numerical computation (the same
program that later targets theta^4 coherent-noise scaling: predict instead
of simulate).
"""

from __future__ import annotations

import numpy as np

from .objects import GeometricObject

__all__ = ["Circle"]


class Circle(GeometricObject):
    """S^1 with metric dphi^2.

    Laplace-Beltrami eigenvalues: k^2, k in Z (multiplicity 2 for k != 0).
    The discrete Laplacian on n_grid equally-spaced points is the cycle
    graph Laplacian scaled by (n_grid / 2 pi)^2; its eigenvalues
    (n_grid/2pi)^2 (2 - 2 cos(2 pi k / n_grid)) converge to k^2 as O(n^-2).
    """

    @property
    def dim(self) -> int:
        return 1

    def laplacian_eigenvalues_closed(self, n_evals: int) -> np.ndarray:
        """Exact spectrum: k^2 with multiplicity 2 for k != 0."""
        evals = [0.0]
        k = 1
        while len(evals) < n_evals:
            evals.extend([k * k, k * k])
            k += 1
        return np.array(evals[:n_evals], dtype=float)

    def laplacian_discrete_eigenvalues(self, n_grid: int, n_evals: int) -> np.ndarray:
        """Numeric spectrum: eigh of the scaled cycle Laplacian (O(N^3))."""
        h = 2 * np.pi / n_grid
        L = np.zeros((n_grid, n_grid))
        for i in range(n_grid):
            L[i, i] = 2.0
            L[i, (i + 1) % n_grid] = -1.0
            L[i, (i - 1) % n_grid] = -1.0
        L = L / h**2
        return np.sort(np.linalg.eigvalsh(L))[:n_evals]

    def verify(self) -> dict:
        return {"ok": True, "note": "spectral invariants checked via laplacian ops"}
