"""Integral input — the physics data (honestly labelled).

The SCF one/two-body integrals are the physical input of the whole
pipeline, taken from the standard library (openfermion + pyscf).
This is NOT a geometric step: the integrals are data (equivalent to
measured constants), and re-deriving them with geometry would be
re-implementing physics input, not geometrising computation.  The
geometrised pipeline starts AFTER the integrals:

    integrals -> exterior sector Hamiltonian -> solve/evolve

(M2 of the project will add the Grassmann-manifold SCF, at which
point the SCF step itself becomes a geometric computation while the
AO integrals remain standard numerical input.)

Two entry points:

  - integrals_from_openfermion : one/two-body tensors + constant
    (physicist notation, t[p,q,r,s] = t[r,s,p,q]) + optional FCI;
  - molecule_terms              : (n, const, z_terms, off_terms) in
    the legacy Pauli-term format (for the sector/fast/merged builds).
"""

import numpy as np

__all__ = ["integrals_from_openfermion", "molecule_terms"]


def integrals_from_openfermion(geometry, basis, run_fci=False):
    """(n, o, t, const, fci_energy) — the SCF integrals (the physics
    data) from openfermion; everything after this point is exterior
    algebra (see geoqc.exterior)."""
    from openfermion import MolecularData
    from openfermionpyscf import run_pyscf
    mol = MolecularData(geometry=geometry, basis=basis, multiplicity=1)
    mol = run_pyscf(mol, run_scf=True, run_fci=run_fci)
    Hf = mol.get_molecular_hamiltonian()
    n = mol.n_qubits
    o = np.asarray(Hf.one_body_tensor)
    t = np.asarray(Hf.two_body_tensor)
    return n, o, t, float(Hf.constant), (float(mol.fci_energy)
                                         if run_fci else None)


def molecule_terms(geometry, basis="sto-3g"):
    """(n, const, z_terms, off_terms) — the legacy Pauli-term format
    (JW expansion, big-endian axes) for the sector/fast/merged builds
    and their machine-precision cross-checks against the exterior
    build.  SCF only; never materialises the full-space vector."""
    from openfermion import MolecularData, jordan_wigner
    from openfermionpyscf import run_pyscf
    mol = MolecularData(geometry=geometry, basis=basis, multiplicity=1)
    mol = run_pyscf(mol, run_scf=True)
    Hq = jordan_wigner(mol.get_molecular_hamiltonian())
    n = mol.n_qubits
    z_terms, off_terms = [], []
    for t, c in Hq.terms.items():
        if not t:
            continue
        axis = ["I"] * n
        for q, p in t:
            axis[q] = p
        ax = "".join(axis)
        if any(ch in "XY" for ch in ax):
            off_terms.append((float(c.real), ax))
        else:
            z_terms.append((float(c.real), ax))
    return n, float(Hq.constant.real), z_terms, off_terms
