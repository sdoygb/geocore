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

For the Grassmann-manifold SCF (geoqc.scf) the atomic-orbital
integrals are needed:

  - ao_integrals                 : (n, h_core, eri, nuc) in the AO
    basis (chemist notation (ij|kl)) from pyscf's GTO numerical
    integrals — the physics input of the SCF step itself.
"""

import numpy as np

__all__ = ["integrals_from_openfermion", "molecule_terms", "ao_integrals",
           "spin_orbital_integrals"]


def spin_orbital_integrals(o, t):
    """Spatial (RHF) -> spin-orbital integrals in the interleaved
    order (alpha_k = 2k, beta_k = 2k+1), in openfermion's stored
    two_body_tensor layout (reverse-engineered element-wise from
    openfermion and verified via InteractionOperator -> FCI):
    t_s[p,q,r,s] = (1/2) (a_p a_s | a_q a_r)  [chemist notation],
    with the SPIN MATCHING sigma_p = sigma_s AND sigma_q = sigma_r,
    and the 1/2 of H2 folded inside the tensor."""
    n = o.shape[0]
    ns = 2 * n
    o_s = np.zeros((ns, ns), dtype=complex)
    t_s = np.zeros((ns, ns, ns, ns), dtype=complex)
    for i in range(n):
        for j in range(n):
            o_s[2 * i, 2 * j] = o[i, j]          # alpha-alpha
            o_s[2 * i + 1, 2 * j + 1] = o[i, j]  # beta-beta
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    v = 0.5 * t[a, d, b, c]      # (a d | b c) / 2
                    t_s[2 * a,     2 * b,     2 * c,     2 * d] = v      # aaaa
                    t_s[2 * a + 1, 2 * b + 1, 2 * c + 1, 2 * d + 1] = v  # bbbb
                    t_s[2 * a,     2 * b + 1, 2 * c + 1, 2 * d] = v      # abba
                    t_s[2 * a + 1, 2 * b,     2 * c,     2 * d + 1] = v  # baab
    return o_s, t_s


def ao_integrals(geometry, basis="sto-3g"):
    """(n, h_core, eri, S, nuc) — the atomic-orbital integrals from
    pyscf's GTO numerical evaluation (the physics input of the SCF
    step, honestly labelled: numerical GTO integrals are standard
    mathematics, not geometry).  h_core = kinetic + nuclear, eri in
    chemist notation (ij|kl), eight-fold symmetric, S the overlap."""
    from pyscf import gto
    mol = gto.M(atom=geometry, basis=basis)
    n = mol.nao_nr()
    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri = mol.intor("int2e")
    S = mol.intor("int1e_ovlp")
    return (n, np.asarray(h_core), np.asarray(eri), np.asarray(S),
            float(mol.energy_nuc()))


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
