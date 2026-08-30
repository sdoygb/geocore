"""geoqc — a geometrised quantum-chemistry / quantum-computation core.

Every step of the pipeline speaks the geometry (article 10.86
§9.04-§9.09; features 45-50 of geocore):

  - sector       : particle-number sectors, combinatorial builds,
                   spectral truncation, commuting merges (F45/46)
  - exterior     : exterior-algebra (Clifford) fermionic action —
                   wedge/contraction, grading signs, the sector
                   Hamiltonian without Jordan-Wigner/Pauli (F47),
                   and the matrix-free H|v> operator (F49)
  - schubert     : the Grassmannian Schubert-cell structure of the
                   sector basis, Bruhat order, S_z bipartiteness (F50)
  - manifold     : Fubini-Study metric of the state space, path
                   geometry, adiabatic-quality measures (F48)
  - descent      : Krylov exponentials — the numerical form of the
                   discrete descent (0.13): evolution and ground
                   states from matvecs only (F49)
  - integrals    : the physics input — SCF one/two-body integrals
                   (openfermion/pyscf interface, honestly labelled:
                   physics data, not geometry)

Honest boundaries (the same ones carried through features 45-50):
  - the SCF integrals are physical input data, taken from the
    standard library and not re-derived by geometry (M2 of the
    project will add the Grassmann-manifold SCF);
  - FCI/CCSD reference energies are independent verification
    benchmarks, never computed by the method under test;
  - every geometric step is machine-verified equal to the standard
    computation it replaces.
"""

from . import sector, exterior, schubert, manifold, descent, integrals  # noqa: F401

__version__ = "0.1.0"
