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
           "spin_orbital_integrals", "mo_transform",
           "natural_orbitals_from_rdm1", "transform_integrals",
           "rdm1_from_civec_sz", "molecule_no_basis", "truncate_orbitals"]


def spin_orbital_integrals(o, t):
    """Spatial (RHF) -> spin-orbital integrals in the interleaved
    order (alpha_k = 2k, beta_k = 2k+1), in openfermion's stored
    two_body_tensor layout (reverse-engineered element-wise from
    openfermion and verified via InteractionOperator -> FCI):
    t_s[p,q,r,s] = (1/2) (a_p a_s | a_q a_r)  [chemist notation],
    with the SPIN MATCHING sigma_p = sigma_s AND sigma_q = sigma_r,
    and the 1/2 of H2 folded inside the tensor.  Vectorised.

    *** THIS IS THE ONLY CORRECT ENTRY POINT — DO NOT HAND-ROLL ***
    A hand-written loop that fills t_s[p,q,r,s] = (p q | r s) with
    the same-spin modes (0,0,0,0)/(1,1,1,1)/(0,1,0,1)/(1,0,1,0)
    silently produces a WRONG spin layout (it needs abba/baab, the
    1/2, and the adbc->abcd rearrangement).  Every energy computed
    from such a tensor is wrong: LiH 6-31G E0 came out -19.25 Ha
    instead of -7.9984 (11 Ha off); the error is silent because all
    internal cross-checks (JW-Pauli, sparse, apply) share the same
    wrong tensor and still agree to machine precision.  Only an
    absolute comparison against pyscf FCI catches it.  Verified:
    LiH 6-31G exact E0 = -7.9983583335 == pyscf FCI (5.6e-8)."""
    n = o.shape[0]
    ns = 2 * n
    o_s = np.zeros((ns, ns), dtype=complex)
    o_s[0::2, 0::2] = o            # alpha-alpha
    o_s[1::2, 1::2] = o            # beta-beta
    t_s = np.zeros((ns, ns, ns, ns), dtype=complex)
    v = 0.5 * np.einsum("adbc->abcd", t)   # (a d | b c) / 2
    t_s[0::2, 0::2, 0::2, 0::2] = v          # aaaa
    t_s[1::2, 1::2, 1::2, 1::2] = v          # bbbb
    t_s[0::2, 1::2, 1::2, 0::2] = v          # abba
    t_s[1::2, 0::2, 0::2, 1::2] = v          # baab
    return o_s, t_s


def mo_transform(C, eri):
    """AO -> MO chemist integrals (ab|cd) = sum C^4 (ij|kl) via four
    sequential 6-index einsums (O(n^6) each instead of O(n^8) — the
    scale path for large bases)."""
    t = np.einsum("ijkl,ia->ajkl", eri, C)
    t = np.einsum("ajkl,jb->abkl", t, C)
    t = np.einsum("abkl,kc->abcl", t, C)
    return np.einsum("abcl,ld->abcd", t, C)


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


# ---------------------------------------------------------------------------
# Natural orbital (NO) basis transformation
# ---------------------------------------------------------------------------
#
# Based on article 10.87 §6.05: the natural-orbital basis (NOON-sorted)
# collapses the FCI wavepacket by up to 289x in effective dimension
# (N2 6-31G frozen core, R=1.1): eff-dim 160423 -> 555, HF% 21% -> 80%,
# and the top-k chemical-accuracy requirement drops by ~50x.
#
# The pipeline is:
#   approximate wavefunction (FCI/MP2/CASSCF/WCI) -> 1-RDM ->
#   diagonalise -> natural orbitals (NOON sorted descending) ->
#   transform one/two-body integrals -> (optionally) truncate low-NOON
#   orbitals -> spin_orbital_integrals.
#
# Honest boundary: the NO basis requires an approximate wavefunction first
# (cost proportional to a small FCI or MP2); for 1e8-scale sectors this
# must be combined with the wavepacket-centre iteration (WCI) rather than
# a full FCI.


def natural_orbitals_from_rdm1(rdm1):
    """Natural orbitals from the one-particle reduced density matrix.

    Diagonalises the spatial-orbital 1-RDM (Hermitian, trace = N_electrons,
    eigenvalues in [0,2] for closed-shell-like systems) and sorts the
    eigenvectors by descending occupation number (NOON).

    Parameters
    ----------
    rdm1 : ndarray, shape (n_orb, n_orb)
        The one-particle density matrix in the current (e.g. RHF MO) basis.
        May be real or complex; for real-basis SCF it is real symmetric.

    Returns
    -------
    noon : ndarray, shape (n_orb,)
        Natural occupation numbers, sorted descending.
    U : ndarray, shape (n_orb, n_orb)
        The unitary transformation matrix from the current basis to the
        natural-orbital basis: phi_NO = phi_MO @ U, so integrals transform
        as o_NO = U^T @ o @ U (or U.conj().T for complex).
    """
    rdm1 = np.asarray(rdm1)
    if np.iscomplexobj(rdm1):
        ev, U = np.linalg.eigh(rdm1)
    else:
        ev, U = np.linalg.eigh(np.asarray(rdm1, dtype=float))
    order = np.argsort(ev)[::-1]
    noon = ev[order]
    U = U[:, order]
    return noon, U


def transform_integrals(o, t, U):
    """Transform one- and two-body integrals by a unitary orbital rotation U.

    o_NO = U^T @ o @ U  (real)  or  U.conj().T @ o @ U  (complex)
    t_NO = einsum('pqrs,pi,qj,rk,sl->ijkl', t, U, U, U, U)

    Parameters
    ----------
    o : ndarray, shape (n, n)
        One-body integrals (physicist notation, Hermitian).
    t : ndarray, shape (n, n, n, n)
        Two-body integrals (physicist notation t[p,q,r,s] = t[r,s,p,q]).
    U : ndarray, shape (n, n)
        Unitary transformation matrix (columns are the new orbitals).

    Returns
    -------
    o_no, t_no : ndarray
        Transformed integrals in the new basis.
    """
    o = np.asarray(o)
    t = np.asarray(t)
    U = np.asarray(U)
    if np.iscomplexobj(o) or np.iscomplexobj(U):
        Uh = U.conj().T
        o_no = Uh @ o @ U
        t_no = np.einsum('pqrs,pi,qj,rk,sl->ijkl', t,
                          U.conj(), U.conj(), U, U)
    else:
        o_no = U.T @ o @ U
        t_no = np.einsum('pqrs,pi,qj,rk,sl->ijkl', t, U, U, U, U)
    return o_no, t_no


def truncate_orbitals(noon, o, t, n_keep):
    """Truncate to the n_keep highest-NOON orbitals.

    The natural-orbital basis is sorted by descending occupation; low-NOON
    orbitals contribute negligibly to the ground state (10.87 §6.05: the
    bottom orbitals have NOON ~1e-6 and can be dropped with <1e-6 Ha error).

    Parameters
    ----------
    noon : ndarray, shape (n,)
        Natural occupation numbers (descending).
    o, t : ndarray
        Integrals in the NO basis.
    n_keep : int
        Number of orbitals to keep (the highest-NOON ones).

    Returns
    -------
    noon_trunc, o_trunc, t_trunc : ndarray
        Truncated NOON and integrals.
    """
    n_keep = int(n_keep)
    return (noon[:n_keep].copy(),
            o[:n_keep, :n_keep].copy(),
            t[:n_keep, :n_keep, :n_keep, :n_keep].copy())


def _spin_sign_alpha(az, bz, k):
    """Exterior grading sign for annihilating/creating alpha orbital k:
    (-1)^{popcount(az & (2^k-1)) + popcount(bz & (2^k-1))}."""
    mask = (1 << k) - 1
    cnt = bin(int(az) & mask).count('1') + bin(int(bz) & mask).count('1')
    return 1.0 if (cnt & 1) == 0 else -1.0


def _spin_sign_beta(az, bz, k):
    """Exterior grading sign for beta orbital k (interleaved order):
    (-1)^{popcount(az & (2^{k+1}-1)) + popcount(bz & (2^k-1))}."""
    mask_a = (1 << (k + 1)) - 1
    mask_b = (1 << k) - 1
    cnt = bin(int(az) & mask_a).count('1') + bin(int(bz) & mask_b).count('1')
    return 1.0 if (cnt & 1) == 0 else -1.0


def rdm1_from_civec_sz(civec, unique_idx, az_of, bz_of, rt_a, rt_b,
                        db, n_orb):
    """One-particle density matrix (spatial orbitals) from an S_z-sector
    CI vector (e.g. a converged WCI wavefunction).

    gamma[p,q] = <psi| a^dag_{p,alpha} a_{q,alpha}
                       + a^dag_{p,beta}  a_{q,beta} |psi>

    Diagonal elements: gamma[p,p] = sum_D |c_D|^2 (n_{p,alpha} + n_{p,beta})
    Off-diagonal: single-excitation expectation values with exterior signs.

    This is the WCI-native route to natural orbitals: converge WCI in the
    MO basis, compute the 1-RDM, transform to NO basis, and re-run WCI
    (which should need far fewer wavepackets — 10.87 §6.05).

    Parameters
    ----------
    civec : ndarray, shape (n_var,)
        CI coefficients in the variational space (real for real-basis).
    unique_idx : ndarray, shape (n_var,)
        Combined sector indices of the variational states.
    az_of, bz_of : ndarray
        Rank -> bitstring maps (from wci.build_rank_tables).
    rt_a, rt_b : ndarray
        Bitstring -> rank maps.
    db : int
        Beta-sector dimension.
    n_orb : int
        Number of spatial orbitals.

    Returns
    -------
    gamma : ndarray, shape (n_orb, n_orb)
        One-particle density matrix (real symmetric for real-basis).
    """
    civec = np.asarray(civec, dtype=float)
    unique_idx = np.asarray(unique_idx, dtype=np.int64)
    n_var = len(unique_idx)

    azs = az_of[unique_idx // db]
    bzs = bz_of[unique_idx % db]
    idx_map = {int(idx): i for i, idx in enumerate(unique_idx)}

    gamma = np.zeros((n_orb, n_orb), dtype=float)
    c2 = civec ** 2

    # diagonal: occupation probabilities
    for p in range(n_orb):
        occ_a = ((azs >> p) & 1).astype(float)
        occ_b = ((bzs >> p) & 1).astype(float)
        gamma[p, p] = float(np.sum(c2 * (occ_a + occ_b)))

    # off-diagonal: single-excitation q -> p (alpha then beta)
    for p in range(n_orb):
        for q in range(n_orb):
            if p == q:
                continue
            # alpha excitation q (occupied) -> p (virtual)
            for i in range(n_var):
                az = int(azs[i]); bz = int(bzs[i])
                if not ((az >> q) & 1) or ((az >> p) & 1):
                    continue
                az2 = az ^ (1 << q) ^ (1 << p)
                idx2 = int(rt_a[az2]) * db + int(bz)
                j = idx_map.get(idx2)
                if j is None:
                    continue
                # sign: annihilate q then create p (on the current state)
                az_q = az ^ (1 << q)
                sgn = (_spin_sign_alpha(az, bz, q)
                       * _spin_sign_alpha(az_q, bz, p))
                gamma[p, q] += civec[j] * civec[i] * sgn

            # beta excitation q -> p
            for i in range(n_var):
                az = int(azs[i]); bz = int(bzs[i])
                if not ((bz >> q) & 1) or ((bz >> p) & 1):
                    continue
                bz2 = bz ^ (1 << q) ^ (1 << p)
                idx2 = int(rt_a[az]) * db + int(rt_b[bz2])
                j = idx_map.get(idx2)
                if j is None:
                    continue
                bz_q = bz ^ (1 << q)
                sgn = (_spin_sign_beta(az, bz, q)
                       * _spin_sign_beta(az, bz_q, p))
                gamma[p, q] += civec[j] * civec[i] * sgn

    # symmetrise (should already be symmetric up to numerical error)
    gamma = (gamma + gamma.T) / 2.0
    return gamma


def molecule_no_basis(geometry, basis="sto-3g", nelec=None, method="mp2",
                      frozen_core=0, n_keep=None, verbose=False):
    """End-to-end natural-orbital basis generation for a molecule.

    Pipeline:
      RHF (pyscf) -> approximate 1-RDM (method: mp2 / fci / casscf) ->
      natural orbitals (NOON sorted) -> transform integrals ->
      (optional) truncate to n_keep orbitals -> spin_orbital_integrals.

    Parameters
    ----------
    geometry : str or list
        Molecular geometry (pyscf format, e.g. 'H 0 0 0; H 0 0 0.74').
    basis : str
        Basis set name.
    nelec : int, optional
        Number of electrons (default: from molecule).
    method : str
        Method for the approximate 1-RDM: 'mp2' (cheap, default), 'fci'
        (exact but only for small systems), 'casscf' (multi-reference).
    frozen_core : int
        Number of frozen core orbitals (excluded from the active space).
    n_keep : int, optional
        If set, truncate to the n_keep highest-NOON orbitals.
    verbose : bool
        Print diagnostic information (NOON, basis-covariance check).

    Returns
    -------
    result : dict
        - 'n': number of (active) spatial orbitals
        - 'o_s': spin-orbital one-body integrals (interleaved)
        - 't_s': spin-orbital two-body integrals
        - 'nuc': nuclear repulsion energy
        - 'noon': natural occupation numbers
        - 'U': MO -> NO transformation matrix
        - 'o_mo', 't_mo': original MO integrals (before NO transform)
        - 'nelec': number of electrons in the active space
    """
    from pyscf import gto, scf, ao2mo, mp, mcscf
    from pyscf.fci import direct_spin1

    mol = gto.M(atom=geometry, basis=basis, verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    n_orb_total = mol.nao_nr()
    if nelec is None:
        nelec = mol.nelectron

    n_act = n_orb_total - frozen_core
    C = mf.mo_coeff[:, frozen_core:]
    h1e_mo = C.T @ mf.get_hcore() @ C
    eri_mo = ao2mo.kernel(mol, C, compact=False).reshape(n_act, n_act, n_act, n_act)
    nuc = float(mol.energy_nuc())
    na = nb = nelec // 2

    # --- approximate 1-RDM ---
    if method == "fci":
        cis = direct_spin1.FCISolver(mol)
        cis.verbose = 0
        cis.max_space = 8
        cis.conv_tol = 1e-8
        E_fci, c_fci = cis.kernel(h1e_mo, eri_mo, n_act, (na, nb))
        dm1a, dm1b = direct_spin1.make_rdm1s(c_fci, n_act, (na, nb))
        rdm1 = np.asarray(dm1a) + np.asarray(dm1b)
        if verbose:
            print(f'  FCI E = {E_fci + nuc:.8f}')
    elif method == "mp2":
        pt = mp.MP2(mf, frozen=frozen_core)
        pt.verbose = 0
        pt.kernel()
        rdm1 = np.asarray(pt.make_rdm1())
        # MP2 rdm1 is in the full MO basis; slice to active
        rdm1 = rdm1[frozen_core:, frozen_core:]
    elif method == "casscf":
        mc = mcscf.CASSCF(mf, n_act, nelec)
        mc.verbose = 0
        mc.frozen = frozen_core
        mc.kernel()
        rdm1 = np.asarray(mc.make_rdm1())
        rdm1 = rdm1[frozen_core:, frozen_core:]
    else:
        raise ValueError(f"Unknown method '{method}'; use 'mp2', 'fci', or 'casscf'")

    # --- natural orbitals ---
    noon, U = natural_orbitals_from_rdm1(rdm1)
    if verbose:
        print(f'  NOON top-6 = {np.round(noon[:6], 4)}')
        print(f'  NOON mid   = {np.round(noon[na-1:na+3], 4)}')
        print(f'  trace(rdm1) = {np.trace(rdm1):.6f} (expect {nelec})')

    # --- transform integrals ---
    o_no, t_no = transform_integrals(h1e_mo, eri_mo, U)

    # --- optional truncation ---
    if n_keep is not None and n_keep < n_act:
        noon, o_no, t_no = truncate_orbitals(noon, o_no, t_no, n_keep)
        n_act = n_keep
        if verbose:
            print(f'  truncated to {n_keep} orbitals')

    # --- basis-covariance check (FCI in NO basis == FCI in MO basis) ---
    if verbose and method == "fci" and n_keep is None:
        cis2 = direct_spin1.FCISolver(mol)
        cis2.verbose = 0
        cis2.max_space = 8
        cis2.conv_tol = 1e-8
        E_no, _ = cis2.kernel(o_no, t_no, n_act, (na, nb))
        print(f'  FCI(NO) E = {E_no + nuc:.8f}, |E_NO - E_MO| = {abs(E_no - E_fci):.2e}')

    # --- spin-orbital integrals (the only correct entry point) ---
    o_s, t_s = spin_orbital_integrals(o_no, t_no)

    return {
        "n": n_act,
        "o_s": o_s,
        "t_s": t_s,
        "nuc": nuc,
        "noon": noon,
        "U": U,
        "o_mo": h1e_mo,
        "t_mo": eri_mo,
        "nelec": nelec,
    }
