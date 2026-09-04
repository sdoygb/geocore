"""Spectral clustering measure — the continuous multireference strength
diagnostic (article 10.89 §6.4).

The mean adjacent gap ratio

    r_i = min(E_{i+1}-E_i, E_i-E_{i-1}) / max(E_{i+1}-E_i, E_i-E_{i-1})
    <r>  = mean_i r_i

distinguishes spectral universality classes:
  - Poisson (integrable / single-reference):     <r> ≈ 0.386
  - Wigner-Dyson GOE (chaotic):                   <r> ≈ 0.536
  - spectral clustering (multireference near-
    degeneracy / broken contacts):                 <r> < 0.386

Article 10.89 §6.4 verified across five systems / three basis sets
(STO-3G / 6-31G / cc-pVDZ, frozen-core and all-electron) that <r> is a
continuous multireference-strength measure:
  - single-reference (equilibrium):        <r> ∈ [0.39, 0.41]
  - weak multireference (LiH charge
    transfer):                              <r> ∈ [0.33, 0.35]
  - strong multireference (N2/H2O
    sigma^2/sigma*^2 competition):         <r> ∈ [0.18, 0.28]
  - extreme (H2O fully dissociated):       <r> ≈ 0.18

The spectrum must be computed within each symmetry block (D2h 8 blocks /
C2v 4 blocks) separately — cross-block gaps are symmetry-forbidden and
would artificially inflate <r>.  This module provides:
  - gap_ratios(eigenvalues): raw r_i sequence
  - mean_gap_ratio(eigenvalues): <r> for a single block
  - spectral_clustering(eigenvalues_by_block): weighted <r> across blocks
  - molecule_spectral_clustering(mol, ...): end-to-end via pyscf FCI
"""

import numpy as np

__all__ = [
    "gap_ratios", "mean_gap_ratio", "spectral_clustering",
    "molecule_spectral_clustering",
]


def gap_ratios(eigenvalues):
    """Adjacent gap ratio sequence r_i for a sorted eigenvalue list.

    r_i = min(d_i, d_{i+1}) / max(d_i, d_{i+1}), where d_i = E_i - E_{i-1}.
    Returns an array of length len(E)-2 (indices 1..n-2).

    Parameters
    ----------
    eigenvalues : ndarray, shape (n,)
        Sorted eigenvalues (ascending) of a single symmetry block.

    Returns
    -------
    r : ndarray, shape (n-2,)
        Gap ratio sequence.
    """
    E = np.sort(np.asarray(eigenvalues, dtype=float))
    if len(E) < 3:
        return np.array([])
    d = np.diff(E)
    d_prev = d[:-1]
    d_next = d[1:]
    # avoid division by zero (exact degeneracies -> r=0)
    denom = np.maximum(d_prev, d_next)
    safe = denom > 0
    r = np.zeros_like(d_prev)
    r[safe] = np.minimum(d_prev[safe], d_next[safe]) / denom[safe]
    return r


def mean_gap_ratio(eigenvalues, min_states=5):
    """Mean gap ratio <r> for a single symmetry block.

    Parameters
    ----------
    eigenvalues : ndarray
        Sorted eigenvalues of one block.
    min_states : int
        Minimum number of states required (blocks with fewer states are
        skipped in the weighted average).

    Returns
    -------
    r_mean : float or None
        Mean gap ratio, or None if the block has fewer than min_states.
    n_states : int
        Number of states in the block.
    """
    E = np.asarray(eigenvalues, dtype=float)
    n = len(E)
    if n < min_states:
        return None, n
    r = gap_ratios(E)
    if len(r) == 0:
        return None, n
    return float(np.mean(r)), n


def spectral_clustering(eigenvalues_by_block, min_states=5):
    """Weighted mean gap ratio <r> across symmetry blocks.

    Each block contributes proportionally to its number of gap ratios
    (n_states - 2).  Blocks with fewer than min_states are skipped.

    Parameters
    ----------
    eigenvalues_by_block : dict or list
        Mapping from block label to sorted eigenvalue array, or a list of
        eigenvalue arrays.
    min_states : int
        Minimum states per block.

    Returns
    -------
    result : dict
        - 'r_mean': weighted mean gap ratio <r>
        - 'blocks': per-block results {label: (r_mean, n_states)}
        - 'n_blocks_used': number of blocks included in the average
        - 'n_total': total states across all blocks
    """
    if isinstance(eigenvalues_by_block, dict):
        items = list(eigenvalues_by_block.items())
    else:
        items = [(i, E) for i, E in enumerate(eigenvalues_by_block)]

    block_results = {}
    total_weight = 0
    weighted_sum = 0.0
    n_total = 0

    for label, E in items:
        r_mean, n_states = mean_gap_ratio(E, min_states=min_states)
        block_results[label] = (r_mean, n_states)
        n_total += n_states
        if r_mean is not None:
            weight = n_states - 2
            weighted_sum += r_mean * weight
            total_weight += weight

    r_mean = weighted_sum / total_weight if total_weight > 0 else None

    return {
        "r_mean": r_mean,
        "blocks": block_results,
        "n_blocks_used": int(total_weight > 0),
        "n_total": n_total,
    }


def molecule_spectral_clustering(mol, mo_coeff=None, h1e=None, eri=None,
                                 nelec=None, nroots=200, tol=1e-10,
                                 min_states=5, verbose=0):
    """End-to-end spectral clustering measure for a molecule via pyscf FCI.

    Computes the FCI spectrum within each symmetry block (using pyscf's
    symmetry-adapted FCI) and returns the weighted mean gap ratio <r>.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        The molecule (with symmetry enabled: mol.symmetry = True).
    mo_coeff : ndarray, optional
        MO coefficients (default: RHF MOs).
    h1e, eri : ndarray, optional
        One- and two-body integrals in the MO basis (default: computed
        from mol and mo_coeff).
    nelec : tuple, optional
        (n_alpha, n_beta) (default: mol.nelec).
    nroots : int
        Number of FCI roots per symmetry block.
    tol : float
        FCI convergence tolerance.
    min_states : int
        Minimum states per block for the gap ratio.
    verbose : int
        pyscf verbosity.

    Returns
    -------
    result : dict
        - 'r_mean': weighted mean gap ratio <r>
        - 'blocks': {irrep_name: (r_mean, n_states, energies)}
        - 'n_total': total states
    """
    from pyscf import scf, fci, ao2mo

    if nelec is None:
        nelec = mol.nelec

    # RHF if no MO coefficients given
    if mo_coeff is None:
        mf = scf.RHF(mol)
        mf.verbose = verbose
        mf.kernel()
        mo_coeff = mf.mo_coeff

    # Handle UHF tuple mo_coeff: use alpha MOs as common basis
    # (for open-shell systems like O2 triplet, alpha/beta share spatial
    #  orbitals in the minimal basis; FCI direct_spin1 handles spin)
    if isinstance(mo_coeff, (tuple, list)):
        mo_coeff = mo_coeff[0]  # alpha coefficients

    if h1e is None or eri is None:
        h1e = mo_coeff.T @ mol.intor("int1e_kin") @ mo_coeff + \
              mo_coeff.T @ mol.intor("int1e_nuc") @ mo_coeff
        eri = ao2mo.kernel(mol, mo_coeff, compact=False)
        n_orb = mo_coeff.shape[1]
        eri = eri.reshape(n_orb, n_orb, n_orb, n_orb)

    n_orb = mo_coeff.shape[1]

    # Get symmetry-adapted orbital basis if molecule has symmetry
    try:
        from pyscf import symm
        orb_irrep_names = symm.label_orb_symm(mol, mol.irrep_id,
                                                mol.irrep_name, mo_coeff)
        # convert names to integer IDs (pyscf orbsym requires ints)
        name_to_id = {name: idx for idx, name in enumerate(mol.irrep_name)}
        orbsym_ids = np.array([name_to_id.get(n, 0) for n in orb_irrep_names],
                               dtype=np.int64)
        has_symm = True
    except Exception:
        has_symm = False
        orbsym_ids = None

    blocks = {}

    if has_symm:
        # FCI within each symmetry block
        for irrep_id, irrep_name in enumerate(mol.irrep_name):
            try:
                cis = fci.direct_spin1.FCISolver(mol)
                cis.verbose = verbose
                cis.conv_tol = tol
                cis.nroots = nroots
                # symmetry-adapted FCI: orbsym = orbital irrep IDs,
                # wfnsym = target wavefunction irrep ID
                e, c = cis.kernel(h1e, eri, n_orb, nelec,
                                  orbsym=orbsym_ids, wfnsym=irrep_id)
                e = np.atleast_1d(e)
                if len(e) >= min_states:
                    blocks[irrep_name] = e
            except Exception:
                continue
    else:
        # No symmetry: single block (S_z sector only)
        cis = fci.direct_spin1.FCISolver(mol)
        cis.verbose = verbose
        cis.conv_tol = tol
        cis.nroots = nroots
        e, c = cis.kernel(h1e, eri, n_orb, nelec)
        e = np.atleast_1d(e)
        blocks["A1"] = e

    result = spectral_clustering(blocks, min_states=min_states)

    # attach full energies per block
    blocks_full = {}
    for label, (r_mean, n_states) in result["blocks"].items():
        blocks_full[label] = (r_mean, n_states, blocks.get(label, np.array([])))
    result["blocks"] = blocks_full

    return result
