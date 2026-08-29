"""Application layer — quantum error correction (coherent-noise scaling).

Coherent rotation noise R_X(theta) on the n-qubit repetition code: the
logical error rate is EXACTLY

    P_L(n, theta) = sum_{k > n/2} C(n,k) sin^{2k}(theta/2) cos^{2(n-k)}(theta/2),

whose leading term is the coherent-noise law

    P_L(n, theta) ~ C(n, (n+1)/2) (theta/2)^{n+1}    (theta -> 0).

MEASURED FINDING (this module): the theta^4 law is the d = 3 special case,
NOT universal — the measured exponents are 4.00, 6.00, 8.00, 10.00 for
distances d = 3, 5, 7, 9, i.e. the general law is theta^{d+1} (the known
coherent-noise enhancement over the theta^d incoherent scaling).  Every
coefficient is verified exactly (3/16, 5/32, 35/256, ...).

The generic path is the O(2^n) exact state-vector simulation; the shortcut
is the O(1) leading-law prediction (prediction instead of simulation).
"""

from __future__ import annotations

import dataclasses

import numpy as np

__all__ = [
    "repetition_code_logical_error",
    "repetition_closed_form",
    "theta4_leading",
    "scaling_leading",
    "measure_scaling_exponent",
    "logical_error_sweep",
    "pseudo_threshold",
    "crossover",
    "diagnose",
    "QECDiagnosticReport",
]


def repetition_code_logical_error(theta: float, n: int = 3) -> float:
    """Generic path: exact state-vector simulation (O(2^n)).

    Encode |0_L> = |000>, apply coherent X-rotation noise R_X(theta) to
    every data qubit, then majority-vote correction.  Returns P_L, the
    total probability mass that corrects to the wrong logical value.
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    nbits = 2**n
    pL = 0.0
    for b in range(nbits):
        ones = bin(b).count("1")
        amp = (-1j) ** ones * s**ones * c ** (n - ones)
        if ones > n // 2:  # majority flips to |1_L>: logical error
            pL += abs(amp) ** 2
    return pL


def repetition_closed_form(theta: float, n: int = 3) -> float:
    """Exact closed form for the n-qubit repetition code under R_X(theta).

    The logical error is the total probability mass whose Hamming weight
    exceeds n/2 (majority correction): sum over k > n/2 of
    C(n,k) sin^{2k}(theta/2) cos^{2(n-k)}(theta/2).
    """
    s, c = np.sin(theta / 2), np.cos(theta / 2)
    from math import comb

    pL = 0.0
    for k in range(n // 2 + 1, n + 1):
        pL += comb(n, k) * s ** (2 * k) * c ** (2 * (n - k))
    return pL


def theta4_leading(theta: float) -> float:
    """The theta^4 law for the distance-3 code: P_L ~ (3/16) theta^4.

    NOT universal: the general law is theta^{d+1} (see scaling_leading).
    """
    return scaling_leading(theta, 3)


def scaling_leading(theta: float, n: int) -> float:
    """The general coherent-noise leading law for the distance-n code:

    P_L ~ C(n, (n+1)/2) (theta/2)^{n+1}   (theta -> 0).
    """
    from math import comb

    k_min = (n + 1) // 2
    return comb(n, k_min) * (float(theta) / 2) ** (n + 1)


def measure_scaling_exponent(n: int = 3, thetas=None) -> tuple[float, float]:
    """Measure the empirical scaling exponent of P_L(theta) (log-log slope).

    Returns (exponent, leading_coefficient).  For the n-qubit repetition
    code under coherent X-noise this measures n+1 and
    C(n, (n+1)/2) / 2^{n+1} — the general theta^{d+1} law, verified
    empirically rather than assumed.
    """
    if thetas is None:
        thetas = np.array([0.01, 0.02, 0.04, 0.08, 0.16])
    PLs = np.array([repetition_code_logical_error(t, n) for t in thetas])
    slope, intercept = np.polyfit(np.log(thetas), np.log(PLs), 1)
    return float(slope), float(np.exp(intercept))


# ---------------------------------------------------------------------------
# QEC diagnostics application layer
# ---------------------------------------------------------------------------


def logical_error_sweep(n: int, thetas) -> np.ndarray:
    """Vectorized P_L(theta) over a sweep of noise strengths (the batch
    core path: one O(n) vectorized sum instead of n_theta scalar calls)."""
    thetas = np.atleast_1d(np.asarray(thetas, dtype=float))
    from math import comb

    s = np.sin(thetas / 2)
    c = np.cos(thetas / 2)
    pL = np.zeros_like(thetas)
    for k in range(n // 2 + 1, n + 1):
        pL = pL + comb(n, k) * s ** (2 * k) * c ** (2 * (n - k))
    return pL


def pseudo_threshold(n: int) -> float:
    """The pseudo-threshold: the largest theta with P_L(n, theta) < P_phys
    (the physical single-qubit error rate sin^2(theta/2)).

    For the distance-3 code this is EXACT: P_L(3) = s^2 (3 - 2s) with
    s = sin^2(theta/2), so P_L = P_phys at s = 1/2, i.e. theta* = pi/2.
    General n: numerical root (brentq), verified by substitution in tests.
    """
    if n == 3:
        return float(np.pi / 2)
    from scipy.optimize import brentq

    def g(theta):
        return repetition_closed_form(theta, n) - np.sin(theta / 2) ** 2

    lo, hi = 1e-8, np.pi - 1e-8
    if g(lo) >= 0 or g(hi) <= 0:
        raise ValueError(f"pseudo_threshold(n={n}): no crossing in (0, pi)")
    return float(brentq(g, lo, hi))


def crossover(n1: int, n2: int) -> float:
    """The noise strength where P_L(n1, theta) = P_L(n2, theta) — the
    distance at which the two codes are equally protected (for coherent
    noise the larger code wins at low theta, so this is the crossover
    point in their favor)."""
    from scipy.optimize import brentq

    def g(theta):
        return repetition_closed_form(theta, n1) - repetition_closed_form(theta, n2)

    lo, hi = 1e-8, np.pi - 1e-8
    g_lo, g_hi = g(lo), g(hi)
    if g_lo * g_hi > 0:
        raise ValueError(f"crossover({n1}, {n2}): no crossing in (0, pi)")
    return float(brentq(g, lo, hi))


@dataclasses.dataclass
class QECDiagnosticReport:
    """The measured diagnostic of a repetition-code family under coherent
    X-noise.  Every field has an analytic counterpart (verified in tests):
    exponents vs n+1, leading coefficients vs C(n,(n+1)/2)/2^{n+1},
    pseudo-threshold of the d=3 code vs the exact pi/2, and any crossover
    verified by substitution."""

    distances: list
    thetas: np.ndarray
    logical_errors: np.ndarray  # (D, T)
    empirical_exponents: np.ndarray  # per distance
    analytic_exponents: np.ndarray  # n + 1
    exponent_errors: np.ndarray
    leading_coefficients: np.ndarray  # measured (fit)
    analytic_coefficients: np.ndarray  # C(n,(n+1)/2) / 2^{n+1}
    coefficient_relative_errors: np.ndarray
    pseudo_thresholds: dict  # n -> theta*
    crossover: float | None = None

    def __repr__(self):
        lines = ["QECDiagnosticReport"]
        for i, n in enumerate(self.distances):
            lines.append(
                f"  d={n}: P_L~{self.leading_coefficients[i]:.4f} "
                f"theta^{self.empirical_exponents[i]:.3f} "
                f"(analytic theta^{n + 1}, coeff "
                f"{self.analytic_coefficients[i]:.4f}); "
                f"pseudo-threshold theta*={self.pseudo_thresholds[n]:.4f}"
            )
        if self.crossover is not None:
            lines.append(f"  crossover P_L(d1)=P_L(d2) at theta={self.crossover:.4f}")
        return "\n".join(lines)


def diagnose(distances=(3, 5, 7), thetas=None) -> QECDiagnosticReport:
    """Run the diagnostic over a code family: sweeps, empirical scaling
    exponents and coefficients, pseudo-thresholds, and (for >= 2 codes)
    the crossover.  ``thetas`` should be a small-to-large sweep (the
    smallest half is used for the leading-law fits)."""
    from math import comb

    if thetas is None:
        thetas = np.array([0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32])
    thetas = np.asarray(thetas, dtype=float)
    fit = thetas[: max(3, len(thetas) // 2)]  # small-theta window
    errors = []
    exponents = []
    coeffs = []
    a_coeffs = []
    c_errors = []
    thresholds = {}
    for n in distances:
        pl = logical_error_sweep(n, thetas)
        pl_fit = logical_error_sweep(n, fit)
        slope, intercept = np.polyfit(np.log(fit), np.log(pl_fit), 1)
        exponents.append(slope)
        errors.append(abs(slope - (n + 1)))
        coeff = float(np.exp(intercept))
        coeffs.append(coeff)
        a_coeff = comb(n, (n + 1) // 2) / 2.0 ** (n + 1)
        a_coeffs.append(a_coeff)
        c_errors.append(abs(coeff - a_coeff) / a_coeff)
        thresholds[n] = pseudo_threshold(n)
    cross = crossover(distances[0], distances[1]) if len(distances) >= 2 else None
    return QECDiagnosticReport(
        distances=list(distances),
        thetas=thetas,
        logical_errors=np.vstack([logical_error_sweep(n, thetas) for n in distances]),
        empirical_exponents=np.array(exponents),
        analytic_exponents=np.array([n + 1 for n in distances]),
        exponent_errors=np.array(errors),
        leading_coefficients=np.array(coeffs),
        analytic_coefficients=np.array(a_coeffs),
        coefficient_relative_errors=np.array(c_errors),
        pseudo_thresholds=thresholds,
        crossover=cross,
    )
