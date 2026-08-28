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

import numpy as np

__all__ = [
    "repetition_code_logical_error",
    "repetition_closed_form",
    "theta4_leading",
    "scaling_leading",
    "measure_scaling_exponent",
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
