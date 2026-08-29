"""Tests for the spectrum-guided parameterization against barren
plateaus (examples/vqe_spectrum_guided.py): the mixed-ansatz analytic
gradient is machine-verified, the spectrum-guided (diagonal-phase)
parameters carry a larger gradient than the random-axis HEA parameters
with the ratio growing in n and a slower decay slope, and the mixed
ansatz converges at least as well and more stably than the pure HEA.

This is the machine-verified instance of the "rebuild the landscape"
lever: the parameterization geometry, not the sampling resolution, is
the handle on the plateau (contrast the warm-start of feature 32,
which moves to a good region instead).
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from vqe_spectrum_guided import (  # noqa: E402
    gradient_scan,
    run_adam_hea,
    run_adam_mixed,
    verify_gradient,
)


def test_mixed_analytic_gradient_machine_verified():
    err = verify_gradient(8)
    assert err < 1e-8


def test_spectrum_params_have_larger_gradient():
    """The diagonal-phase (problem-spectrum) parameters carry a larger
    gradient than the random-axis HEA parameters."""
    for n in (8, 10):
        h, s = gradient_scan(n, npts=15)
        assert s > 2 * h


def test_advantage_grows_with_n():
    """The ratio spectrum/HEA gradient is larger at n=12 than at n=6."""
    h6, s6 = gradient_scan(6, npts=20)
    h12, s12 = gradient_scan(12, npts=20)
    assert (s12 / h12) > (s6 / h6)


def test_spectrum_params_decay_slower():
    """log10 slope per qubit: spectrum params > HEA params (both decay,
    but the guided ones slower)."""
    ns = list(range(6, 13))
    rh, rg = [], []
    for n in ns:
        h, s = gradient_scan(n, npts=12)
        rh.append(h)
        rg.append(s)
    sl_h, _ = np.polyfit(ns, np.log10(rh), 1)
    sl_g, _ = np.polyfit(ns, np.log10(rg), 1)
    assert sl_g > sl_h
    assert sl_h < -0.2   # the HEA part is genuinely barren
    assert sl_g < 0.0    # the guided part still decays (honest)


def test_mixed_ansatz_converges_better_or_equal():
    """Adam 300 steps, n=8, 4 starts: the mixed ansatz median fidelity
    is >= the pure HEA median, and its worst start is better."""
    from vqe_barren_plateaus import hea_gates, _base_state  # noqa: F401
    rng = np.random.default_rng(0)
    r_hea, r_mix = [], []
    for s in range(4):
        th0 = rng.uniform(-np.pi, np.pi, len(hea_gates(8, 2)))
        gm0 = rng.uniform(0, 0.5, 2)
        r_hea.append(run_adam_hea(8, th0))
        r_mix.append(run_adam_mixed(8, th0, gm0))
    assert np.median(r_mix) >= np.median(r_hea) - 1e-6
    assert min(r_mix) >= min(r_hea) - 1e-6
