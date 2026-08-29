"""Tests for the relative-plateau demonstration
(examples/vqe_relative_plateau.py): the SAME Ising n=12 system is
barren under the pure-continuous tool, partially protected under the
continuous+spectrum-anchor tool, and converged under the discrete-
evolution tool — there is no absolute plateau, only a tool-system
mismatch (article 10.86 §9).
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from vqe_relative_plateau import (  # noqa: E402
    N,
    anchored_gradient_rms,
    hea_gradient_rms,
)
from vqe_barren_plateaus import hea_gates, _base_state, ising_hamiltonian  # noqa: E402
from vqe_barren_prewarm import ising_ground_state  # noqa: E402
from vqe_discrete_evolution import diag_values  # noqa: E402
from vqe_evolution_scaling import (  # noqa: E402
    ising_gs_parity,
    sector_alternating_init,
    sector_pure_evolution,
)


@pytest.fixture(scope="module")
def setup():
    gates = hea_gates(N, 2)
    base = _base_state(N)
    _, gs = ising_ground_state(N)
    C = diag_values(N, ising_hamiltonian(N))
    rng = np.random.default_rng(0)
    thA = rng.uniform(-np.pi, np.pi, len(gates))
    th2 = rng.uniform(-np.pi, np.pi, len(gates))
    gm2 = rng.uniform(0, 0.5, 2)
    return dict(gates=gates, base=base, gs=gs, C=C, thA=thA, th2=th2,
                gm2=gm2)


def test_continuous_tool_is_barren(setup):
    """Pure-continuous HEA: gradient RMS < 1e-5 (barren)."""
    s = setup
    g = hea_gradient_rms(s["thA"], s["gates"], s["base"], s["gs"])
    assert g < 1e-5


def test_discrete_tool_converges(setup):
    """Discrete evolution: fidelity > 0.9 to the exact GS."""
    s = setup
    par = ising_gs_parity(N)
    psi = sector_pure_evolution(N, 1000, 100, s["C"],
                                sector_alternating_init(N, par))
    assert abs(np.vdot(s["gs"], psi)) ** 2 > 0.9


def test_spectrum_anchor_partially_protects(setup):
    """Continuous + spectrum anchor: anchored gradient > continuous."""
    s = setup
    gA = hea_gradient_rms(s["thA"], s["gates"], s["base"], s["gs"])
    gA2 = anchored_gradient_rms(s["th2"], s["gm2"], s["gates"], s["base"],
                                s["gs"], s["C"])
    assert gA2 > 100 * gA


def test_anchor_existence_not_layer_count(setup):
    """The protection is the existence of the anchor (L >= 1), not the
    layer count: L=1..4 medians are all well above the barren scale."""
    s = setup
    rng = np.random.default_rng(1)
    for L in (1, 2, 4):
        vals = []
        for _ in range(5):
            th = rng.uniform(-np.pi, np.pi, len(s["gates"]))
            gm = rng.uniform(0, 0.5, L)
            vals.append(anchored_gradient_rms(
                th, gm, s["gates"], s["base"], s["gs"], s["C"], L))
        assert np.median(vals) > 1e-5
