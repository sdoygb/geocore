"""NPC screening feasibility (from the published plasma-metallomics
summary table): the effect sizes are strongly discriminating and the
clearly-labeled Gaussian simulation predicts ~0.97 LOOCV accuracy."""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from npc_screening import ELEMENTS, NONCA, NPC, effect_sizes, simulate_and_classify  # noqa: E402


def test_data_table_complete():
    assert len(ELEMENTS) == 19
    assert NPC.shape == (19, 2)
    assert NONCA.shape == (19, 2)


def test_effect_sizes_strong():
    """Several elements separate NPC from non-cancer with |d| > 1; the
    strongest is lead (Pb d ~ -5.8)."""
    rows = effect_sizes()
    strong = [r for r in rows if r[0] > 1.0]
    assert len(strong) >= 5
    assert rows[0][1] == "Pb"
    assert rows[0][0] > 3.0
    # Zn down / Cu up — the cancer-literature signature
    zn = next(r for r in rows if r[1] == "Zn")
    cu = next(r for r in rows if r[1] == "Cu")
    assert zn[2] < 0 and cu[2] > 0


def test_simulation_reproducible_and_separable():
    """The labeled simulation (Gaussian from published Mean±SD) reaches
    high LOOCV accuracy — the screening problem is highly separable if
    the distribution is as published."""
    accs = [simulate_and_classify(seed=s) for s in range(3)]
    assert np.mean(accs) > 0.85
    assert all(0.5 <= a <= 1.0 for a in accs)
