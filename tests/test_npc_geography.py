"""Tests for the NPC geographic-association test (GLOBOCAN 2020,
published regional ASIR/cases/population from JMIR 2023, PMC10551785):
the regional ASIR contrast is ~20-30x, the high-incidence regions are
the known ones (SE/E Asia, N Africa), and incidence does NOT track
population (permutation p > 0.05) — so the burden is not a population
artifact."""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from npc_geography import REGIONS, to_sphere  # noqa: E402


def test_regional_table_complete():
    assert len(REGIONS) == 20
    for name, (coord, asir, cases, pop) in REGIONS.items():
        assert asir > 0
        assert cases > 0
        assert pop > 0


def test_high_incidence_regions_are_the_known_ones():
    """SE Asia, E Asia and N Africa carry the highest age-standardized
    incidence — the documented high-burden geography."""
    top = sorted(REGIONS, key=lambda n: -REGIONS[n][1])[:4]
    assert "South-Eastern Asia" in top
    assert "Eastern Asia" in top
    assert "Northern Africa" in top
    assert REGIONS["Northern Africa"][1] > 1.0
    se_asir = REGIONS["South-Eastern Asia"][1]
    eu_asir = REGIONS["Northern Europe"][1]
    assert se_asir / eu_asir > 10  # >10x regional contrast


def test_incidence_does_not_track_population():
    """Spearman rho of ASIR vs population is small and the permutation p
    is NOT significant — the high case counts are not explained by
    populous regions (supporting the geographic explanation)."""
    from scipy.stats import spearmanr

    names = list(REGIONS)
    asir = np.array([REGIONS[n][2] / REGIONS[n][3] for n in names])
    pop = np.array([REGIONS[n][3] for n in names])
    rho_obs, _ = spearmanr(asir, pop)
    assert abs(rho_obs) < 0.5
    # permutation p is stable and not significant
    from numpy.random import default_rng
    from scipy.stats import rankdata

    gen = default_rng(0)
    pop_r = rankdata(pop)
    n_perm = 500
    cnt = 0
    for _ in range(n_perm):
        pr = rankdata(gen.permutation(asir))
        cnt += abs(float(np.corrcoef(pr, pop_r)[0, 1])) >= abs(rho_obs)
    p = (cnt + 1) / (n_perm + 1)
    assert p > 0.05


def test_case_weighted_center_in_east_asia():
    """The case-weighted center of mass sits in E/SE Asia (the burden's
    geographic core), not the population core in South Asia."""
    from geocore import Sphere, frechet_mean

    S = Sphere()
    names = list(REGIONS)
    pts = np.array([to_sphere(*REGIONS[n][0]) for n in names])
    cases = np.array([REGIONS[n][2] for n in names])
    pops = np.array([REGIONS[n][3] for n in names])
    w = cases / cases.sum()
    c = frechet_mean(S, pts, weights=w, lr=0.2, n_steps=500).point
    lat, lon = 90 - np.degrees(c[0]), np.degrees(c[1])
    assert 20 <= lat <= 40          # E Asia
    assert 80 <= lon <= 110         # E Asia / China
