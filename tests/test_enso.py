"""Tests on the real NOAA ONI data: El Nino / La Nina event detection
reproduces the famous events, the geometric phase-space statistics
separate the three regimes, and the winter phase-locking of El Nino
peaks is verified."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from enso import detect_events, load_oni  # noqa: E402


@pytest.fixture(scope="module")
def data():
    years, months, anom = load_oni()
    events = detect_events(anom)
    el = [e for e in events if e[4] > 0]
    la = [e for e in events if e[4] < 0]
    return years, months, anom, events, el, la


def test_event_counts_are_plausible(data):
    """1950-2026: 22 El Nino / 20 La Nina events, each >= 5 months."""
    _, _, _, _, el, la = data
    assert 15 <= len(el) <= 30
    assert 15 <= len(la) <= 30
    for e in el + la:
        assert e[1] - e[0] + 1 >= 5


def test_famous_events_via_explicit_index(data):
    years, _, _, events, _, _ = data
    for yr, lo in [(1982, 2.1), (1997, 2.3), (2015, 2.5), (2010, -1.5)]:
        hits = [e for e in events if years[e[2]] in (yr, yr + 1)]
        peak = max((e[3] for e in hits), key=abs, default=None)
        assert peak is not None, yr
        if lo > 0:
            assert peak >= lo
        else:
            assert peak <= lo


def test_winter_phase_locking(data):
    """El Nino |ONI| is strongest in the boreal winter months (11-2)."""
    _, months, anom, _, _, _ = data
    mask = anom >= 0.5
    mmean = [np.mean(np.abs(anom[mask & (months == mo)])) for mo in range(1, 13)]
    peak_month = int(np.argmax(mmean)) + 1
    assert peak_month in (11, 12, 1, 2)
    # winter mean beats the annual mean
    winter = (mmean[10] + mmean[11] + mmean[0]) / 3  # NDJ
    assert winter > np.mean(mmean)


def test_phase_space_regimes_separate(data):
    """Frechet means of the three regimes in (ONI, dONI/dt) are clearly
    separated (El Nino positive, La Nina negative, neutral near zero)."""
    from geocore import PolarPlane, frechet_mean

    _, _, anom, _, _, _ = data
    d_anom = np.concatenate([[0.0], np.diff(anom)])
    to_polar = np.column_stack([np.hypot(anom, d_anom), np.arctan2(d_anom, anom)])
    P = PolarPlane()
    means = {}
    for name, mask in [("el", anom >= 0.5), ("neu", np.abs(anom) < 0.5), ("la", anom <= -0.5)]:
        sel = to_polar[mask]
        m = frechet_mean(P, sel, lr=0.1, n_steps=400).point
        means[name] = np.array([m[0] * np.cos(m[1]), m[0] * np.sin(m[1])])
    assert means["el"][0] > 0.7
    assert means["la"][0] < -0.7
    assert abs(means["neu"][0]) < 0.3
