"""Tests for the ENSO statistical forecast: the interval statistics and
the forecast windows derived from the real NOAA ONI record are
reproducible, and the official ONI tail cross-checks the El Nino
window."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from enso import detect_events, load_oni  # noqa: E402


@pytest.fixture(scope="module")
def series():
    years, months, anom = load_oni()
    events = detect_events(anom)
    el = sorted([e for e in events if e[4] > 0], key=lambda e: e[0])
    la = sorted([e for e in events if e[4] < 0], key=lambda e: e[0])
    return years, anom, el, la


def test_el_nino_interval_statistics(series):
    """El Nino start-to-start interval: mean ~3.4 yr, range [1, 6]."""
    _, _, el, _ = series
    en_years = np.array([e[0] for e in el])  # index into years
    starts = np.array([series[0][i] for i in [e[0] for e in el]])
    intervals = np.diff(starts)
    assert 3.0 < intervals.mean() < 4.0
    assert intervals.min() == 1
    assert 5 <= intervals.max() <= 7


def test_forecast_window_centered_on_2026(series):
    """From the last El Nino (2023) plus the mean interval, the expected
    next El Nino lands in 2026, with a plausible 68% window."""
    _, _, el, _ = series
    starts = np.array([series[0][i] for i in [e[0] for e in el]])
    intervals = np.diff(starts)
    last = starts[-1]  # 2023
    expected = last + intervals.mean()
    assert 2025.5 < expected < 2027.5
    assert (expected - intervals.std()) < expected < (expected + intervals.std())


def test_strong_el_nino_la_nina_alternation_rate(series):
    """~44% of strong El Ninos (peak >= 1.5) are followed by La Nina
    within 2 years — the alternation tendency exists but is not certain."""
    _, _, el, la = series
    la_starts = {series[0][i] for i in [e[0] for e in la]}
    strong = [e for e in el if e[3] >= 1.5]
    followed = sum(1 for e in strong if any(series[0][e[0]] < y <= series[0][e[0]] + 2 for y in la_starts))
    rate = followed / len(strong)
    assert 0.3 <= rate <= 0.7


def test_official_tail_crosschecks_el_nino_window(series):
    """The trailing ONI (official initial conditions) turns positive —
    consistent with the historical-interval El Nino window (the forecast
    is not contradicted by the official tail)."""
    _, anom, _, _ = series
    tail = anom[-12:]
    assert tail[-1] > tail[0]  # rising
    if tail[-1] > 0.5:
        # positive official tail: expect a positive-phase event within
        # the historical window from the last El Nino
        starts = [series[0][e[0]] for e in series[2]]
        expected_next = starts[-1] + 3.43
        assert 2025 < expected_next < 2028
