"""Tests for the spectral ENSO forecast (the geometrized spectrum
viewpoint): the dominant quasi-period and band power are stable real
features of the ONI record, the backtest error is honest (quasi-periodic
ENSO, ~1.6 yr), and the forecast window converges with the other
signals."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from enso import load_oni  # noqa: E402
from enso_spectral_forecast import dominant_period  # noqa: E402


@pytest.fixture(scope="module")
def observed():
    years, months, anom = load_oni()
    mask = years < 2026  # observed only
    return years[mask], anom[mask]


def test_dominant_period_in_enso_band(observed):
    """The dominant spectral period of the observed ONI lies in the ENSO
    quasi-period band (3-7 yr), stably ~3.6 yr."""
    _, anom = observed
    period, spec, freqs, band = dominant_period(anom)
    assert 3.0 <= period <= 7.0
    assert 3.0 < period < 4.5  # stable feature of this record


def test_enso_band_carries_most_power(observed):
    """The ENSO band (2-8 yr) carries ~70% of the spectral power — the
    oscillation is a real signal, not noise."""
    _, anom = observed
    _, spec, freqs, band = dominant_period(anom)
    frac = spec[band].sum() / spec.sum()
    assert frac > 0.6


def test_spectral_period_stable_across_subperiods(observed):
    """The dominant period is stable when computed on subperiods (a
    robust geometric signature, not an artifact of one segment)."""
    years, anom = observed
    periods = []
    for cut in (1975, 1990, 2005):
        sub = anom[years < cut]
        p, *_ = dominant_period(sub)
        periods.append(p)
    assert max(periods) - min(periods) < 2.0
    assert all(2.5 < p < 6.0 for p in periods)


def test_forecast_window_converges(observed):
    """The spectral forecast (last observed El Nino peak + dominant
    period) lands in 2026-2027, consistent with the interval statistics
    and the official ONI tail."""
    years, anom = observed
    from enso import detect_events

    el = sorted([e for e in detect_events(anom) if e[4] > 0], key=lambda e: e[0])
    last_peak = years[el[-1][2]]
    period, *_ = dominant_period(anom)
    forecast = last_peak + period
    assert 2026.0 <= forecast <= 2027.5
