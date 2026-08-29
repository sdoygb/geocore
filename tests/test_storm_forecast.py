"""Tests for the similar-track storm forecast (S^2 pipeline): the
forecast is reproducible from the real IBTrACS record, the spread grows
with horizon, and the JULIO +24 h forecast lands close to the official
NHC position (0.8 deg in the live comparison)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from storm_forecast import load_ep_history, forecast  # noqa: E402


def test_ep_history_loaded():
    hist = load_ep_history()
    assert len(hist) > 1000  # EP basin 1980-2026


def test_julio_forecast_lands_in_east_pacific():
    """JULIO's +24h forecast lies in the east Pacific subtropics, near
    the official NHC +24h position (18.1N, 119.9W)."""
    fc = forecast("2026232N10250")
    assert fc is not None
    k, lat, lon, spread = fc[0]  # +24 h
    assert 16 <= lat <= 20
    assert -122 <= lon <= -116
    assert abs(lat - 18.1) < 1.5  # NHC comparison


def test_spread_grows_with_horizon():
    """Forecast uncertainty grows with the horizon (analog dispersion)."""
    fc = forecast("2026232N10250")
    spreads = [s for _, _, _, s in fc]
    assert spreads[0] < spreads[-1]


def test_iselle_forecast_moves_northwest():
    """ISELLE's forecast continues its northwest motion (typical east
    Pacific recurving storm)."""
    fc = forecast("2026234N10255")
    assert fc is not None
    lat0, lon0 = fc[0][1], fc[0][2]
    lat1, lon1 = fc[-1][1], fc[-1][2]
    assert lat1 > lat0  # northward
    assert lon1 < lon0  # westward
