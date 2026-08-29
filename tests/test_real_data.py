"""Tests on the real USGS earthquake data (2024, M >= 5): the geometric
statistics pipeline reproduces verifiable geographical facts, and the
naive Euclidean (lat, lon) average is demonstrably wrong across the
±180° meridian — real-data evidence that the tool does real work."""

import csv
import os

import numpy as np

from geocore import Sphere, frechet_mean, frechet_variance, principal_directions

DATA = os.path.join(os.path.dirname(__file__), "..", "examples", "data", "earthquakes_2024.csv")


def load_quakes():
    with open(os.path.normpath(DATA)) as f:
        rows = list(csv.DictReader(f))
    lats = np.array([float(r["latitude"]) for r in rows])
    lons = np.array([float(r["longitude"]) for r in rows])
    return lats, lons


def to_sphere(lats, lons):
    return np.column_stack([np.pi / 2 - np.radians(lats), np.radians(lons)])


def analyze(mask):
    lats, lons = load_quakes()
    lats, lons = lats[mask], lons[mask]
    S = Sphere()
    pts = to_sphere(lats, lons)
    mean = frechet_mean(S, pts, lr=0.2, n_steps=500).point
    evals, evecs = principal_directions(S, pts, mean=mean)
    mlat, mlon = np.degrees(np.pi / 2 - mean[0]), np.degrees(mean[1])
    return mlat, mlon, evals


def test_japan_centroid_is_in_japan():
    """The geometric centroid of 2024 Japan-arc events lies in central
    Japan (and matches the naive average, since no meridian crossing)."""
    lats, lons = load_quakes()
    mask = (lats >= 24) & (lats <= 46) & (lons >= 125) & (lons <= 146)
    mlat, mlon, _ = analyze(mask)
    assert 28 <= mlat <= 38        # Honshu
    assert 132 <= mlon <= 142
    assert abs(mlat - lats[mask].mean()) < 2  # agrees with naive here


def test_tonga_geometric_mean_is_correct_across_meridian():
    """Tonga-Fiji events straddle ±180°: the S^2 centroid stays at ~180°E,
    while the naive (lat, lon) average lands in the wrong hemisphere —
    the real error the geometric treatment avoids."""
    lats, lons = load_quakes()
    mask = (lats >= -26) & (lats <= -14) & ((lons >= 170) | (lons <= -175))
    mlat, mlon, evals = analyze(mask)
    # geometric centroid near the true location (20-22 S, ~180 E)
    assert -25 <= mlat <= -15
    assert abs(mlon - 180) <= 12
    # naive average is badly wrong across the meridian
    assert abs(lons[mask].mean() - 180) > 60
    # the belt is elongated (strike), so the PCA ratio is clearly > 1
    assert np.sqrt(evals[1] / evals[0]) > 1.5


def test_global_centroid_is_on_the_pacific_ring():
    """The global seismicity centroid of 2024 lies in the southwest
    Pacific (the mass center of the circum-Pacific belt), while the naive
    longitude average cancels to ~25°E (Africa) — meaningless."""
    lats, lons = load_quakes()
    mlat, mlon, evals = analyze(np.ones(len(lats), dtype=bool))
    assert -30 <= mlat <= 10
    assert 130 <= mlon <= 200          # southwest Pacific
    assert abs(lons.mean() - 180) > 100  # naive average is meaningless
    assert evals[1] > evals[0]          # the ring is anisotropic
