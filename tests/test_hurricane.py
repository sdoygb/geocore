"""Tests on the real IBTrACS hurricane tracks (NA basin, 1980-2026, 746
storms): the S^2 pipeline reproduces verifiable climatology — the
activity region, the August-October season with a September peak, the
NNW mean movement, and three track clusters matching the known track
types (Cape Verde / Gulf-Caribbean / mid-latitude recurving)."""

import os
import sys
from collections import Counter

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from hurricane import fast_sphere_mean, load_tracks, to_sphere  # noqa: E402
from geocore import Sphere, frechet_mean, principal_directions  # noqa: E402
from geocore.geostats import geodesic_distance  # noqa: E402


@pytest.fixture(scope="module")
def data():
    tracks = load_tracks()
    S = Sphere()
    centroids = []
    move_dirs = []
    months = Counter()
    for sid, pts in tracks.items():
        sph = np.array([to_sphere(float(p[0]), float(p[1])) for p in pts])
        centroids.append(fast_sphere_mean(S, sph))
        lat0, lon0 = float(pts[0][0]), float(pts[0][1])
        lat1, lon1 = float(pts[-1][0]), float(pts[-1][1])
        dy = lat1 - lat0
        dx = (lon1 - lon0) * np.cos(np.radians(lat0))
        move_dirs.append(np.degrees(np.arctan2(dx, dy)) % 360.0)
        for p in pts:
            months[int(str(p[2])[5:7])] += 1
    return tracks, S, np.array(centroids), np.array(move_dirs), months


def test_storm_count(data):
    tracks, *_ = data
    assert 700 <= len(tracks) <= 800


def test_centroids_in_atlantic(data):
    _, _, centroids, _, _ = data
    lats = 90 - np.degrees(centroids[:, 0])
    lons = np.degrees(centroids[:, 1])
    assert lats.min() > 5 and lats.max() < 55
    assert lons.min() > -105 and lons.max() < -15


def test_activity_region_center(data):
    """The centroid of track centroids lies in the central-western
    tropical Atlantic (20-35N, 40-80W) — the real activity region."""
    _, S, centroids, _, _ = data
    mean = frechet_mean(S, centroids, lr=0.2, n_steps=300).point
    lat = 90 - np.degrees(mean[0])
    lon = np.degrees(mean[1])
    assert 20 <= lat <= 35
    assert -80 <= lon <= -40


def test_activity_region_anisotropic(data):
    """The activity region is elongated (PCA anisotropy > 1.5)."""
    _, S, centroids, _, _ = data
    mean = frechet_mean(S, centroids, lr=0.2, n_steps=300).point
    evals, _ = principal_directions(S, centroids, mean=mean)
    assert np.sqrt(evals[1] / evals[0]) > 1.5


def test_mean_movement_is_nnw(data):
    """The mean storm movement is toward the north-northwest (~347 deg)."""
    _, _, _, move_dirs, _ = data
    vx = np.mean(np.cos(np.radians(move_dirs)))
    vy = np.mean(np.sin(np.radians(move_dirs)))
    circ = np.degrees(np.arctan2(vy, vx)) % 360.0
    assert circ >= 330 or circ <= 10


def test_season_peak_in_september(data):
    """June-November season with the September peak (real climatology)."""
    _, _, _, _, months = data
    assert months[9] == max(months.values())
    # June-November carries the bulk
    season = sum(months[m] for m in (6, 7, 8, 9, 10, 11))
    assert season / sum(months.values()) > 0.9


def test_three_track_clusters_separate(data):
    """Three clusters separate the known track types by longitude: a
    Gulf-Caribbean cluster (west, ~86W), a mid-latitude cluster (~63W)
    and a Cape Verde cluster (east, ~42W)."""
    tracks, S, centroids, _, _ = data
    k = 3
    rng = np.random.default_rng(0)
    centers = centroids[rng.choice(len(centroids), k, replace=False)]
    labels = np.zeros(len(centroids), dtype=int)
    for _ in range(25):
        dists = np.array([[geodesic_distance(S, c, p) for p in centroids] for c in centers])
        labels = np.argmin(dists, axis=0)
        for j in range(k):
            if (labels == j).sum() > 0:
                centers[j] = fast_sphere_mean(S, centroids[labels == j])
    lons = np.sort(np.degrees([c[1] for c in centers]))
    assert lons[0] < -75          # Gulf-Caribbean
    assert lons[1] < -55          # mid-latitude
    assert lons[2] > -55          # Cape Verde / eastern
    sizes = [(labels == j).sum() for j in range(k)]
    assert all(100 <= s <= 400 for s in sizes)
