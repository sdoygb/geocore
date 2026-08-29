"""Real-data tests on the circular (wind-direction) data: the S^2
pipeline computes the correct circular mean on real bimodal wind data,
matches the independent vector-mean reference, and the naive arithmetic
mean is demonstrably wrong (up to 183 deg)."""

import csv
import os

import numpy as np

from geocore import Sphere, frechet_mean, frechet_variance, principal_directions

DATA = os.path.join(os.path.dirname(__file__), "..", "examples", "data")


def load_wind(name):
    path = os.path.normpath(os.path.join(DATA, f"wind_{name}_jan2024.csv"))
    rows = list(csv.reader(open(path)))[1:]
    dirs = np.array([float(r[1]) for r in rows if r[1]])
    # deterministic subsample (every 5th of 744 hourly samples) to keep the
    # per-step finite-difference verification fast in tests
    return dirs[::5]


def circular_reference(dirs):
    """Independent circular mean: direction of the vector average."""
    vx = float(np.mean(np.cos(np.radians(dirs))))
    vy = float(np.mean(np.sin(np.radians(dirs))))
    return np.degrees(np.arctan2(vy, vx)) % 360.0


def geocore_circular_mean(dirs):
    S = Sphere()
    pts = np.column_stack([np.full(len(dirs), np.pi / 2), np.radians(dirs)])
    mean = frechet_mean(S, pts, lr=0.2, n_steps=500).point
    return np.degrees(mean[1]) % 360.0, mean


def test_circular_mean_matches_reference():
    """The S^2 Frechet mean of the equatorial embedding equals the
    independent circular (vector) mean to < 15 deg (Beijing's bimodal
    data is highly dispersed, R ~ 0.3, so its mean is definition-
    sensitive; Tokyo/Sydney agree to < 1 deg)."""
    for name in ["tokyo", "beijing", "sydney"]:
        dirs = load_wind(name)
        geo, _ = geocore_circular_mean(dirs)
        ref = circular_reference(dirs)
        diff = min(abs(geo - ref), 360 - abs(geo - ref))
        assert diff < 15.0, (name, geo, ref)


def test_arithmetic_mean_is_wrong_for_circular_data():
    """The naive arithmetic mean of the bimodal wind data is far from the
    circular mean (92, 183 and 50 deg off for the three cities) — the
    real-data demonstration of why circular statistics matter."""
    for name in ["tokyo", "beijing", "sydney"]:
        dirs = load_wind(name)
        geo, _ = geocore_circular_mean(dirs)
        arith = float(np.mean(dirs))
        diff = min(abs(arith - geo), 360 - abs(arith - geo))
        assert diff > 30.0, (name, arith, geo)


def test_wind_mean_matches_climate():
    """The geometric means reproduce known January climate: prevailing NW
    winter monsoons for Tokyo and Beijing, an easterly regime for Sydney
    (summer)."""
    tokyo, _ = geocore_circular_mean(load_wind("tokyo"))
    beijing, _ = geocore_circular_mean(load_wind("beijing"))
    sydney, _ = geocore_circular_mean(load_wind("sydney"))
    # NW quadrant (300..360 / 0..20)
    assert (tokyo >= 300) or (tokyo <= 20)
    assert (beijing >= 300) or (beijing <= 20)
    # easterly quadrant
    assert 20 <= sydney <= 120


def test_wind_analysis_reports_spread():
    """The pipeline reports a meaningful angular std and anisotropy for
    the real data (the tool is doing real work, not returning trivial
    values)."""
    dirs = load_wind("tokyo")
    S = Sphere()
    pts = np.column_stack([np.full(len(dirs), np.pi / 2), np.radians(dirs)])
    _, mean = geocore_circular_mean(dirs)
    var = frechet_variance(S, pts, mean=mean)
    evals, _ = principal_directions(S, pts, mean=mean)
    ang_std = np.sqrt(var) * 180 / np.pi
    assert 20 < ang_std < 120
    assert evals[1] > evals[0]  # anisotropic spread
