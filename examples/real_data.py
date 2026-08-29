#!/usr/bin/env python3
"""Real-data application: 2024 global seismicity analyzed with the
geometric statistics pipeline.

Data: USGS earthquake catalog, all M >= 5.0 in 2024 (1507 events),
downloaded 2025-08-29 from
https://earthquake.usgs.gov/fdsnws/event/1/query (format=csv,
starttime=2024-01-01, endtime=2025-01-01, minmagnitude=5.0).
Stored at examples/data/earthquakes_2024.csv.

Analysis: earthquake hypocenters are points on the sphere (lat, lon).
We use the S^2 Frechet mean / variance / tangent PCA (geocore) to
extract real structure — the mean is the seismicity centroid, the PCA
principal axis is the strike of the seismic belt — and compare with the
naive Euclidean (lat, lon) average, which is wrong across the ±180°
meridian.

Run:  PYTHONPATH=src python3 examples/real_data.py
"""

import csv
import os

import numpy as np

from geocore import Sphere, frechet_mean, frechet_variance, principal_directions
from geocore.derivatives import log_map
from geocore.geostats import geodesic_distance

DATA = os.path.join(os.path.dirname(__file__), "data", "earthquakes_2024.csv")


def load_quakes():
    with open(DATA) as f:
        rows = list(csv.DictReader(f))
    lats = np.array([float(r["latitude"]) for r in rows])
    lons = np.array([float(r["longitude"]) for r in rows])
    mags = np.array([float(r["mag"]) for r in rows])
    return lats, lons, mags


def to_sphere(lats, lons):
    """(lat, lon) in degrees -> (theta, phi) on the unit sphere."""
    return np.column_stack([np.pi / 2 - np.radians(lats), np.radians(lons)])


def to_latlon(pts):
    return np.degrees(np.pi / 2 - pts[:, 0]), np.degrees(pts[:, 1])


def axis_bearing(manifold, mean, evecs):
    """Bearing (deg from north, clockwise) of the principal axis on the
    ground: map the orthonormal-frame eigenvector to the lab frame
    (e_theta points south, e_phi_hat points east)."""
    th_m, ph_m = mean
    a, b = evecs[:, 1]
    north = -a  # e_theta points toward decreasing theta = south
    east = b
    return np.degrees(np.arctan2(east, north)) % 360.0


def analyze(region_name, mask):
    """Frechet mean / variance / PCA of the selected events, plus the
    naive Euclidean (lat, lon) average for comparison."""
    lats, lons, mags = load_quakes()
    lats, lons, mags = lats[mask], lons[mask], mags[mask]
    S = Sphere()
    pts = to_sphere(lats, lons)
    mean = frechet_mean(S, pts, lr=0.2, n_steps=500).point
    var = frechet_variance(S, pts, mean=mean)
    evals, evecs = principal_directions(S, pts, mean=mean)
    mlat, mlon = to_latlon(np.asarray([mean]))
    bearing = axis_bearing(S, mean, evecs)

    # naive Euclidean average of (lat, lon) — the textbook mistake
    eu_lat, eu_lon = lats.mean(), lons.mean()

    print(f"\n=== {region_name} ({len(lats)} events, M >= 5) ===")
    print(f"geometric centroid : {mlat[0]:.2f}N/S, {mlon[0]:.2f} "
          f"(angular std {np.sqrt(var) * 180 / np.pi:.2f} deg)")
    print(f"naive (lat,lon) avg: {eu_lat:.2f}, {eu_lon:.2f}")
    print(f"principal axis     : bearing {bearing:.1f} deg "
          f"(strike of the seismic belt), eigenvalue ratio "
          f"{np.sqrt(evals[1] / evals[0]):.1f}")
    return mean, evals, evecs, bearing


def main():
    lats, lons, _ = load_quakes()

    # --- Region 1: Japan — the belt runs NE-SW along the island arc ---
    analyze("Japan arc", (lats >= 24) & (lats <= 46) & (lons >= 125) & (lons <= 146))

    # --- Region 2: Tonga-Fiji — straddles the ±180° meridian ---
    mask_tonga = (lats >= -26) & (lats <= -14) & ((lons >= 170) | (lons <= -175))
    analyze("Tonga-Fiji (crosses 180° meridian)", mask_tonga)

    # --- Region 3: global — the circum-Pacific belt ---
    analyze("Global", np.ones(len(lats), dtype=bool))

    # Conclusion: the naive average is a real error across the meridian.
    lats_t, lons_t = lats[mask_tonga], lons[mask_tonga]
    print("\nAcross the ±180° meridian the naive (lat, lon) average of the "
          "Tonga events lands near longitude 0° (175° and −175° average to "
          "0°), while the geometric centroid stays at ~180° — the S² "
          "treatment is the correct one for directions on a sphere.")


if __name__ == "__main__":
    main()
