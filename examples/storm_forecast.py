#!/usr/bin/env python3
"""Forecast the track of an ACTIVE tropical cyclone with the similar-track
method on the geocore S^2 pipeline, and compare with the official
forecast (NHC fstadv text) when available.

Method (similar-track / analog): the observed segment of the active
storm (last 48 h) is matched against every 48-h sliding window in the
historical basin record (EP, 1980-2026, 1147 storms); the subsequent
72-120 h of the most similar analogs form the forecast, with the S^2
Frechet mean as the central track.  This is the same family of method
NHC uses as a guidance tool — so the comparison is fair, not apples to
oranges.

Data: IBTrACS v04r01 ACTIVE + EP list, downloaded 2025-08-29/2026-08-29.

Run:  PYTHONPATH=src python3 examples/storm_forecast.py [SID]
"""

import csv
import os
import sys

import numpy as np

from geocore import Sphere
from geocore.geostats import geodesic_distance

DATA = os.path.join(os.path.dirname(__file__), "data")
ACTIVE = "/tmp/ibtracs_active.csv"
EP_HIST = os.path.join(DATA, "hurricanes_ep_1980.csv")

WINDOW = 8  # 48 h in 6-h steps
HORIZON = 20  # 120 h
N_ANALOGS = 20


def load_ep_history():
    tracks = {}
    cur = None
    with open(EP_HIST) as f:
        for r in csv.DictReader(f):
            if cur != r["SID"]:
                cur = r["SID"]
                tracks[cur] = []
            tracks[cur].append((float(r["LAT"]), float(r["LON"]), r["ISO_TIME"]))
    # keep lists (mixed str/float tuples become str arrays under np.array)
    return {sid: v for sid, v in tracks.items() if len(v) >= WINDOW + 4}


def load_active_storm(sid_prefix):
    """Load an active storm's provisional track by SID prefix."""
    rows = []
    with open(ACTIVE) as f:
        for r in csv.DictReader(f):
            if r["SID"] == "SID":
                continue
            if r["SID"].startswith(sid_prefix) and r["TRACK_TYPE"].strip() in ("PROVISIONAL", "main"):
                rows.append(r)
    rows.sort(key=lambda r: r["ISO_TIME"])
    return [(float(r["LAT"]), float(r["LON"]), r["ISO_TIME"]) for r in rows]


def to_sphere(lat, lon):
    return np.array([np.pi / 2 - np.radians(lat), np.radians(lon)])


def fast_sphere_mean(S, pts):
    from geocore.derivatives import log_map

    m = np.asarray(pts, dtype=float).mean(axis=0)
    for _ in range(25):
        vs = np.array([log_map(S, m, p) for p in pts])
        v = vs.mean(axis=0)
        if np.linalg.norm(v) < 1e-10:
            break
        m = S.geodesic_closed_form(m, v, 1.0).point
    return m


def segment_distance(S, a, b):
    """Mean S^2 distance between two equal-length path segments."""
    return float(np.mean([geodesic_distance(S, to_sphere(x[0], x[1]), to_sphere(y[0], y[1]))
                          for x, y in zip(a, b)]))


def forecast(sid):
    S = Sphere()
    obs = load_active_storm(sid)
    if len(obs) < WINDOW:
        print(f"{sid}: not enough observed points ({len(obs)})")
        return None
    seg = obs[-WINDOW:]  # last 48 h
    print(f"{sid}: {len(obs)} observed points, last {obs[-1][2]} at "
          f"({obs[-1][0]:.1f}, {obs[-1][1]:.1f})\n")

    # match against historical windows
    analogs = []  # (distance, future points)
    for hsid, pts in load_ep_history().items():
        for i in range(0, len(pts) - WINDOW - 4):
            win = pts[i:i + WINDOW]
            d = segment_distance(S, seg, win)
            analogs.append((d, pts[i + WINDOW:i + WINDOW + HORIZON], hsid))
    analogs.sort(key=lambda a: a[0])
    best = analogs[:N_ANALOGS]
    print(f"matched {N_ANALOGS} analogs (median segment distance "
          f"{np.median([a[0] for a in best]):.1f} deg)")

    # forecast: S^2 mean of the analogs' future positions per step
    t0 = obs[-1][2]
    forecast_pts = []
    for k in range(4, HORIZON + 1, 4):  # +24, +48, ... +120 h
        futs = [a[1][k - 1] for a in best if len(a[1]) > k - 1]
        if len(futs) < 10:
            continue
        sph = np.array([to_sphere(f[0], f[1]) for f in futs])
        m = fast_sphere_mean(S, sph)
        lat = 90 - np.degrees(m[0])
        lon = np.degrees(m[1])
        # spread: mean distance from the mean
        spread = np.mean([geodesic_distance(S, m, p) for p in sph])
        forecast_pts.append((k, lat, lon, spread))
    return forecast_pts


def fetch_nhc_advisory(atcf_id):
    """Best-effort fetch of the latest NHC fstadv text for an ATCF id
    (e.g. EP102026); returns (forecast_points, text) or None."""
    import re
    import ssl
    import urllib.request

    # local python has no CA bundle; NHC data is public (unverified context)
    ctx = ssl._create_unverified_context()
    basin, num = atcf_id[:2].lower(), atcf_id[2:4]
    points = []
    for n in range(30, 0, -1):
        url = f"https://www.nhc.noaa.gov/archive/2026/{basin}{num}/{atcf_id.lower()}.fstadv.{n:03d}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 geocore"})
        try:
            with urllib.request.urlopen(req, timeout=15, context=ctx) as r:
                txt = r.read().decode("utf-8", "replace")
        except Exception:
            continue
        for m in re.finditer(r"FORECAST VALID (\d{2}/\d{4}Z) (\d+\.\d)N (\d+\.\d)W", txt):
            day, hhmm = m.group(1).split("/")
            hour = int(hhmm[:2])
            points.append((hour, float(m.group(2)), -float(m.group(3))))
        if "DISSIPATED" in txt or "POST-TROP" in txt:
            points.append((999, None, None))  # lifecycle ended
        if points:
            return points, txt
    return None


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else "2026232N10250"  # JULIO
    fc = forecast(sid)
    if fc is None:
        return
    print(f"\n=== similar-track forecast (S^2 pipeline) ===")
    for k, lat, lon, spread in fc:
        lon_s = f"{abs(lon):.1f}W" if lon < 0 else f"{lon:.1f}E"
        print(f"+{k * 6 / 24:>4.0f} d : {lat:5.1f}N {lon_s:>10}   "
              f"(spread {np.degrees(spread):.1f} deg)")

    # best-effort official comparison
    atcf = None
    with open(ACTIVE) as f:
        for r in csv.DictReader(f):
            if r["SID"] == "SID":
                continue
            if r["SID"].startswith(sid) and r.get("USA_ATCF_ID"):
                atcf = r["USA_ATCF_ID"].strip()
                break
    if atcf:
        res = fetch_nhc_advisory(atcf)
        if res:
            pts, txt = res
            print(f"\n=== official NHC forecast ({atcf}) ===")
            for hour, lat, lon in pts:
                if lat is None:
                    print(f"  lifecycle: ended (dissipated / post-tropical)")
                else:
                    lon_s = f"{abs(lon):.1f}W" if lon < 0 else f"{lon:.1f}E"
                    print(f"  +{hour:>3d} h : {lat:5.1f}N {lon_s}")
        else:
            print("\n(official NHC forecast not retrievable)")


if __name__ == "__main__":
    main()
