#!/usr/bin/env python3
"""El Nino / La Nina diagnosis with the geocore statistics pipeline.

Data: NOAA CPC Oceanic Nino Index (ONI) — the 3-month running mean of the
Nino-3.4 sea-surface-temperature anomaly, monthly, 1950–present.
Downloaded 2025-08-29 from
https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
(stored in examples/data/oni.txt).

What we compute (honest scope): the ONI is NOAA's product; we use it to
(1) detect El Nino / La Nina events with the standard criterion
(|ONI| >= 0.5 for >= 5 consecutive overlapping seasons), (2) verify
against the famous events, (3) analyze the ENSO phase space
(ONI, dONI/dt) as a point set with the geocore statistics pipeline
(Frechet mean / variance / tangent PCA / spread ellipses) — the three
regimes (El Nino / neutral / La Nina) occupy distinct locations, and
(4) verify the winter phase-locking of El Nino peaks.

Run:  PYTHONPATH=src python3 examples/enso.py
"""

import os

import numpy as np

from geocore import PolarPlane, frechet_mean, frechet_variance, principal_directions
from geocore.viz import plot_spread

DATA = os.path.join(os.path.dirname(__file__), "data", "oni.txt")

SEASONS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]


def load_oni():
    """Return (years, months, anom) arrays: the season's middle month is
    used as its month index (DJF -> 1, JFM -> 2, ..., NDJ -> 12)."""
    years, months, anom = [], [], []
    with open(DATA) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 4 or parts[0] not in SEASONS:
                continue
            seas, yr, _total, a = parts
            years.append(int(yr))
            months.append(SEASONS.index(seas) + 1)
            anom.append(float(a))
    return np.array(years), np.array(months), np.array(anom)


def detect_events(anom, threshold=0.5, min_len=5):
    """Standard ONI event detection: |anom| >= threshold for >= min_len
    consecutive seasons.  Returns events = [(start, end, peak_index,
    peak, sign)] with sign = +1 (El Nino) or -1 (La Nina)."""
    events = []
    run_start, run_sign = None, 0
    best = (None, None, -1e9)
    for i, a in enumerate(anom):
        sign = 1 if a >= threshold else (-1 if a <= -threshold else 0)
        if sign == 0:
            # end any active run
            if run_start is not None and i - run_start >= min_len:
                events.append((run_start, i - 1, best[0], best[1], run_sign))
            run_start, run_sign, best = None, 0, (None, None, -1e9)
        elif run_sign == 0 or sign == run_sign:
            # start or continue a run
            if run_sign == 0:
                run_start, run_sign = i, sign
            if abs(a) > best[2]:
                best = (i, a, abs(a))
        else:
            # sign change: end the old run, start a new one
            if i - run_start >= min_len:
                events.append((run_start, i - 1, best[0], best[1], run_sign))
            run_start, run_sign, best = i, sign, (i, a, abs(a))
    if run_start is not None and len(anom) - run_start >= min_len:
        events.append((run_start, len(anom) - 1, best[0], best[1], run_sign))
    return events


def main():
    years, months, anom = load_oni()
    print(f"ONI series: {years.min()}-{years.max()}, {len(anom)} months\n")

    # --- 1. event detection ---
    events = detect_events(anom)
    el_nino = [e for e in events if e[4] > 0]
    la_nina = [e for e in events if e[4] < 0]
    print(f"El Nino events: {len(el_nino)}, La Nina events: {len(la_nina)}\n")
    print(f"{'event':<18}{'peak':>7}{'peak yr':>9}{'length':>8}")
    for e in sorted(el_nino, key=lambda e: -e[3])[:5]:
        start = years[e[0]]
        end = years[e[1]]
        print(f"{start}-{end:<14}{e[3]:>7.2f}{years[e[2]]:>9}{e[1]-e[0]+1:>8}")
    print()
    for e in sorted(la_nina, key=lambda e: e[3])[:5]:
        start = years[e[0]]
        end = years[e[1]]
        print(f"{start}-{end:<14}{e[3]:>7.2f}{years[e[2]]:>9}{e[1]-e[0]+1:>8}")

    # --- 2. verify famous events ---
    famous = {1982: "1982-83 El Nino", 1997: "1997-98 El Nino", 2015: "2015-16 El Nino",
              2010: "2010-11 La Nina"}
    print("\nverification (detected peak around the famous year):")
    for yr, label in famous.items():
        hits = [e for e in events if years[e[2]] in (yr, yr + 1)]
        peak = max((e[3] for e in hits), key=abs, default=None)
        print(f"  {label}: peak ONI {peak:.2f}" if peak else f"  {label}: not detected")

    # --- 3. phase-space geometry (geocore): (ONI, dONI/dt) ---
    d_anom = np.concatenate([[0.0], np.diff(anom)])
    pts = np.column_stack([anom, d_anom])  # 2D Euclidean phase space
    to_polar = np.column_stack([np.hypot(anom, d_anom), np.arctan2(d_anom, anom)])
    P = PolarPlane()
    regimes = {"El Nino": pts[anom >= 0.5], "neutral": pts[np.abs(anom) < 0.5],
               "La Nina": pts[anom <= -0.5]}
    print("\nphase space (ONI, dONI/dt) — Frechet mean / spread:")
    for name, R in regimes.items():
        if len(R) < 5:
            print(f"  {name:8s}: n={len(R)}")
            continue
        res = frechet_mean(P, to_polar[anom >= 0.5] if name == "El Nino" else
                           (to_polar[np.abs(anom) < 0.5] if name == "neutral" else to_polar[anom <= -0.5]),
                           lr=0.1, n_steps=400)
        m = res.point
        m_cart = np.array([m[0] * np.cos(m[1]), m[0] * np.sin(m[1])])
        var = frechet_variance(P, to_polar[anom >= 0.5] if name == "El Nino" else
                               (to_polar[np.abs(anom) < 0.5] if name == "neutral" else to_polar[anom <= -0.5]),
                               mean=m)
        print(f"  {name:8s}: n={len(R):4d}  centroid=({m_cart[0]:+.2f}, {m_cart[1]:+.2f})  "
              f"std={np.sqrt(var):.2f}")

    # --- 4. winter phase-locking ---
    print("\nphase-locking: mean |ONI| by calendar month (El Nino years only, |ONI|>=0.5):")
    el_mask = anom >= 0.5
    mmean = [np.mean(np.abs(anom[el_mask & (months == mo)])) for mo in range(1, 13)]
    peak_month = int(np.argmax(mmean)) + 1
    print("  " + " ".join(f"{mmean[i]:.2f}" for i in range(12)))
    print(f"  El Nino |ONI| peaks in calendar month {peak_month} (winter = 11-2)")

    # --- 5. spread ellipse of the El Nino regime ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sel = to_polar[anom >= 0.5]
        if len(sel) > 10:
            fig, ax = plt.subplots(figsize=(5, 5))
            plot_spread(P, sel, ax=ax, title="El Nino regime in (ONI, dONI/dt)")
            fig.savefig(os.path.join(os.path.dirname(__file__), "enso_phasespace.png"), dpi=100)
            print("\nsaved examples/enso_phasespace.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
