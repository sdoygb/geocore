#!/usr/bin/env python3
"""Statistical forecast of the next El Nino / La Nina from the historical
NOAA ONI record (1950-2026).

Honest scope: this is a STATISTICAL estimate from the observed event
intervals and the El Nino -> La Nina alternation pattern — NOT a
physical forecast.  The authoritative outlooks are NOAA CPC / IRI; we
only cross-check our statistical window against the official initial
conditions at the end of the ONI file.

Method:
  - El Nino -> El Nino and La Nina -> La Nina interval distributions
    (from the detected events),
  - the strong-El-Nino -> La Nina alternation probability (1982-83,
    1997-98, 2015-16 all were followed by La Nina within ~1-2 years),
  - the current state: last event peaks and the trailing ONI.

Run:  PYTHONPATH=src python3 examples/enso_forecast.py
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from enso import load_oni, detect_events  # noqa: E402


def main():
    years, months, anom = load_oni()
    events = detect_events(anom)
    el = sorted([e for e in events if e[4] > 0], key=lambda e: e[0])
    la = sorted([e for e in events if e[4] < 0], key=lambda e: e[0])
    en_years = np.array([years[e[0]] for e in el])
    la_years = np.array([years[e[0]] for e in la])
    en_peaks = np.array([years[e[2]] for e in el])
    la_peaks = np.array([years[e[2]] for e in la])

    # --- 1. interval statistics ---
    en_int = np.diff(en_years)
    la_int = np.diff(la_years)
    en_peak_int = np.diff(en_peaks)
    print("=== historical intervals (years) ===")
    print(f"El Nino  start-to-start: mean {en_int.mean():.2f}  "
          f"std {en_int.std():.2f}  range [{en_int.min()}, {en_int.max()}]  (n={len(en_int)})")
    print(f"El Nino  peak-to-peak  : mean {en_peak_int.mean():.2f}  "
          f"std {en_peak_int.std():.2f}  range [{en_peak_int.min()}, {en_peak_int.max()}]")
    print(f"La Nina  start-to-start: mean {la_int.mean():.2f}  "
          f"std {la_int.std():.2f}  range [{la_int.min()}, {la_int.max()}]  (n={len(la_int)})")

    # --- 2. strong El Nino -> La Nina alternation ---
    strong = [e for e in el if e[3] >= 1.5]
    followed = 0
    for e in strong:
        yr = years[e[0]]
        if any(yr < y <= yr + 2 for y in la_years):
            followed += 1
    print(f"\n=== alternation ===")
    print(f"strong El Nino (peak>=1.5): {len(strong)}; "
          f"{followed} followed by La Nina within 2 years "
          f"({100.0 * followed / max(len(strong), 1):.0f}%)")

    # --- 3. current state ---
    last_en_start, last_en_peak = en_years[-1], en_peaks[-1]
    last_la_start = la_years[-1]
    trailing = anom[-6:]
    print(f"\n=== current state (trailing ONI: {[f'{a:+.2f}' for a in trailing]}) ===")
    print(f"last El Nino event: {last_en_start} (peak {last_en_peak}), "
          f"last La Nina: {last_la_start}")
    print(f"years since last El Nino start: {years[-1] - last_en_start}")

    # --- 4. forecast windows ---
    print(f"\n=== statistical forecast ===")
    next_en = last_en_start + en_int.mean()
    next_la = last_la_start + la_int.mean()
    print(f"next El Nino : expected ~{next_en:.1f} "
          f"(68% window [{next_en - en_int.std():.1f}, {next_en + en_int.std():.1f}])")
    print(f"next La Nina : expected ~{next_la:.1f} "
          f"(68% window [{next_la - la_int.std():.1f}, {next_la + la_int.std():.1f}])")
    # peak-to-peak based estimate from the 2023 peak
    next_en_peak = last_en_peak + en_peak_int.mean()
    print(f"next El Nino (peak-to-peak from {last_en_peak}): ~{next_en_peak:.1f}")

    # --- 5. cross-check with the official tail ---
    tail = anom[-12:]
    print(f"\n=== cross-check with the official ONI tail (last 12 months) ===")
    print(f"trailing ONI trend: {tail[0]:+.2f} -> {tail[-1]:+.2f}")
    if tail[-1] > 0.5:
        print("official tail already turns positive — consistent with the "
              "El Nino window from the historical interval statistics")
    else:
        print("official tail not yet positive")

    print("\nDisclaimer: statistical estimate from the observed record; "
          "see NOAA CPC/IRI for authoritative ENSO outlooks.")


if __name__ == "__main__":
    main()
