# geocore — geometric computation core

A geometric computation core, architected with PyTorch as the structural
reference: geometric objects (≈ Tensor), operator dispatch (≈ aten/c10),
automatic verification (≈ autograd), and closed-form shortcuts
(≈ torch.compile) — every result verified to machine precision, every
speedup measured.

**22 features · 178 tests · ~700 fuzz cases · 12 measured shortcuts ·
5 real datasets.**

## Documents

- [User guide](user-guide.md) — installation, quick start, the four
  layers, the feature catalog with verification + measured numbers
- [Architecture](architecture.md) — the PyTorch-referenced four-layer
  blueprint
- [Retrospective](retrospective.md) — the honest archive: features,
  measured findings (θ⁴ → θ^{d+1}, π/2 pseudo-threshold, …), the bugs the
  verification discipline caught, and the conclusions
- [Walkthrough notebook (HTML)](geocore_demo.html) — the executed demo of
  all features

## Real-data applications (all verified against known facts)

| Application | Data | What the pipeline recovers |
|---|---|---|
| Seismicity | USGS 2024, 1507 M≥5 events | S² centroid/PCA; naive (lat,lon) ±180° error exposed |
| Wind directions | Open-Meteo, 3 cities × 744 h | circular mean matches climate; arithmetic up to 183° off |
| El Niño/La Niña | NOAA ONI 1950-2026 | event detection (famous peaks), winter phase-locking |
| ENSO forecast | same + spectral geometry | quasi-period 3.62 yr, next event ~2026.6 |
| Hurricane tracks | IBTrACS NA 1980-2026, 746 storms | activity region, NNW movement, 3 real track clusters |

## Quick start

```bash
pip install -e .
python -m pytest tests/        # 178 tests
PYTHONPATH=src python3 examples/real_data.py       # seismicity
PYTHONPATH=src python3 examples/real_data_multi.py # wind
PYTHONPATH=src python3 examples/enso.py            # El Niño
PYTHONPATH=src python3 examples/hurricane.py       # hurricanes
```

The theory is the engine, not the claim: what ships is standard math,
verified to machine precision, with measured performance numbers.
