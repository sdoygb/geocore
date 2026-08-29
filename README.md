# geocore — geometric computation core

[![CI](https://github.com/sdoygb/geocore/actions/workflows/ci.yml/badge.svg)](https://github.com/sdoygb/geocore/actions/workflows/ci.yml)
[![version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/sdoygb/geocore)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/sdoygb/geocore)
[![license](https://img.shields.io/badge/license-MIT-green)](https://github.com/sdoygb/geocore)
[![tests](https://img.shields.io/badge/tests-24%20passed-brightgreen)](https://github.com/sdoygb/geocore)

An independent project whose core is built on **geometric structure**:
Clifford (Pauli) algebra as the foundation, geometric invariants as the
organizing principle, and **machine-precision verification** as a first-class
feature.

**Goal**: computation that is both geometrically principled *and* cheaper —
closed-form and spectral shortcuts wherever they exist, with every component
verified to machine precision and every speedup *measured*.

```
┌─────────────────────────────────────────────────────────────────┐
│ Application layers (future)                                     │
│   QEC diagnostics, geometric statistics, physics models         │
├─────────────────────────────────────────────────────────────────┤
│ L3  Reduce computation (≈ torch.compile)                        │
│     ShortcutRegistry + BenchmarkLog                             │
│     closed-form/spectral fast paths, auto-verified vs generic   │
│     path, measured FLOPs/time speedup — no unmeasured claims    │
├─────────────────────────────────────────────────────────────────┤
│ L2  Automatic verification (≈ autograd)                         │
│     Invariant / VerificationContext / no_verify()               │
│     operators declare geometric invariants; the core self-      │
│     checks them to machine precision on every call              │
├─────────────────────────────────────────────────────────────────┤
│ L1  Operator dispatch (≈ aten/c10)                              │
│     Operator / @dispatch / registry                             │
│     geometric operators dispatch by geometric object type       │
├─────────────────────────────────────────────────────────────────┤
│ L0  Geometric object engine (≈ torch.Tensor)                    │
│     Pauli (2n-bit symplectic encoding + r-bit phase),           │
│     Rotation (R_P(θ), closure semantics)                       │
└─────────────────────────────────────────────────────────────────┘
```

## Core principle: engine vs presentation

- The **derivation engine** (how we think, choose, and verify) is the
  geometry theory: closure/completeness reasoning, spectral discipline,
  exact-invariant verification.
- The **presentation** is standard mathematics + machine-precision
  verification — every claim must be reproducible by anyone.
- A claim is accepted only if verified to machine precision; an unmeasured
  claim is not a claim.

## Quick start

```python
import geocore as gc
from geocore import Pauli, Rotation, get_op
import numpy as np

# Layer 1: geometric operator dispatch
get_op("pauli.commutes")(Pauli("X"), Pauli("Z"))        # False (anticommute)
get_op("rotation.merge")(Rotation("XX", 0.3), Rotation("XX", 0.4))
# Rotation('XX', 0.7)  — closure of phase addition

# Layer 2: automatic verification (strict by default)
opt, cl = get_op("circuit.optimize")([
    ("XXII", -np.pi/4), ("ZZIY", np.pi/4), ("ZZYI", np.pi/4),
    ("YYXX", np.pi/4), ("XXII", np.pi/4),
])  # 5 -> 3 rotations, unitarily equivalent (checked automatically)

# Layer 3: reduce computation — closed-form Pauli rotation vs expm
from geocore import shortcuts
r, s = Rotation("XX", 0.7), np.random.randn(4)
res, report = shortcuts.registry.apply("rotation.closed_form", r, s, verify=True)
print(report)              # machine-precision check vs the generic path
print(shortcuts.registry.benchmark("rotation.closed_form", r, s))
```

See [`examples/geocore_demo.ipynb`](examples/geocore_demo.ipynb) for an
executed walkthrough of all features (regenerate with
`PYTHONPATH=src python3 examples/build_demo.py`),
[`docs/user-guide.md`](docs/user-guide.md) for the full user guide,
[`docs/retrospective.md`](docs/retrospective.md) for the honest project
retrospective, and the **project site**:
<https://sdoygb.github.io/geocore/> (GitHub Pages, docs/ directory).

## First measured result (n = 10 qubits)

Closed-form Pauli rotation `R_P(θ)|ψ⟩ = cos(θ/2)|ψ⟩ − i sin(θ/2) P|ψ⟩`
(from the orbit closure `P² = I`) versus the generic dense matrix
exponential:

| n | wall-time speedup | FLOPs speedup |
|---|---|---|
| 4 | 20× | 260× |
| 6 | 72× | 4,100× |
| 8 | 1,258× | 66,000× |
| 10 | **31,920×** | **1,000,000×** |

FLOPs ratio scales ~4ⁿ as predicted (`expm` is O(8ⁿ), the Pauli action is
O(2ⁿ)).  Every number above is produced by the benchmark harness
(`BenchmarkLog`), and every shortcut is verified against the generic path to
machine precision before it is trusted.

## Verification status

| Check | Result |
|---|---|
| Conjugation primitives vs matrix truth (all gates × all Paulis, n ≤ 3) | 0 failures |
| Pauli action vs explicit matrices (all 3ⁿ Paulis × basis states) | pass |
| Closed form vs expm (n = 1..4, random) | pass to 1e-9 |
| Unitary equivalence on random circuits | pass |
| Known cases (merge, cancel, π/2 absorption, issue example 5→3) | pass |

## Repository layout

```
geocore/
├── docs/architecture.md     # PyTorch-referenced 4-layer blueprint
├── examples/geocore_demo.ipynb  # walkthrough of all layers
├── src/geocore/
│   ├── clifford.py          # symplectic encoding, r-bit phase, Pauli action
│   ├── objects.py           # L0: Pauli / Rotation
│   ├── ops.py               # L1: geometric operator dispatch
│   ├── invariants.py        # L2: automatic verification
│   ├── shortcuts.py         # L3: reduce computation + BenchmarkLog
│   ├── optim.py             # Riemannian optimizer (≈ torch.optim)
│   ├── sphere.py            # S²: spherical coords, great-circle geodesics
│   ├── hyperbolic.py        # H²: upper half-plane, semicircle geodesics
│   ├── derivatives.py       # analytic derivatives (≈ autograd)
│   ├── rotations.py         # rotation-chain optimization (verified)
│   └── verify.py            # machine-precision verification harness
└── tests/                   # 192 tests
```

## Riemannian optimizer (≈ torch.optim)

PyTorch moves parameters in Euclidean space; geocore moves them *on the
manifold*.  The gradient is the Riesz representative of df w.r.t. the
metric (`optim.gradient`, verified by the duality `g(grad f, v) = df(v)`);
each step follows the exponential map (`optim.step`, verified against the
closed form to machine precision):

```python
from geocore import PolarPlane, minimize

m = PolarPlane()  # ds² = dr² + r² dy²; exp map = straight line in Cartesian
f = lambda p: (p[0] - 1.5)**2 + (p[1] - 0.7)**2

res = minimize(m, f, [2.0, 0.3], lr=0.05, n_steps=500, minimizer=[1.5, 0.7])
# converged=True, minimizer_error≈7e-11, f strictly non-increasing
```

Every step is verified (exp-map validity, manifold constraint `r > 0`,
descent property).  The stateful `RiemannianSGD(lr, momentum)` mirrors
`torch.optim.SGD`.  Measured per-step cost of the closed-form exponential
map vs the RK4 ODE integration: **370× wall time, 60× FLOPs** (n=2).

### RiemannianAdam (≈ torch.optim.Adam)

Adaptive moment estimation on the manifold: first/second moment buffers
live in the tangent space, and after each step they are **parallel
transported** along the step's geodesic to the new point (the second
moment via its square root, so it stays positive).  Transport is verified
to be an isometry; the buffers' metric norm is preserved to ~1e-17.

```python
from geocore import Sphere, minimize
import numpy as np

S = Sphere()
target = [1.4, 2.0]
f = lambda p: np.arccos(np.clip(np.sin(p[0])*np.sin(target[0])*np.cos(p[1]-target[1])
                                + np.cos(p[0])*np.cos(target[0]), -1, 1))**2
res = minimize(S, f, [0.9, 0.4], lr=0.1, n_steps=500, optimizer="adam",
               minimizer=target, atol=1e-5)
# converged=True, minimizer_error≈3e-11 (descent_ok reported honestly —
# Adam overshoots early flat-region steps; a step leaving the chart
# raises a clear error instead of failing silently)
```

Adam converges on all three manifolds to ~1e-11 (squared geodesic
distance to a target).  Parallel transport is a first-class geometric op
(`geodesic.parallel_transport`) with the isometry invariant: sphere =
SO(3) rotation, hyperbolic = rotation + scaling, polar plane = coordinate
rotation + scaling (the flat plane's transport is the identity only in
Cartesian coordinates — the invariant caught this when tested).

## Vectorized / batched core paths (≈ batched tensor ops / vmap)

The batch analogues of the geodesic and transport ops: inputs (B, 2),
outputs (B, 2), verified to equal the per-point paths to machine
precision (the vectorized closed form vs the per-point loop; the
vectorized RK4 vs the per-point RK4, exactly 0.0).  The batch closed-form
shortcut (`geodesic.batch_closed_form`) is measured:

| B | wall-time speedup | FLOPs speedup (analytic) |
|---|---|---|
| 10 | 1,581× | 30× |
| 100 | 14,163× | 30× |
| 500 | 46,290× | 30× |

`minimize_batch` runs one gradient flow per starting point with
vectorized steps (all points verified, chart-guarded), agreeing with the
per-point `minimize` loop.  The batch verification caught a real bug: a
refactor of `geodesic_ode` to batched indexing had swapped the velocity
component for the position — the closed-form-vs-RK4 batch comparison
flagged it immediately (both paths would otherwise have been wrong in
the same way).

## Analytic derivatives (≈ autograd's gradient computation)

PyTorch's autograd computes gradients; geocore provides *analytic*
closed-form derivatives, verified against finite differences to ~1e-10:

- `rotation.derivative` — d/dθ R_P(θ)|ψ⟩ = −(i/2) P R_P(θ)|ψ⟩ (closed form
  from P² = I), O(2^n) vs two dense `expm`; measured **2,390× / 66,000×**
  (n=8).
- `geodesic.jacobian` — the Jacobians of the geodesic endpoint w.r.t. the
  initial point and velocity (sensitivity / tangent propagation), closed
  form per manifold (polar: chain rule through the Cartesian line; sphere:
  through the R³ embedding including the rotating frame; hyperbolic:
  through the semicircle parameters c, R, α, β).  Verified against central
  differences to ~1e-10 on all three manifolds, plus the homogeneity
  identity Jv·v₀ = t·γ′(t).  Modest measured speedup (~6× — its real
  value is exactness over finite differences).

```python
from geocore import Rotation, get_op
import numpy as np

state = np.random.randn(8) + 1j*np.random.randn(8)
d = get_op("rotation.derivative")(Rotation("XYZ", 0.7), state)  # verified
```

## Geometric statistics (≈ torch.mean) + analytic gradients in optimizers

The Frechet mean minimizes the weighted sum of squared geodesic
distances; its gradient is closed form (grad_p d(p,q)² = −2·log_p(q)),
so the optimizers now accept an **analytic gradient** (`minimize(…,
grad_f=…)`) that is *verified against finite differences on every step*
(disagreement beyond 1e-4 raises — a wrong analytic gradient is
surfaced, not hidden; the worst deviation is reported in
`max_grad_error`).

```python
from geocore import PolarPlane, frechet_mean
import numpy as np

pts = np.array([[2.0, 0.3], [1.2, -0.5], [1.9, 1.2], [1.1, 0.9]])
res = frechet_mean(PolarPlane(), pts, lr=0.1, n_steps=500)
# res.point is the Frechet mean — on the flat polar plane it equals the
# Cartesian arithmetic mean to 4e-16; on the sphere/hyperbolic plane two
# points give the geodesic midpoint exactly (d(m,q1)=d(m,q2)=d12/2)
```

The log map (inverse exponential) is closed form per manifold and
verified by exp_p(log_p(q)) = q to ~1e-12; the geodesic distance uses
numerically stable formulas (haversine / 2·asinh(√(δ/2))) — the naive
acos/acosh loses precision for nearly coincident points (measured
2.1e-8 error before the fix).

### Manifold variance / covariance (≈ torch.std + tangent PCA)

The Frechet variance is (1/N) Σ d(m, pᵢ)²; the tangent covariance maps
every point to the mean's tangent space via the log map and computes
(1/N) Σ log_m(pᵢ) log_m(pᵢ)ᵀ in an **orthonormal frame** (the coordinate
charts are not orthonormal — g = diag(1, r²) etc. — so the components are
scaled by √g_diag(m); the raw coordinates would distort the geometry,
caught by the tr(Cov) = variance check).  Principal directions are the
eigendecomposition (tangent PCA).

```python
from geocore import PolarPlane, frechet_variance, principal_directions

var = frechet_variance(PolarPlane(), pts)
evals, evecs = principal_directions(PolarPlane(), pts)  # ascending
```

Verified truths (machine precision): tr(Cov) = variance on all three
manifolds (because |log_m(p)|_g = d(m, p)); for a uniform ellipse of
semi-axes (a, b) the variance is exactly (a²+b²)/2, the covariance
eigenvalues exactly (b²/2, a²/2), and the top principal direction is the
long axis (|dot| = 1); a single point has zero spread.

Spread ellipses (`geocore.viz`): the tangent PCA ellipse flowed through
the exponential map onto each manifold (every ellipse point satisfies
d(m, exp_m(v)) = |v|_g to machine precision), drawn in each manifold's
natural chart:

![spread ellipses](examples/spread_ellipses.png)

(left: polar plane in Cartesian; middle: sphere in azimuthal-equidistant
projection about the mean; right: hyperbolic plane in the upper
half-plane.)

## QEC diagnostics application layer

`geocore.qec.diagnose` runs the coherent-noise diagnostic over a code
family (repetition codes, distance d, noise R_X(θ)):

```python
from geocore.qec import diagnose

rep = diagnose((3, 5, 7))
# d=3: P_L ~ 0.1874 theta^4.000 (analytic theta^4, coeff 0.1875)
# d=5: P_L ~ 0.1561 theta^6.000 (analytic theta^6, coeff 0.1562)
# d=7: P_L ~ 0.1365 theta^8.000 (analytic theta^8, coeff 0.1367)
# pseudo-threshold theta* = 1.5708 (pi/2) for every distance
# crossover P_L(d1) = P_L(d2) at theta = 1.5708
```

Measured truths (all machine-verified): the empirical exponents are
d+1 to 0.001 and the leading coefficients match
C(n,(n+1)/2)/2^{n+1} to <2%; the pseudo-threshold is *exactly* π/2 for
every distance (at θ = π/2, P_L(n) = 1/2 = P_phys for all n — encoding
helps below π/2 and hurts above, verified exactly for d=3); crossovers
are roots verified by substitution to 1e-10.  The vectorized sweep
(`logical_error_sweep`) is measured 18.7× faster than the per-point
closed-form loop (and ~600× vs the O(2^n) state-vector simulation).

## Is this a tool? — property-based evidence

Every invariant is also verified on **randomly generated** inputs, not
just the fixed scenarios in the other tests (`tests/test_property.py`,
fixed seeds, ~700 random cases): random Paulis/rotations (including edge
angles 0, 2π, ±4π), random deep Clifford circuits, random manifold
points/velocities, random point sets, random QEC parameters — all
invariants hold machine-precision.  A real end-to-end use case
(`examples/real_use.py`) recovers the true direction and the anisotropic
noise structure (magnitudes + orientation) of a directional sensor from
raw sphere measurements:

```
true direction             : [0.931  0.5191]
geocore Frechet mean       : [0.9303 0.5118] | error 5.91e-03 rad
tangent PCA eigenvalues    : [0.0058 0.0631] (true 0.0064, 0.0625)
recovered long-axis angle  : 3.08 deg off the true noise axis
```

The geometric pipeline is the only one that recovers the anisotropic
spread of directions on a sphere — the naive Euclidean mean has no
principled analogue.  The API is dynamic (arbitrary inputs, runtime
verification, runtime shortcut dispatch); the tests prove it.

### Real data: 2024 global seismicity

`examples/real_data.py` analyzes the real USGS earthquake catalog (1507
M≥5 events in 2024, stored in `examples/data/`) with the S² pipeline —
the centroid is the seismicity's mass center, the PCA principal axis is
the strike of the seismic belt:

```
=== Tonga-Fiji (crosses 180° meridian) (91 events) ===
geometric centroid : -20.91, 179.71   (correct — the belt is at ~180°E)
naive (lat,lon) avg: -20.84, -69.53   (lands in South America!)
principal axis     : bearing 82.6 deg, eigenvalue ratio 1.9

=== Global (1507 events) ===
geometric centroid : -10.57, 167.10   (southwest Pacific — ring's center)
naive (lat,lon) avg: -1.49,  25.84    (Africa — meaningless)
```

The naive Euclidean average of latitudes/longitudes is *demonstrably
wrong* across the ±180° meridian (175° and −175° average to 0°) — the
geometric treatment is the correct one for directions on a sphere, and
the difference is visible in real data, not a synthetic toy.

### Real data: circular wind directions (three cities)

`examples/real_data_multi.py` analyzes hourly wind direction (January
2024, Open-Meteo archive, stored in `examples/data/`) — a *circular*
dataset — with the same S² pipeline, no special-casing:

```
              geocore mean   naive arithmetic   reference
tokyo   (1月)   346.1° (NW)   253.0°  (err 92°)   345.2°
beijing (1月)   342.9° (NW)   173.3°  (err 183°)  356.2°
sydney  (1月)    74.5° (ENE)  125.4°  (err 50°)    75.5°
```

The bimodal wind data makes the arithmetic mean average the two modes
into the gap between them (up to 183° off — almost the opposite
direction); the geometric mean matches the independent vector-mean
reference to <1° and the known climate (NW winter monsoons for
Tokyo/Beijing, an easterly regime for Sydney in January).  The same
module handled S² seismicity and S¹ wind directions correctly — the
dynamic-capability check on real, differently-shaped data.

### Real data: El Niño / La Niña (NOAA ONI 1950–2026)

`examples/enso.py` detects ENSO events with the standard criterion
(|ONI| ≥ 0.5 for ≥ 5 seasons) and analyzes the phase space with the
geocore statistics pipeline:

```
top El Nino  : 2014-16 (2.59), 1997-98 (2.37), 1982-83 (2.14) — matches NOAA
top La Nina  : 1973-74 (-2.04), 1988-89 (-1.85), 2007-08 (-1.76)
phase locking: |ONI| peaks in December (boreal winter) — verified
phase space  : El Nino centroid (+1.04, +0.03), neutral (-0.02, 0.00),
               La Nina (-0.91, -0.02) — the three regimes separate cleanly
```

The ONI is NOAA's product (we use it, not rebuild it); our contribution
is the verified detection logic and the geometric phase-space view.
Event detection caught a real bug in its own first version (runs
starting from neutral were never recorded — fixed).

### ENSO statistical forecast (`examples/enso_forecast.py`)

A statistical estimate from the 1950-2026 record (not a physical
forecast — see NOAA CPC/IRI for authoritative outlooks):

```
El Nino start-to-start interval: mean 3.43 yr, range [1, 6]
La Nina start-to-start interval: mean 3.74 yr, range [1, 9]
strong El Nino -> La Nina within 2 yr: 44%
next El Nino : expected ~2026.4, 68% window [2025.1, 2027.8]
official ONI tail turns positive (+1.39 by mid-2026) — consistent
```

The statistical window agrees with the official initial conditions at
the end of the ONI file (the trailing values turn positive through
2026), and the 2025-10..12 ONI dip below −0.5 flags a possible weak
La Nina that the 2026 official tail then interrupts — both signals
cross-check each other.

### Hurricane track geometry (IBTrACS, NA basin 1980-2026)

`examples/hurricane.py` analyzes 746 real storm tracks as curves on the
sphere: per-track centroids, the activity region (Frechet mean + PCA +
spread ellipse), circular movement statistics, and a simple centroid
clustering:

```
activity region center : 27.7N, 64.6W (angular std 19.4 deg), anisotropy 2.2
mean movement          : 347.6 deg from north (NNW)
season peak            : September (June-November season)
track clusters         : Gulf-Caribbean (24.3N 86.3W, 263 storms),
                         mid-latitude (30.7N 63.0W, 249),
                         Cape Verde (24.4N 41.7W, 234)
```

All results match real climatology: the central-western tropical
Atlantic activity center, the NNW mean movement, the September peak,
and the three clusters matching the known track types (Cape Verde
storms crossing the Atlantic, Gulf-Caribbean storms, mid-latitude
recurving storms).

### Live storm track forecast vs NHC (`examples/storm_forecast.py`)

The similar-track (analog) method on the S² pipeline — the observed
48-h segment of an ACTIVE storm is matched against the historical EP
record (1147 storms) and the analogs' futures form the forecast.  Tested
against the live 2026 storms and compared with the official NHC
forecasts:

```
JULIO (EP102026), forecast from 2026-08-25:
  our +24h : 18.0N 119.0W     NHC +24h : 18.1N 119.9W   (0.8 deg, ~90 km)
  NHC: dissipated 26 Aug (confirmed); ours predicts continued motion —
  lifecycle prediction is our honest blind spot (track-only method)

ISELLE (EP092026): direction agrees (NW); position ~2-2.5 deg off over
  5 days (within the typical official-forecast error); NHC marks
  post-tropical decay, we do not model intensity.
```

Honest conclusion: the SHORT-TERM POSITION forecast (24-48 h) is close
to the professional agencies (within their typical error); the LIFECYCLE
(decay / dissipation) is not predicted — the similar-track method
forecasts tracks, not intensity.

### EuclideanSpace: the N-dimensional fix

`EuclideanSpace(n)` is R^n as a flat manifold — geodesics are straight
lines, transport is the identity — which removes the "2-d only"
limitation: the optimizers now run on arbitrary-dimensional problems.
Verified: quadratic optimization converges in n = 3/10/50 to ~1e-14; a
d = 10-feature logistic regression on EuclideanSpace(11) reaches 99.4%
accuracy with the learned weight direction at cos = +0.999 of the true
one — matching torch's nn.Linear + BCEWithLogits (0.994, cos +1.000).

### PyTorch's classic examples, re-run with geocore

`examples/pytorch_comparison.py` runs three official-tutorial problems
side by side:

| Example | PyTorch | geocore | Agreement |
|---|---|---|---|
| Linear regression (learn y=3x−2) | nn.Linear + SGD → w=2.9999, b=−1.9963 | minimize() → w=3.0036, b=−1.9919 | both at the noise floor |
| Autograd Jacobian of a geodesic | `torch.autograd.functional.jacobian` | analytic `geodesic.jacobian` | **1.1e-16** |
| Circle-Laplacian spectrum | `torch.linalg.eigvalsh` (discrete) | closed form k² | converges O(1/N²) |

The Jacobian row is the cleanest statement of the project's
verification discipline: PyTorch's automatic differentiation and our
analytic closed form compute the same derivative to machine precision —
the analytic path is not an approximation, it is the exact answer.

Extended comparisons (`examples/pytorch_comparison.py`):

| Example | PyTorch | geocore | Agreement |
|---|---|---|---|
| Logistic regression (boundary x=0.5) | nn.Linear + BCE → x≈0.50 | minimize() → x≈0.41 | both near the truth |
| **High-dim logistic (d=10)** | nn.Linear(10) + BCE → acc 0.994 | minimize() on **EuclideanSpace(11)** → acc 0.994 | w direction cos +1.000 / +0.999 |
| Hessian of Re⟨ψ\|R₁R₂\|ψ⟩ | `torch.autograd.functional.hessian` | analytic 2×2 Hessian | **1.1e-16** |
| Adam on Rosenbrock | torch.optim.Adam → f≈2e-7 | RiemannianAdam → f≈4e-6 | both converge to (1,1) |

The Hessian row is another verification win: the torch comparison caught
a wrong naive derivation of the mixed term (A = dR/dθ is anti-Hermitian,
so ⟨ψ\|AB\|ψ⟩ ≠ ⟨d₁\|d₂⟩) — fixed, and the analytic Hessian now agrees
with autograd to machine precision.

### Spectral (geometrized) ENSO forecast (`examples/enso_spectral_forecast.py`)

The spectrum as a geometric invariant: the observed ONI's dominant
quasi-period is **3.62 yr** (ENSO band 2-8 yr carries 72% of the
power) — a stable feature across subperiods.  Backtesting the spectral
extrapolation against the historical record gives a mean |error| of
~1.6 yr (ENSO is quasi-periodic, not periodic — an honest window, not
a date).  The final forecast:

```
last El Nino peak (2023) + dominant period (3.62 yr) = 2026.6
winter-locked: 2026-27 DJF
cross-checks: interval statistics ~2026.4; official ONI tail positive
through 2026 (+1.39) — independent signals converge on 2026.5-2027
```

Method comparison on the backtest (next-event prediction): spectral
1.62 yr mean error vs interval statistics 2.09 yr — the spectral
(geometrized) view is the better of the two, and both are honest about
ENSO's intrinsic irregularity.

## Circuit object

A `Circuit` is a gate sequence of Clifford gates (`h, s, sd, sx, sxdg,
cx`) and Pauli rotations `R_P(θ)`, with `to_matrix`, `apply_to_state`
(closed-form rotations, O(2ⁿ)) and `optimize`:

```python
from geocore import Circuit

c = Circuit([("h", 0), ("r", "XI", 0.4), ("cx", 0, 1), ("r", "YY", 0.6)])
opt, cliff = c.optimize()   # Clifford pulled through, rotations merged
# U(input) == U(clifford) @ U(optimized), verified to 1e-9 automatically
```

The optimizer pulls Clifford gates through the rotations
(`C R_P(θ) = R_{C†PC}(θ) C`, an exact identity), reduces the rotation
chain to its fixed point (merge / 2π cancel / π/2 absorption), and
verifies unitary equivalence to machine precision (a failure raises).
Verified: 40 random 2-qubit + 15 random 3-qubit mixed circuits all
unitary-equivalent after optimization; the textbook 5→3 reduction; the
π/2 absorption is exact (the absorbed piece stays an exact Clifford
rotation — expressing it with S gates would leak S's e^{iπ/4} global
phase, a real subtlety caught by the verification).  Fixing the
optimizer exposed a pre-existing bug in the π/2 piece construction
(fold order) that no test had covered.

## Clifford group elements (L0 extension)

A `Clifford` object on n qubits: the symplectic tableau (2n×2n binary
matrix + 2n-bit phase, Aaronson-Gottesman), constructed from a gate
sequence (`h, s, sd, sx, sxdg, cx`).  Composition and Pauli conjugation
are binary linear algebra on the tableau; the dense matrix is rebuilt
from the tableau for verification.

```python
from geocore import Clifford, Pauli

C = Clifford([("cx", (0, 1))], 2)
C.conjugate(Pauli("XI"))          # (Pauli('XX'), 0) — the classic CNOT rule
C.compose(C).conjugate(Pauli("XI"))  # group product, verified
```

Verified (machine precision): tableau vs gates-dense agree to 1e-16 up
to a global phase (the tableau determines the Clifford only
projectively — C and e^{iθ}C share a tableau, so the comparison allows
the 8th-root phase); conjugation (axis and phase) equals C P C† exactly;
composition equals the dense product up to global phase; associativity
exact; H² = I, S² = Z; CNOT rules (X⊗I→X⊗X, I⊗Z→Z⊗Z, Y⊗I→Y⊗X) exact.
The r-bit phase of the existing machinery was extended to the full 2-bit
phase i^q — the single sign bit cannot track the i carried by Y factors
in multiplication (caught by the composition verification).

## Manifolds (all with closed-form geodesics, verified)

| Manifold | Metric | Geodesic (closed form) | Verified truths |
|---|---|---|---|
| `PolarPlane` | dr² + r² dy² | straight line in Cartesian | energy drift ≤ 2e-16 |
| `Sphere` (S²) | dθ² + sin²θ dφ² | great circle in R³ | \|p_t\| = 1, coplanarity det ≈ 1e-16, energy drift 6e-17 |
| `HyperbolicPlane` (H²) | (dx² + dy²)/y² | semicircle / vertical line | (x−c)² + y² = R² exact, Poincaré distance additive to 1e-9 |

Closed form vs RK4 ODE: agreement to ~5e-14.  Measured shortcuts:
sphere **130× / 30×**, hyperbolic **910× / 60×** (wall time / FLOPs).
The Riemannian optimizer runs on all three: on S² and H² it converges to
the closed-form minimizer of the (squared) geodesic distance to a target
to ~1e-10 with verified descent.

```python
from geocore import Sphere, minimize
import numpy as np

S = Sphere()
target = [1.4, 2.0]
f = lambda p: np.arccos(np.clip(np.sin(p[0])*np.sin(target[0])*np.cos(p[1]-target[1])
                                + np.cos(p[0])*np.cos(target[0]), -1, 1))**2
res = minimize(S, f, [0.9, 0.4], lr=0.3, n_steps=300, minimizer=target)
# converged=True, minimizer_error≈4e-11, descent_ok=True
```

## Roadmap (honest)

Done so far — each with machine-precision verification + measured benchmark:

1. ✅ Closed-form Pauli rotation vs `expm` (31,920× / 1,000,000× at n=10).
2. ✅ Closed-form geodesics (polar plane) vs ODE integration (675× / 60×).
3. ✅ Closed-form Laplacian spectrum vs discrete eigensolve (2,299× / 8×10⁴×).
4. ✅ Coherent-noise leading law: measured **θ^{d+1}**, not θ⁴ — θ⁴ is the
   d=3 special case; exponents 4, 6, 8, 10 for d = 3, 5, 7, 9 with
   coefficients C(n,(n+1)/2)/2^{n+1} exact (48,595× / 1.6×10³×).
5. ✅ Riemannian optimizer step (≈ torch.optim): closed-form exponential map
   vs RK4 (370× / 60×), gradient = Riesz representative, verified descent.
6. ✅ Sphere S² and hyperbolic plane H²: closed-form great-circle /
   semicircle geodesics vs RK4 (130× / 30× and 910× / 60×); optimizer
   converges on both to ~1e-10 (geodesic-distance potentials).
7. ✅ RiemannianAdam (≈ torch.optim.Adam): adaptive steps + parallel
   transport of moment buffers (verified isometry, positivity-preserving
   via √v); converges on all three manifolds to ~1e-11.
8. ✅ Vectorized/batched core paths (≈ vmap): batch geodesics (closed form
   + RK4) and batch transport, verified identical to the per-point paths;
   batch closed form measured 46,290× at B=500; `minimize_batch` agrees
   with the per-point loop.
9. ✅ Analytic derivatives (≈ autograd): `rotation.derivative` closed form
   (2,390× / 66,000× at n=8) and per-manifold `geodesic.jacobian`
   (verified to ~1e-10, incl. the Jv·v₀ = t·γ′(t) identity).
10. ✅ Geometric statistics (≈ torch.mean): log map (inverse exp,
    exp∘log = q to ~1e-12) + Frechet mean with the analytic gradient
    (verified vs FD every step); polar-plane mean = Cartesian arithmetic
    mean to 4e-16; stable distance formulas (haversine / asinh).
11. ✅ QEC diagnostics application layer: vectorized sweeps, the
    θ^{d+1} law reproduced to 0.001, leading coefficients to <2%,
    pseudo-threshold exactly π/2 for every distance, verified crossovers.
12. ✅ Manifold variance/covariance (≈ torch.std + tangent PCA):
    Frechet variance, orthonormal-frame tangent covariance with
    tr(Cov) = variance to machine precision, ellipse statistics exact
    ((a²+b²)/2, (b²/2, a²/2), long-axis direction |dot|=1).
13. ✅ Spread-ellipse visualization (geocore.viz): the tangent PCA
    ellipse flowed through the exponential map (d(m, exp_m(v)) = |v|_g
    to 1e-15), in each manifold's natural chart.
14. ✅ Clifford group elements (L0 extension): symplectic tableau with
    full 2-bit phases, composition/conjugation verified against dense
    truth (up to the projective global phase).
15. ✅ Property-based (fuzz) tests: ~700 random cases across every
    feature (arbitrary inputs, edge angles, deep circuits) — the
    dynamic-tool evidence — plus real end-to-end use cases
    (examples/real_use.py, examples/real_data.py).
16. ✅ Real-data application: 2024 USGS seismicity (1507 M≥5 events) —
    the S² centroid/PCA reproduce verifiable geography (Japan centroid,
    Tonga belt strike), and expose the naive (lat, lon) average's real
    ±180° error.
17. ✅ Circuit object: Clifford gates + Pauli rotations with the
    geometric optimizer (pull-through + merge/2π/π/2), unitary
    equivalence verified to machine precision; fixed a pre-existing
    π/2-piece fold-order bug no test covered.
18. ✅ Real circular data (wind, 3 cities): the S² pipeline computes the
    correct circular mean on bimodal wind data (arithmetic mean up to
    183° off; geometric matches climate — NW monsoons, easterly
    Sydney).
19. ✅ El Niño/La Niña diagnosis (NOAA ONI 1950-2026): event detection
    reproduces the famous peaks (2015-16: 2.59, 1997-98: 2.37); winter
    phase-locking verified; phase-space regimes separate.
20. ✅ ENSO statistical forecast: interval/alternation statistics from
    the real record — next El Nino expected ~2026.4 (window
    [2025.1, 2027.8]), cross-checked against the official ONI tail.
21. ✅ Spectral (geometrized) ENSO forecast: dominant quasi-period
    3.62 yr (72% band power), backtested (mean error ~1.6 yr), final
    forecast 2026.6 winter-locked to 2026-27 DJF; spectral beats
    interval statistics on the backtest.
22. ✅ Hurricane track geometry (IBTrACS, 746 NA storms): activity
    region center/PCA, NNW movement, September peak, and three clusters
    matching the real track types.
23. ✅ Live storm track forecast vs NHC: similar-track (analog) method
    on the S² pipeline; +24h within 0.8 deg of NHC for JULIO 2026;
    lifecycle prediction honestly absent.
24. ✅ PyTorch classic examples re-run: linear regression, autograd
    Jacobian (1.1e-16 vs torch), spectrum — the same math, verified.
25. ✅ More PyTorch examples: logistic regression, Hessian (1.1e-16 vs
    torch autograd; caught a wrong mixed-term derivation), Adam on
    Rosenbrock (both converge).
26. ✅ EuclideanSpace(n): arbitrary-dimension optimization/classification
    (the 2D-only limitation fixed) — high-dim logistic matches torch
    (acc 0.994 both, w cos +1.000 vs +0.999).

Next candidates (hypotheses to measure, not claims):
- Intensity/lifecycle modeling (the honest gap the comparison exposed).

The theory is the engine, not the claim: what ships is standard math,
verified to machine precision, with measured performance numbers.
