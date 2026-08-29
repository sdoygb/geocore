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

See [`examples/geocore_demo.ipynb`](examples/geocore_demo.ipynb) for a
walkthrough of all four layers.

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
│   ├── rotations.py         # rotation-chain optimization (verified)
│   └── verify.py            # machine-precision verification harness
└── tests/                   # 82 tests
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

Next candidates (hypotheses to measure, not claims):
- Application layers (QEC diagnostics, geometric statistics primitives).

The theory is the engine, not the claim: what ships is standard math,
verified to machine precision, with measured performance numbers.
