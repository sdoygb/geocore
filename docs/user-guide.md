# geocore — User Guide

A geometric computation core, architected with PyTorch as the structural
reference.  Every result is verified to machine precision; every speedup is
measured.  The derivation engine is the geometry theory; the presentation
is standard mathematics.

**Project site**: <https://sdoygb.github.io/geocore/> (GitHub Pages).

- [Installation](#installation)
- [Quick start](#quick-start)
- [The four layers](#the-four-layers)
- [Feature catalog](#feature-catalog)
- [Verification discipline](#verification-discipline)
- [PyTorch reference](#pytorch-reference)
- [Roadmap](#roadmap)

---

## Installation

```bash
pip install -e .        # numpy + scipy only
python -m pytest tests/ # 118 tests
```

## Quick start

```python
import numpy as np
from geocore import Pauli, Rotation, PolarPlane, minimize, frechet_mean

# geometric objects (≈ Tensor)
Pauli("XX").commutes_with(Pauli("ZY"))           # True — symplectic form
Rotation("XX", 0.3).merge_with(Rotation("XX", 0.4))  # Rotation('XX', 0.7)

# optimization on a manifold (≈ torch.optim)
m = PolarPlane()                                  # ds² = dr² + r² dy²
f = lambda p: (p[0]-1.5)**2 + (p[1]-0.7)**2
res = minimize(m, f, [2.0, 0.3], lr=0.05, n_steps=500, minimizer=[1.5, 0.7])
# converged=True, minimizer_error≈7e-11

# statistics (≈ torch.mean)
res = frechet_mean(m, [[2.0, 0.3], [1.2, -0.5], [1.9, 1.2], [1.1, 0.9]])
```

A full walkthrough of all features: `examples/geocore_demo.ipynb`
(regenerate with `PYTHONPATH=src python3 examples/build_demo.py`).

## The four layers

```
Layer 3  Reduce computation (≈ torch.compile)   shortcuts.py
Layer 2  Automatic verification (≈ autograd)    invariants.py
Layer 1  Operator dispatch (≈ aten/c10)         ops.py
Layer 0  Geometric object engine (≈ tensor)     objects.py, manifolds.py,
                                                sphere.py, hyperbolic.py
```

**Layer 0 — objects.** `Pauli` (2n-bit symplectic encoding, commutation,
Clifford conjugation), `Rotation` (closure semantics: merge on the same
axis, cancel at 2π), and three 2-d manifolds with one interface —
`metric_diag`, `geodesic_ode` (generic RK4), `geodesic_closed_form`
(exact), `in_chart`, `parallel_transport` (isometry):

| Manifold | Metric | Closed-form geodesic |
|---|---|---|
| `PolarPlane` | dr² + r²dy² | straight line in Cartesian |
| `Sphere` | dθ² + sin²θ dφ² | great circle in R³ |
| `HyperbolicPlane` | (dx² + dy²)/y² | semicircle / vertical line |

**Layer 1 — operators.** First-class operators with type dispatch,
declared invariants, and documented geometric theorems: `pauli.commutes`,
`pauli.conjugate_by`, `rotation.merge`, `rotation.cancel`,
`circuit.optimize`, `geodesic.polar_point`, `geodesic.parallel_transport`,
`geodesic.batch`, `laplacian.eigenvalues`, `qec.logical_error`,
`optim.gradient`, `optim.step`, `rotation.derivative`,
`geodesic.jacobian`.

**Layer 2 — verification.** Operators declare invariants; the core checks
them to machine precision on every call.  `no_verify()` disables checking
(the analogue of `torch.no_grad()`); a failing invariant raises
`VerificationError` — a wrong implementation is surfaced, not hidden.

**Layer 3 — shortcuts.** Closed-form/spectral paths replace generic
numerical paths.  Each shortcut is auto-verified against the generic path
and reports a measured `BenchmarkLog` (wall time + FLOPs speedup).

## Feature catalog

| # | Feature | ≈ PyTorch | Verified | Measured (wall/FLOPs) |
|---|---|---|---|---|
| 1 | Closed-form Pauli rotation | — | 5.6e-17 vs expm | 31,920× / 1,000,000× (n=10) |
| 2 | Closed-form geodesics (polar) | — | 2.2e-16 energy drift | 675× / 60× |
| 3 | Closed-form Laplacian spectrum | torch.linalg | 1e-8 (O(n⁻²) conv.) | 2,299× / 8×10⁴× (N=400) |
| 4 | Coherent-noise law θ^{d+1} | — | exponents d+1 to 0.001 | 48,595× / 1.6×10³× |
| 5 | Riemannian SGD | torch.optim.SGD | convergence ~1e-11 | — |
| 6 | Parallel transport | — | isometry drift ≤ 1e-16 | — |
| 7 | Riemannian Adam | torch.optim.Adam | convergence ~1e-11 | — |
| 8 | Sphere/Hyperbolic geodesics | — | great-circle/semicircle truths | 130× / 30× and 910× / 60× |
| 9 | Batch geodesics / transport | vmap | batch == per-point (≤1e-14) | 46,290× (B=500) |
| 10 | Analytic derivatives | autograd | ~1e-10 vs finite diff. | 2,390× / 66,000× (n=8) |
| 11 | Frechet mean / variance / PCA | torch.mean/std | tr(Cov)=var; ellipse exact | — |
| 13 | Spread-ellipse visualization | — | exp image faithful to 1e-15 | — |
| 14 | Clifford group elements | — | tableau vs dense 1e-16 (up to phase) | — |
| 15 | Property (fuzz) tests | — | ~700 random cases, all invariants | — |
| 16 | Real-data application (USGS 2024 seismicity) | — | centroids/PCA reproduce geography; naive ±180° error exposed | — |
| 17 | Circuit object (gates + optimize) | — | unitary equivalence to 1e-9 | — |
| 18 | Real circular data (wind, 3 cities) | — | circular mean matches climate; arithmetic up to 183° off | — |
| 19 | El Niño/La Niña diagnosis (NOAA ONI) | — | famous peaks reproduced; winter locking verified | — |
| 20 | ENSO statistical forecast | — | window [2025.1, 2027.8] cross-checked vs official tail | — |
| 21 | Spectral (geometrized) ENSO forecast | — | period 3.62 yr; backtest ~1.6 yr; forecast 2026.6 | — |
| 22 | Hurricane track geometry (IBTrACS) | — | activity region, NNW, Sept peak, 3 track clusters | — |
| 23 | Live storm forecast vs NHC | — | +24h within 0.8 deg of NHC (JULIO 2026) | — |
| 24 | PyTorch classic examples re-run | — | Jacobian 1.1e-16 vs torch autograd | — |
| 25 | More PyTorch examples (logistic, Hessian, Adam) | — | Hessian 1.1e-16; caught mixed-term bug | — |
| 26 | EuclideanSpace(n) high-dim | — | d=10 logistic acc 0.994, w cos +0.999 vs torch | — |
| 27 | Breast-cancer diagnosis (real, 30 features) | — | acc 0.9649, identical to torch | — |
| 28 | NPC screening feasibility | — | effect sizes + labeled simulation (LOOCV 0.97) | — |
| 29 | NPC geographic association | — | 20-30x ASIR contrast; not a population artifact | — |
| 30 | Quantum: H2 VQE | — | exact ground state to 5e-10; analytic grad verified | — |
| 31 | Quantum: barren-plateau diagnostics | — | global cost falls ~1.8× faster/qubit than local; no plateau at n=2 | — |
| 32 | Quantum: pre-training vs barren plateaus | — | warm start restores gradient ~5 orders; naive zero-fill trap exposed | — |
| 33 | Quantum: geometric root of barren plateaus | — | intrinsic == euclidean decay (not a coordinate artifact); root = concentration × alignment | — |
| 34 | Quantum: noise-aware VQE geometry | — | affine energy (ZNE linear exact); QFI contracts by c(λ); natural gradient immune | — |
| 35 | Quantum: non-depolarizing noise geometry | — | AD scalar QFI (1−γ) but basis-dependent energy; PD anisotropic; twirl shrinks residual | — |
| 36 | Quantum: coherent rotation noise geometry | — | pure rank-1; E=A cos²+B sin²+C sin; FS-QFI preserved (zero contraction); nat. grad not immune | — |
| 37 | Quantum: QAOA (MaxCut) gradient geometry | — | gradient O(1-10) across n,p (no barren plateau); gamma/beta split; no param concentration | — |
| 38 | Quantum: spectrum-guided vs plateaus | — | guided params 3-7× larger gradient, slower decay (−0.309 vs −0.356/q); better convergence | — |
| 39 | Quantum: discrete dynamic evolution | — | zero-gradient adiabatic: fid 0.978 (n=10) while HEA is barren; plateau never enters | — |
| 40 | Quantum: evolution scaling + molecule | — | Ising Δ~3/n, T~O(n²) (polynomial); H2 to chemical accuracy (1e-4 Ha) zero-gradient; odd-n = Z2-odd sector (frustrated boundary), symmetry-reduced path fixes (fid 1.0) | — |
| 12 | QEC diagnostics | application | pseudo-threshold = π/2 exact | 18.7× (vectorized sweep) |

## Verification discipline

- **Machine precision**: results are checked to 1e-9…1e-16; anything not
  measured is not claimed.
- **Dual paths**: every shortcut is verified against its generic path
  (closed form vs expm / RK4 / eigensolve / simulation / finite
  differences).
- **Geometric truths** where available: energy conservation, great-circle
  coplanarity, semicircle equation, Poincaré-distance additivity,
  exp∘log = q, tr(Cov) = variance, ellipse statistics, pseudo-threshold
  π/2, Jv·v₀ = t·γ′(t).
- **Measurement corrected the theory**: θ⁴ was measured to be the d=3
  special case of the general θ^{d+1} law; the naive 'flat ⇒ identity'
  parallel transport was caught by the isometry invariant.

## PyTorch reference

| PyTorch | geocore |
|---|---|
| Tensor | GeometricObject (Pauli, Rotation, manifolds) |
| aten/c10 operator library | Operator dispatch (17 operators) |
| autograd | analytic derivatives + automatic verification |
| torch.compile | ShortcutRegistry (12 shortcuts, all measured) |
| torch.optim (SGD, Adam) | RiemannianSGD, RiemannianAdam (+transport) |
| vmap / batched tensors | vectorized batch geodesics, sweeps |
| torch.mean / std / linalg PCA | Frechet mean, variance, tangent PCA |
| — | QEC diagnostics application layer |

Honest note: the architecture mirrors PyTorch's structure; the objects are
specific (Pauli strings, three 2-d manifolds) rather than generic tensors.
The value proposition is depth — every feature machine-verified with
measured performance — not breadth.

## Roadmap

Done: 40 features, 266 tests (incl. ~700 fuzz cases + real-data tests +
PyTorch comparisons + VQE + barren-plateau diagnostics + pre-training
mitigation + geometric-root analysis + noise geometry series (4
fingerprints) + QAOA gradient geometry + spectrum-guided
parameterization + discrete dynamic evolution + evolution scaling),
12 measured shortcuts (see catalog).
