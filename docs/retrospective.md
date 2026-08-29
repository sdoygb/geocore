# geocore — Project Retrospective

*Status: 17 features, 155 tests, 12 measured shortcuts, ~700 fuzz cases.
This document is the honest archive: what was built, what was measured,
what the verification discipline caught, and what the theory's actual
role turned out to be.*

- [Delivered features](#delivered-features)
- [Measured findings (measurement corrected the theory)](#measured-findings)
- [What the verification discipline caught](#what-the-verification-discipline-caught)
- [Is it a tool? (dynamic evidence)](#is-it-a-tool)
- [Honest conclusions](#honest-conclusions)
- [PyTorch reference: what maps, what does not](#pytorch-reference)
- [Roadmap review](#roadmap-review)

---

## Delivered features

Architected with PyTorch as the structural reference, layer by layer:

| # | Feature | ≈ PyTorch | Verified | Measured (wall/FLOPs) |
|---|---|---|---|---|
| 1 | Closed-form Pauli rotation | — | 5.6e-17 vs expm | 31,920× / 1,000,000× (n=10) |
| 2 | Closed-form geodesics (polar) | — | 2.2e-16 energy drift | 675× / 60× |
| 3 | Closed-form Laplacian spectrum | torch.linalg | O(n⁻²) conv. | 2,299× / 8×10⁴× (N=400) |
| 4 | Coherent-noise law θ^{d+1} | — | exponents d+1 to 0.001 | 48,595× / 1.6×10³× |
| 5 | Riemannian SGD | torch.optim.SGD | convergence ~1e-11 | — |
| 6 | Parallel transport | — | isometry drift ≤ 1e-16 | — |
| 7 | Riemannian Adam | torch.optim.Adam | convergence ~1e-11 | — |
| 8 | Sphere/Hyperbolic geodesics | — | great-circle/semicircle truths | 130× / 30×; 910× / 60× |
| 9 | Batch geodesics / transport | vmap | batch == per-point ≤1e-14 | 46,290× (B=500) |
| 10 | Analytic derivatives | autograd | ~1e-10 vs finite diff. | 2,390× / 66,000× (n=8) |
| 11 | Frechet mean / variance / PCA | torch.mean/std | tr(Cov)=var; ellipse exact | — |
| 12 | QEC diagnostics | application | pseudo-threshold = π/2 exact | 18.7× (vectorized) |
| 13 | Spread-ellipse visualization | — | exp image faithful to 1e-15 | — |
| 14 | Clifford group elements | — | tableau vs dense 1e-16 (up to phase) | — |
| 15 | Property (fuzz) tests | — | ~700 random cases, all invariants | — |
| 16 | Real-data application (USGS 2024) | — | centroids/PCA reproduce geography | — |
| 17 | Circuit object + optimizer | — | unitary equivalence to 1e-9 | — |
| 18 | Real circular data (wind, 3 cities) | — | circular mean matches climate; arithmetic 183° off | — |
| 19 | El Niño/La Niña diagnosis (NOAA ONI) | — | famous peaks 2.59/2.37/2.14; winter locking | — |
| 20 | ENSO statistical forecast | — | interval stats; window cross-checked vs official tail | — |

Every shortcut is verified against its generic path (closed form vs
expm / RK4 / eigensolve / simulation / finite differences) and every
speedup is a measured `BenchmarkLog`, not an estimate.

## Measured findings

The project's most valuable output is the set of places where
*measurement corrected a prior belief* — the core of the
"measurement-over-assertion" discipline:

1. **θ⁴ is not universal — the law is θ^{d+1}.** The θ⁴ coherent-noise
   scaling is the d = 3 special case; measured exponents 4.00, 6.00,
   8.00, 10.00 for d = 3, 5, 7, 9, leading coefficients matching
   C(n,(n+1)/2)/2^{n+1} to machine precision.  This is the canonical
   example of the theory being corrected by its own verification.
2. **"Flat ⇒ identity transport" is wrong.** The polar plane's
   parallel transport was initially assumed to be the identity; the
   isometry invariant caught it — the polar coordinate frame is not
   parallel, and the correct transport rotates and scales (tangent
   preservation then exact to 0.0).
3. **The naive distance formulas are numerically unstable.** acos/acosh
   lose precision for nearly coincident points (measured 2.1e-8 false
   error); the haversine / 2·asinh(√(δ/2)) forms restored 1e-15.
4. **Raw-coordinate covariance is not the tangent covariance.** The
   charts are not orthonormal (g = diag(1, r²)); the tr(Cov) = variance
   identity (machine-checkable because |log_m(p)|_g = d(m, p)) forced
   the orthonormal-frame computation.
5. **Adam's normalized step explodes in flat regions** (|m̂/√v̂| ~ 7–11
   on the hyperbolic plane); a chart guard turns the failure into a
   clear error ("reduce lr or increase eps") instead of a crash.
6. **The optimizer's conditioning is the manifold's, not ours.**
   The polar y-direction is ill-conditioned by ~r²; convergence needs a
   matching budget — gradient descent's own behavior, documented.
7. **The pseudo-threshold of every repetition code under coherent
   X-noise is exactly π/2** — P_L(n, π/2) = 1/2 = P_phys for all n; the
   diagnostic report verified it rather than assuming a per-code value.

## What the verification discipline caught

Every "double path" (closed form vs generic; tableau vs dense; analytic
vs finite difference) is a bug trap.  Real bugs it caught:

| Bug | Where | How caught |
|---|---|---|
| θ⁴ assumed universal | qec | measurement (exponent fit) |
| Identity transport on polar plane | manifolds | isometry invariant |
| acos/acosh cancellation | geostats | near-coincident check |
| Coordinate-frame covariance | geostats | tr(Cov) = variance identity |
| Missing rotating-frame term in sphere Jacobian | derivatives | finite-difference comparison (err 0.22) |
| Velocity component swapped for position in `geodesic_ode` | manifolds | batch closed-form-vs-RK4 (err 6.28) |
| 1-bit phase cannot track Y's i | clifford | composition verification (err 0.54) |
| Input-convention clash in tableau conjugation | clifford | conjugation consistency |
| π/2 piece fold order (pre-existing, untested) | rotations | circuit optimize verification |
| S gate leaks e^{iπ/4} global phase | circuit | exact unitary equivalence |
| Naive (lat, lon) average wrong across ±180° | real data | Tonga events → 179.7°E vs −69.5° |
| Arithmetic mean wrong for circular (wind) data | real data | up to 183° off; geometric matches climate |

None of these were found by reading code — all by running the
verification against an independent path.

## Is it a tool?

Dynamic evidence, not a demo:
- **~700 fuzz cases** (fixed seeds): every invariant must hold for
  random Paulis/rotations (edge angles 0, 2π, ±4π), deep Clifford
  circuits, random manifold points, random point sets, random QEC.
- **Real end-to-end pipelines**: directional-sensor parameter recovery
  (`examples/real_use.py`) and 2024 USGS seismicity analysis
  (`examples/real_data.py`), both reproducing verifiable facts.
- The API is parameterized end to end; verification and shortcut
  dispatch are runtime, not test-time.

## Honest conclusions

- **The theory's actual role.** The geometric-derivation framework
  (geometrized closure, Riesz duality, exponential-map geometry, the
  machine-precision discipline) selected and justified the closed forms;
  its *foundation* is standard Clifford algebra and Riemannian geometry
  — public mathematics.  The theory's unique axioms (Bott periodicity,
  Cl(8), the three sectors, δ, the spectral action) played **no
  substantive role** in any delivered feature.  This is recorded
  deliberately: "the theory is the engine, not the claim" — the engine
  here being the standard-math core plus the verification discipline.
- **No speed advantage is attributed to the theory.** Every speedup is
  a closed-form/shortcut speedup, measured, and independent of the
  theory's unique content.  The honest claim is: standard math, verified
  to machine precision, with measured numbers.
- **"We derived X" is only true in the weak sense** — the foundations
  are sufficient for X; the derivation was standard mathematics.  Our
  public role is implementation plus machine-precision verification
  (adjudication of others' theories), not "first implementation"
  (e.g. Rustiq already implemented Clifford-tableau machinery).
- **Measurement corrected the theory once, visibly** (θ⁴ → θ^{d+1});
  that is the discipline working as intended, and it is the project's
  main transferable lesson.

## PyTorch reference

Structural mapping holds; functional mapping is partial:

| PyTorch | geocore | Functional parity |
|---|---|---|
| Tensor | GeometricObject (Pauli, Rotation, Clifford, Circuit, 3 manifolds) | objects specific, not generic |
| aten/c10 | Operator dispatch (19 ops) | mechanism, not library |
| autograd | analytic derivatives + verification | derivatives yes; arbitrary-f autodiff no |
| torch.compile | ShortcutRegistry (12 shortcuts) | hand-registered, no JIT |
| torch.optim | Riemannian SGD/Adam | closest parity |
| vmap | batch geodesics/sweeps | hand-written, not general |
| torch.mean/std/PCA | Frechet statistics | real |
| nn / data / GPU / distributed | — | absent (out of scope) |

The honest framing: geocore is a *verified geometric core* using
PyTorch's architecture as a reference, not a PyTorch reimplementation.

## Roadmap review

Done: all 17 items above.  Open candidates (hypotheses, not claims):

- Circuit object hardening (Clifford-circuit global-phase tracking,
  depth/2-qubit-gate counting, noise-model application).
- Analytic-gradient mechanism for arbitrary f (the autograd gap).
- New manifolds (S¹ circular statistics, matrix manifolds).
- Real-data studies beyond seismicity.
- Documentation hosting (GitHub Pages).
