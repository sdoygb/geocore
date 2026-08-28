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
│   ├── rotations.py         # rotation-chain optimization (verified)
│   └── verify.py            # machine-precision verification harness
└── tests/                   # 24 tests
```

## Roadmap (honest)

1. Second measured shortcut (closed-form geodesics vs ODE integration).
2. Vectorized/batched core paths.
3. Application layers (quantum error correction diagnostics, geometric
   statistics primitives) — each with verification + benchmark.
4. Spectral shortcuts (Laplacian eigenvalues, θ⁴ scaling) with measured
   savings vs. simulation baselines.

The theory is the engine, not the claim: what ships is standard math,
verified to machine precision, with measured performance numbers.
