# geocore — geometric computation core

An independent project whose core is built on **geometric structure**: Clifford
(Pauli) algebra as the foundation, geometric invariants as the organizing
principle, and **machine-precision verification** as a first-class feature.

## Goal

The long-term goal is computation that is *both* geometrically principled
*and* cheaper: closed-form and spectral shortcuts wherever they exist, with
every component verified to machine precision. The short-term goal is a
verified geometric core.

## Core principle: engine vs presentation

- The **derivation engine** (how we think, choose, and verify) is the geometry
  theory: closure/completeness reasoning, spectral discipline, exact-invariant
  verification.
- The **presentation** (what is written) is standard mathematics + machine-
  precision verification — every claim must be reproducible by anyone.
- A claim is only accepted if verified to machine precision; an unmeasured
  claim is not a claim.

## What is in the core (v0.1)

- `geocore.clifford`: Pauli representation (`2n`-bit symplectic encoding), the
  symplectic commutation form, and Clifford conjugation with tableau-style
  `r`-bit phase tracking — verified against explicit matrix truth for every
  gate × every Pauli (0 failures).
- `geocore.rotations`: geometric rotation objects `R_P(θ)` and the
  merge/cancel/optimize machinery (same-axis merge = closure of phase
  addition; cancellation at `2π` closure; Clifford pull-through by the dagger
  conjugation; fixed-point termination). Verified by unitary equivalence on
  random circuits.
- `geocore.verify`: the machine-precision verification harness itself
  (matrix-truth conjugation checks, unitary-equivalence checks).

## Verification status

| Check | Result |
|---|---|
| Conjugation primitives vs matrix truth (all gates × all Paulis, n ≤ 3) | 0 failures |
| Unitary equivalence on 1200 random circuits (n = 1..3) | all pass at 1e-9 |
| Known cases (merge, cancel, π/2 absorption, issue example 5→3) | pass |

## Roadmap (honest)

1. Verify the core further (batch/vectorized paths, more gates).
2. Add closed-form/spectral shortcuts with measured FLOPs/memory savings
   vs. naive baselines (the *reduce computation* goal, measured).
3. Domain applications (quantum error correction diagnostics, geometric
   statistics primitives) — each with verification + benchmark.

The theory is the engine, not the claim: what ships is standard math,
verified to machine precision, with measured performance numbers.
