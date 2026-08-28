# geocore — Architecture (PyTorch-referenced)

A geometric computation core, architected with PyTorch as the structural
reference: a geometric-object engine (the "tensor"), an operator-dispatch
layer ("aten/c10"), an automatic-verification layer (our counterpart of
"autograd"), and a reduce-computation layer (our counterpart of
"torch.compile"). Every layer ships with machine-precision verification; the
derivation engine is the geometry theory, the presentation is standard
mathematics.

```
+---------------------------------------------------------------+
| Application layers (future)                                    |
|   QEC diagnostics, geometric statistics, physics models        |
+---------------------------------------------------------------+
| Layer 3  Reduce-computation (≈ torch.compile)                  |
|   ShortcutRegistry: closed-form/spectral fast paths,           |
|   auto-verified vs generic path, measured speedup log          |
+---------------------------------------------------------------+
| Layer 2  Automatic verification (≈ autograd)                   |
|   Invariant, VerificationContext, verify_invariants()          |
|   every op declares invariants; core self-checks to machine    |
|   precision (no_verify() to disable, like no_grad())           |
+---------------------------------------------------------------+
| Layer 1  Operator dispatch (≈ aten/c10)                        |
|   Operator, @dispatch, op registry; ops carry invariants,      |
|   optional fast path, optional generic path                    |
+---------------------------------------------------------------+
| Layer 0  Geometric object engine (≈ tensor)                    |
|   GeometricObject: Pauli, Rotation (later: ManifoldPoint,      |
|   TangentVector, Metric) with unified encoding + backend       |
+---------------------------------------------------------------+
```

## Design principles

1. **PyTorch-referenced structure**: the layer boundaries mirror PyTorch's
   (engine / dispatch / automatic layer / compile / applications), so the
   architecture is familiar and each layer has a clear geometric analogue.
2. **Verification is a core feature, not a test**: like autograd is built
   into PyTorch's core, invariant verification is built into ours. Every
   operator declares what it preserves; the core can self-check to machine
   precision.
3. **Measured claims only**: any "this is faster" claim must come with a
   benchmark log (FLOPs / time / memory vs. the generic path). No unmeasured
   claims.
4. **Engine vs presentation**: the geometry theory is the derivation engine;
   everything written is standard mathematics, verified to machine precision.

---

## Layer 0 — Geometric object engine (≈ `torch.Tensor`, `aten`/`c10`)

The "tensor" of this library is a geometric object with a unified algebraic
encoding. v0.1 has two objects; the interface is designed so manifold
objects can join without breaking dispatch.

```python
# geocore/objects.py (interface)

class GeometricObject:
    """Base class. Analogue of torch.Tensor: the unit of computation.

    Subclasses must define:
      - a canonical encoding (e.g. symplectic bits for Paulis),
      - `.verify()` returning a VerificationReport,
      - the invariants they participate in (via ops).
    """

    @property
    def dim(self) -> int:
        """Number of qubits / manifold dimension (analogue of Tensor.ndim)."""

    def verify(self) -> "VerificationReport":
        """Machine-precision self-check of all declared invariants."""


class Pauli(GeometricObject):
    """P = i^m prod_k X_k^{x_k} Z_k^{z_k}; canonical 2n-bit symplectic encoding."""

    def __init__(self, axis: str): ...
    def commutes_with(self, other: "Pauli") -> bool: ...
    def conjugate_by(self, gates) -> "tuple[Pauli, int]":
        """Return (conjugated Pauli, phase r) — the r-bit tracked conjugation."""
    def to_matrix(self) -> np.ndarray: ...


class Rotation(GeometricObject):
    """R_P(theta) = exp(-i theta P / 2); the geometric rotation object."""

    def __init__(self, axis: str, theta: float): ...
    def merge_with(self, other: "Rotation") -> "Rotation | None":
        """Same-axis merge (closure of phase addition) or None."""
    def cancels(self) -> bool:
        """True iff theta ≡ 0 (mod 2 pi) — the 2-pi closure."""
    def to_matrix(self) -> np.ndarray: ...
```

## Layer 1 — Operator dispatch (≈ `aten`/`c10`)

Operators are first-class objects with a name, input/output types, declared
invariants, and (optionally) fast and generic implementations. Dispatch
routes by input types (Python-level multi-method dispatch).

```python
# geocore/ops.py (interface)

class Operator:
    """Analogue of an aten operator: name, types, invariants, impls."""

    def __init__(self, name: str, invariants: list["Invariant"]): ...
    def __call__(self, *args, **kwargs): ...


def dispatch(*types):
    """Decorator registering an implementation for a type signature.

    geocore.op("rotation.merge")
    @dispatch(Rotation, Rotation)
    def _(a, b): ...
    """
```

Key operators in v0.1:

| Operator | Inputs | Invariant preserved |
|---|---|---|
| `pauli.commutes` | (Pauli, Pauli) | symplectic form ω(a,b) = 0 |
| `pauli.conjugate_by` | (Pauli, gates) | conjugation = symplectic action + r-bit phase |
| `rotation.merge` | (Rotation, Rotation) | R_P(t)R_P(s) = R_P(t+s), unitary |
| `rotation.cancel` | (Rotation,) | θ ≡ 0 (2π) ⇒ identity |
| `circuit.optimize` | (rotations,) | U(in) = C1 C2 … U(out) (unitary equivalence) |
| `geodesic.polar_point` | (PolarPlane, p, v, t) | metric norm of velocity conserved |
| `laplacian.eigenvalues` | (Circle, k, N) | valid ascending spectrum, PSD |
| `qec.logical_error` | (θ, n) | closed form P_L to machine precision |
| `optim.gradient` | (PolarPlane, df, p) | Riesz duality g(grad f, v) = df(v) |
| `optim.step` | (PolarPlane, p, v, lr) | exp-map validity, r > 0, descent |

## Layer 2 — Automatic verification (≈ `autograd`)

The analogue of autograd: instead of automatically differentiating, the core
automatically *verifies*. Operators declare invariants; the verification
layer checks them to machine precision; `no_verify()` disables it (the
analogue of `torch.no_grad()`).

```python
# geocore/verify_core.py (interface)

class Invariant:
    """A machine-precision check attached to an operator."""

    def check(self, *args, **kwargs) -> "VerificationReport": ...


class VerificationReport:
    ok: bool
    max_error: float
    details: str


def verify_invariants(op: Operator, *args, **kwargs) -> VerificationReport:
    """Run all invariants of op on the given call; report machine precision."""


@contextlib.contextmanager
def no_verify():
    """Disable automatic verification (analogue of torch.no_grad())."""
```

Built-in invariants for v0.1:

| Invariant | Checks |
|---|---|
| `MatrixTruth` | op result == explicit matrix computation (axis AND phase) |
| `UnitaryEquivalence` | U(in) == U(clifford) @ U(out) to 1e-9 |
| `SymplecticForm` | commutation decisions match ω(a,b) |

## Layer 3 — Reduce computation (≈ `torch.compile`)

The goal of the project: replace generic numerical paths with closed-form /
spectral shortcuts, *verified* and *measured*. A shortcut declares what it
replaces, is automatically checked against the generic path to machine
precision, and reports its speedup.

```python
# geocore/shortcuts.py (interface)

class Shortcut:
    """A fast path for an operator, with auto-verification + benchmark."""

    def __init__(self, name: str, replaces: Operator, impl): ...
    def verify_against(self, generic) -> VerificationReport:
        """Machine-precision check: fast path == generic path."""
    def profile(self, *args, n_trials=100) -> "BenchmarkLog": ...


class ShortcutRegistry:
    """Registry of fast paths; the 'compile' entry point."""

    def register(self, shortcut: Shortcut): ...
    def apply(self, op: Operator, *args):
        """Route to the registered shortcut, verify, and log the speedup."""


class BenchmarkLog:
    """The measured claim: flops, wall time, peak memory, speedup vs generic."""
    flops_generic: float
    flops_shortcut: float
    time_generic: float
    time_shortcut: float
    speedup: float
```

First candidate shortcuts (to be measured, not assumed):

| Shortcut | Generic path | Expected benefit (to measure) |
|---|---|---|
| Closed-form Pauli-rotation optimization | naive merge simulation | fewer rotations → fewer gates |
| Closed-form geodesics (warped-product / sphere-like) | numeric ODE integration | orders of magnitude (no ODE) |
| Spectral shortcuts (Laplacian eigenvalues, θ^{d+1} scaling) | full simulation / Monte Carlo | orders of magnitude (prediction vs simulation) |
| Closed-form exponential-map optimizer step | RK4 geodesic integration per step | orders of magnitude (no ODE per step) |

## v0.1 → target mapping

| v0.1 module | Target layer |
|---|---|
| `clifford.py` | Layer 0 (Pauli encoding) + Layer 1 (`pauli.*` ops) |
| `rotations.py` | Layer 0 (Rotation) + Layer 1 (`rotation.*`, `circuit.optimize`) |
| `manifolds.py` / `spectral.py` | Layer 0 (manifold objects) + Layer 1 (`geodesic.*`, `laplacian.*`) |
| `qec.py` | Application layer (QEC diagnostics) + Layer 1 (`qec.logical_error`) |
| `optim.py` | Layer 1 (`optim.gradient`, `optim.step`) + application API (≈ `torch.optim`) |
| `verify.py` | Layer 2 (first Invariant implementations) |
| `shortcuts.py` | Layer 3 (ShortcutRegistry + BenchmarkLog) |

## Honest notes

- The architecture is standard engineering (object model, dispatch,
  registry, benchmark harness) — nothing here depends on the theory's
  unique axioms.
- The theory's role: the *derivation engine* that selects which closed-form /
  spectral shortcuts to pursue, and the *verification discipline* that
  every layer inherits. The first measured speedup will be the first
  evidence of whether the engine produces computation savings — until then,
  speedups are hypotheses, not claims.
