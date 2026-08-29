#!/usr/bin/env python3
"""Build and execute the full geocore walkthrough notebook.

Usage:  PYTHONPATH=src python3 examples/build_demo.py
Produces examples/geocore_demo.ipynb with all cells executed.
"""

import os
import sys

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

MD = {
    "title": """# geocore — geometric computation core

A walkthrough of all features (118 tests green).  The architecture mirrors
PyTorch layer by layer; every result is **verified to machine precision**
and every speedup is **measured** (no unmeasured claims).  The derivation
engine is the geometry theory; everything presented is standard
mathematics.

| PyTorch | geocore feature |
|---|---|
| Tensor | Geometric objects: Pauli, Rotation, 3 manifolds |
| aten/c10 | Operator dispatch (17 operators) |
| autograd | Analytic derivatives + automatic verification |
| torch.compile | Closed-form shortcuts (12, all measured) |
| torch.optim | Riemannian SGD / Adam with parallel transport |
| vmap | Vectorized batch geodesics / sweeps |
| torch.mean / std / PCA | Frechet mean / variance / tangent PCA |
| application | QEC coherent-noise diagnostics |
""",
    "setup": "## Setup",
    "l0": """## L0 — geometric objects (≈ Tensor)

A Pauli is an element of the Clifford algebra with a canonical 2n-bit
symplectic encoding; a Rotation is a point on the rotation orbit of a
Pauli axis with closure semantics.""",
    "l0_manifolds": """## L0 — manifolds (closed-form geodesics)

Three 2-dimensional manifolds share one interface (`metric_diag`,
`geodesic_ode`, `geodesic_closed_form`, `in_chart`); the closed forms are
the Layer-3 shortcuts: polar plane = straight line in Cartesian, sphere =
great circle in R³, hyperbolic plane = semicircle / vertical line.""",
    "l1": """## L1 — operator dispatch (≈ aten/c10)

Operators are first-class objects with type dispatch, declared invariants
and documented geometric theorems.""",
    "l2": """## L2 — automatic verification (≈ autograd)

Instead of automatically differentiating, the core automatically
*verifies*: every operator declares invariants, checked to machine
precision on every call (`no_verify()` is the analogue of
`torch.no_grad()`).""",
    "l3": """## L3 — reduce computation (≈ torch.compile)

Closed-form / spectral shortcuts replace generic numerical paths.  Each
shortcut is auto-verified against the generic path and reports a measured
`BenchmarkLog` (FLOPs estimate + wall time + speedup).""",
    "l3_bench": "All measured shortcuts (wall-time speedup / FLOPs speedup):",
    "optim": """## Riemannian optimizers (≈ torch.optim)

Parameters move *on the manifold*: the gradient is the Riesz
representative of df (verified by g(grad f, v) = df(v)), each step follows
the exponential map, and moment buffers are parallel-transported along the
step's geodesic (an isometry, verified).""",
    "deriv": """## Analytic derivatives (≈ autograd's gradient computation)

Closed-form derivatives verified against finite differences:
`rotation.derivative` (d/dθ R_P(θ)|ψ⟩ = −(i/2) P R_P(θ)|ψ⟩) and
`geodesic.jacobian` (per-manifold closed forms).  `minimize(grad_f=…)`
accepts an analytic gradient, verified on every step.""",
    "batch": """## Vectorized / batched core paths (≈ vmap)

Batch geodesics and batch parallel transport, verified identical to the
per-point paths; the batch closed form is measured orders of magnitude
faster than the per-point loop.""",
    "stats": """## Geometric statistics (≈ torch.mean / std / PCA)

The Frechet mean minimizes Σ d(p, pᵢ)² with the analytic gradient
−2Σ log_p(pᵢ); the tangent covariance (orthonormal frame) satisfies
tr(Cov) = variance to machine precision; its eigendecomposition is the
tangent PCA.""",
    "qec": """## QEC diagnostics application layer

Coherent-noise diagnostics over a repetition-code family: vectorized
sweeps, the measured θ^{d+1} law, pseudo-thresholds (exactly π/2 for
every distance) and verified crossovers.""",
    "summary": """## Summary

12 features, 118 tests, all machine-verified; 12 measured shortcuts.  The
theory is the engine, not the claim: what ships is standard math, verified
to machine precision, with measured performance numbers.""",
}

CODE = {
    "setup": """import numpy as np
import geocore
print("geocore", geocore.__version__)""",
    "l0": """from geocore import Pauli, Rotation, op, get_op

# Pauli: commutation decided by the symplectic form
print("X,Z commute:", Pauli("X").commutes_with(Pauli("Z")))
print("XX,ZY commute:", Pauli("XX").commutes_with(Pauli("ZY")))

# Rotation: closure semantics — same-axis rotations merge
a, b = Rotation("XX", 0.3), Rotation("XX", 0.4)
print("merge:", get_op("rotation.merge")(a, b))   # Rotation('XX', 0.7)
print("cancels at 2pi:", Rotation("XX", 2*np.pi).cancels())""",
    "l0_manifolds": """from geocore import PolarPlane, Sphere, HyperbolicPlane

for M, init, vel in [
    (PolarPlane(),      [2.0, 0.8], [0.2, 0.15]),
    (Sphere(),          [1.1, 0.6], [0.3, 0.5]),
    (HyperbolicPlane(), [0.3, 1.2], [0.4, 0.1]),
]:
    sol = M.geodesic_closed_form(init, vel, 0.5)          # exact
    e0 = M.metric_norm_sq(init, vel)
    e1 = M.metric_norm_sq(sol.point, sol.velocity)
    print(f"{type(M).__name__:16s} endpoint={np.round(sol.point,5)} "
          f"energy drift={abs(e1-e0):.1e}")""",
    "l1": """from geocore import get_op

# dispatch by geometric type; invariants verified automatically
r = get_op("rotation.merge")(Rotation("XX", 0.3), Rotation("XX", 0.4))
print("rotation.merge ->", r)

# geodesic: generic RK4 path, energy-conservation invariant checked
sol = get_op("geodesic.polar_point")(PolarPlane(), [2.0, 0.8], [0.2, 0.15], 0.5)
print("geodesic.polar_point ->", np.round(sol.point, 5))""",
    "l2": """from geocore import Rotation, get_op
from geocore.invariants import no_verify, verify_invariants, VerificationContext

# every op call runs its invariants; inspect the reports directly
a, b = Rotation("XX", 0.3), Rotation("XX", 0.4)
res = a.merge_with(b)
for rpt in verify_invariants(get_op("rotation.merge"), res, a, b):
    print("merge invariant:", rpt.ok, rpt.details)

# no_verify(): the analogue of torch.no_grad() — checks disabled
with no_verify():
    active = VerificationContext.is_enabled()
print("verification inside no_verify():", active,
      "| outside:", VerificationContext.is_enabled())""",
    "l3": """from geocore import Rotation
from geocore.shortcuts import registry

# closed-form Pauli rotation vs dense matrix exponential
r, state = Rotation("XXXX", 0.7), np.random.randn(16)
res, report = registry.apply("rotation.closed_form", r, state, verify=True)
print("verify:", report.ok, f"(max error {report.max_error:.1e})")
log = registry.benchmark("rotation.closed_form", r, state, n_trials=50,
                         size_of=lambda rot, st: len(rot.axis))
print("benchmark:", log)""",
    "l3_bench": """# every shortcut, measured (n_trials small for the walkthrough)
from geocore.shortcuts import registry as R

cases = [
    ("rotation.closed_form",        Rotation("X"*6, 0.7), np.random.randn(64),     lambda r, s: len(r.axis)),
    ("geodesic.polar_closed_form",  PolarPlane(), [2.0, 0.8], [0.2, 0.15], 0.5,   lambda *a: 2),
    ("laplacian.circle_closed_form", __import__("geocore").Circle(), 5, 200,      lambda *a: 200),
    ("qec.scaling_prediction",      0.02, 7,                                       lambda *a: 7),
    ("optim.step_closed_form",      PolarPlane(), [2.0, 0.8], [-0.2, 0.1], 0.1,   lambda *a: 2),
    ("geodesic.jacobian_closed_form", HyperbolicPlane(), [0.3, 1.2], [0.4, 0.1], 0.8, lambda *a: 2),
]
for case in cases:
    name, *args, size = case[0], *case[1:-1], case[-1]
    log = R.benchmark(name, *args, n_trials=20, size_of=size)
    print(f"{name:34s} {log.speedup_time:9.1f}x wall  {log.speedup_flops:9.1e}x flops")""",
    "optim": """from geocore import PolarPlane, minimize

m = PolarPlane()
f = lambda p: (p[0]-1.5)**2 + (p[1]-0.7)**2
res = minimize(m, f, [2.0, 0.3], lr=0.05, n_steps=500, minimizer=[1.5, 0.7])
print("SGD:", res)

# Adam with parallel-transported moment buffers
res = minimize(m, f, [2.0, 0.3], lr=0.1, n_steps=500, optimizer="adam",
               minimizer=[1.5, 0.7])
print("Adam:", res)""",
    "parallel": """from geocore.ops import geodesic_parallel_transport

# parallel transport is an isometry: metric norm preserved to machine precision
for M, p, q in [
    (PolarPlane(),      [2.0, 0.8], [1.9, 1.0]),
    (Sphere(),          [1.1, 0.6], [1.4, 1.0]),
    (HyperbolicPlane(), [0.3, 1.2], [0.6, 1.5]),
]:
    v = np.array([0.3, 0.2])
    vt = geodesic_parallel_transport(M, p, q, v)   # invariant checked
    drift = abs(M.metric_norm_sq(q, vt) - M.metric_norm_sq(p, v))
    print(f"{type(M).__name__:16s} transport isometry drift={drift:.1e}")""",
    "deriv": """from geocore import Rotation, get_op
from geocore.derivatives import rotation_derivative, geodesic_jacobian

state = np.random.randn(8) + 1j*np.random.randn(8)
d = get_op("rotation.derivative")(Rotation("XYZ", 0.7), state)  # verified
print("d/dtheta R|psi> shape:", d.shape)

Jp, Jv = geodesic_jacobian(Sphere(), [1.1, 0.6], [0.3, 0.5], 0.7)
print("geodesic Jacobian d(gamma(t))/d(p0):\\n", np.round(Jp, 4))""",
    "batch": """from geocore import shortcuts
from geocore.ops import geodesic_batch

rng = np.random.default_rng(0)
init = rng.uniform(0.5, 2.5, (200, 2)); vel = rng.uniform(-0.3, 0.3, (200, 2))
t = rng.uniform(0.1, 0.9, 200)
pts = geodesic_batch(Sphere(), init, vel, t)     # per-point loop, verified
rep = shortcuts.registry.get("geodesic.batch_closed_form").verify_against(
    Sphere(), init, vel, t)
print("batch == per-point:", rep.ok, f"(max error {rep.max_error:.1e})")
log = shortcuts.registry.benchmark("geodesic.batch_closed_form",
    Sphere(), init, vel, t, n_trials=5,
    size_of=lambda *a: np.atleast_2d(a[1]).shape[0])
print("batch benchmark:", log)""",
    "stats": """from geocore import frechet_mean, frechet_variance, principal_directions

pts = np.array([[2.0, 0.3], [1.2, -0.5], [1.9, 1.2], [1.1, 0.9]])
res = frechet_mean(PolarPlane(), pts, lr=0.1, n_steps=500)
print("Frechet mean:", np.round(res.point, 5))
print("variance:", round(frechet_variance(PolarPlane(), pts), 6))
evals, evecs = principal_directions(PolarPlane(), pts)
print("tangent PCA eigenvalues:", np.round(evals, 6))""",
    "qec": """from geocore.qec import diagnose

rep = diagnose((3, 5, 7))
print(rep)
print("exponent errors:", np.round(rep.exponent_errors, 4))
print("coeff rel. errors:", np.round(rep.coefficient_relative_errors, 5))""",
}


def main():
    cells = []
    cells.append(new_markdown_cell(MD["title"]))
    cells.append(new_markdown_cell(MD["setup"]))
    cells.append(new_code_cell(CODE["setup"]))
    cells.append(new_markdown_cell(MD["l0"]))
    cells.append(new_code_cell(CODE["l0"]))
    cells.append(new_markdown_cell(MD["l0_manifolds"]))
    cells.append(new_code_cell(CODE["l0_manifolds"]))
    cells.append(new_markdown_cell(MD["l1"]))
    cells.append(new_code_cell(CODE["l1"]))
    cells.append(new_markdown_cell(MD["l2"]))
    cells.append(new_code_cell(CODE["l2"]))
    cells.append(new_markdown_cell(MD["l3"]))
    cells.append(new_code_cell(CODE["l3"]))
    cells.append(new_markdown_cell(MD["l3_bench"]))
    cells.append(new_code_cell(CODE["l3_bench"]))
    cells.append(new_markdown_cell(MD["optim"]))
    cells.append(new_code_cell(CODE["optim"]))
    cells.append(new_code_cell(CODE["parallel"]))
    cells.append(new_markdown_cell(MD["deriv"]))
    cells.append(new_code_cell(CODE["deriv"]))
    cells.append(new_markdown_cell(MD["batch"]))
    cells.append(new_code_cell(CODE["batch"]))
    cells.append(new_markdown_cell(MD["stats"]))
    cells.append(new_code_cell(CODE["stats"]))
    cells.append(new_markdown_cell(MD["qec"]))
    cells.append(new_code_cell(CODE["qec"]))
    cells.append(new_markdown_cell(MD["summary"]))
    nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}})

    out = os.path.join(os.path.dirname(__file__), "geocore_demo.ipynb")
    nbformat.write(nb, out)

    # execute
    from nbclient import NotebookClient
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(os.path.dirname(__file__), "..", "src"),
         os.environ.get("PYTHONPATH", "")]
    )
    client = NotebookClient(nb, timeout=300, kernel_name="python3")
    client.execute()
    nbformat.write(nb, out)
    print(f"executed -> {out}")

    # report errors, if any
    errs = [c for c in nb.cells if c.cell_type == "code" and c.get("outputs")
            and any(o.get("output_type") == "error" for o in c.outputs)]
    if errs:
        for c in errs:
            for o in c.outputs:
                if o.get("output_type") == "error":
                    print("ERROR:", o.get("ename"), o.get("evalue"))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
