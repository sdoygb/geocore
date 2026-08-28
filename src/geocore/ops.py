"""Layer 1 — operator dispatch (geometric operators).

Operators are first-class geometric mappings between geometric objects.
Each operator:

- is registered under a geometric operation family (pauli.*, rotation.*,
  circuit.*),
- dispatches by the geometric type of its arguments (Layer 0 objects),
- declares the geometric invariants it preserves (Layer 2 checks them),
- documents the geometric theorem it implements.

All theory is geometrized: e.g. ``rotation.merge`` is the closure of phase
addition on the orbit of a Pauli axis; ``circuit.optimize`` preserves the
total unitary to machine precision and terminates at the fixed point
(completeness / closure).
"""

from __future__ import annotations

import numpy as np

from .invariants import (
    CancellationClosure,
    ConjugationMatrixTruth,
    GeodesicEnergyConservation,
    LogicalErrorValidity,
    MergeClosure,
    RotationActionClosure,
    SpectralValidity,
    SymplecticForm,
    UnitaryEquivalence,
    VerificationContext,
    VerificationError,
)
from .manifolds import PolarPlane
from .spectral import Circle
from .objects import Pauli, Rotation

__all__ = ["Operator", "op", "get_op", "registry", "dispatch"]


class Operator:
    """A named geometric operation with type dispatch and invariants.

    Parameters
    ----------
    name : str
        Dotted geometric name, e.g. ``pauli.commutes``.
    invariants : list of Invariant
        Geometric invariants this operator preserves (Layer 2).
    theorem : str
        The geometric theorem/principle this operator implements.
    """

    def __init__(self, name: str, invariants=None, theorem: str = ""):
        self.name = name
        self.invariants = list(invariants or [])
        self.theorem = theorem
        self._implementations: dict[tuple, callable] = {}
        self._default: callable | None = None

    def register(self, *types):
        """Decorator: register an implementation for a geometric type signature."""

        def deco(fn):
            self._implementations[tuple(types)] = fn
            return fn

        return deco

    def register_default(self, fn):
        """Register the fallback implementation."""
        self._default = fn
        return fn

    def __call__(self, *args, **kwargs):
        key = tuple(type(a) for a in args)
        impl = self._implementations.get(key)
        if impl is None:
            impl = self._default
        if impl is None:
            raise NotImplementedError(
                f"{self.name}: no implementation for argument types {key}"
            )
        result = impl(*args, **kwargs)
        # Layer 2: automatic verification (strict by default)
        if VerificationContext.is_enabled() and self.invariants:
            for inv in self.invariants:
                report = inv.check(result, *args, **kwargs)
                if not report.ok:
                    raise VerificationError(
                        f"{self.name}: invariant '{inv.name}' failed: {report.details}"
                    )
        return result

    def __repr__(self):
        n = len(self._implementations)
        return f"Operator({self.name!r}, {n} type implementations)"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

registry: dict[str, Operator] = {}


def op(name: str, invariants=None, theorem: str = "") -> Operator:
    """Get an operator by name, creating and registering it if absent."""
    if name not in registry:
        registry[name] = Operator(name, invariants, theorem)
    return registry[name]


def get_op(name: str) -> Operator:
    return registry[name]


def dispatch(*types, name: str | None = None):
    """Convenience decorator: create/get an operator and register a type impl.

    Example::

        @dispatch(Rotation, Rotation, name="rotation.merge")
        def _(a, b): ...
    """

    def deco(fn):
        op_reg = op(name or fn.__name__)
        op_reg.register(*types)(fn)
        return fn

    return deco


# ---------------------------------------------------------------------------
# The v0.1 geometric operators
# ---------------------------------------------------------------------------

pauli_commutes = op(
    "pauli.commutes",
    invariants=[SymplecticForm()],
    theorem="Commutation is decided by the symplectic form omega(a,b) = 0.",
)


@pauli_commutes.register(Pauli, Pauli)
def _(a: Pauli, b: Pauli) -> bool:
    return a.commutes_with(b)


pauli_conjugate = op(
    "pauli.conjugate_by",
    invariants=[ConjugationMatrixTruth()],
    theorem=(
        "Conjugation by a Clifford is a symplectic transformation; the "
        "tableau r-bit tracks the +/- phase exactly (verified vs matrix truth)."
    ),
)


@pauli_conjugate.register(Pauli, tuple)
@pauli_conjugate.register(Pauli, list)
def _(a: Pauli, gates) -> tuple[Pauli, int]:
    return a.conjugate_by(tuple(gates))


rotation_merge = op(
    "rotation.merge",
    invariants=[MergeClosure()],
    theorem=(
        "Closure of phase addition on the orbit of a Pauli axis: "
        "R_P(t) R_P(s) = R_P(t+s) for same-axis rotations."
    ),
)


@rotation_merge.register(Rotation, Rotation)
def _(a: Rotation, b: Rotation):
    return a.merge_with(b)


rotation_cancel = op(
    "rotation.cancels",
    invariants=[CancellationClosure()],
    theorem="2-pi closure: R_P(theta) = 1 iff theta ≡ 0 (mod 2 pi).",
)


@rotation_cancel.register(Rotation)
def _(a: Rotation) -> bool:
    return a.cancels()


circuit_optimize = op(
    "circuit.optimize",
    invariants=[UnitaryEquivalence()],
    theorem=(
        "Fixed-point completeness (closure): iterate merge/cancel with "
        "Clifford pull-through (dagger conjugation) until no merge is "
        "possible; the output is unitarily equivalent to the input."
    ),
)


@circuit_optimize.register(list)
def _(rotations):
    from .rotations import optimize_pauli_rotations

    return optimize_pauli_rotations(rotations)


rotation_apply_to_state = op(
    "rotation.apply_to_state",
    invariants=[RotationActionClosure()],
    theorem=(
        "R_P(theta)|psi> = cos(theta/2)|psi> - i sin(theta/2) P|psi> "
        "(P^2 = I: the rotation orbit of the Pauli axis closes in two "
        "steps).  The registered implementation is the generic dense "
        "matrix-exponential path; the closed form is a Layer-3 shortcut."
    ),
)


@rotation_apply_to_state.register(Rotation, np.ndarray)
def _(rotation, state):
    from scipy.linalg import expm

    from .verify import _pauli_matrix

    P = _pauli_matrix(rotation.axis)
    U = expm(-1j * rotation.theta / 2 * P)
    return U @ state


geodesic_polar_point = op(
    "geodesic.polar_point",
    invariants=[GeodesicEnergyConservation()],
    theorem=(
        "The geodesic of ds^2 = dr^2 + r^2 dy^2 satisfies the second-order "
        "ODE from Gamma^r_yy = -r, Gamma^y_ry = 1/r; the metric norm of the "
        "velocity is conserved (a Levi-Civita invariant).  The registered "
        "implementation is the generic RK4 integration; the closed form "
        "(straight line in Cartesian coordinates) is a Layer-3 shortcut."
    ),
)


@geodesic_polar_point.register_default
def _(manifold, initial, velocity, t, **kwargs):
    if not hasattr(manifold, "geodesic_generic"):
        raise NotImplementedError(
            f"geodesic.polar_point: {type(manifold).__name__} has no geodesic_generic"
        )
    return manifold.geodesic_generic(initial, velocity, float(t))


laplacian_eigenvalues = op(
    "laplacian.eigenvalues",
    invariants=[SpectralValidity()],
    theorem=(
        "The Laplace-Beltrami spectrum is a geometric invariant of the "
        "manifold; on S^1 the eigenvalues are k^2 (multiplicity 2).  The "
        "generic path diagonalizes the discrete Laplacian (cycle graph, "
        "scaled by 1/h^2), which converges to the exact spectrum as O(n^-2)."
    ),
)


@laplacian_eigenvalues.register(Circle, int, int)
def _(manifold, n_evals, n_grid):
    return manifold.laplacian_discrete_eigenvalues(n_grid, n_evals)


qec_logical_error = op(
    "qec.logical_error",
    invariants=[LogicalErrorValidity()],
    theorem=(
        "Coherent X-rotation noise on the repetition code: P_L(theta) = "
        "3 sin^4(theta/2) cos^2(theta/2) + sin^6(theta/2) for the 3-qubit "
        "code, leading term (3/16) theta^4 (measured exponent ~4).  The "
        "generic path is the O(2^n) state-vector simulation."
    ),
)


@qec_logical_error.register(float, int)
def _(theta, n):
    from .qec import repetition_code_logical_error

    return repetition_code_logical_error(float(theta), n)


# ---------------------------------------------------------------------------
# v0.3 geometric operators: Riemannian optimization (the analogue of
# torch.optim).  The gradient is the Riesz representative of df w.r.t. the
# metric; the step moves along the exponential map (a geodesic), which on
# the polar plane has a closed form (a straight line in Cartesian
# coordinates) — the Layer-3 shortcut for optimization.
# ---------------------------------------------------------------------------

from .invariants import (  # noqa: E402
    DescentProperty,
    ExponentialMapValidity,
    ManifoldConstraint,
    ParallelTransportIsometry,
    RieszGradientValidity,
)

optim_gradient = op(
    "optim.gradient",
    invariants=[RieszGradientValidity()],
    theorem=(
        "Riesz representation: the Riemannian gradient is grad f = g^{-1} df, "
        "i.e. g(grad f, v) = df(v) for all tangent vectors v.  On the polar "
        "plane g = diag(1, r^2), so grad f = (df_r, df_y / r^2)."
    ),
)


@optim_gradient.register_default
def _(manifold, df, point, **kwargs):
    df = np.asarray(df, dtype=float)
    g0, g1 = manifold.metric_diag(np.asarray(point, dtype=float))
    return np.array([df[0] / g0, df[1] / g1])


optim_step = op(
    "optim.step",
    invariants=[ExponentialMapValidity(), ManifoldConstraint(), DescentProperty()],
    theorem=(
        "Riemannian gradient step: p' = exp_p(lr * v) along the exponential "
        "map (a geodesic of unit parameter).  On the polar plane the "
        "exponential map is a straight line in Cartesian coordinates, so the "
        "step has a closed form; the registered implementation is the generic "
        "RK4 geodesic integration (the Layer-3 shortcut replaces it)."
    ),
)


@optim_step.register_default
def _(manifold, point, descent_vector, lr, **kwargs):
    return manifold.geodesic_generic(
        point, lr * np.asarray(descent_vector, dtype=float), 1.0
    ).point


geodesic_parallel_transport = op(
    "geodesic.parallel_transport",
    invariants=[ParallelTransportIsometry()],
    theorem=(
        "Parallel transport of a tangent vector along the connecting "
        "geodesic is an isometry: g_{to}(V', V') = g_{from}(V, V).  On the "
        "sphere it is a rotation about the great-circle axis; on the "
        "hyperbolic plane a rotation plus scaling; on the flat polar plane "
        "the identity.  This is what carries optimizer moment buffers "
        "between tangent spaces (Riemannian Adam)."
    ),
)


@geodesic_parallel_transport.register_default
def _(manifold, point_from, point_to, vector, **kwargs):
    return manifold.parallel_transport(point_from, point_to, vector)
