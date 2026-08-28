"""Layer 3 — reduce computation (the analogue of torch.compile).

Closed-form / spectral shortcuts replace generic numerical paths.  Every
shortcut:

- declares the operator it replaces,
- is automatically verified against the generic path to machine precision
  (``verify_against``),
- reports a measured benchmark log (FLOPs estimate + wall time + speedup);
  no unmeasured claims.

The first shortcut is geometric in origin: a Pauli rotation obeys
R_P(theta) = cos(theta/2) I - i sin(theta/2) P because P^2 = I (the rotation
orbit of the Pauli axis closes in two steps).  This replaces the generic
dense matrix-exponential path (O(8^n) FLOPs for n qubits) with an O(2^n)
path — an exponential reduction, verified to machine precision.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np

from .invariants import VerificationReport

__all__ = ["BenchmarkLog", "Shortcut", "ShortcutRegistry", "registry", "rotation_closed_form", "geodesic_polar_closed_form", "geodesic_sphere_closed_form", "geodesic_hyperbolic_closed_form", "laplacian_circle_closed_form", "qec_scaling_prediction", "qec_theta4_prediction", "optim_step_closed_form"]


@dataclasses.dataclass
class BenchmarkLog:
    """The measured claim: FLOPs estimate + wall time + speedup.

    Fields prefixed ``time_`` are measured; ``flops_`` are documented
    analytic estimates (complexity class, not cycle counts).
    """

    name: str
    n: int | None = None  # system size (e.g. qubits)
    time_generic: float = 0.0
    time_shortcut: float = 0.0
    flops_generic: float = 0.0
    flops_shortcut: float = 0.0

    @property
    def speedup_time(self) -> float:
        return self.time_generic / max(self.time_shortcut, 1e-30)

    @property
    def speedup_flops(self) -> float:
        return self.flops_generic / max(self.flops_shortcut, 1e-30)

    def __repr__(self):
        return (
            f"BenchmarkLog({self.name}, n={self.n}, "
            f"time {self.time_generic:.2e}->{self.time_shortcut:.2e}s "
            f"({self.speedup_time:.1f}x), "
            f"flops {self.flops_generic:.1e}->{self.flops_shortcut:.1e} "
            f"({self.speedup_flops:.1e}x))"
        )


class Shortcut:
    """A fast path for an operator, with auto-verification and benchmarking.

    Parameters
    ----------
    name : str
        Shortcut name, e.g. ``rotation.closed_form``.
    replaces : Operator
        The operator this shortcut replaces (its generic implementation is
        the baseline).
    impl : callable
        The fast implementation, with the same signature as the operator.
    flops_generic / flops_shortcut : callable or float
        FLOPs estimate as a function of the system size (or a constant).
    """

    def __init__(
        self,
        name: str,
        replaces,
        impl,
        flops_generic=None,
        flops_shortcut=None,
    ):
        self.name = name
        self.replaces = replaces
        self.impl = impl
        self._flops_generic = flops_generic
        self._flops_shortcut = flops_shortcut
        self._generic = None

    def generic(self, *args):
        """Run the replaced operator's registered implementation (baseline)."""
        return self.replaces(*args)

    def verify_against(self, *args, atol=1e-9) -> VerificationReport:
        """Machine-precision check: shortcut result == generic result.

        Handles plain arrays and objects exposing ``point``/``velocity``
        (e.g. geodesic solutions).
        """
        generic_result = self.generic(*args)
        shortcut_result = self.impl(*args)
        if hasattr(generic_result, "point") and hasattr(shortcut_result, "point"):
            err = max(
                float(np.abs(np.asarray(generic_result.point) - np.asarray(shortcut_result.point)).max()),
                float(np.abs(np.asarray(generic_result.velocity) - np.asarray(shortcut_result.velocity)).max()),
            )
            return VerificationReport(
                ok=err < atol, max_error=err, details=f"point/velocity max error {err:.2e}"
            )
        a = np.asarray(generic_result)
        b = np.asarray(shortcut_result)
        if a.dtype.kind == "O" or b.dtype.kind == "O" or a.shape != b.shape:
            return VerificationReport(False, 1.0, f"incomparable results: {a!r} vs {b!r}")
        err = float(np.abs(a - b).max())
        return VerificationReport(
            ok=err < atol, max_error=err, details=f"max error {err:.2e}"
        )

    def profile(self, *args, n_trials=30, size_of=None) -> BenchmarkLog:
        """Measure wall time of generic vs shortcut; produce a BenchmarkLog."""
        size_of = size_of or (lambda *a: None)
        for _ in range(3):  # warmup
            self.generic(*args)
            self.impl(*args)
        t0 = time.perf_counter()
        for _ in range(n_trials):
            self.generic(*args)
        t_generic = (time.perf_counter() - t0) / n_trials
        t0 = time.perf_counter()
        for _ in range(n_trials):
            self.impl(*args)
        t_shortcut = (time.perf_counter() - t0) / n_trials
        n = size_of(*args)
        fg = self._flops_generic(n) if callable(self._flops_generic) else (self._flops_generic or 0.0)
        fs = self._flops_shortcut(n) if callable(self._flops_shortcut) else (self._flops_shortcut or 0.0)
        return BenchmarkLog(
            name=self.name,
            n=n,
            time_generic=t_generic,
            time_shortcut=t_shortcut,
            flops_generic=fg,
            flops_shortcut=fs,
        )


class ShortcutRegistry:
    """Registry of fast paths — the 'compile' entry point."""

    def __init__(self):
        self._shortcuts: dict[str, Shortcut] = {}
        self.logs: list[BenchmarkLog] = []

    def register(self, shortcut: Shortcut):
        self._shortcuts[shortcut.name] = shortcut
        return shortcut

    def get(self, name: str) -> Shortcut:
        return self._shortcuts[name]

    def apply(self, name: str, *args, verify=True, **kwargs):
        """Run a shortcut: verify against the generic path, then compute.

        Returns (result, VerificationReport).  ``kwargs`` are forwarded to
        ``verify_against`` (e.g. ``atol`` for convergence-aware checks).
        """
        shortcut = self._shortcuts[name]
        report = shortcut.verify_against(*args, **kwargs) if verify else None
        return shortcut.impl(*args), report

    def benchmark(self, name: str, *args, **kwargs) -> BenchmarkLog:
        log = self._shortcuts[name].profile(*args, **kwargs)
        self.logs.append(log)
        return log


registry = ShortcutRegistry()


# ---------------------------------------------------------------------------
# First shortcut: closed-form Pauli rotation (geometric: P^2 = I).
# ---------------------------------------------------------------------------

from .clifford import rotation_action_closed_form  # noqa: E402
from .ops import rotation_apply_to_state  # noqa: E402


def _size_of(rotation, state):
    return len(rotation.axis)


rotation_closed_form = registry.register(
    Shortcut(
        name="rotation.closed_form",
        replaces=rotation_apply_to_state,
        impl=lambda rotation, state: rotation_action_closed_form(rotation.axis, rotation.theta, state),
        flops_generic=lambda n: (2**n) ** 3,          # expm: O(d^3), d = 2^n
        flops_shortcut=lambda n: 2**n,                # Pauli action: O(d)
    )
)


# ---------------------------------------------------------------------------
# Second shortcut: closed-form geodesics on the polar plane (vs RK4 ODE).
# ---------------------------------------------------------------------------

from .manifolds import PolarPlane  # noqa: E402
from .ops import geodesic_polar_point  # noqa: E402


def _geodesic_polar_closed_form(manifold, initial, velocity, t):
    return manifold.geodesic_closed_form(initial, velocity, float(t))


def _geodesic_size(*args):
    return 2  # 2-dimensional manifold


geodesic_polar_closed_form = registry.register(
    Shortcut(
        name="geodesic.polar_closed_form",
        replaces=geodesic_polar_point,
        impl=_geodesic_polar_closed_form,
        flops_generic=lambda n: 200 * 6,   # RK4: n_steps x ~6 ODE evals
        flops_shortcut=lambda n: 20,        # closed form: a handful of trig ops
    )
)


# ---------------------------------------------------------------------------
# Third shortcut: closed-form Laplacian spectrum (vs discrete eigensolve).
# ---------------------------------------------------------------------------

from .ops import laplacian_eigenvalues  # noqa: E402
from .spectral import Circle  # noqa: E402


def _laplacian_circle_closed(manifold, n_evals, n_grid):
    return manifold.laplacian_eigenvalues_closed(n_evals)


def _spectral_size(manifold, n_evals, n_grid):
    return n_grid


laplacian_circle_closed_form = registry.register(
    Shortcut(
        name="laplacian.circle_closed_form",
        replaces=laplacian_eigenvalues,
        impl=_laplacian_circle_closed,
        flops_generic=lambda N: N**3,     # eigvalsh of N x N: O(N^3)
        flops_shortcut=lambda N: 2 * N,    # closed form: a handful of squares
    )
)


# ---------------------------------------------------------------------------
# Fourth shortcut: theta^4 logical-error prediction (vs O(2^n) simulation).
# ---------------------------------------------------------------------------

from .ops import qec_logical_error  # noqa: E402


def _theta4_predict(theta, n):
    return (3.0 / 16.0) * float(theta) ** 4


def _scaling_predict(theta, n):
    from .qec import scaling_leading

    return scaling_leading(float(theta), int(n))


def _qec_size(theta, n):
    return n


qec_scaling_prediction = registry.register(
    Shortcut(
        name="qec.scaling_prediction",
        replaces=qec_logical_error,
        impl=_scaling_predict,
        flops_generic=lambda n: 2**n,    # state-vector simulation: O(2^n)
        flops_shortcut=lambda n: 10,      # leading-law prediction: O(1)
    )
)


qec_theta4_prediction = registry.register(
    Shortcut(
        name="qec.theta4_prediction",
        replaces=qec_logical_error,
        impl=_theta4_predict,
        flops_generic=lambda n: 2**n,    # state-vector simulation: O(2^n)
        flops_shortcut=lambda n: 10,      # closed form: a handful of ops
    )
)


# ---------------------------------------------------------------------------
# Fifth shortcut: closed-form exponential-map optimizer step (vs RK4 ODE).
# ---------------------------------------------------------------------------

from .ops import optim_step  # noqa: E402


def _optim_step_closed(manifold, point, descent_vector, lr):
    return manifold.geodesic_closed_form(
        point, lr * np.asarray(descent_vector, dtype=float), 1.0
    ).point


def _optim_size(manifold, point, descent_vector, lr):
    return 2  # 2-dimensional manifold


optim_step_closed_form = registry.register(
    Shortcut(
        name="optim.step_closed_form",
        replaces=optim_step,
        impl=_optim_step_closed,
        flops_generic=lambda n: 200 * 6,   # RK4: n_steps x ~6 ODE evals
        flops_shortcut=lambda n: 20,        # closed form: a handful of trig ops
    )
)


# ---------------------------------------------------------------------------
# Sixth shortcut: closed-form geodesics on the sphere and the hyperbolic
# plane (great circles / semicircles vs RK4 ODE integration).
# ---------------------------------------------------------------------------

from .ops import geodesic_polar_point  # noqa: E402
from .sphere import Sphere  # noqa: E402
from .hyperbolic import HyperbolicPlane  # noqa: E402


def _geodesic_sphere_closed(manifold, initial, velocity, t):
    return manifold.geodesic_closed_form(initial, velocity, float(t))


def _geodesic_hyperbolic_closed(manifold, initial, velocity, t):
    return manifold.geodesic_closed_form(initial, velocity, float(t))


geodesic_sphere_closed_form = registry.register(
    Shortcut(
        name="geodesic.sphere_closed_form",
        replaces=geodesic_polar_point,
        impl=_geodesic_sphere_closed,
        flops_generic=lambda n: 200 * 6,   # RK4: n_steps x ~6 ODE evals
        flops_shortcut=lambda n: 40,        # closed form: trig + inverse chart
    )
)


geodesic_hyperbolic_closed_form = registry.register(
    Shortcut(
        name="geodesic.hyperbolic_closed_form",
        replaces=geodesic_polar_point,
        impl=_geodesic_hyperbolic_closed,
        flops_generic=lambda n: 200 * 6,   # RK4: n_steps x ~6 ODE evals
        flops_shortcut=lambda n: 20,        # closed form: a handful of trig ops
    )
)
