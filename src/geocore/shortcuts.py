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

__all__ = ["BenchmarkLog", "Shortcut", "ShortcutRegistry", "registry", "rotation_closed_form"]


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
        """Machine-precision check: shortcut result == generic result."""
        generic_result = self.generic(*args)
        shortcut_result = self.impl(*args)
        a = np.asarray(generic_result)
        b = np.asarray(shortcut_result)
        if a.shape != b.shape:
            return VerificationReport(False, 1.0, "shape mismatch")
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

    def apply(self, name: str, *args, verify=True):
        """Run a shortcut: verify against the generic path, then compute.

        Returns (result, VerificationReport).
        """
        shortcut = self._shortcuts[name]
        report = shortcut.verify_against(*args) if verify else None
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
