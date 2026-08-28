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
        return impl(*args, **kwargs)

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
    theorem="Commutation is decided by the symplectic form omega(a,b) = 0.",
)


@pauli_commutes.register(Pauli, Pauli)
def _(a: Pauli, b: Pauli) -> bool:
    return a.commutes_with(b)


pauli_conjugate = op(
    "pauli.conjugate_by",
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
    theorem="2-pi closure: R_P(theta) = 1 iff theta ≡ 0 (mod 2 pi).",
)


@rotation_cancel.register(Rotation)
def _(a: Rotation) -> bool:
    return a.cancels()


circuit_optimize = op(
    "circuit.optimize",
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
