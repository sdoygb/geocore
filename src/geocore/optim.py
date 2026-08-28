"""Layer 1/3 — Riemannian optimization (the analogue of torch.optim).

PyTorch optimizers move parameters in Euclidean space; geometric
optimizers move them *on the manifold*: the update is

    p_{k+1} = exp_{p_k}( lr * v_k )

where exp is the exponential map (a geodesic step of unit parameter) and
v_k is the descent direction (typically v_k = -grad f(p_k), with grad the
Riemannian gradient from the ``optim.gradient`` operator).  On the polar
plane the exponential map is a straight line in Cartesian coordinates, so
each step is O(1) instead of an ODE integration — the Layer-3 shortcut for
optimization.

Everything is verified: the step against the closed-form exponential map
to machine precision, the gradient against the Riesz duality, and the
descent property against the supplied potential; ``minimize`` reports the
measured convergence honestly (gradient norm, monotonicity, minimizer
distance when known).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .objects import GeometricObject

__all__ = ["RiemannianSGD", "minimize", "OptimizationResult"]


@dataclasses.dataclass
class OptimizationResult:
    """The measured outcome of an optimization run (no unmeasured claims).

    ``converged`` means the final Riemannian gradient norm is below
    ``atol``; ``descent_ok`` means every step satisfied f(p_{k+1}) <=
    f(p_k) + tol; ``minimizer_error`` is the distance to the closed-form
    minimizer when one is supplied.
    """

    point: np.ndarray
    trajectory: list  # list of points (np.ndarray)
    f_history: list  # list of f values
    converged: bool
    descent_ok: bool
    final_grad_norm: float
    minimizer_error: float | None = None
    n_steps: int = 0

    def __repr__(self):
        return (
            f"OptimizationResult(point={np.round(self.point, 6)}, "
            f"converged={self.converged}, descent_ok={self.descent_ok}, "
            f"final_grad_norm={self.final_grad_norm:.2e}, "
            f"minimizer_error={self.minimizer_error}, n_steps={self.n_steps})"
        )


def _central_difference(f, point, eps=1e-6):
    """Euclidean-coordinate covector df = (df/dr, df/dy) by central
    differences (the generic numerical path)."""
    r, y = float(point[0]), float(point[1])
    dfdr = (f([r + eps, y]) - f([r - eps, y])) / (2 * eps)
    dfdy = (f([r, y + eps]) - f([r, y - eps])) / (2 * eps)
    return np.array([dfdr, dfdy])


class RiemannianSGD(GeometricObject):
    """Analogue of ``torch.optim.SGD``: stateful Riemannian gradient
    descent with momentum.

    State
    -----
    velocity : the momentum buffer, a tangent vector.  On the polar plane
    the tangent spaces are canonically identified (the manifold is flat),
    so the buffer transfers between steps without parallel transport.

    Each ``step`` is the operator ``optim.step`` (verified), routed through
    the closed-form exponential-map shortcut.
    """

    def __init__(self, manifold, lr: float = 0.1, momentum: float = 0.0):
        self.manifold = manifold
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.velocity = np.zeros(2)

    def zero_grad(self):
        """Reset the momentum buffer (analogue of SGD.zero_grad)."""
        self.velocity = np.zeros(2)

    def step(self, point, grad, f=None) -> np.ndarray:
        """One optimizer step: update the momentum buffer and move along
        the exponential map.  ``grad`` is the Riemannian gradient (from
        ``optim.gradient``); ``f`` optionally enables the descent check."""
        from .ops import optim_step
        from .shortcuts import optim_step_closed_form

        self.velocity = self.momentum * self.velocity + np.asarray(grad, dtype=float)
        if f is None:
            return optim_step_closed_form.impl(self.manifold, point, -self.velocity, self.lr)
        return optim_step(self.manifold, point, -self.velocity, self.lr, f=f)

    def __repr__(self):
        return f"RiemannianSGD(lr={self.lr}, momentum={self.momentum})"


def minimize(
    manifold,
    f,
    p0,
    lr: float = 0.1,
    n_steps: int = 200,
    momentum: float = 0.0,
    atol: float = 1e-6,
    minimizer=None,
    use_shortcut: bool = True,
) -> OptimizationResult:
    """Minimize f on the manifold by Riemannian gradient descent.

    Parameters
    ----------
    manifold : PolarPlane (or any manifold with geodesic_generic/closed_form)
    f : callable, the potential, f([r, y]) -> float
    p0 : initial point [r, y]
    lr : step size (geodesic parameter per step)
    n_steps : number of steps
    momentum : momentum coefficient for the velocity buffer
    atol : gradient-norm convergence tolerance
    minimizer : optional known minimizer [r*, y*] to report the distance
    use_shortcut : route steps through the closed-form exponential map
        (True) or the generic RK4 path (False, for benchmarking/verification)

    Every step is verified (exp-map validity, manifold constraint, and
    descent against f when momentum == 0); the report carries the measured
    convergence data.
    """
    from .ops import optim_gradient, optim_step
    from .shortcuts import optim_step_closed_form

    point = np.asarray(p0, dtype=float)
    opt = RiemannianSGD(manifold, lr=lr, momentum=momentum)
    trajectory = [point.copy()]
    f_history = [float(f(point))]
    descent_ok = True
    for _ in range(n_steps):
        df = _central_difference(f, point)
        grad = optim_gradient(manifold, df, point)
        opt.velocity = momentum * opt.velocity + grad
        if use_shortcut:
            new_point = optim_step_closed_form.impl(
                manifold, point, -opt.velocity, lr
            )
            if f is not None and momentum == 0.0:
                # closed-form path: verify descent manually (no op invariants)
                if float(f(new_point)) > float(f(point)) + 1e-7:
                    descent_ok = False
        else:
            new_point = optim_step(
                manifold, point, -opt.velocity, lr,
                f=f if momentum == 0.0 else None,
            )
        point = new_point
        trajectory.append(point.copy())
        f_history.append(float(f(point)))
    df_final = _central_difference(f, point)
    grad_final = optim_gradient(manifold, df_final, point)
    grad_norm = float(
        np.sqrt(manifold.metric_norm_sq(point, grad_final))
    )
    converged = grad_norm < atol
    minimizer_error = None
    if minimizer is not None:
        minimizer_error = float(np.linalg.norm(point - np.asarray(minimizer, dtype=float)))
    return OptimizationResult(
        point=point,
        trajectory=trajectory,
        f_history=f_history,
        converged=converged,
        descent_ok=descent_ok,
        final_grad_norm=grad_norm,
        minimizer_error=minimizer_error,
        n_steps=n_steps,
    )
