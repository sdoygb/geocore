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

__all__ = ["RiemannianSGD", "RiemannianAdam", "minimize", "minimize_batch", "OptimizationResult", "BatchOptimizationResult"]


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
    max_grad_error: float | None = None  # analytic vs FD gradient, when grad_f given

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

    def step(self, point, grad, f=None, use_shortcut=True) -> np.ndarray:
        """One optimizer step: update the momentum buffer and move along
        the exponential map.  ``grad`` is the Riemannian gradient (from
        ``optim.gradient``); ``f`` optionally enables the descent check.
        The momentum buffer is parallel-transported along the step's
        geodesic (an isometry, verified), so it stays a genuine tangent
        vector at the new point."""
        from .ops import geodesic_parallel_transport, optim_step
        from .shortcuts import optim_step_closed_form

        self.velocity = self.momentum * self.velocity + np.asarray(grad, dtype=float)
        if use_shortcut:
            new_point = optim_step_closed_form.impl(
                self.manifold, point, -self.velocity, self.lr
            )
        else:
            new_point = optim_step(self.manifold, point, -self.velocity, self.lr, f=f)
        if self.momentum != 0.0:
            self.velocity = geodesic_parallel_transport(
                self.manifold, point, new_point, self.velocity
            )
        return new_point

    def __repr__(self):
        return f"RiemannianSGD(lr={self.lr}, momentum={self.momentum})"


class RiemannianAdam(GeometricObject):
    """Analogue of ``torch.optim.Adam``: adaptive moment estimation on the
    manifold.

    The first and second moment buffers live in the tangent space:

        m <- beta1 m + (1 - beta1) g
        v <- beta2 v + (1 - beta2) g^2          (g^2 elementwise)
        p' = exp_p( -lr * m_hat / (sqrt(v_hat) + eps) )

    with the standard bias correction m_hat = m/(1-beta1^t),
    v_hat = v/(1-beta2^t).  After each step both buffers are
    parallel-transported along the step's geodesic, so they remain tangent
    vectors at the new point (this is the geometric part; on a flat
    manifold it is the identity).

    Positivity: the second moment v is a diagonal (0,2)-tensor in
    coordinates; transporting its components directly can rotate them
    negative.  We therefore transport the step-scale vector s = sqrt(v)
    (a genuine tangent vector) and set v = s^2 afterwards — v stays
    positive semidefinite by construction.
    """

    def __init__(self, manifold, lr: float = 0.1, betas=(0.9, 0.999), eps: float = 1e-8):
        self.manifold = manifold
        self.lr = float(lr)
        self.betas = tuple(betas)
        self.eps = float(eps)
        self.m = np.zeros(2)
        self.v = np.zeros(2)
        self.t = 0

    def zero_grad(self):
        """Reset the moment buffers and the step counter."""
        self.m = np.zeros(2)
        self.v = np.zeros(2)
        self.t = 0

    def step(self, point, grad, f=None, use_shortcut=True) -> np.ndarray:
        """One Adam step: update the moment buffers, apply the adaptive
        direction along the exponential map, and parallel-transport the
        buffers (v via its square root, preserving positivity).  ``f``
        optionally enables the descent check (Adam may overshoot, so the
        check is reported, not enforced).

        Adam's normalized step can be large in flat regions (the second
        moment decays while the first does not); if the step leaves the
        manifold chart, a clear error is raised — reduce ``lr`` or increase
        ``eps``."""
        from .ops import geodesic_parallel_transport, optim_step
        from .shortcuts import optim_step_closed_form
        from .invariants import VerificationError

        self.t += 1
        b1, b2 = self.betas
        g = np.asarray(grad, dtype=float)
        self.m = b1 * self.m + (1.0 - b1) * g
        self.v = b2 * self.v + (1.0 - b2) * g * g
        m_hat = self.m / (1.0 - b1**self.t)
        v_hat = self.v / (1.0 - b2**self.t)
        direction = m_hat / (np.sqrt(v_hat) + self.eps)
        if use_shortcut:
            new_point = optim_step_closed_form.impl(
                self.manifold, point, -direction, self.lr
            )
        else:
            new_point = optim_step(self.manifold, point, -direction, self.lr, f=f)
        if not self.manifold.in_chart(np.asarray(new_point)):
            raise VerificationError(
                f"RiemannianAdam step left the manifold chart "
                f"({np.round(new_point, 4)}); reduce lr or increase eps"
            )
        self.m = geodesic_parallel_transport(self.manifold, point, new_point, self.m)
        s = geodesic_parallel_transport(
            self.manifold, point, new_point, np.sqrt(self.v)
        )
        self.v = s * s
        return new_point

    def __repr__(self):
        b1, b2 = self.betas
        return f"RiemannianAdam(lr={self.lr}, betas=({b1}, {b2}))"


@dataclasses.dataclass
class BatchOptimizationResult:
    """The measured outcome of a batched optimization run: one trajectory
    per starting point, vectorized."""

    points: np.ndarray  # (B, 2)
    f_final: np.ndarray  # (B,)
    converged: np.ndarray  # (B,)
    minimizer_error: np.ndarray | None = None  # (B,) when minimizer given
    n_steps: int = 0
    descent_ok: bool = True

    def __repr__(self):
        conv = int(self.converged.sum()) if self.converged.ndim else int(self.converged)
        return (
            f"BatchOptimizationResult(B={len(self.points)}, "
            f"converged={conv}/{len(self.points)}, "
            f"descent_ok={self.descent_ok}, n_steps={self.n_steps})"
        )


def _central_difference_vec(f, points, eps=1e-6):
    """Vectorized central-difference covector df over a batch: f must
    accept (B, 2) and return (B,)."""
    p = np.atleast_2d(np.asarray(points, dtype=float))
    d0 = (f(p + np.array([eps, 0.0])) - f(p - np.array([eps, 0.0]))) / (2 * eps)
    d1 = (f(p + np.array([0.0, eps])) - f(p - np.array([0.0, eps]))) / (2 * eps)
    return np.column_stack([np.asarray(d0, dtype=float), np.asarray(d1, dtype=float)])


def minimize_batch(
    manifold,
    f,
    p0,
    lr: float = 0.1,
    n_steps: int = 200,
    atol: float = 1e-6,
    minimizer=None,
) -> BatchOptimizationResult:
    """Vectorized Riemannian gradient descent over a batch of starting
    points (the analogue of a batched optimizer step in torch).

    ``f`` must accept (B, 2) and return (B,); ``p0`` is (B, 2).  Each
    point follows its own gradient flow; the exponential-map steps and the
    gradient (Riesz representative) are computed with vectorized numpy ops
    (the batch core path).  The result is verified to agree with the
    per-point loop in tests; a step leaving the chart raises a clear error.
    """
    from .invariants import VerificationError

    points = np.atleast_2d(np.asarray(p0, dtype=float))
    B = points.shape[0]
    descent_ok = True
    for _ in range(n_steps):
        df = _central_difference_vec(f, points)
        g0, g1 = manifold.metric_diag(points)
        grad = np.column_stack([df[:, 0] / np.asarray(g0), df[:, 1] / np.asarray(g1)])
        new_points, _ = manifold.geodesic_closed_form_batch(points, -lr * grad, 1.0)
        if not bool(np.all(manifold.in_chart(new_points))):
            raise VerificationError(
                f"minimize_batch step left the manifold chart "
                f"({np.round(new_points, 4)}); reduce lr"
            )
        if bool(np.any(np.asarray(f(new_points)) > np.asarray(f(points)) + 1e-7)):
            descent_ok = False
        points = new_points
    df_final = _central_difference_vec(f, points)
    g0, g1 = manifold.metric_diag(points)
    grad_final = np.column_stack([df_final[:, 0] / np.asarray(g0), df_final[:, 1] / np.asarray(g1)])
    g_norm = manifold.metric_norm_sq(points, grad_final)
    converged = np.sqrt(np.asarray(g_norm)) < atol
    minimizer_error = None
    if minimizer is not None:
        minimizer_error = np.linalg.norm(
            points - np.broadcast_to(np.asarray(minimizer, dtype=float), points.shape),
            axis=1,
        )
    return BatchOptimizationResult(
        points=points,
        f_final=np.asarray(f(points), dtype=float),
        converged=converged,
        minimizer_error=minimizer_error,
        n_steps=n_steps,
        descent_ok=descent_ok,
    )


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
    optimizer: str = "sgd",
    betas=(0.9, 0.999),
    eps: float = 1e-8,
    grad_f=None,
    verify_grad: bool = True,
) -> OptimizationResult:
    """Minimize f on the manifold by Riemannian gradient descent.

    Parameters
    ----------
    manifold : a RiemannianManifold (polar plane, sphere, hyperbolic plane)
    f : callable, the potential, f([u, v]) -> float
    p0 : initial point [u, v]
    lr : step size (geodesic parameter per step)
    n_steps : number of steps
    momentum : SGD momentum coefficient (ignored for ``optimizer="adam"``)
    atol : gradient-norm convergence tolerance
    minimizer : optional known minimizer [u*, v*] to report the distance
    use_shortcut : route steps through the closed-form exponential map
        (True) or the generic RK4 path (False, for benchmarking/verification)
    optimizer : "sgd" (RiemannianSGD) or "adam" (RiemannianAdam)
    betas, eps : Adam hyperparameters
    grad_f : optional analytic Riemannian gradient, grad_f(p) -> tangent
        vector (the analogue of a user-supplied autograd gradient).  When
        given, it replaces the finite-difference gradient and is *verified*
        against finite differences on every step; the worst deviation is
        reported in ``max_grad_error``, and a disagreement beyond 1e-4
        raises (your analytic gradient is wrong — surfaced, not hidden).
    verify_grad : verify grad_f against finite differences (default True)

    Every step is verified (exp-map validity, manifold constraint, and for
    SGD with zero momentum the descent against f); moment buffers are
    parallel-transported along each step's geodesic.  The report carries
    the measured convergence data (Adam may overshoot early: ``descent_ok``
    reports it honestly).
    """
    from .ops import optim_gradient

    point = np.asarray(p0, dtype=float)
    if optimizer == "adam":
        opt = RiemannianAdam(manifold, lr=lr, betas=betas, eps=eps)
    else:
        opt = RiemannianSGD(manifold, lr=lr, momentum=momentum)
    trajectory = [point.copy()]
    f_history = [float(f(point))]
    descent_ok = True
    max_grad_error = 0.0
    for _ in range(n_steps):
        if grad_f is not None:
            grad = np.asarray(grad_f(point), dtype=float)
            if verify_grad:
                df = _central_difference(f, point)
                g0, g1 = manifold.metric_diag(point)
                grad_fd = np.array([df[0] / g0, df[1] / g1])
                max_grad_error = max(
                    max_grad_error, float(np.abs(grad - grad_fd).max())
                )
        else:
            df = _central_difference(f, point)
            grad = optim_gradient(manifold, df, point)
        new_point = opt.step(point, grad, f=f, use_shortcut=use_shortcut)
        if float(f(new_point)) > float(f(point)) + 1e-7:
            descent_ok = False
        point = new_point
        trajectory.append(point.copy())
        f_history.append(float(f(point)))
    if grad_f is not None and verify_grad and max_grad_error > 1e-4:
        from .invariants import VerificationError

        raise VerificationError(
            f"analytic gradient disagrees with finite differences "
            f"(max error {max_grad_error:.2e} > 1e-4)"
        )
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
        max_grad_error=max_grad_error if grad_f is not None else None,
    )
