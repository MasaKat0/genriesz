"""Population (weighted) GRR solutions for Stage 0 (registration §1.9).

A finite-support or quadrature population is a set of support rows ``X`` with
probabilities ``w`` (summing to one). For a compatible pair the population
dual objective is

    F(beta) = sum_i w_i g*(u_i) - beta' b,   u_i = u_ref(X_i) + phi(X_i)' beta,
    b = sum_i w_i m(W_i, phi),

whose gradient is the population imbalance ``sum_i w_i alpha_i phi_i - b`` and
whose Hessian is ``sum_i w_i (d alpha / d v)_i phi_i phi_i'``. :func:`solve`
minimizes it by damped Newton steps (registration §1.3 A-2: halve the step
until every support and counterfactual row is admissible and the Armijo
condition with constant 1e-4 holds, at most 60 halvings, at most 500
iterations). The acceptance test is genriesz's own
``genriesz.solvers._accept_newton`` (the reviewed implementation of A-2): Armijo,
or, when the change of the objective is below rounding, a decrease of the
gradient norm. A row is admissible when its dual coordinate is in the link
domain and its representer is finite and in the domain of ``g``. It never
clips; a step that cannot satisfy both conditions stops with ``linesearch``.

The returned quantities are those of the §1.9 certificate on a finite support:
the largest absolute gradient, the smallest Hessian eigenvalue, and the
smallest distance of the dual coordinate to the finite end of its range
(``generator.dual_margin``) over support and counterfactual rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from genriesz.solvers import _accept_newton

ARMIJO = 1e-4


@dataclass(frozen=True)
class PopulationSolution:
    beta: np.ndarray
    status: str  # "ok", "maxit", "linesearch", "singular"
    n_iter: int
    max_gradient: float
    min_hessian_eig: float
    alpha: np.ndarray  # on the support rows
    min_dual_margin: float  # over support and counterfactual rows

    def certified(self, grad_tol: float = 1e-12, margin_tol: float = 1e-4) -> bool:
        """§1.9 (ii)–(iii) on a finite support: gradient, positive Hessian, dual margin."""
        return (
            self.status == "ok"
            and self.max_gradient <= grad_tol
            and self.min_hessian_eig > 0.0
            and self.min_dual_margin >= margin_tol
        )


def _admissible(generator, X, v):
    """``(g*, alpha, dalpha)`` when every row is admissible, else ``None``."""
    if not np.all(generator.link_domain(X, v)):
        return None
    g_star, alpha, dalpha = generator.dual_eval(X, v)
    finite = (
        np.all(np.isfinite(g_star)) and np.all(np.isfinite(alpha)) and np.all(np.isfinite(dalpha))
    )
    if not finite or not np.all(generator.alpha_domain(X, alpha)):
        return None
    return g_star, alpha, dalpha


def solve(
    *,
    generator,
    X: np.ndarray,
    w: np.ndarray,
    Phi: np.ndarray,
    M: np.ndarray,
    offset: np.ndarray,
    check_X: np.ndarray | None = None,
    check_Phi: np.ndarray | None = None,
    check_offset: np.ndarray | None = None,
    beta0: np.ndarray | None = None,
    tol: float = 1e-13,
    max_iter: int = 500,
    max_halvings: int = 60,
) -> PopulationSolution:
    """Minimize the population dual objective by damped Newton steps.

    ``M`` is the matrix ``m(W_i, phi_j)`` on the support rows; ``check_*`` are the
    counterfactual rows at which ``m`` evaluates the representer.
    """
    X = np.asarray(X, float)
    w = np.asarray(w, float)
    Phi = np.asarray(Phi, float)
    M = np.asarray(M, float)
    offset = np.asarray(offset, float)
    n, p = Phi.shape
    if w.shape != (n,) or M.shape != (n, p) or offset.shape != (n,):
        raise ValueError("w, M and offset must match the support rows")
    if np.any(w < 0) or not np.isclose(w.sum(), 1.0, rtol=0, atol=1e-14):
        raise ValueError("w must be non-negative probabilities summing to one")
    has_checks = check_X is not None
    b = w @ M

    def evaluate(beta_):
        point = _admissible(generator, X, offset + Phi @ beta_)
        if point is None:
            return None
        if has_checks and _admissible(generator, check_X, check_offset + check_Phi @ beta_) is None:
            return None
        return point

    beta = np.zeros(p) if beta0 is None else np.asarray(beta0, float).copy()
    point = evaluate(beta)
    if point is None:
        raise ValueError("the starting point is outside the link domain")
    status, n_iter = "maxit", 0
    for _ in range(max_iter):
        g_star, alpha, dalpha = point
        F = float(w @ g_star) - beta @ b
        grad = Phi.T @ (w * alpha) - b
        if np.max(np.abs(grad)) <= tol:
            status = "ok"
            break
        n_iter += 1
        H = Phi.T @ (Phi * (w * dalpha)[:, None])
        if np.min(np.linalg.eigvalsh((H + H.T) / 2)) <= 1e-14 * max(1.0, np.max(np.abs(H))):
            status = "singular"  # the Newton system has no unique solution
            break
        step = -np.linalg.solve(H, grad)
        t, accepted = 1.0, False
        for _ in range(max_halvings + 1):  # the full step, then up to 60 halvings
            cand = beta + t * step
            new = evaluate(cand)
            if new is not None:
                F_new = float(w @ new[0]) - cand @ b
                g_new = float(np.max(np.abs(Phi.T @ (w * new[1]) - b)))
                if _accept_newton(
                    F, F_new, -(grad @ step), t, ARMIJO, float(np.max(np.abs(grad))), g_new
                ):
                    beta, point, accepted = cand, new, True
                    break
            t /= 2.0
        if not accepted:
            status = "linesearch"
            break
    g_star, alpha, dalpha = point
    grad = Phi.T @ (w * alpha) - b
    H = Phi.T @ (Phi * (w * dalpha)[:, None])
    margin = float(np.min(generator.dual_margin(X, offset + Phi @ beta)))
    if has_checks:
        margin = min(
            margin, float(np.min(generator.dual_margin(check_X, check_offset + check_Phi @ beta)))
        )
    return PopulationSolution(
        beta=beta,
        status=status,
        n_iter=n_iter,
        max_gradient=float(np.max(np.abs(grad))),
        min_hessian_eig=float(np.min(np.linalg.eigvalsh((H + H.T) / 2))),
        alpha=alpha,
        min_dual_margin=margin,
    )


def weighted_projection(values: np.ndarray, Phi: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``L_2(P)`` projection of ``values`` onto the span of the columns of ``Phi``."""
    sw = np.sqrt(np.asarray(w, float))
    coef, *_ = np.linalg.lstsq(Phi * sw[:, None], values * sw, rcond=None)
    return Phi @ coef


def l2_norm(values: np.ndarray, w: np.ndarray) -> float:
    return float(np.sqrt(np.asarray(w, float) @ np.asarray(values, float) ** 2))
