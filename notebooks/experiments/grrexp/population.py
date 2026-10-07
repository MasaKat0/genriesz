"""Population (weighted) GRR solutions for Stage 0 (registration §1.9).

A finite-support or quadrature population is a set of support rows ``X`` with
probabilities ``w`` (summing to one). For a compatible pair the population
dual objective is

    F(beta) = sum_i w_i g*(u_i) - beta' b,   u_i = u_ref(X_i) + phi(X_i)' beta,
    b = sum_i w_i m(W_i, phi),

whose gradient is the population imbalance ``sum_i w_i alpha_i phi_i - b`` and
whose Hessian is ``sum_i w_i (d alpha / d v)_i phi_i phi_i'``. :func:`solve`
minimizes it by damped Newton steps that keep every support row and every
counterfactual row in the link domain, and returns the quantities of the §1.9
certificate: the largest absolute gradient, the smallest Hessian eigenvalue and
the smallest distance of the representer to the domain boundary. It never
clips; a step that cannot stay in the domain stops with a status.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PopulationSolution:
    beta: np.ndarray
    status: str  # "ok", "maxit", "linesearch", "singular"
    n_iter: int
    max_gradient: float
    min_hessian_eig: float
    alpha: np.ndarray  # on the support rows
    min_margin: float  # smallest boundary margin over support and counterfactual rows

    def certified(self, grad_tol: float = 1e-12, margin_tol: float = 1e-4) -> bool:
        """§1.9 (ii)–(iii) on a finite support: gradient, positive Hessian, interior margin."""
        return (
            self.status == "ok"
            and self.max_gradient <= grad_tol
            and self.min_hessian_eig > 0.0
            and self.min_margin >= margin_tol
        )


def _evaluate(generator, X, Phi, offset, beta, w):
    v = offset + Phi @ beta
    if not np.all(generator.link_domain(X, v)):
        return None
    g_star, alpha, dalpha = generator.dual_eval(X, v)
    if not (
        np.all(np.isfinite(g_star)) and np.all(np.isfinite(alpha)) and np.all(np.isfinite(dalpha))
    ):
        return None
    return float(w @ g_star), alpha, dalpha


def _margin(generator, X, alpha) -> float:
    return float(np.min(generator.boundary_margin(X, alpha)))


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
    b = w @ M
    beta = np.zeros(p) if beta0 is None else np.asarray(beta0, float).copy()

    def feasible_checks(beta_):
        if check_X is None:
            return True
        v = check_offset + check_Phi @ beta_
        return bool(np.all(generator.link_domain(check_X, v)))

    point = _evaluate(generator, X, Phi, offset, beta, w)
    if point is None or not feasible_checks(beta):
        raise ValueError("the starting point is outside the link domain")
    status, n_iter = "maxit", 0
    for _ in range(max_iter):
        n_iter += 1
        F, alpha, dalpha = point
        F -= beta @ b
        grad = Phi.T @ (w * alpha) - b
        if np.max(np.abs(grad)) <= tol:
            status = "ok"
            break
        H = Phi.T @ (Phi * (w * dalpha)[:, None])
        if np.min(np.linalg.eigvalsh((H + H.T) / 2)) <= 1e-14 * max(1.0, np.max(np.abs(H))):
            status = "singular"  # the Newton system has no unique solution
            break
        step = -np.linalg.solve(H, grad)
        t = 1.0
        for _ in range(max_halvings):
            cand = beta + t * step
            new = _evaluate(generator, X, Phi, offset, cand, w)
            if new is not None and feasible_checks(cand):
                F_new = new[0] - cand @ b
                if F_new <= F + 1e-4 * t * (grad @ step) or np.max(np.abs(t * step)) < 1e-15:
                    beta, point = cand, new
                    break
            t /= 2.0
        else:
            status = "linesearch"
            break
    _, alpha, dalpha = point
    grad = Phi.T @ (w * alpha) - b
    H = Phi.T @ (Phi * (w * dalpha)[:, None])
    margin = _margin(generator, X, alpha)
    if check_X is not None:
        v_c = check_offset + check_Phi @ beta
        _, a_c, _ = generator.dual_eval(check_X, v_c)
        margin = min(margin, _margin(generator, check_X, a_c))
    return PopulationSolution(
        beta=beta,
        status=status,
        n_iter=n_iter,
        max_gradient=float(np.max(np.abs(grad))),
        min_hessian_eig=float(np.min(np.linalg.eigvalsh((H + H.T) / 2))),
        alpha=alpha,
        min_margin=margin,
    )


def weighted_projection(values: np.ndarray, Phi: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``L_2(P)`` projection of ``values`` onto the span of the columns of ``Phi``."""
    sw = np.sqrt(np.asarray(w, float))
    coef, *_ = np.linalg.lstsq(Phi * sw[:, None], values * sw, rcond=None)
    return Phi @ coef


def l2_norm(values: np.ndarray, w: np.ndarray) -> float:
    return float(np.sqrt(np.asarray(w, float) @ np.asarray(values, float) ** 2))
