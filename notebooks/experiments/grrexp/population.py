"""Population (weighted) GRR solutions for Stage 0 (registration §1.9).

A finite-support or quadrature population is a set of support rows ``X`` with
probabilities ``w`` (summing to one). For a compatible pair the population
dual objective is

    F(beta) = sum_i w_i g*(u_i) - beta' b,   u_i = u_ref(X_i) + phi(X_i)' beta,
    b = sum_i w_i m(W_i, phi),

whose gradient is the population imbalance ``sum_i w_i alpha_i phi_i - b`` and
whose Hessian is ``sum_i w_i (d alpha / d v)_i phi_i phi_i'``. :func:`solve`
minimizes it with SciPy's default unconstrained BFGS routine. Infeasible
points have objective value +inf. Library success determines the fit status;
gradients, Hessian eigenvalues and domain margins are reported as diagnostics.

The returned quantities are those of the §1.9 certificate on a finite support:
the largest absolute gradient, the smallest Hessian eigenvalue, and the
smallest distance of the dual coordinate to the finite end of its range
(``generator.dual_margin``) over support and counterfactual rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize


@dataclass(frozen=True)
class PopulationSolution:
    beta: np.ndarray
    status: str  # "ok", "maxit", "linesearch", "singular"
    n_iter: int
    max_gradient: float
    min_hessian_eig: float
    alpha: np.ndarray  # on the support rows
    min_dual_margin: float  # over support and counterfactual rows
    message: str = ""

    def certified(self, grad_tol: float | None = None, margin_tol: float = 1e-4) -> bool:
        """§1.9 (ii)–(iii) on a finite support: gradient, positive Hessian, dual margin."""
        return (
            self.status == "ok"
            and (grad_tol is None or self.max_gradient <= grad_tol)
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
    tol: float | None = None,
    max_iter: int | None = None,
    options: dict | None = None,
) -> PopulationSolution:
    """Minimize the population dual objective with SciPy defaults.

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
    # imported here, after the notebook selected this checkout (grrexp.env.use_submodule_genriesz)
    from genriesz.solvers import scipy_status

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

    def value(coef):
        point = evaluate(coef)
        return np.inf if point is None else float(w @ point[0] - coef @ b)

    def gradient(coef):
        point = evaluate(coef)
        return np.full_like(coef, np.nan) if point is None else Phi.T @ (w * point[1]) - b

    opts = {} if options is None else dict(options)
    if max_iter is not None:
        opts["maxiter"] = int(max_iter)
    res = optimize.minimize(value, beta, jac=gradient, tol=tol, options=opts)
    beta = res.x
    point = evaluate(beta)
    status = scipy_status(res)
    if point is None or not np.isfinite(res.fun) or not np.all(np.isfinite(beta)):
        return PopulationSolution(beta, "nonfinite" if point is not None or not np.all(np.isfinite(beta))
                                  else "infeasible_start",
                                  int(getattr(res, "nit", 0)), np.nan, np.nan,
                                  np.full(n, np.nan), np.nan, str(res.message))
    n_iter = int(getattr(res, "nit", 0))
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
        message=str(res.message),
    )


def weighted_projection(values: np.ndarray, Phi: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``L_2(P)`` projection of ``values`` onto the span of the columns of ``Phi``."""
    sw = np.sqrt(np.asarray(w, float))
    coef, *_ = np.linalg.lstsq(Phi * sw[:, None], values * sw, rcond=None)
    return Phi @ coef


def l2_norm(values: np.ndarray, w: np.ndarray) -> float:
    return float(np.sqrt(np.asarray(w, float) @ np.asarray(values, float) ** 2))
