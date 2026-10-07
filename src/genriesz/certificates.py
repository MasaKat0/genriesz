r"""Sample-level balancing-weight program (PR-4a, ``prop:dual_balancing_program``).

For a model linear in the dual coordinate with offset ``u_ref`` and
``B_0 = R^p``, the GRR coefficient problem (``eq:beta_erm`` with ``q = 1``) is
dual to the weight program

.. math::

    \min_{\alpha_i\in\overline{\mathcal A}_i}\;
    D_\lambda(\alpha) = \frac1n\sum_i\bigl(g(\alpha_i) - u_{\mathrm{ref}}(X_i)\alpha_i\bigr)
    \quad\text{s.t.}\quad
    \Bigl|\frac1n\sum_i \alpha_i\phi_j(X_i) - b_j\Bigr| \le \lambda,

over the *closures* of the sign components. By part (a) of the proposition, if a
GRR minimizer exists, its weights are the unique minimizer of ``D_lambda`` and
lie in the open components. So a minimizer of ``D_lambda`` **exactly** on the
boundary (``|alpha_i| = C``) would show that this sample's GRR problem has no
minimizer.

What :func:`weight_program_certificate` computes is numerical: a solution of the
program, a dual coefficient vector ``beta``, and the duality gap
``P(beta) + D(alpha)`` with the closure-extended conjugates (``eq:dual_gap_identity``).
A numerical solution within ``margin_tol`` of the boundary with a small gap is
reported as ``"uncertified_boundary"``: it is **not** a proof that the sample
problem has no minimizer (an interior minimizer arbitrarily close to the
boundary is not excluded), and no rigorous certificate is attempted here. It is
never evidence about the *population* problem, and a solver failure is never
read as nonexistence.

Verdicts
--------
``"interior"``
    The numerical solution is at least ``margin_tol`` from the boundary, with
    gap and balance within tolerance.
``"uncertified_boundary"``
    Some ``|alpha_i| - C <= margin_tol`` (``min(|alpha|, 1-|alpha|)`` for PU),
    with gap and balance within tolerance.
``"uncertified_infeasible"``
    The solver reports the program infeasible (the balance set does not meet
    the closed components). Solver-based; not a proof.
``"inconclusive"``
    Anything else (solver not optimal, gap or balance above tolerance,
    non-finite values).

Backends: ``"cvxpy"`` (Clarabel; the ``cvxpy`` package is an optional
dependency) and ``"scipy"`` (SLSQP on the weights, with the dual coefficients
recovered by least squares from the interior stationarity conditions). The gap
is computed by this module in both cases, from the returned weights and dual
coefficients, so the reported gap does not depend on a solver's own accounting.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

from .basis import Basis
from .functionals import LinearFunctional
from .generators import (
    BKLGenerator,
    BPGenerator,
    BregmanGenerator,
    PUGenerator,
    SquaredGenerator,
    UKLGenerator,
)
from .glm import OffsetSpec, _ensure_basis_fitted, evaluate_offset
from .utils import as_2d

VERDICTS = ("interior", "uncertified_boundary", "uncertified_infeasible", "inconclusive")


@dataclass(frozen=True)
class WeightProgramCertificate:
    """Outcome of :func:`weight_program_certificate`.

    Attributes
    ----------
    verdict:
        One of :data:`VERDICTS`. None of them certifies the (non)existence of a
        GRR minimizer; see the module docstring.
    alpha:
        Weights of the numerical solution (projected onto the closed
        components), or ``None``.
    beta:
        Dual coefficient vector (``u_ref + Phi beta`` is the dual coordinate).
    primal_value, dual_value:
        ``D_lambda(alpha)`` and ``-P(beta)``.
    gap:
        ``P(beta) + D(alpha)`` (nonnegative up to rounding when ``alpha`` is
        feasible).
    balance_violation:
        ``max_j (|Delta_j(alpha)| - lambda)_+``.
    min_margin, n_at_boundary:
        Smallest distance of a weight to the boundary, and the number of weights
        within ``margin_tol`` of it.
    backend, solver_status:
        Backend used and its raw status.
    """

    verdict: str
    alpha: NDArray[np.float64] | None
    beta: NDArray[np.float64] | None
    primal_value: float
    dual_value: float
    gap: float
    balance_violation: float
    min_margin: float
    n_at_boundary: int
    backend: str
    solver_status: str
    margin_tol: float
    gap_tol: float


# ---------------------------------------------------------------------------
# Generator-specific pieces on the closed components
# ---------------------------------------------------------------------------
def _kind(generator: BregmanGenerator) -> str:
    if isinstance(generator, SquaredGenerator):
        return "sq"
    if isinstance(generator, UKLGenerator):
        return "ukl"
    if isinstance(generator, BPGenerator):
        return "bp"
    if isinstance(generator, BKLGenerator):
        return "bkl"
    if isinstance(generator, PUGenerator):
        return "pu"
    raise TypeError(
        "weight_program_certificate supports SquaredGenerator, UKLGenerator, BPGenerator, "
        f"BKLGenerator and PUGenerator; got {type(generator).__name__}."
    )


def _signs(generator: BregmanGenerator, X: NDArray[np.float64], kind: str) -> NDArray[np.float64]:
    if kind == "sq":
        return np.ones(len(X))
    if generator.branch_fn is None:
        raise ValueError(
            "The weight program needs the sign component of every observation: build the "
            "generator with branch_fn."
        )
    return generator._sign(X, np.zeros(len(X)))


def _alpha_from_t(t: NDArray[np.float64], s: NDArray[np.float64], C: float, kind: str):
    if kind == "sq":
        return t
    if kind == "pu":
        return s * t
    return s * (C + t)


def _project_t(t: NDArray[np.float64], kind: str) -> NDArray[np.float64]:
    if kind == "sq":
        return t
    t = np.maximum(t, 0.0)
    if kind == "pu":
        t = np.minimum(t, 1.0)
    return t


def _margin(alpha: NDArray[np.float64], C: float, kind: str) -> NDArray[np.float64]:
    if kind == "sq":
        return np.full(alpha.shape[0], np.inf)
    if kind == "pu":
        a = np.abs(alpha)
        return np.minimum(a, 1.0 - a)
    return np.abs(alpha) - C


def _closure_conjugate(
    generator: BregmanGenerator, v: NDArray[np.float64], s: NDArray[np.float64], kind: str
) -> NDArray[np.float64]:
    """``sup_{a in closure(A_i)} (a v - g(a))`` for each row."""

    C = float(generator.C)
    if kind == "sq":
        return C * v + 0.25 * v * v
    u = s * v
    if kind == "ukl":
        with np.errstate(over="ignore"):
            return C * u + C + np.exp(u)
    if kind == "bp":
        k = 1.0 + 1.0 / generator.omega
        return C * u + np.power(np.maximum(1.0 + u / k, 0.0), k)
    if kind == "bkl":
        out = np.full(u.shape[0], np.inf)
        neg = u < 0.0
        out[neg] = (
            C * u[neg] + 2.0 * C * np.log(2.0 * C) - 2.0 * C * np.log(-np.expm1(u[neg]))
        )
        return out
    # pu
    return C * np.logaddexp(0.0, u / C)


def _g_closure(generator: BregmanGenerator, X, alpha: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(generator.g(X, alpha), dtype=float)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def _solve_cvxpy(Phi, b, u0, s, generator, kind, lam, solver_options=None):
    import cvxpy as cp

    n, p = Phi.shape
    C = float(generator.C)
    if kind == "sq":
        t = cp.Variable(n)
        alpha = t
        gsum = cp.sum_squares(alpha - C)
    else:
        t = cp.Variable(n, nonneg=True)
        alpha = cp.multiply(s, t) if kind == "pu" else cp.multiply(s, C + t)
        if kind == "ukl":
            gsum = cp.sum(-cp.entr(t) - t)
        elif kind == "bp":
            om = float(generator.omega)
            gsum = cp.sum(cp.power(t, 1.0 + om) / om - (1.0 + om) / om * t)
        elif kind == "bkl":
            gsum = cp.sum(cp.rel_entr(t, t + 2.0 * C) - 2.0 * C * cp.log(t + 2.0 * C))
        else:  # pu
            gsum = C * cp.sum(-cp.entr(t) - cp.entr(1.0 - t))
    objective = cp.Minimize((gsum - u0 @ alpha) / n)
    delta = Phi.T @ alpha / n - b
    constraints = []
    if kind == "pu":
        constraints.append(t <= 1.0)
    if lam == 0.0:
        bal = delta == 0
        constraints.append(bal)
    else:
        up = delta <= lam
        lo = -delta <= lam
        constraints += [up, lo]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver="CLARABEL", **(solver_options or {}))
    status = str(prob.status)
    if status not in ("optimal", "optimal_inaccurate") or t.value is None:
        return None, None, status
    if lam == 0.0:
        beta = -np.asarray(bal.dual_value, dtype=float).reshape(-1)
    else:
        beta = np.asarray(lo.dual_value, dtype=float).reshape(-1) - np.asarray(
            up.dual_value, dtype=float
        ).reshape(-1)
    return np.asarray(t.value, dtype=float).reshape(-1), beta, status


def _solve_scipy(Phi, b, u0, s, generator, kind, lam, X, margin_tol):
    n, p = Phi.shape
    C = float(generator.C)

    def alpha_of(t):
        return _alpha_from_t(t, s, C, kind)

    def dalpha_dt():
        return np.ones(n) if kind == "sq" else s

    def fun(t):
        a = alpha_of(_project_t(t, kind))
        return float(np.mean(_g_closure(generator, X, a) - u0 * a))

    def jac(t):
        a = alpha_of(_project_t(t, kind))
        g1 = np.asarray(generator.grad(X, a), dtype=float)
        # At a UKL/BKL/PU boundary g' is infinite; SLSQP then fails and the
        # verdict is "inconclusive" (use the cvxpy backend for those).
        return (g1 - u0) * dalpha_dt() / n

    def delta(t):
        return Phi.T @ alpha_of(t) / n - b

    J = (Phi * dalpha_dt()[:, None]).T / n
    if lam == 0.0:
        cons = [{"type": "eq", "fun": delta, "jac": lambda t: J}]
    else:
        cons = [
            {"type": "ineq", "fun": lambda t: lam - delta(t), "jac": lambda t: -J},
            {"type": "ineq", "fun": lambda t: lam + delta(t), "jac": lambda t: J},
        ]
    if kind == "sq":
        bounds = None
        t0 = np.full(n, C)
    elif kind == "pu":
        bounds = [(0.0, 1.0)] * n
        t0 = np.full(n, 0.5)
    else:
        bounds = [(0.0, None)] * n
        t0 = np.ones(n)
    res = optimize.minimize(
        fun, t0, jac=jac, method="SLSQP", bounds=bounds, constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-15},
    )
    if not res.success:
        return None, None, f"slsqp: {res.message}"
    t = _project_t(np.asarray(res.x, dtype=float), kind)
    a = alpha_of(t)
    interior = _margin(a, C, kind) > margin_tol
    if not np.any(interior):
        return t, None, "slsqp: no interior weight to recover the dual coefficients"
    rhs = np.asarray(generator.grad(X[interior], a[interior]), dtype=float) - u0[interior]
    beta, *_ = np.linalg.lstsq(Phi[interior], rhs, rcond=None)
    return t, beta, "optimal"


def weight_program_certificate(
    *,
    X: ArrayLike,
    basis: Basis,
    functional: LinearFunctional,
    generator: BregmanGenerator,
    offset: OffsetSpec = None,
    lam: float = 0.0,
    backend: str = "auto",
    margin_tol: float = 1e-7,
    gap_tol: float = 1e-9,
    balance_tol: float = 1e-9,
    solver_options: dict | None = None,
) -> WeightProgramCertificate:
    """Solve the sample weight program for a GRR problem and classify its solution.

    Parameters
    ----------
    X, basis, functional, generator, offset:
        As for :class:`~genriesz.GRRGLM` (``B_0 = R^p``, ``l1`` penalty ``lam``).
        Branch-wise generators need ``branch_fn`` (the sign component of each
        observation).
    lam:
        ``l1`` penalty level (``0``: exact balance).
    backend:
        ``"cvxpy"``, ``"scipy"``, or ``"auto"`` (``"cvxpy"`` when the package is
        installed, otherwise ``"scipy"``; the choice is recorded in the result).
    margin_tol, gap_tol, balance_tol:
        Thresholds of the verdict (registration §1.3 A-6: ``1e-7`` and ``1e-9``).
    solver_options:
        Keyword settings passed to Clarabel through cvxpy (for example
        ``{"tol_gap_abs": 1e-12}``); ``None`` keeps Clarabel's defaults. Only the
        ``"cvxpy"`` backend accepts them.

    Returns
    -------
    WeightProgramCertificate
        See the module docstring for what the verdicts do and do not establish.
    """

    X_ = as_2d(X)
    _ensure_basis_fitted(basis, X_)
    Phi = np.asarray(basis(X_), dtype=float)
    M = np.asarray(functional.m_basis_matrix(X_, basis), dtype=float)
    b = M.mean(axis=0)
    u0 = evaluate_offset(offset, X_)
    lam = float(lam)
    if lam < 0.0:
        raise ValueError("lam must be >= 0")
    kind = _kind(generator)
    s = _signs(generator, X_, kind)
    C = float(generator.C)

    backend_ = str(backend).lower()
    if backend_ == "auto":
        backend_ = "cvxpy" if importlib.util.find_spec("cvxpy") is not None else "scipy"
    if solver_options and backend_ != "cvxpy":
        raise ValueError("solver_options apply only to the 'cvxpy' backend")
    if backend_ == "cvxpy":
        t, beta, status = _solve_cvxpy(Phi, b, u0, s, generator, kind, lam, solver_options)
    elif backend_ == "scipy":
        t, beta, status = _solve_scipy(Phi, b, u0, s, generator, kind, lam, X_, margin_tol)
    else:
        raise ValueError("backend must be 'auto', 'cvxpy' or 'scipy'")

    nan = float("nan")
    if t is None:
        verdict = "uncertified_infeasible" if "infeasible" in status else "inconclusive"
        return WeightProgramCertificate(
            verdict=verdict, alpha=None, beta=None, primal_value=nan, dual_value=nan, gap=nan,
            balance_violation=nan, min_margin=nan, n_at_boundary=0, backend=backend_,
            solver_status=status, margin_tol=margin_tol, gap_tol=gap_tol,
        )

    t = _project_t(t, kind)
    alpha = _alpha_from_t(t, s, C, kind)
    primal = float(np.mean(_g_closure(generator, X_, alpha) - u0 * alpha))
    delta = Phi.T @ alpha / len(X_) - b
    balance = float(np.max(np.maximum(np.abs(delta) - lam, 0.0))) if delta.size else 0.0
    margins = _margin(alpha, C, kind)
    min_margin = float(np.min(margins)) if margins.size else float("inf")
    n_bd = int(np.sum(margins <= margin_tol))
    if beta is None:
        dual = nan
        gap = nan
    else:
        v = u0 + Phi @ beta
        P = float(np.mean(_closure_conjugate(generator, v, s, kind)) - b @ beta) + lam * float(
            np.sum(np.abs(beta))
        )
        dual = -P
        gap = P + primal

    ok = (
        np.isfinite(primal)
        and np.isfinite(gap)
        and abs(gap) <= gap_tol
        and balance <= balance_tol
    )
    if not ok:
        verdict = "inconclusive"
    elif n_bd > 0:
        verdict = "uncertified_boundary"
    else:
        verdict = "interior"
    return WeightProgramCertificate(
        verdict=verdict,
        alpha=alpha,
        beta=beta,
        primal_value=primal,
        dual_value=dual,
        gap=gap,
        balance_violation=balance,
        min_margin=min_margin,
        n_at_boundary=n_bd,
        backend=backend_,
        solver_status=status,
        margin_tol=margin_tol,
        gap_tol=gap_tol,
    )
