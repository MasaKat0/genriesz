r"""Generalized Riesz regression with an arbitrary (possibly incompatible) link.

For a generator ``g`` and a link ``alpha_beta(x) = link(x, v_beta(x))`` with the
index ``v_beta(x) = v_ref(x) + phi(x)'beta`` (``v_ref`` a fixed index offset) that
need not be the generator's own (compatible) link, the empirical objective is
the sample analogue of ``BD_g(alpha_beta)`` (``eq:bd_emp_beta``),

.. math::

    F(\beta) = \frac1n\sum_i\bigl\{-g(\alpha_i) + \alpha_i\,g'(\alpha_i)\bigr\}
               - \frac1n\sum_i m\bigl(W_i, g'\circ\alpha_\beta\bigr) + \mathrm{pen}(\beta).

Proposition ``prop:arbitrary_pair_foc`` gives its gradient as the empirical
imbalance in the *tangent* directions
:math:`\psi_{j,\beta} = g''(\alpha_\beta)\,\partial_{\beta_j}\alpha_\beta =
g''(\alpha_\beta)\,\mathrm{link}'(v_\beta)\,\phi_j`:

.. math::

    \partial_{\beta_j} F = \widehat\Delta(\alpha_\beta, \psi_{j,\beta})
    = \frac1n\sum_i \alpha_i\psi_j(X_i) - \frac1n\sum_i m(W_i, \psi_j),

and its Hessian is

.. math::

    \partial_{\beta_k}\partial_{\beta_j} F
    = \frac1n\sum_i g''\,\mathrm{link}'^2\,\phi_j\phi_k(X_i)
      + \frac1n\sum_i \alpha_i\,e\,\phi_j\phi_k(X_i)
      - \frac1n\sum_i m(W_i, e\,\phi_j\phi_k),
    \qquad e = g'''(\alpha)\,\mathrm{link}'^2 + g''(\alpha)\,\mathrm{link}''.

The objective is in general not convex, and its sample version need not be
bounded below (for example, SQ with the exponential link when a control unit lies
outside the convex hull of the treated units' regressors); the iterates then
diverge and the status is not ``"ok"``. :class:`GRRGeneralLink` minimizes it by
a damped (modified) Newton method that keeps every training row and every row
evaluated by ``m`` inside the domain of ``g``, and reports a stationary point
with an explicit status; ``"ok"`` additionally requires a positive-definite
Hessian at the solution.

Only functionals that evaluate their argument at points (ATE, ATT, DID, callable
functionals) are supported; derivative functionals (AME) are not.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .basis import Basis
from .functionals import LinearFunctional
from .generators import BregmanGenerator, DomainError
from .glm import OffsetSpec, _ensure_basis_fitted, evaluate_offset
from .solvers import BOUNDARY, INFEASIBLE_START, LINESEARCH, MAXIT, NONFINITE, OK, SINGULAR
from .utils import as_2d

LinkFn = Callable[[NDArray[np.float64], NDArray[np.float64]], ArrayLike]

_EPS = float(np.finfo(float).eps)


class _FunctionBasis:
    """A "basis" whose columns are ``c(x) * phi_j(x)`` (or products), for ``m_basis_matrix``."""

    def __init__(self, fn: Callable[[NDArray[np.float64]], NDArray[np.float64]], width: int):
        self._fn = fn
        self._width = int(width)

    @property
    def n_features(self) -> int:
        return self._width

    def fit(self, X, y=None):  # pragma: no cover - never refitted
        return self

    def __call__(self, X: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        return self._fn(X_)

    def derivative(self, X: ArrayLike, coordinate: int) -> NDArray[np.float64]:
        raise NotImplementedError(
            "GRRGeneralLink does not support derivative functionals (e.g. the AME)."
        )


@dataclass
class GeneralLinkFitResult:
    """Result of :meth:`GRRGeneralLink.fit`.

    Attributes
    ----------
    beta, status, success, message, n_iter:
        Last iterate, status (one of :data:`genriesz.solvers.STATUSES`), whether
        ``status == "ok"``, a message, and the number of Newton iterations.
    objective_value:
        Penalized objective at ``beta``.
    tangent_imbalance:
        ``max_j |Delta(alpha, psi_j)|`` -- the first-order residual of the
        unpenalized objective (``I_psi`` in the registration).
    regressor_imbalance:
        ``max_j |Delta(alpha, phi_j)|`` in the intended regressors (``I_phi``);
        zero at a stationary point only for compatible pairs.
    kkt_residual:
        ``max_j |dF/dbeta_j|`` including the penalty.
    hessian_min_eig:
        Smallest eigenvalue of the Hessian at ``beta``.
    n_hessian_shifts:
        Newton steps whose Hessian had to be shifted to be positive definite.
    n_boundary:
        Training rows numerically at the boundary of the domain of ``g``.
    """

    beta: NDArray[np.float64]
    status: str
    message: str
    n_iter: int
    objective_value: float = float("nan")
    tangent_imbalance: float = float("nan")
    regressor_imbalance: float = float("nan")
    kkt_residual: float = float("nan")
    hessian_min_eig: float = float("nan")
    n_hessian_shifts: int = 0
    n_boundary: int = 0
    gradient: NDArray[np.float64] | None = field(default=None, repr=False)

    @property
    def success(self) -> bool:
        return self.status == OK


@dataclass
class _Eval:
    feasible: bool
    finite: bool = False
    F: float = float("nan")
    grad: NDArray[np.float64] | None = None
    alpha: NDArray[np.float64] | None = None
    u: NDArray[np.float64] | None = None
    beta: NDArray[np.float64] | None = None


class GRRGeneralLink:
    """GRR with a generator ``g`` and an arbitrary link (registration E-13).

    Parameters
    ----------
    generator:
        Bregman generator; uses ``g``, ``grad``, ``grad2``, ``grad3``,
        ``alpha_domain`` and ``boundary_mask``.
    link, dlink, d2link:
        ``link(X, v)`` returns ``alpha`` row-wise for the index
        ``v = v_ref(X) + phi(X)' beta``; ``dlink`` and ``d2link`` are its first
        and second derivatives in ``v`` (both are needed for the analytic
        Hessian).
    index_offset:
        Fixed index offset ``v_ref`` (``None``, a constant, or a callable
        ``X -> (n,)``; registration §1.3 A-1: ``v_ref = link^{-1}(alpha_ref)``
        per sign component, e.g. ``v_ref = 1`` for I-UKLlin). Coefficients are
        deviations from it: ``beta = 0`` gives ``alpha_ref``, and the penalty
        acts on ``beta``. It is evaluated, unchanged, at the fitting rows, at
        the counterfactual rows evaluated by ``m``, and at prediction rows.
    basis, functional:
        Regressors ``phi`` and the linear functional ``m``.
    penalty, lam:
        ``None`` or ``"l2"`` (``lam/2 ||beta||^2``).
    boundary_tol:
        Distance to the domain boundary below which the status is
        ``"uncertified_numerical_boundary"`` (never for SQ).
    """

    def __init__(
        self,
        generator: BregmanGenerator,
        link: LinkFn,
        dlink: LinkFn,
        d2link: LinkFn,
        *,
        basis: Basis,
        functional: LinearFunctional,
        penalty: str | None = None,
        lam: float = 0.0,
        boundary_tol: float = 1e-8,
        index_offset: OffsetSpec = None,
    ):
        if penalty not in (None, "l2"):
            raise ValueError("GRRGeneralLink supports penalty=None or 'l2'.")
        if lam < 0.0:
            raise ValueError("lam must be >= 0")
        self.generator = generator
        self.link = link
        self.dlink = dlink
        self.d2link = d2link
        self.basis = basis
        self.functional = functional
        self.lam = 0.0 if penalty is None else float(lam)
        self.boundary_tol = float(boundary_tol)
        self.index_offset = index_offset
        self.beta_: NDArray[np.float64] | None = None
        self.fit_result_: GeneralLinkFitResult | None = None

    # ------------------------------------------------------------------
    # Pointwise pieces
    # ------------------------------------------------------------------
    def _alpha_at(self, X: NDArray[np.float64], beta: NDArray[np.float64]):
        eta = evaluate_offset(self.index_offset, X) + np.asarray(self.basis(X), dtype=float) @ beta
        alpha = np.asarray(self.link(X, eta), dtype=float).reshape(-1)
        return eta, alpha

    def _checked_u(self, beta: NDArray[np.float64], record: list[int]):
        """``x -> g'(alpha_beta(x))``, recording rows outside the domain of ``g``."""

        def u_fn(Z: NDArray[np.float64]) -> NDArray[np.float64]:
            Z_ = as_2d(Z)
            _, a = self._alpha_at(Z_, beta)
            ok = np.asarray(self.generator.alpha_domain(Z_, a), dtype=bool)
            record[0] += int(np.sum(~ok))
            out = np.full(a.shape[0], np.nan)
            if np.any(ok):
                out[ok] = self.generator.grad(Z_[ok], a[ok])
            return out

        return u_fn

    def _coef(self, Z: NDArray[np.float64], beta: NDArray[np.float64]):
        """``(c, e)`` with ``psi_j = c phi_j`` and ``d psi_j/d beta_k = e phi_j phi_k``."""

        eta, a = self._alpha_at(Z, beta)
        g2 = np.asarray(self.generator.grad2(Z, a), dtype=float)
        g3 = np.asarray(self.generator.grad3(Z, a), dtype=float)
        d1 = np.asarray(self.dlink(Z, eta), dtype=float).reshape(-1)
        d2 = np.asarray(self.d2link(Z, eta), dtype=float).reshape(-1)
        return g2 * d1, g3 * d1 * d1 + g2 * d2

    def _tangent_basis(self, beta: NDArray[np.float64], p: int) -> _FunctionBasis:
        def fn(Z: NDArray[np.float64]) -> NDArray[np.float64]:
            c, _ = self._coef(Z, beta)
            return c[:, None] * np.asarray(self.basis(Z), dtype=float)

        return _FunctionBasis(fn, p)

    def _product_basis(self, beta: NDArray[np.float64], p: int) -> _FunctionBasis:
        def fn(Z: NDArray[np.float64]) -> NDArray[np.float64]:
            _, e = self._coef(Z, beta)
            P = np.asarray(self.basis(Z), dtype=float)
            return (e[:, None, None] * P[:, :, None] * P[:, None, :]).reshape(len(Z), p * p)

        return _FunctionBasis(fn, p * p)

    # ------------------------------------------------------------------
    # Objective, gradient, Hessian
    # ------------------------------------------------------------------
    def _evaluate(self, X: NDArray[np.float64], beta: NDArray[np.float64]) -> _Eval:
        # Trial points may overflow the user's link (e.g. exp); the result is
        # then non-finite and the point is rejected or reported ("nonfinite"),
        # so the floating-point warnings carry no extra information.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return self._evaluate_raw(X, beta)

    def _evaluate_raw(self, X: NDArray[np.float64], beta: NDArray[np.float64]) -> _Eval:
        beta = np.asarray(beta, dtype=float).reshape(-1)
        _, a = self._alpha_at(X, beta)
        if not np.all(self.generator.alpha_domain(X, a)):
            return _Eval(feasible=False, beta=beta)
        record = [0]
        u_fn = self._checked_u(beta, record)
        u = u_fn(X)
        m_u = np.asarray(self.functional.m_from_function(X, predict=u_fn), dtype=float)
        if record[0] > 0:
            return _Eval(feasible=False, beta=beta)
        g = np.asarray(self.generator.g(X, a), dtype=float)
        F = float(np.mean(-g + a * u) - np.mean(m_u)) + 0.5 * self.lam * float(beta @ beta)
        p = beta.size
        Phi = np.asarray(self.basis(X), dtype=float)
        c, _ = self._coef(X, beta)
        M_psi = np.asarray(self.functional.m_basis_matrix(X, self._tangent_basis(beta, p)), float)
        grad = (a[:, None] * c[:, None] * Phi - M_psi).mean(axis=0) + self.lam * beta
        finite = bool(np.isfinite(F) and np.all(np.isfinite(grad)))
        return _Eval(feasible=True, finite=finite, F=F, grad=grad, alpha=a, u=u, beta=beta)

    def objective(self, X: ArrayLike, beta: ArrayLike) -> float:
        """Penalized objective ``F(beta)`` (NaN outside the domain)."""

        ev = self._evaluate(as_2d(X), np.asarray(beta, dtype=float))
        return ev.F if ev.feasible else float("nan")

    def gradient(self, X: ArrayLike, beta: ArrayLike) -> NDArray[np.float64]:
        """``dF/dbeta = Delta(alpha_beta, psi_beta) + lam beta`` (``prop:arbitrary_pair_foc``)."""

        ev = self._evaluate(as_2d(X), np.asarray(beta, dtype=float))
        if not ev.feasible:
            raise ValueError("beta is outside the domain of the generator")
        assert ev.grad is not None
        return ev.grad

    def hessian(self, X: ArrayLike, beta: ArrayLike) -> NDArray[np.float64]:
        """Analytic Hessian of ``F`` (see the module docstring)."""

        X_ = as_2d(X)
        b = np.asarray(beta, dtype=float).reshape(-1)
        p = b.size
        Phi = np.asarray(self.basis(X_), dtype=float)
        eta, a = self._alpha_at(X_, b)
        g2 = np.asarray(self.generator.grad2(X_, a), dtype=float)
        d1 = np.asarray(self.dlink(X_, eta), dtype=float).reshape(-1)
        _, e = self._coef(X_, b)
        n = len(X_)
        H = (Phi.T * (g2 * d1 * d1 + a * e)) @ Phi / n
        M_prod = np.asarray(
            self.functional.m_basis_matrix(X_, self._product_basis(b, p)), dtype=float
        )
        H = H - M_prod.mean(axis=0).reshape(p, p)
        H = 0.5 * (H + H.T)
        return H + self.lam * np.eye(p)

    def tangent_imbalance(self, X: ArrayLike, beta: ArrayLike) -> NDArray[np.float64]:
        """``Delta(alpha_beta, psi_j)`` for every ``j`` (unpenalized gradient)."""

        b = np.asarray(beta, dtype=float).reshape(-1)
        return self.gradient(X, b) - self.lam * b

    def regressor_imbalance(self, X: ArrayLike, beta: ArrayLike) -> NDArray[np.float64]:
        """``Delta(alpha_beta, phi_j)`` for every ``j`` (the intended regressors)."""

        X_ = as_2d(X)
        b = np.asarray(beta, dtype=float).reshape(-1)
        _, a = self._alpha_at(X_, b)
        Phi = np.asarray(self.basis(X_), dtype=float)
        M = np.asarray(self.functional.m_basis_matrix(X_, self.basis), dtype=float)
        return (a[:, None] * Phi - M).mean(axis=0)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self,
        X: ArrayLike,
        *,
        beta0: ArrayLike | None = None,
        max_iter: int = 500,
        max_halvings: int = 60,
        tol: float = 1e-10,
        armijo: float = 1e-4,
        margin_shrink: float = 0.01,
    ) -> GeneralLinkFitResult:
        """Damped (modified) Newton from ``beta0`` (default zeros).

        Stops with ``"ok"`` when ``max_j |dF/dbeta_j| <= tol`` at an interior
        point with a positive-definite Hessian (``"singular"`` if the Hessian is
        not positive definite there). Indefinite Hessians along the way are
        shifted to positive definiteness (counted in ``n_hessian_shifts``).
        Backtracking keeps the training rows and every row evaluated by ``m``
        in the domain of ``g``.
        """

        X_ = as_2d(X)
        _ensure_basis_fitted(self.basis, X_)
        p = int(np.asarray(self.basis(X_[:1]), dtype=float).shape[1])
        beta = np.zeros(p) if beta0 is None else np.asarray(beta0, dtype=float).reshape(-1)
        if beta.shape != (p,):
            raise ValueError(f"beta0 must have length {p}.")

        def margin(ev: _Eval) -> NDArray[np.float64]:
            fn = getattr(self.generator, "boundary_margin", None)
            if not callable(fn):
                return np.full(len(X_), np.inf)
            return np.asarray(fn(X_, ev.alpha), dtype=float)

        ev = self._evaluate(X_, beta)
        if not ev.feasible:
            return self._finish(X_, ev, beta, INFEASIBLE_START,
                                "the starting point is outside the domain", 0, 0, float("nan"))
        if not ev.finite:
            return self._finish(X_, ev, beta, NONFINITE, "non-finite objective at the start",
                                0, 0, float("nan"))

        it = 0
        shifts = 0
        hmin = float("nan")
        while True:
            assert ev.grad is not None and ev.alpha is not None and ev.u is not None
            nb = int(np.sum(self.generator.boundary_mask(X_, ev.u, ev.alpha,
                                                        tol=self.boundary_tol)))
            if nb > 0:
                return self._finish(X_, ev, ev.beta, BOUNDARY,
                                    f"{nb} row(s) at the domain boundary", it, shifts, hmin)
            with np.errstate(over="ignore", invalid="ignore"):
                H = self.hessian(X_, ev.beta)
            if not np.all(np.isfinite(H)):
                return self._finish(X_, ev, ev.beta, NONFINITE, "non-finite Hessian", it,
                                    shifts, hmin)
            evals, evecs = np.linalg.eigh(H)
            hmin = float(evals.min())
            emax = float(np.abs(evals).max())
            gnorm = float(np.max(np.abs(ev.grad)))
            if gnorm <= tol:
                if hmin > 1e-12 * max(1.0, emax):
                    return self._finish(X_, ev, ev.beta, OK, "", it, shifts, hmin)
                return self._finish(X_, ev, ev.beta, SINGULAR,
                                    f"stationary point with Hessian min eig {hmin:.3e}",
                                    it, shifts, hmin)
            if it >= max_iter:
                return self._finish(X_, ev, ev.beta, MAXIT, f"reached max_iter={max_iter}",
                                    it, shifts, hmin)
            floor = 1e-8 * max(1.0, emax)
            if hmin <= floor:
                evals = evals + (floor - hmin)
                shifts += 1
            d = evecs @ ((evecs.T @ ev.grad) / evals)
            slope = float(ev.grad @ d)
            step = 1.0
            accepted = None
            m0 = margin(ev)
            fin = np.isfinite(m0)
            for _ in range(max_halvings + 1):
                trial = self._evaluate(X_, ev.beta - step * d)
                if trial.feasible and trial.finite:
                    m1 = margin(trial)
                    ok_margin = bool(np.all(m1[fin] >= margin_shrink * m0[fin]))
                    decrease = trial.F <= ev.F - armijo * step * slope
                    assert trial.grad is not None
                    roundoff = (abs(trial.F - ev.F) <= 8.0 * _EPS * max(1.0, abs(ev.F))
                                and float(np.max(np.abs(trial.grad))) < gnorm)
                    if ok_margin and (decrease or roundoff):
                        accepted = trial
                        break
                step *= 0.5
            it += 1
            if accepted is None:
                return self._finish(X_, ev, ev.beta, LINESEARCH,
                                    f"no acceptable step after {max_halvings} halvings",
                                    it, shifts, hmin)
            ev = accepted

    def _finish(self, X_, ev: _Eval, beta, status: str, message: str, it: int, shifts: int,
                hmin: float) -> GeneralLinkFitResult:
        beta = np.asarray(beta, dtype=float)
        if ev.feasible and ev.grad is not None:
            tang = ev.grad - self.lam * beta
            res = GeneralLinkFitResult(
                beta=beta,
                status=status,
                message=message or status,
                n_iter=it,
                objective_value=ev.F,
                tangent_imbalance=float(np.max(np.abs(tang))),
                regressor_imbalance=float(np.max(np.abs(self.regressor_imbalance(X_, beta)))),
                kkt_residual=float(np.max(np.abs(ev.grad))),
                hessian_min_eig=hmin,
                n_hessian_shifts=shifts,
                n_boundary=int(np.sum(self.generator.boundary_mask(
                    X_, ev.u, ev.alpha, tol=self.boundary_tol))),
                gradient=ev.grad,
            )
        else:
            res = GeneralLinkFitResult(beta=beta, status=status, message=message or status,
                                       n_iter=it)
        self.beta_ = beta if status == OK else None
        self.fit_result_ = res
        return res

    def classify(
        self, X: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
        """``(alpha, outside, nonfinite)`` of the fitted representer at ``X``.

        ``outside``: a finite value outside the open domain of ``g`` or on the
        wrong branch; ``nonfinite``: a non-finite link value. ``alpha`` is NaN on
        both.
        """

        if self.beta_ is None:
            raise RuntimeError("Model is not fit.")
        X_ = as_2d(X)
        with np.errstate(over="ignore", invalid="ignore"):
            _, a = self._alpha_at(X_, self.beta_)
        nonfinite = ~np.isfinite(a)
        outside = np.zeros(a.shape[0], dtype=bool)
        fin = ~nonfinite
        if np.any(fin):
            outside[fin] = ~np.asarray(self.generator.alpha_domain(X_[fin], a[fin]), bool)
        alpha = np.where(outside | nonfinite, np.nan, a)
        return alpha, outside, nonfinite

    def domain_mask(self, X: ArrayLike) -> NDArray[np.bool_]:
        """Rows at which the fitted representer is finite and in the domain of ``g``."""

        _, outside, nonfinite = self.classify(X)
        return ~(outside | nonfinite)

    def predict_alpha(self, X: ArrayLike, *, out_of_domain: str = "raise") -> NDArray[np.float64]:
        """``alpha(x) = link(x, v_ref(x) + phi(x)' beta)``, validated.

        ``out_of_domain="raise"`` (default) raises :class:`DomainError` when a
        row is outside the domain of ``g`` (including the wrong sign branch) or
        non-finite; ``"nan"`` returns NaN there. Raises if not fitted
        successfully.
        """

        if out_of_domain not in {"raise", "nan"}:
            raise ValueError("out_of_domain must be 'raise' or 'nan'")
        alpha, outside, nonfinite = self.classify(X)
        if out_of_domain == "raise" and (np.any(outside) or np.any(nonfinite)):
            raise DomainError(
                f"{int(np.sum(outside))}/{alpha.shape[0]} row(s) are outside the domain of "
                f"generator '{self.generator.name}' and {int(np.sum(nonfinite))} are non-finite."
            )
        return alpha
