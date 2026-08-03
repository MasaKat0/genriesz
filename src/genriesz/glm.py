"""GLM-style solvers used by Generalized Riesz Regression.

The library uses a simple finite-dimensional model:

    v(x) = phi(x)^T beta,

and a generator-specific link that maps the linear predictor ``v`` to a Riesz
representer ``alpha(x)``.

The GRR objective for beta can be written as

    min_beta  E[ g*(v(X)) - m(X, v) ] + penalty(beta),

where ``g*`` is the convex conjugate of the generator and ``m(X,v)`` is linear
in ``v``.

We solve the resulting convex (often smooth) problem with L-BFGS-B.
"""

from __future__ import annotations

import contextlib
import re
import time
import warnings
from contextlib import AbstractContextManager
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

from .basis import Basis
from .functionals import LinearFunctional
from .generators import BregmanGenerator, SquaredGenerator
from .generators import (
    DomainError as DomainError,  # noqa: F401  # re-exported: public API kept after the move to generators
)
from .utils import as_1d_of_length, as_2d, sigmoid


def _branch_cache_of(generator: object) -> AbstractContextManager[None]:
    """Memoize a generator's branch signs, if it knows how (duck-typed)."""

    cache = getattr(generator, "branch_cache", None)
    return cache() if callable(cache) else contextlib.nullcontext()

# ``DomainError`` is now defined in ``generators`` (the lower layer, which raises
# it directly from broken links). glm.py both uses it (in ``fit``) and re-exports
# it, so ``from genriesz.glm import DomainError`` keeps working.


def _ensure_basis_fitted(
    basis: Basis,
    X: NDArray[np.float64],
    y: NDArray[np.float64] | None = None,
    *,
    fit_basis: bool = True,
) -> None:
    """Fit ``basis`` on the solver's training observations when requested."""

    if not fit_basis:
        return
    if y is None:
        basis.fit(X)
    else:
        basis.fit(X, y)


@dataclass(frozen=True)
class _StationaritySolution:
    beta: NDArray[np.float64]
    success: bool
    message: str


def _solve_squared_stationarity(
    matrix: NDArray[np.float64], right_hand_side: NDArray[np.float64]
) -> _StationaritySolution:
    """Solve a positive-semidefinite stationarity equation without substitution."""

    A = np.asarray(matrix, dtype=float)
    b = np.asarray(right_hand_side, dtype=float).reshape(-1)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] != b.size:
        raise ValueError("The stationarity system has incompatible dimensions.")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("The stationarity system must contain finite values.")

    symmetric = 0.5 * (A + A.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    spectral_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    negative_tolerance = 100.0 * np.finfo(float).eps * spectral_scale
    if float(np.min(eigenvalues)) < -negative_tolerance:
        return _StationaritySolution(
            beta=np.zeros_like(b),
            success=False,
            message="The squared-loss stationarity matrix is not positive semidefinite.",
        )

    rank_tolerance = 1e-10 * max(float(np.max(eigenvalues)), np.finfo(float).tiny)
    keep = eigenvalues > rank_tolerance
    coordinates = eigenvectors.T @ b
    null_coordinates = coordinates[~keep]
    null_tolerance = 1e-10 * max(float(np.linalg.norm(b)), 1.0)
    if null_coordinates.size and float(np.linalg.norm(null_coordinates)) > null_tolerance:
        return _StationaritySolution(
            beta=np.zeros_like(b),
            success=False,
            message=(
                "The unpenalized squared-loss objective is unbounded below: "
                "the right-hand side has a component outside the range of the "
                "stationarity matrix."
            ),
        )

    beta = np.zeros_like(b)
    if np.any(keep):
        beta = eigenvectors[:, keep] @ (coordinates[keep] / eigenvalues[keep])
    residual = symmetric @ beta - b
    residual_tolerance = 1e-8 * max(float(np.linalg.norm(b)), 1.0)
    if not np.all(np.isfinite(beta)) or float(np.linalg.norm(residual)) > residual_tolerance:
        return _StationaritySolution(
            beta=np.zeros_like(b),
            success=False,
            message="The squared-loss stationarity equation could not be solved reliably.",
        )
    return _StationaritySolution(
        beta=np.asarray(beta, dtype=float), success=True, message="closed_form"
    )


@dataclass(frozen=True)
class _LinearDualDomain:
    """Observationwise linear bounds for a generator's exact dual domain."""

    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    initial_beta: NDArray[np.float64]


def _exact_linear_dual_domain(
    generator: BregmanGenerator,
    X: NDArray[np.float64],
    Phi: NDArray[np.float64],
    *,
    margin: float,
) -> _LinearDualDomain | None:
    """Return linear bounds that keep an exact branchwise link representable.

    Let ``u_i = s_i v_i``, where ``s_i`` is fixed by the estimand. The exact
    UKL, BKL, BP, and PU links impose an interval on ``u_i``. Since
    ``v = Phi beta``, the interval gives observationwise linear constraints on
    ``beta``. The numerical margin moves the constraints into the open domain;
    it does not replace an invalid link value.
    """

    if generator.branch_fn is None:
        return None
    interval = generator.signed_dual_interval(margin=margin)
    if interval is None:
        return None

    sign = np.asarray(generator._sign(X, np.zeros(X.shape[0])), dtype=float)
    lower_u, upper_u, target_u = interval

    if not lower_u < upper_u:
        return _LinearDualDomain(
            lower=np.full(X.shape[0], np.nan),
            upper=np.full(X.shape[0], np.nan),
            initial_beta=np.full(Phi.shape[1], np.nan),
        )

    lower = np.where(sign > 0.0, lower_u, -upper_u).astype(float)
    upper = np.where(sign > 0.0, upper_u, -lower_u).astype(float)
    target = sign * target_u

    initial_beta = np.linalg.lstsq(Phi, target, rcond=None)[0]
    values = Phi @ initial_beta
    feasible = bool(np.all(values >= lower) and np.all(values <= upper))
    if not feasible:
        rows: list[NDArray[np.float64]] = []
        bounds: list[float] = []
        finite_upper = np.isfinite(upper)
        if np.any(finite_upper):
            rows.extend(Phi[finite_upper])
            bounds.extend(upper[finite_upper])
        finite_lower = np.isfinite(lower)
        if np.any(finite_lower):
            rows.extend(-Phi[finite_lower])
            bounds.extend(-lower[finite_lower])
        phase = optimize.linprog(
            np.zeros(Phi.shape[1], dtype=float),
            A_ub=np.asarray(rows, dtype=float),
            b_ub=np.asarray(bounds, dtype=float),
            bounds=[(None, None)] * Phi.shape[1],
            method="highs",
        )
        if not phase.success or phase.x is None:
            return _LinearDualDomain(
                lower=lower,
                upper=upper,
                initial_beta=np.full(Phi.shape[1], np.nan),
            )
        initial_beta = np.asarray(phase.x, dtype=float)

    return _LinearDualDomain(lower=lower, upper=upper, initial_beta=initial_beta)


def _linear_constraint_kkt_residual(
    gradient: NDArray[np.float64],
    beta: NDArray[np.float64],
    matrix: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    *,
    tolerance: float,
) -> float:
    """KKT residual for ``lower <= matrix @ beta <= upper``."""

    values = matrix @ beta
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    lower_slack = values[finite_lower] - lower[finite_lower]
    upper_slack = upper[finite_upper] - values[finite_upper]
    primal = 0.0
    if lower_slack.size:
        primal = max(primal, float(np.max(np.maximum(-lower_slack, 0.0))))
    if upper_slack.size:
        primal = max(primal, float(np.max(np.maximum(-upper_slack, 0.0))))

    active_tolerance = max(1e-7, 10.0 * float(tolerance))
    normals: list[NDArray[np.float64]] = []
    slacks: list[float] = []
    lower_rows = matrix[finite_lower]
    for row, slack in zip(lower_rows, lower_slack, strict=True):
        if slack <= active_tolerance:
            normals.append(-row)
            slacks.append(float(slack))
    upper_rows = matrix[finite_upper]
    for row, slack in zip(upper_rows, upper_slack, strict=True):
        if slack <= active_tolerance:
            normals.append(row)
            slacks.append(float(slack))

    if normals:
        normal_matrix = np.asarray(normals, dtype=float)
        multipliers, _ = optimize.nnls(normal_matrix.T, -gradient)
        stationarity = gradient + normal_matrix.T @ multipliers
        complementarity = float(
            np.max(np.abs(multipliers * np.asarray(slacks, dtype=float)))
        )
    else:
        stationarity = gradient
        complementarity = 0.0
    stationarity_residual = float(np.max(np.abs(stationarity))) if stationarity.size else 0.0
    return max(primal, complementarity, stationarity_residual)


@dataclass
class FitResult:
    """Result of a nuisance optimization.

    Attributes
    ----------
    beta, success, message, n_iter:
        Solution, optimizer success flag, optimizer message, iteration count.
        On every failure path the model itself stays unpredictable
        (``beta_ is None``); ``beta`` then holds the last iterate (or the
        initial point when no solution was ever computed) for diagnostics and
        shape introspection only.
    status:
        One of ``"closed_form"``, ``"converged"``, ``"optimizer_failure"``,
        ``"domain_error"``, ``"domain_error_at_solution"``,
        ``"degenerate_functional"`` (the functional's basis evaluations are
        identically zero on the training data, e.g. an ATT fold with no treated
        unit, so the Riesz problem has no information to fit), ``"singular"``
        (the closed-form system is numerically rank-deficient -- usually an
        unpenalized path -- and has no stationary point that can be stood behind:
        either none exists and the objective is unbounded below, or reaching it
        would mean dividing by an eigenvalue at the numerical rank threshold.
        ``message`` says which), or ``""`` for results produced by callers that do
        not set it.
    objective_value:
        Penalized objective evaluated at ``beta``.
    gradient_norm:
        Infinity norm of the full (loss + penalty) gradient at ``beta``.
    kkt_residual:
        Stationarity residual. Equal to ``gradient_norm`` for smooth
        penalties; for l1 it is the exact subgradient residual.
    clip_binding_rate:
        Fraction of observations at which a bounded link or the numerical
        margin of an exact open dual domain is active (``nan`` when not
        applicable). The field name is retained for API compatibility; an
        exact-domain fit does not replace an invalid link value by a cap.
    fit_time:
        Wall-clock seconds spent in ``fit``.
    """

    beta: NDArray[np.float64]
    success: bool
    message: str
    n_iter: int
    status: str = ""
    objective_value: float = field(default=float("nan"))
    gradient_norm: float = field(default=float("nan"))
    kkt_residual: float = field(default=float("nan"))
    clip_binding_rate: float = field(default=float("nan"))
    fit_time: float = field(default=float("nan"))


class _Penalty:
    def __init__(self, penalty: str | None, lam: float, p_norm: float | None):
        self.penalty = None if penalty is None else str(penalty).lower()
        self.lam = float(lam)
        self.p_norm = 2.0 if p_norm is None else float(p_norm)
        if self.lam < 0:
            raise ValueError("lam must be >= 0")

        # Convenience shorthand: allow strings like "l1.5" to mean an l_p penalty.
        if (
            self.penalty is not None
            and re.fullmatch(r"l(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)", self.penalty)
            and self.penalty not in {"l1", "l2"}
        ):
            self.p_norm = float(self.penalty[1:])
            self.penalty = "lp"
        if self.penalty in {"l1", "lasso"}:
            self.p_norm = 1.0
        elif self.penalty in {"l2", "ridge"}:
            self.p_norm = 2.0
        elif self.penalty in {"lp", "l_p", "p"}:
            if self.p_norm < 1.0:
                raise ValueError("p_norm must be >= 1")
        elif self.penalty in {None, "none", ""}:
            self.penalty = None
        else:
            raise ValueError(f"Unknown penalty: {penalty}")

        # Smoothing for the l1 gradient (subgradient at 0)
        self._eps = 1e-8

    def value(self, beta: NDArray[np.float64]) -> float:
        if self.penalty is None or self.lam == 0.0:
            return 0.0
        if self.p_norm == 1.0:
            # Keep the objective differentiable in the same way as grad().
            # L-BFGS-B expects a gradient that is consistent with the objective.
            # The additive constant sqrt(eps) is irrelevant for optimisation, so
            # we do not subtract it.
            return float(self.lam * np.sum(np.sqrt(beta * beta + self._eps)))
        return float((self.lam / self.p_norm) * np.sum(np.abs(beta) ** self.p_norm))

    def grad(self, beta: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.penalty is None or self.lam == 0.0:
            return np.zeros_like(beta)
        if self.p_norm == 1.0:
            return self.lam * beta / np.sqrt(beta * beta + self._eps)
        return self.lam * np.sign(beta) * (np.abs(beta) ** (self.p_norm - 1.0))


class GRRGLM:
    """Finite-dimensional generalized Riesz regression (GLM form)."""

    def __init__(
        self,
        *,
        basis: Basis,
        generator: BregmanGenerator,
        functional: LinearFunctional,
        penalty: str | None = "l2",
        lam: float = 1e-3,
        p_norm: float | None = None,
    ):
        self.basis = basis
        self.generator = generator
        self.functional = functional
        self.penalty = _Penalty(penalty, lam, p_norm)

        self._Phi: NDArray[np.float64] | None = None
        self._M: NDArray[np.float64] | None = None
        self.beta_: NDArray[np.float64] | None = None
        self.fit_result_: FitResult | None = None

    def fit(
        self,
        X: ArrayLike,
        *,
        beta0: ArrayLike | None = None,
        max_iter: int = 500,
        tol: float = 1e-8,
        verbose: bool = False,
        fit_basis: bool = True,
    ) -> FitResult:
        t0 = time.perf_counter()
        X_ = as_2d(X)
        _ensure_basis_fitted(self.basis, X_, fit_basis=fit_basis)
        Phi = np.asarray(self.basis(X_), dtype=float)
        M = np.asarray(self.functional.m_basis_matrix(X_, self.basis), dtype=float)

        n, p = Phi.shape
        if M.shape != (n, p):
            raise ValueError(
                f"m_basis_matrix returned shape {M.shape}, expected {(n, p)}."
            )

        # A functional whose basis evaluations vanish identically on this
        # training data (e.g. an ATT/DID M-matrix on a fold with no treated
        # unit, or an AME derivative of a piecewise-constant basis) makes the
        # Riesz problem degenerate: the "solution" is an artifact of the
        # penalty alone (beta = 0 for the closed form), yet it would be
        # reported as a successful fit and produce alpha_hat = const with a
        # deceptively tight downstream CI (audit EST-07 / K-01).
        if M.size and not np.any(M):
            out = FitResult(
                beta=np.zeros(p, dtype=float),
                success=False,
                message=(
                    "m_basis_matrix(X) is identically zero on this training "
                    "data, so the Riesz problem is degenerate (for a "
                    "treatment-type functional this typically means the "
                    "training fold contains no treated unit)."
                ),
                n_iter=0,
                status="degenerate_functional",
                fit_time=time.perf_counter() - t0,
            )
            self.beta_ = None
            self.fit_result_ = out
            self._Phi = None
            self._M = None
            return out

        if beta0 is None:
            beta0_ = np.zeros(p, dtype=float)
        else:
            beta0_ = np.asarray(beta0, dtype=float).reshape(-1)
            if beta0_.shape[0] != p:
                raise ValueError(f"beta0 must have length {p}. Got {beta0_.shape}.")

        # Closed form for the squared generator with an L2 (or no) penalty.
        # g(alpha) = (alpha - C)^2 gives g*(v) = C v + v^2/4, so the objective
        # is quadratic and the stationarity condition is
        #     (0.5 Phi'Phi/n + lam I) beta = mean(M) - C mean(Phi).
        if isinstance(self.generator, SquaredGenerator) and (
            self.penalty.penalty is None or self.penalty.p_norm == 2.0
        ):
            lam = self.penalty.lam if self.penalty.penalty is not None else 0.0
            A = 0.5 * (Phi.T @ Phi) / n + lam * np.eye(p)
            b = M.mean(axis=0) - self.generator.C * Phi.mean(axis=0)
            # Unlike a least-squares normal equation, b = mean(M) - C mean(Phi)
            # need not lie in the range of A, so an unpenalized rank-deficient
            # fit can have no stationary point at all. Report that as a failure
            # instead of passing off lstsq's finite non-solution as a fit.
            stationarity = _solve_squared_stationarity(A, b)
            if not stationarity.success:
                out = FitResult(
                    beta=beta0_,
                    success=False,
                    message=stationarity.message,
                    n_iter=0,
                    status="singular",
                    fit_time=time.perf_counter() - t0,
                )
                self.beta_ = None
                self.fit_result_ = out
                self._Phi = None
                self._M = None
                return out
            return self._finalize_fit(
                X_,
                Phi,
                M,
                stationarity.beta,
                success=True,
                message="closed_form",
                status="closed_form",
                n_iter=1,
                t0=t0,
            )


        # Invalid generator evaluations return NaN, so the optimizer reports
        # failure. No finite objective or gradient is substituted for an
        # undefined link.
        def fun(beta: NDArray[np.float64]) -> float:
            v = Phi @ beta
            evaluated = self.generator.conjugate_status(X_, v)
            if not bool(np.all(evaluated.valid)):
                return float("nan")
            loss = float(np.mean(evaluated.conjugate - (M @ beta)))
            return loss + self.penalty.value(beta)

        def jac(beta: NDArray[np.float64]) -> NDArray[np.float64]:
            v = Phi @ beta
            evaluated = self.generator.conjugate_status(X_, v)
            if not bool(np.all(evaluated.valid)):
                return np.full(Phi.shape[1], np.nan, dtype=float)
            grad = (evaluated.alpha[:, None] * Phi - M).mean(axis=0)
            return grad + self.penalty.grad(beta)

        opts: dict = {"maxiter": int(max_iter), "ftol": float(tol)}
        if verbose:
            opts["disp"] = True

        # The branch signs depend on X only. The same signs determine the
        # exact linear dual domain and enter every objective evaluation, so one
        # cache must cover both operations.
        with _branch_cache_of(self.generator):
            dual_domain = _exact_linear_dual_domain(
                self.generator,
                X_,
                Phi,
                margin=max(1e-10, 10.0 * float(tol)),
            )
            if dual_domain is not None:
                if not np.all(np.isfinite(dual_domain.initial_beta)):
                    out = FitResult(
                        beta=beta0_,
                        success=False,
                        message=(
                            "The fixed branch and the fitted basis have no common "
                            "linear predictor inside the exact generator domain."
                        ),
                        n_iter=0,
                        status="domain_infeasible",
                        fit_time=time.perf_counter() - t0,
                    )
                    self.beta_ = None
                    self.fit_result_ = out
                    self._Phi = None
                    self._M = None
                    return out
                if beta0 is None:
                    beta0_ = dual_domain.initial_beta
                else:
                    values0 = Phi @ beta0_
                    if np.any(values0 < dual_domain.lower) or np.any(
                        values0 > dual_domain.upper
                    ):
                        raise ValueError("beta0 lies outside the exact generator domain.")

            initial_evaluation = self.generator.conjugate_status(X_, Phi @ beta0_)
            if not bool(np.all(initial_evaluation.valid)):
                out = FitResult(
                    beta=beta0_,
                    success=False,
                    message=(
                        f"generator '{self.generator.name}' is undefined at beta0; "
                        "no optimizer step was taken"
                    ),
                    n_iter=0,
                    status="domain_error",
                    fit_time=time.perf_counter() - t0,
                )
                self.beta_ = None
                self.fit_result_ = out
                self._Phi = None
                self._M = None
                return out

            if dual_domain is None:
                res = optimize.minimize(
                    fun=fun, x0=beta0_, jac=jac, method="L-BFGS-B", options=opts
                )
            else:
                constraint = optimize.LinearConstraint(
                    Phi,
                    dual_domain.lower,
                    dual_domain.upper,
                )
                res = optimize.minimize(
                    fun=fun,
                    x0=beta0_,
                    jac=jac,
                    method="SLSQP",
                    constraints=(constraint,),
                    options=opts,
                )

            beta_hat = np.asarray(res.x, dtype=float)
            return self._finalize_fit(
                X_,
                Phi,
                M,
                beta_hat,
                success=bool(res.success),
                message=str(res.message),
                status="converged" if bool(res.success) else "optimizer_failure",
                n_iter=int(getattr(res, "nit", -1)),
                t0=t0,
                constraint_matrix=Phi if dual_domain is not None else None,
                constraint_lower=dual_domain.lower if dual_domain is not None else None,
                constraint_upper=dual_domain.upper if dual_domain is not None else None,
                tolerance=tol,
            )

    def _finalize_fit(
        self,
        X_: NDArray[np.float64],
        Phi: NDArray[np.float64],
        M: NDArray[np.float64],
        beta_hat: NDArray[np.float64],
        *,
        success: bool,
        message: str,
        status: str,
        n_iter: int,
        t0: float,
        constraint_matrix: NDArray[np.float64] | None = None,
        constraint_lower: NDArray[np.float64] | None = None,
        constraint_upper: NDArray[np.float64] | None = None,
        tolerance: float = 1e-8,
    ) -> FitResult:
        """Compute solution diagnostics and store the fit result."""

        objective = float("nan")
        gradient_norm = float("nan")
        kkt = float("nan")
        binding = float("nan")
        v = Phi @ beta_hat
        evaluated = self.generator.conjugate_status(X_, v)
        if bool(np.all(evaluated.valid)):
            objective = float(np.mean(evaluated.conjugate - (M @ beta_hat)))
            objective += self.penalty.value(beta_hat)
            grad_loss = (evaluated.alpha[:, None] * Phi - M).mean(axis=0)
            grad_total = grad_loss + self.penalty.grad(beta_hat)
            gradient_norm = (
                float(np.max(np.abs(grad_total))) if grad_total.size else 0.0
            )
            if (
                constraint_matrix is not None
                and constraint_lower is not None
                and constraint_upper is not None
            ):
                kkt = _linear_constraint_kkt_residual(
                    grad_total,
                    beta_hat,
                    constraint_matrix,
                    constraint_lower,
                    constraint_upper,
                    tolerance=tolerance,
                )
            else:
                kkt = self._kkt_residual(grad_loss, beta_hat)
            binding_fn = getattr(self.generator, "domain_binding", None)
            if callable(binding_fn):
                bind = np.asarray(binding_fn(X_, v), dtype=bool)
                binding = float(np.mean(bind)) if bind.size else 0.0
        else:
            success = False
            status = "domain_error_at_solution"
            message = (
                f"{message} | generator '{self.generator.name}' is undefined "
                "at the reported solution"
            )


        out = FitResult(
            beta=beta_hat,
            success=success,
            message=message,
            n_iter=n_iter,
            status=status,
            objective_value=objective,
            gradient_norm=gradient_norm,
            kkt_residual=kkt,
            clip_binding_rate=binding,
            fit_time=time.perf_counter() - t0,
        )
        # Only a successful fit is allowed to predict (audit P0-07): an
        # optimizer that hit max_iter or failed its diagnostics at the last
        # iterate must not leave a silently predictable state behind. The
        # iterate itself stays available on ``fit_result_.beta``.
        self.beta_ = beta_hat if success else None
        self.fit_result_ = out
        # Do not keep the (n, p) design matrices alive on the fitted object.
        self._Phi = None
        self._M = None
        return out

    def _kkt_residual(
        self, grad_loss: NDArray[np.float64], beta: NDArray[np.float64]
    ) -> float:
        pen = self.penalty
        if beta.size == 0:
            return 0.0
        if pen.penalty is None or pen.lam == 0.0:
            return float(np.max(np.abs(grad_loss)))
        if pen.p_norm == 1.0:
            nz = beta != 0.0
            resid = np.where(
                nz,
                np.abs(grad_loss + pen.lam * np.sign(beta)),
                np.maximum(0.0, np.abs(grad_loss) - pen.lam),
            )
            return float(np.max(resid))
        return float(np.max(np.abs(grad_loss + pen.grad(beta))))

    def predict_v(self, X: ArrayLike) -> NDArray[np.float64]:
        if self.beta_ is None:
            raise RuntimeError("Model is not fit.")
        Phi = np.asarray(self.basis(as_2d(X)), dtype=float)
        return Phi @ self.beta_

    def predict_alpha(self, X: ArrayLike) -> NDArray[np.float64]:
        v = self.predict_v(X)
        return self.generator.inv_grad(as_2d(X), v)

    def derivative_alpha(self, X: ArrayLike, coordinate: int) -> NDArray[np.float64]:
        """Derivative of alpha(x) wrt x_coordinate.

        Uses the identity grad_g(alpha(x)) = v(x) and the inverse function theorem:

            g''(alpha) * d alpha/dx = d v/dx.
        """

        if self.beta_ is None:
            raise RuntimeError("Model is not fit.")
        X_ = as_2d(X)
        dPhi = self.basis.derivative(X_, coordinate)
        dv = dPhi @ self.beta_
        alpha = self.predict_alpha(X_)
        g2 = np.asarray(self.generator.grad2(X_, alpha), dtype=float)
        bad = ~np.isfinite(g2) | (g2 <= 0.0)
        if np.any(bad):
            warnings.warn(
                "derivative_alpha encountered non-positive or non-finite curvature "
                f"g''(alpha) for {int(bad.sum())} observation(s); returning NaN there.",
                RuntimeWarning,
                stacklevel=2,
            )
        out = np.full_like(dv, np.nan, dtype=float)
        ok = ~bad
        out[ok] = dv[ok] / g2[ok]
        return out


class OutcomeGLM:
    """Simple (penalized) outcome regression on top of a basis."""

    def __init__(
        self,
        *,
        basis: Basis,
        link: str = "identity",
        penalty: str | None = "l2",
        lam: float = 1e-3,
        p_norm: float | None = None,
    ):
        self.basis = basis
        self.link = str(link).lower()
        if self.link not in {"identity", "logit"}:
            raise ValueError("link must be 'identity' or 'logit'")
        self.penalty = _Penalty(penalty, lam, p_norm)

        self.theta_: NDArray[np.float64] | None = None

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        theta0: ArrayLike | None = None,
        max_iter: int = 500,
        tol: float = 1e-8,
        verbose: bool = False,
        fit_basis: bool = True,
    ) -> FitResult:
        X_ = as_2d(X)
        y_ = as_1d_of_length(y, n=X_.shape[0], name="y")
        _ensure_basis_fitted(self.basis, X_, y_, fit_basis=fit_basis)
        Phi = np.asarray(self.basis(X_), dtype=float)
        n, p = Phi.shape
        if y_.shape[0] != n:
            raise ValueError(f"y must have length {n}. Got {y_.shape}.")

        if theta0 is None:
            theta0_ = np.zeros(p, dtype=float)
        else:
            theta0_ = np.asarray(theta0, dtype=float).reshape(-1)
            if theta0_.shape[0] != p:
                raise ValueError(f"theta0 must have length {p}. Got {theta0_.shape}.")

        # Closed form for identity + l2 (ridge)
        if self.link == "identity" and self.penalty.penalty in {"l2", "ridge"}:
            if self.penalty.lam == 0.0:
                # Ordinary least squares (with pseudo-inverse)
                theta = np.linalg.pinv(Phi) @ y_
            elif p > n:
                # Dual (kernel ridge / Woodbury) form: O(n^3) instead of O(p^3)
                K = (Phi @ Phi.T) / n
                theta = (Phi.T @ np.linalg.solve(K + self.penalty.lam * np.eye(n), y_)) / n
            else:
                A = (Phi.T @ Phi) / n + self.penalty.lam * np.eye(p)
                b = (Phi.T @ y_) / n
                theta = np.linalg.solve(A, b)
            self.theta_ = np.asarray(theta, dtype=float)
            return FitResult(
                beta=self.theta_, success=True, message="closed_form", n_iter=1,
                status="closed_form",
            )

        def fun(theta: NDArray[np.float64]) -> float:
            eta = Phi @ theta
            if self.link == "identity":
                resid = y_ - eta
                loss = 0.5 * float(np.mean(resid * resid))
            else:
                # Bernoulli negative log-likelihood
                # mean(log(1+exp(eta)) - y*eta)
                loss = float(np.mean(np.logaddexp(0.0, eta) - y_ * eta))
            return loss + self.penalty.value(theta)

        def jac(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            eta = Phi @ theta
            if self.link == "identity":
                resid = y_ - eta
                grad = -(Phi.T @ resid) / n
            else:
                p_hat = sigmoid(eta)
                grad = (Phi.T @ (p_hat - y_)) / n
            return grad + self.penalty.grad(theta)

        opts_: dict = {"maxiter": int(max_iter), "ftol": float(tol)}
        if verbose:
            opts_["iprint"] = 1
        res = optimize.minimize(fun=fun, x0=theta0_, jac=jac, method="L-BFGS-B", options=opts_)

        theta_hat = np.asarray(res.x, dtype=float)
        # Same contract as GRRGLM (audit P0-07): a failed fit must not leave a
        # predictable state behind. The last iterate stays on the FitResult.
        self.theta_ = theta_hat if bool(res.success) else None
        return FitResult(
            beta=theta_hat,
            success=bool(res.success),
            message=str(res.message),
            n_iter=int(getattr(res, "nit", -1)),
            status="converged" if bool(res.success) else "optimizer_failure",
        )

    def predict_link(self, X: ArrayLike) -> NDArray[np.float64]:
        if self.theta_ is None:
            raise RuntimeError("OutcomeGLM is not fit.")
        Phi = np.asarray(self.basis(as_2d(X)), dtype=float)
        return Phi @ self.theta_

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        eta = self.predict_link(X)
        if self.link == "identity":
            return eta
        return sigmoid(eta)

    def derivative(self, X: ArrayLike, coordinate: int) -> NDArray[np.float64]:
        if self.theta_ is None:
            raise RuntimeError("OutcomeGLM is not fit.")
        X_ = as_2d(X)
        dPhi = self.basis.derivative(X_, coordinate)
        deta = dPhi @ self.theta_
        if self.link == "identity":
            return deta
        mu = self.predict(X_)
        return mu * (1.0 - mu) * deta
