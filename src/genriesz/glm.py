"""GLM-style solvers used by Generalized Riesz Regression.

The library uses a simple finite-dimensional model:

    v(x) = phi(x)^T beta,

and a generator-specific link that maps the linear predictor ``v`` to a Riesz
representer ``alpha(x)``.

The GRR objective for beta can be written as

    min_beta  E[ g*(v(X)) - m(X, v) ] + penalty(beta),

where ``g*`` is the convex conjugate of the generator and ``m(X,v)`` is linear
in ``v``.

With an offset ``u_ref`` (a fixed function, never fitted on the estimation
sample) the dual coordinate is ``v(x) = u_ref(x) + phi(x)^T beta``.

By default the problem is solved by the strict solvers of
:mod:`genriesz.solvers` (damped Newton for smooth penalties, FISTA for ``l1``),
which keep every iterate in the domain of the generator and report an explicit
status. ``solver="lbfgs"`` selects the L-BFGS-B path of releases <= 0.2.6.
"""

from __future__ import annotations

import contextlib
import time
import warnings
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

from .basis import Basis
from .functionals import LinearFunctional
from .generators import BregmanGenerator, DomainError, SquaredGenerator
from .solvers import (
    NONFINITE,
    OK,
    DualProblem,
    SolverResult,
    fista_solve,
    newton_solve,
)
from .utils import as_1d_of_length, as_2d, sigmoid, solve_stationarity

#: Accepted values of ``GRRGLM(solver=...)``.
GRR_SOLVERS = ("auto", "newton", "fista", "lbfgs")

OffsetSpec = Callable[[NDArray[np.float64]], ArrayLike] | float | None


def evaluate_offset(offset: OffsetSpec, X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate an offset specification ``u_ref`` at the rows of ``X``.

    ``None`` means no offset (zeros), a number is a constant offset, and a
    callable ``offset(X)`` must return one finite value per row. The offset must
    be a fixed function: it may not be fitted on the sample used to estimate
    ``beta`` (it is evaluated, unchanged, at held-out and counterfactual rows).
    """

    n = X.shape[0]
    if offset is None:
        return np.zeros(n, dtype=float)
    if callable(offset):
        out = np.asarray(offset(X), dtype=float)
        if out.ndim == 2 and out.shape[1] == 1:
            out = out[:, 0]
        if out.shape != (n,):
            raise ValueError(f"offset(X) must return shape ({n},). Got {out.shape}.")
        return out
    val = float(offset)  # type: ignore[arg-type]
    return np.full(n, val, dtype=float)


def offset_derivative(
    offset: OffsetSpec, X: NDArray[np.float64], coordinate: int
) -> NDArray[np.float64]:
    """Derivative of the offset in ``X[:, coordinate]``.

    Zero for ``None`` and constant offsets. A callable offset must expose
    ``offset.derivative(X, coordinate)``; otherwise :class:`NotImplementedError`
    is raised (the derivative is needed only by derivative functionals such as
    the AME, when ``m(alpha)`` is evaluated).
    """

    if offset is None or not callable(offset):
        return np.zeros(X.shape[0], dtype=float)
    deriv = getattr(offset, "derivative", None)
    if not callable(deriv):
        raise NotImplementedError(
            "This offset has no derivative(X, coordinate) method, which is needed to "
            "differentiate the fitted representer."
        )
    return np.asarray(deriv(X, coordinate), dtype=float).reshape(-1)


def offset_from_alpha(
    generator: BregmanGenerator,
    alpha_ref: Callable[[NDArray[np.float64]], ArrayLike],
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Offset ``u_ref = g'(alpha_ref)`` for a fixed reference representer.

    ``alpha_ref(X)`` returns the reference values row-wise, for example
    ``lambda X: np.where(X[:, 0] == 1, 2.0, -2.0)`` for ``alpha_ref = +-2`` in an
    ATE problem. The coefficient vector ``beta = 0`` then reproduces
    ``alpha_ref``, toward which a penalty shrinks. The result has a
    ``derivative`` method only when ``alpha_ref`` has one.
    """

    def _offset(X: NDArray[np.float64]) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = np.asarray(alpha_ref(X_), dtype=float).reshape(-1)
        return np.asarray(generator.grad(X_, a), dtype=float)

    return _offset


def _branch_cache_of(generator: object) -> AbstractContextManager[None]:
    """Memoize a generator's branch signs, if it knows how (duck-typed)."""

    cache = getattr(generator, "branch_cache", None)
    return cache() if callable(cache) else contextlib.nullcontext()

# ``DomainError`` is now defined in ``generators`` (the lower layer, which raises
# it directly from broken links). glm.py both uses it (in ``fit``) and re-exports
# it, so ``from genriesz.glm import DomainError`` keeps working.


def _ensure_basis_fitted(basis: Basis, X: NDArray[np.float64]) -> None:
    """Fit ``basis`` on the training data if it has not been fit yet.

    Stateful bases raise when ``n_features`` is accessed before ``fit``; in
    that case fitting on the solver's own training sample is the correct
    (leakage-free) default. Already-fitted bases are left untouched so that
    cross-fitting callers keep control over what data the basis sees.
    """

    try:
        _ = basis.n_features
    except Exception:
        basis.fit(X)


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
        Strict solvers (``solver`` in ``{"auto", "newton", "fista"}``): one of
        :data:`genriesz.solvers.STATUSES` -- ``"ok"``, ``"boundary"``,
        ``"maxit"``, ``"linesearch"``, ``"singular"``, ``"infeasible_start"``,
        ``"nonfinite"``, ``"degenerate_functional"``. Only ``"ok"`` is a
        success. Legacy ``solver="lbfgs"``:
        one of ``"closed_form"``, ``"converged"``, ``"optimizer_failure"``,
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
        Fraction of observations for which the generator's internal domain
        clip was active at the solution (``nan`` when not applicable; ``0.0``
        for the strict solvers, which never clip).
    fit_time:
        Wall-clock seconds spent in ``fit``.
    solver:
        ``"newton"``, ``"fista"``, ``"lbfgs"`` or ``"closed_form"``.
    max_abs_imbalance:
        ``max_j |Delta_j|`` on the fitting sample, where ``Delta`` is the loss
        gradient (the empirical Riesz imbalance of the regressors).
    n_boundary:
        Rows numerically at the domain boundary at ``beta`` (strict solvers).
    l1_ball_active, ball_multiplier:
        Whether the ``l1``-ball restriction is active at the solution and the
        multiplier of its normal cone used in ``kkt_residual``.
    polished:
        Whether a FISTA solution was refined by Newton on its support.
    hessian_min_eig:
        Smallest eigenvalue of the Hessian at the last Newton iterate.
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
    solver: str = "lbfgs"
    max_abs_imbalance: float = field(default=float("nan"))
    n_boundary: int = 0
    l1_ball_active: bool = False
    ball_multiplier: float = 0.0
    polished: bool = False
    hessian_min_eig: float = field(default=float("nan"))


class _Penalty:
    def __init__(self, penalty: str | None, lam: float, p_norm: float | None):
        self.penalty = None if penalty is None else str(penalty).lower()
        self.lam = float(lam)
        self.p_norm = 2.0 if p_norm is None else float(p_norm)
        if self.lam < 0:
            raise ValueError("lam must be >= 0")

        # Convenience shorthand: allow strings like "l1.5" to mean an l_p penalty.
        # This keeps the public API concise while still supporting general l_p.
        if (
            self.penalty is not None
            and self.penalty.startswith("l")
            and self.penalty not in {"l1", "l2", "lp", "l_p"}
        ):
            try:
                p = float(self.penalty[1:])
                self.penalty = "lp"
                self.p_norm = p
            except Exception:
                # Fall through to the standard parser below.
                pass
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
    """Finite-dimensional generalized Riesz regression (GLM form).

    The representer model is linear in the dual coordinate,
    ``v(x) = u_ref(x) + phi(x)^T beta`` and ``alpha(x) = (g')^{-1}(v(x))``, and
    ``beta`` minimizes ``mean[g*(v) - m(W, v)] + penalty(beta)``.

    Parameters
    ----------
    basis, generator, functional:
        Regressors ``phi``, Bregman generator ``g`` and linear functional ``m``.
    penalty, lam, p_norm:
        ``None`` (no penalty), ``"l2"`` (``lam/2 ||beta||^2``), ``"l1"``
        (``lam ||beta||_1``, exact), or ``"lp"`` (``lam/q ||beta||_q^q``).
    offset:
        Offset ``u_ref``: ``None`` (zero), a constant, or a callable ``X -> (n,)``.
        It must be a fixed function -- chosen before estimation or built from
        observations independent of the fitting sample -- because it is
        evaluated unchanged at held-out and counterfactual rows. ``beta = 0``
        corresponds to ``alpha_ref = (g')^{-1}(u_ref)``, toward which the penalty
        shrinks; see :func:`offset_from_alpha`. BKL requires an offset (or a
        starting point) with ``s * u_ref < 0``.
    solver:
        ``"auto"`` (default): damped Newton for no penalty, ``l2`` and
        ``l_q`` with ``q >= 2``; FISTA for ``l1`` and ``l_q`` with ``1 < q < 2``.
        ``"newton"``/``"fista"`` force a solver. ``"lbfgs"``: the L-BFGS-B path
        of releases <= 0.2.6 (closed form for SQ with ``l2``), kept for old
        notebooks; it does not keep iterates in the domain and reports the
        legacy statuses.
    l1_radius:
        Optional coefficient restriction ``||beta||_1 <= l1_radius`` (requires
        the ``l1`` penalty or no penalty; solved by FISTA). Convergence is then
        judged by the constrained residual of
        ``0 in Delta + lam subdiff||beta||_1 + N_B(beta)``.
    boundary_tol:
        Rows with a distance to the domain boundary at most ``boundary_tol``
        (``|alpha| - C`` for UKL/BP/BKL, also ``s v >= -1e-12`` for BKL; never
        for SQ) make the status ``"boundary"``.
    """

    def __init__(
        self,
        *,
        basis: Basis,
        generator: BregmanGenerator,
        functional: LinearFunctional,
        penalty: str | None = "l2",
        lam: float = 1e-3,
        p_norm: float | None = None,
        offset: OffsetSpec = None,
        solver: str = "auto",
        l1_radius: float | None = None,
        boundary_tol: float = 1e-8,
    ):
        self.basis = basis
        self.generator = generator
        self.functional = functional
        self.penalty = _Penalty(penalty, lam, p_norm)
        self.offset = offset
        solver_ = str(solver).lower()
        if solver_ not in GRR_SOLVERS:
            raise ValueError(f"solver must be one of {GRR_SOLVERS}. Got {solver!r}.")
        self.solver = solver_
        if l1_radius is not None:
            l1_radius = float(l1_radius)
            if not np.isfinite(l1_radius) or l1_radius <= 0.0:
                raise ValueError(f"l1_radius must be positive and finite. Got {l1_radius!r}.")
            if self.penalty.penalty is not None and self.penalty.p_norm != 1.0:
                raise ValueError("l1_radius requires penalty='l1' or penalty=None.")
            if solver_ in {"newton", "lbfgs"}:
                raise ValueError("l1_radius is solved by FISTA; use solver='auto' or 'fista'.")
        self.l1_radius = l1_radius
        self.boundary_tol = float(boundary_tol)

        self._Phi: NDArray[np.float64] | None = None
        self._M: NDArray[np.float64] | None = None
        self.beta_: NDArray[np.float64] | None = None
        self.fit_result_: FitResult | None = None

    def _resolve_solver(self) -> str:
        """Pick the strict solver for the penalty (``"newton"`` or ``"fista"``)."""

        pen = self.penalty
        q = None if (pen.penalty is None or pen.lam == 0.0) else pen.p_norm
        smooth = q is None or q >= 2.0
        if self.solver == "newton":
            if not smooth or self.l1_radius is not None:
                raise ValueError(
                    "solver='newton' needs a twice-differentiable penalty (none, l2, or "
                    "lp with p >= 2) and no l1_radius."
                )
            return "newton"
        if self.solver == "fista":
            if q is not None and q >= 2.0:
                raise ValueError("solver='fista' supports the l1 and l_q (1 < q < 2) penalties.")
            return "fista"
        if self.l1_radius is not None or not smooth:
            return "fista"
        return "newton"

    def fit(
        self,
        X: ArrayLike,
        *,
        beta0: ArrayLike | None = None,
        max_iter: int | None = None,
        tol: float | None = None,
        verbose: bool = False,
    ) -> FitResult:
        """Fit ``beta`` on the rows of ``X``.

        ``max_iter`` and ``tol`` default to the solver's registered values:
        Newton 500 iterations and ``max_j |dF/dbeta_j| <= 1e-10`` (scaled by
        ``max(1, max_j |mean m(W, phi_j)|)`` without a penalty); FISTA 100000
        iterations and a residual ``<= 1e-6 * lam`` (``1e-10`` when ``lam = 0``);
        L-BFGS-B 500 iterations and ``ftol = 1e-8``.
        """

        t0 = time.perf_counter()
        X_ = as_2d(X)
        _ensure_basis_fitted(self.basis, X_)
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

        u0 = evaluate_offset(self.offset, X_)
        if self.solver != "lbfgs":
            return self._fit_strict(
                X_, Phi, M, u0, beta0_, max_iter=max_iter, tol=tol, t0=t0
            )
        max_iter = 500 if max_iter is None else int(max_iter)
        tol = 1e-8 if tol is None else float(tol)

        # Closed form for the squared generator with an L2 (or no) penalty.
        # g(alpha) = (alpha - C)^2 gives g*(v) = C v + v^2/4, so the objective
        # is quadratic and the stationarity condition is
        #     (0.5 Phi'Phi/n + lam I) beta = mean(M) - C mean(Phi).
        if isinstance(self.generator, SquaredGenerator) and (
            self.penalty.penalty is None or self.penalty.p_norm == 2.0
        ):
            lam = self.penalty.lam if self.penalty.penalty is not None else 0.0
            A = 0.5 * (Phi.T @ Phi) / n + lam * np.eye(p)
            b = M.mean(axis=0) - self.generator.C * Phi.mean(axis=0) - 0.5 * (Phi.T @ u0) / n
            # Unlike a least-squares normal equation, b = mean(M) - C mean(Phi)
            # need not lie in the range of A, so an unpenalized rank-deficient
            # fit can have no stationary point at all. Report that as a failure
            # instead of passing off lstsq's finite non-solution as a fit.
            try:
                beta_hat = solve_stationarity(A, b)
            except np.linalg.LinAlgError as exc:
                out = FitResult(
                    beta=beta0_,
                    success=False,
                    message=str(exc),
                    n_iter=0,
                    status="singular",
                    fit_time=time.perf_counter() - t0,
                )
                # No solution was ever computed: leave the model unpredictable
                # rather than letting predict_alpha() silently evaluate the
                # (meaningless) initial point. The failure lives in fit_result_.
                self.beta_ = None
                self.fit_result_ = out
                self._Phi = None
                self._M = None
                return out
            return self._finalize_fit(
                X_,
                Phi,
                M,
                u0,
                np.asarray(beta_hat, dtype=float),
                success=True,
                message="closed_form",
                status="closed_form",
                n_iter=1,
                t0=t0,
            )

        # Objective and gradient. Generator failures are raised (and turned
        # into an explicit FitResult failure below), never converted into a
        # huge objective value with a zero gradient.
        def fun(beta: NDArray[np.float64]) -> float:
            v = u0 + Phi @ beta
            try:
                g_star, _ = self.generator.conjugate(X_, v)
            except Exception as exc:
                raise DomainError(
                    f"generator '{self.generator.name}' failed to evaluate its "
                    f"conjugate during optimization: {exc}"
                ) from exc
            loss = float(np.mean(g_star - (M @ beta)))
            return loss + self.penalty.value(beta)

        def jac(beta: NDArray[np.float64]) -> NDArray[np.float64]:
            v = u0 + Phi @ beta
            try:
                _, alpha = self.generator.conjugate(X_, v)
            except Exception as exc:
                raise DomainError(
                    f"generator '{self.generator.name}' failed to evaluate its "
                    f"link during optimization: {exc}"
                ) from exc
            grad = (alpha[:, None] * Phi - M).mean(axis=0)
            return grad + self.penalty.grad(beta)

        opts: dict = {"maxiter": int(max_iter), "ftol": float(tol)}
        if verbose:
            opts["iprint"] = 1

        # The branch signs depend on X only, but fun/jac are evaluated many
        # times on the same X. Memoize them for the duration of the fit.
        with _branch_cache_of(self.generator):
            try:
                res = optimize.minimize(
                    fun=fun, x0=beta0_, jac=jac, method="L-BFGS-B", options=opts
                )
            except DomainError as exc:
                out = FitResult(
                    beta=beta0_,
                    success=False,
                    message=str(exc),
                    n_iter=0,
                    status="domain_error",
                    fit_time=time.perf_counter() - t0,
                )
                # Same as the singular closed-form path: no solution exists, so
                # do not leave the initial point behind as a predictable state.
                self.beta_ = None
                self.fit_result_ = out
                self._Phi = None
                self._M = None
                return out

            beta_hat = np.asarray(res.x, dtype=float)
            return self._finalize_fit(
                X_,
                Phi,
                M,
                u0,
                beta_hat,
                success=bool(res.success),
                message=str(res.message),
                status="converged" if bool(res.success) else "optimizer_failure",
                n_iter=int(getattr(res, "nit", -1)),
                t0=t0,
            )

    def _finalize_fit(
        self,
        X_: NDArray[np.float64],
        Phi: NDArray[np.float64],
        M: NDArray[np.float64],
        u0: NDArray[np.float64],
        beta_hat: NDArray[np.float64],
        *,
        success: bool,
        message: str,
        status: str,
        n_iter: int,
        t0: float,
    ) -> FitResult:
        """Compute solution diagnostics and store the fit result."""

        objective = float("nan")
        gradient_norm = float("nan")
        kkt = float("nan")
        binding = float("nan")
        imbalance = float("nan")
        v = u0 + Phi @ beta_hat
        try:
            g_star, alpha = self.generator.conjugate(X_, v)
            objective = float(np.mean(g_star - (M @ beta_hat))) + self.penalty.value(beta_hat)
            grad_loss = (alpha[:, None] * Phi - M).mean(axis=0)
            imbalance = float(np.max(np.abs(grad_loss))) if grad_loss.size else 0.0
            grad_total = grad_loss + self.penalty.grad(beta_hat)
            gradient_norm = float(np.max(np.abs(grad_total))) if grad_total.size else 0.0
            kkt = self._kkt_residual(grad_loss, beta_hat)
            binding_fn = getattr(self.generator, "domain_binding", None)
            if callable(binding_fn):
                bind = np.asarray(binding_fn(X_, v), dtype=bool)
                binding = float(np.mean(bind)) if bind.size else 0.0
        except Exception as exc:
            success = False
            status = "domain_error_at_solution"
            message = f"{message} | diagnostics failed at solution: {exc}"

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
            solver="closed_form" if status == "closed_form" else "lbfgs",
            max_abs_imbalance=imbalance,
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

    def _fit_strict(
        self,
        X_: NDArray[np.float64],
        Phi: NDArray[np.float64],
        M: NDArray[np.float64],
        u0: NDArray[np.float64],
        beta0_: NDArray[np.float64],
        *,
        max_iter: int | None,
        tol: float | None,
        t0: float,
    ) -> FitResult:
        """Strict (domain-feasible, unclipped) solve with an explicit status."""

        kind = self._resolve_solver()
        pen = self.penalty
        lam = 0.0 if pen.penalty is None else float(pen.lam)
        b = M.mean(axis=0)

        if not np.all(np.isfinite(u0)):
            res = SolverResult(beta=beta0_, status=NONFINITE, n_iter=0,
                               message="the offset is not finite on the fitting rows")
        else:
            problem = DualProblem(generator=self.generator, X=X_, Phi=Phi, offset=u0, target=b)
            cache = (
                _branch_cache_of(self.generator)
                if getattr(self.generator, "branch_fn", None) is not None
                else contextlib.nullcontext()
            )
            with cache:
                if kind == "newton":
                    q = None if lam == 0.0 else pen.p_norm
                    scale = max(1.0, float(np.max(np.abs(b)))) if (q is None and b.size) else 1.0
                    res = newton_solve(
                        problem,
                        beta0=beta0_,
                        lam=lam,
                        q=q,
                        max_iter=500 if max_iter is None else int(max_iter),
                        tol=1e-10 if tol is None else float(tol),
                        tol_scale=scale,
                        boundary_tol=self.boundary_tol,
                    )
                else:
                    q = 1.0 if (pen.penalty is None or lam == 0.0) else pen.p_norm
                    default_tol = 1e-6 * lam if lam > 0.0 else 1e-10
                    res = fista_solve(
                        problem,
                        beta0=beta0_,
                        lam=lam,
                        q=q,
                        radius=self.l1_radius,
                        max_iter=100_000 if max_iter is None else int(max_iter),
                        tol=default_tol if tol is None else float(tol),
                        boundary_tol=self.boundary_tol,
                    )

        grad_loss = res.gradient
        imbalance = (
            float(np.max(np.abs(grad_loss))) if grad_loss is not None and grad_loss.size else
            float("nan")
        )
        if grad_loss is not None and kind == "newton" and lam > 0.0:
            gnorm = float(np.max(np.abs(grad_loss + pen.grad(res.beta)))) if grad_loss.size else 0.0
        else:
            gnorm = imbalance
        out = FitResult(
            beta=res.beta,
            success=res.status == OK,
            message=res.message or res.status,
            n_iter=res.n_iter,
            status=res.status,
            objective_value=res.objective,
            gradient_norm=gnorm,
            kkt_residual=res.kkt_residual,
            clip_binding_rate=0.0 if res.status == OK else float("nan"),
            fit_time=time.perf_counter() - t0,
            solver=kind,
            max_abs_imbalance=imbalance,
            n_boundary=res.n_boundary,
            l1_ball_active=res.l1_ball_active,
            ball_multiplier=res.ball_multiplier,
            polished=res.polished,
            hessian_min_eig=res.hessian_min_eig,
        )
        self.beta_ = res.beta if res.status == OK else None
        self.fit_result_ = out
        self._Phi = None
        self._M = None
        return out

    def predict_v(self, X: ArrayLike) -> NDArray[np.float64]:
        """Dual coordinate ``v(x) = u_ref(x) + phi(x)^T beta``."""

        if self.beta_ is None:
            raise RuntimeError("Model is not fit.")
        X_ = as_2d(X)
        Phi = np.asarray(self.basis(X_), dtype=float)
        return evaluate_offset(self.offset, X_) + Phi @ self.beta_

    def domain_mask(self, X: ArrayLike) -> NDArray[np.bool_]:
        """Rows at which the fitted representer is defined.

        True where the predicted dual coordinate lies in the open range of
        ``g'`` on the row's branch and the representer value is finite. Points
        outside it (e.g. evaluation-fold or counterfactual rows beyond the range
        of a BP/BKL link) have no representer value: the strict
        ``predict_alpha`` raises there rather than clipping.
        """

        X_ = as_2d(X)
        v = self.predict_v(X_)
        ok = np.asarray(self.generator.link_domain(X_, v), dtype=bool)
        if np.any(ok):
            _, a, _ = self.generator.dual_eval(X_[ok], v[ok])
            ok_idx = np.flatnonzero(ok)
            ok[ok_idx[~np.isfinite(a)]] = False
        return ok

    def predict_alpha(self, X: ArrayLike, *, out_of_domain: str = "raise") -> NDArray[np.float64]:
        """Fitted representer ``alpha(x) = (g')^{-1}(v(x))``.

        Parameters
        ----------
        out_of_domain:
            ``"raise"`` (default): raise :class:`DomainError` if any row is
            outside :meth:`domain_mask`. ``"nan"``: return NaN at those rows.
            Nothing is clipped in either case (generators built with
            ``legacy_clip=True`` clip inside their own link).
        """

        if out_of_domain not in {"raise", "nan"}:
            raise ValueError("out_of_domain must be 'raise' or 'nan'")
        X_ = as_2d(X)
        v = self.predict_v(X_)
        if self.solver == "lbfgs":
            return self.generator.inv_grad(X_, v)
        ok = self.domain_mask(X_)
        if not np.all(ok) and out_of_domain == "raise":
            raise DomainError(
                f"{int(np.sum(~ok))}/{ok.shape[0]} prediction row(s) are outside the domain "
                f"of generator '{self.generator.name}' (no finite representer value)."
            )
        out = np.full(v.shape[0], np.nan)
        if np.any(ok):
            out[ok] = self.generator.inv_grad(X_[ok], v[ok])
        return out

    def derivative_alpha(
        self, X: ArrayLike, coordinate: int, *, out_of_domain: str = "raise"
    ) -> NDArray[np.float64]:
        """Derivative of alpha(x) wrt x_coordinate.

        Uses the identity grad_g(alpha(x)) = v(x) and the inverse function theorem:

            g''(alpha) * d alpha/dx = d v/dx,

        with ``d v/dx`` including the derivative of the offset.
        """

        if self.beta_ is None:
            raise RuntimeError("Model is not fit.")
        X_ = as_2d(X)
        dPhi = self.basis.derivative(X_, coordinate)
        dv = dPhi @ self.beta_ + offset_derivative(self.offset, X_, coordinate)
        alpha = self.predict_alpha(X_, out_of_domain=out_of_domain)
        out = np.full_like(dv, np.nan, dtype=float)
        ok = np.isfinite(alpha)
        if not np.any(ok):
            return out
        g2 = np.full_like(dv, np.nan, dtype=float)
        g2[ok] = np.asarray(self.generator.grad2(X_[ok], alpha[ok]), dtype=float)
        bad = ok & (~np.isfinite(g2) | (g2 <= 0.0))
        if np.any(bad):
            warnings.warn(
                "derivative_alpha encountered non-positive or non-finite curvature "
                f"g''(alpha) for {int(bad.sum())} observation(s); returning NaN there.",
                RuntimeWarning,
                stacklevel=2,
            )
        good = ok & ~bad
        out[good] = dv[good] / g2[good]
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
    ) -> FitResult:
        X_ = as_2d(X)
        _ensure_basis_fitted(self.basis, X_)
        Phi = np.asarray(self.basis(X_), dtype=float)
        n, p = Phi.shape
        y_ = as_1d_of_length(y, n=n, name="y")

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
