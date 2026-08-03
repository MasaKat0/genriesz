r"""Density ratio estimation via generalized Bregman divergence minimization.

This module provides :func:`fit_density_ratio` for the
covariate-shift density ratio

    r(x) = p(x) / q(x),

given two samples:

- ``X_num`` ~ p (numerator)
- ``X_den`` ~ q (denominator)

The estimator is a special case of the generalized Riesz-regression (GRR)
framework. Given a Bregman generator ``g(x, alpha)``, we fit a model

    v(x) = phi(x)^T beta,

and map the linear predictor to the ratio via the *canonical link*

    r(x) = alpha(x) = (\partial g(x,\cdot))^{-1}( v(x) ).

The objective is the empirical version of the Bregman-divergence risk
(see the paper's density-ratio estimation section):

    E_q[ g*(X, v(X)) ] - E_p[ v(X) ] + penalty(beta),

where ``g*`` is the convex conjugate of ``g`` and ``p``/``q`` correspond to the
numerator/denominator samples.

By default we use a Gaussian-kernel RKHS basis. You can optionally select the
RBF bandwidth ``sigma`` and regularization ``lam`` via cross validation.

Notes
-----
- For general generators we solve the convex problem numerically (L-BFGS-B).
- For the squared generator (``SquaredGenerator`` / ``generator='sq'``) with an
  L2 penalty, the objective is quadratic and we use a closed-form ridge solve.

"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

from .basis import Basis, GaussianRKHSBasis, coerce_basis
from .generators import (
    BKLGenerator,
    BregmanGenerator,
    DomainError,
    SquaredGenerator,
    coerce_generator,
)
from .glm import (
    _branch_cache_of,
    _exact_linear_dual_domain,
    _linear_constraint_kkt_residual,
    _Penalty,
    _solve_squared_stationarity,
)
from .utils import as_2d, kfold_splits, sigmoid


def _positive_branch(_x: NDArray[np.float64]) -> int:
    """A density ratio is nonnegative, so every row takes the positive branch."""

    return 1


def _coerce_generator(
    *,
    generator: BregmanGenerator | str | None,
    g: Callable | None,
    grad_g: Callable | None,
    inv_grad_g: Callable | None,
    grad2_g: Callable | None,
) -> BregmanGenerator:
    if generator is not None and g is not None:
        raise ValueError('Pass either generator=... or g=... (not both).')

    if g is not None:
        return BregmanGenerator(g=g, grad=grad_g, inv_grad=inv_grad_g, grad2=grad2_g)

    if generator is None:
        return SquaredGenerator(C=0.0)

    return coerce_generator(generator, branch_fn=_positive_branch)


@dataclass(frozen=True)
class _DensityRatioFit:
    beta: NDArray[np.float64] | None
    success: bool
    status: str
    message: str
    kkt_residual: float = float("nan")
    domain_binding_rate: float = float("nan")


@dataclass(frozen=True)
class DensityRatioResult:
    """Result of :func:`fit_density_ratio`."""

    basis: Basis
    generator: BregmanGenerator
    beta: NDArray[np.float64]
    penalty: str | None
    lam: float
    p_norm: float
    centers: NDArray[np.float64] | None = None
    sigma: float | None = None
    standardize: bool | None = None
    class_prior_ratio: float | None = None
    route: str = "bregman"
    cv_path: tuple[dict[str, object], ...] = ()
    n_failed_candidates: int = 0

    def predict_v(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict the linear score ``v(x) = phi(x)^T beta``."""

        X_ = as_2d(X, name="X")
        Phi = np.asarray(self.basis(X_), dtype=float)
        return Phi @ self.beta

    def predict_ratio(
        self, X: ArrayLike, *, clip_nonnegative: bool = False
    ) -> NDArray[np.float64]:
        """Predict the density ratio.

        ``clip_nonnegative=True`` is an explicit post-fit modification. It is
        never applied by default.
        """

        X_ = as_2d(X, name="X")
        v = self.predict_v(X_)
        if self.class_prior_ratio is not None:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                ratio = float(self.class_prior_ratio) * np.exp(v)
            valid = np.isfinite(v) & np.isfinite(ratio) & (ratio > 0.0)
            if not np.all(valid):
                n_bad = int(np.sum(~valid))
                raise DomainError(
                    "The logistic-classification density ratio is not representable "
                    f"in float64 for {n_bad}/{len(v)} observation(s)."
                )
        else:
            ratio = self.generator.inv_grad(X_, v)
        ratio = np.asarray(ratio, dtype=float).reshape(-1)
        if clip_nonnegative:
            ratio = np.maximum(ratio, 0.0)
        return ratio


def _solve_squared_closed_form(
    *,
    Phi_num: NDArray[np.float64],
    Phi_den: NDArray[np.float64],
    C: float,
    penalty: _Penalty,
) -> NDArray[np.float64]:
    """Closed-form solution for the squared generator with an L2 penalty.

    For ``SquaredGenerator`` we have ``alpha = C + 0.5 v`` and the density-ratio
    objective is quadratic in ``beta``.

    With an L2 penalty (or no penalty) the normal equations are

        (0.5 H + lam I) beta = h - C m,

    where

        H = E_q[Phi^T Phi],  m = E_q[Phi],  h = E_p[Phi].
    """

    Phi_num = np.asarray(Phi_num, dtype=float)
    Phi_den = np.asarray(Phi_den, dtype=float)

    n_den = Phi_den.shape[0]
    if n_den <= 0:
        raise ValueError('Empty denominator sample')

    H = (Phi_den.T @ Phi_den) / float(n_den)
    m = Phi_den.mean(axis=0)
    h = Phi_num.mean(axis=0)

    lam = penalty.lam if penalty.penalty is not None else 0.0
    A = 0.5 * H + lam * np.eye(H.shape[0])
    b = h - float(C) * m

    solution = _solve_squared_stationarity(A, b)
    if not solution.success:
        raise np.linalg.LinAlgError(solution.message)
    return solution.beta


def _solve_squared_closed_form_status(
    *,
    Phi_num: NDArray[np.float64],
    Phi_den: NDArray[np.float64],
    C: float,
    penalty: _Penalty,
) -> _DensityRatioFit:
    Phi_num_ = np.asarray(Phi_num, dtype=float)
    Phi_den_ = np.asarray(Phi_den, dtype=float)
    if Phi_den_.shape[0] <= 0:
        raise ValueError("Empty denominator sample")
    H = (Phi_den_.T @ Phi_den_) / float(Phi_den_.shape[0])
    m = Phi_den_.mean(axis=0)
    h = Phi_num_.mean(axis=0)
    lam = penalty.lam if penalty.penalty is not None else 0.0
    solution = _solve_squared_stationarity(
        0.5 * H + lam * np.eye(H.shape[0]), h - float(C) * m
    )
    return _DensityRatioFit(
        beta=solution.beta if solution.success else None,
        success=solution.success,
        status="closed_form" if solution.success else "singular",
        message=solution.message,
        kkt_residual=0.0 if solution.success else float("nan"),
        domain_binding_rate=0.0 if solution.success else float("nan"),
    )


def _fit_bkl_classification(
    *,
    Phi_num: NDArray[np.float64],
    Phi_den: NDArray[np.float64],
    penalty: _Penalty,
    max_iter: int,
    tol: float,
    verbose: bool,
) -> _DensityRatioFit:
    """Fit the probabilistic-classification route for BKL density ratios."""

    Phi_num_ = np.asarray(Phi_num, dtype=float)
    Phi_den_ = np.asarray(Phi_den, dtype=float)
    Phi = np.vstack([Phi_num_, Phi_den_])
    y = np.concatenate(
        [np.ones(Phi_num_.shape[0], dtype=float), np.zeros(Phi_den_.shape[0], dtype=float)]
    )
    beta0 = np.zeros(Phi.shape[1], dtype=float)
    n = float(Phi.shape[0])

    def fun(beta: NDArray[np.float64]) -> float:
        eta = Phi @ beta
        return float(np.mean(np.logaddexp(0.0, eta) - y * eta)) + penalty.value(beta)

    def jac(beta: NDArray[np.float64]) -> NDArray[np.float64]:
        eta = Phi @ beta
        gradient = (Phi.T @ (sigmoid(eta) - y)) / n
        return gradient + penalty.grad(beta)

    solver_ftol = min(float(tol), 1e-10)
    solver_gtol = min(float(tol), 1e-7)
    options: dict[str, object] = {
        "maxiter": int(max_iter),
        "ftol": solver_ftol,
        "gtol": solver_gtol,
        "maxls": 50,
    }
    if verbose:
        options["iprint"] = 1
    result = optimize.minimize(
        fun=fun, x0=beta0, jac=jac, method="L-BFGS-B", options=options
    )
    beta = np.asarray(result.x, dtype=float)
    gradient = jac(beta)
    kkt = float(np.max(np.abs(gradient))) if gradient.size else 0.0
    threshold = max(1e-5, 100.0 * float(tol))
    success = (
        bool(result.success)
        and bool(np.all(np.isfinite(beta)))
        and np.isfinite(float(result.fun))
        and np.isfinite(kkt)
        and kkt <= threshold
    )
    return _DensityRatioFit(
        beta=beta if success else None,
        success=success,
        status="converged" if success else "optimization_failure",
        message=str(result.message),
        kkt_residual=kkt,
        domain_binding_rate=0.0,
    )


def _fit_numeric(
    *,
    X_num: NDArray[np.float64],
    X_den: NDArray[np.float64],
    Phi_num: NDArray[np.float64],
    Phi_den: NDArray[np.float64],
    generator: BregmanGenerator,
    penalty: _Penalty,
    max_iter: int,
    tol: float,
    verbose: bool,
) -> _DensityRatioFit:
    """Solve the general density-ratio objective with exact domain constraints."""

    X_num_ = np.asarray(X_num, dtype=float)
    X_den_ = np.asarray(X_den, dtype=float)
    Phi_num_ = np.asarray(Phi_num, dtype=float)
    Phi_den_ = np.asarray(Phi_den, dtype=float)
    Phi_all = np.vstack([Phi_num_, Phi_den_])
    X_all = np.vstack([X_num_, X_den_])
    p = Phi_den_.shape[1]
    beta0 = np.zeros(p, dtype=float)
    phi_num_mean = Phi_num_.mean(axis=0)

    def fun(beta: NDArray[np.float64]) -> float:
        evaluation = generator.conjugate_status(X_den_, Phi_den_ @ beta)
        if not bool(np.all(evaluation.valid)):
            return float("nan")
        loss = float(np.mean(evaluation.conjugate) - np.mean(Phi_num_ @ beta))
        return loss + penalty.value(beta)

    def jac(beta: NDArray[np.float64]) -> NDArray[np.float64]:
        evaluation = generator.conjugate_status(X_den_, Phi_den_ @ beta)
        if not bool(np.all(evaluation.valid)):
            return np.full(p, np.nan, dtype=float)
        gradient = (evaluation.alpha[:, None] * Phi_den_).mean(axis=0) - phi_num_mean
        return gradient + penalty.grad(beta)

    solver_ftol = min(float(tol), 1e-10)
    options: dict[str, object] = {"maxiter": int(max_iter), "ftol": solver_ftol}
    if verbose:
        options["disp"] = True

    with _branch_cache_of(generator):
        dual_domain = _exact_linear_dual_domain(
            generator,
            X_all,
            Phi_all,
            margin=max(1e-10, 10.0 * float(tol)),
        )
        if dual_domain is not None:
            if not np.all(np.isfinite(dual_domain.initial_beta)):
                return _DensityRatioFit(
                    beta=None,
                    success=False,
                    status="domain_infeasible",
                    message=(
                        "The positive density-ratio branch and the fitted basis have "
                        "no common linear predictor inside the exact generator domain."
                    ),
                )
            beta0 = dual_domain.initial_beta

        initial = generator.conjugate_status(X_den_, Phi_den_ @ beta0)
        if not bool(np.all(initial.valid)):
            return _DensityRatioFit(
                beta=None,
                success=False,
                status="domain_error",
                message="The initial dual coordinate is outside the exact generator domain.",
            )

        if dual_domain is None:
            result = optimize.minimize(
                fun=fun, x0=beta0, jac=jac, method="L-BFGS-B", options=options
            )
        else:
            constraint = optimize.LinearConstraint(
                Phi_all, dual_domain.lower, dual_domain.upper
            )
            result = optimize.minimize(
                fun=fun,
                x0=beta0,
                jac=jac,
                method="SLSQP",
                constraints=(constraint,),
                options=options,
            )

        beta = np.asarray(result.x, dtype=float)
        final_den = generator.conjugate_status(X_den_, Phi_den_ @ beta)
        final_all = generator.inv_grad_status(X_all, Phi_all @ beta)
        objective = fun(beta)
        gradient = jac(beta)
        finite_gradient = bool(np.all(np.isfinite(gradient)))
        if dual_domain is None and finite_gradient:
            kkt = float(np.max(np.abs(gradient))) if gradient.size else 0.0
        elif dual_domain is not None and finite_gradient:
            kkt = _linear_constraint_kkt_residual(
                gradient,
                beta,
                Phi_all,
                dual_domain.lower,
                dual_domain.upper,
                tolerance=tol,
            )
        else:
            kkt = float("nan")
        binding = np.asarray(generator.domain_binding(X_all, Phi_all @ beta), dtype=bool)
        binding_rate = float(np.mean(binding)) if binding.size else 0.0
        threshold = max(1e-5, 100.0 * float(tol))
        success = (
            bool(result.success)
            and bool(np.all(np.isfinite(beta)))
            and bool(np.all(final_den.valid))
            and bool(np.all(final_all.valid))
            and np.isfinite(objective)
            and np.isfinite(kkt)
            and kkt <= threshold
        )
        if success:
            return _DensityRatioFit(
                beta=beta,
                success=True,
                status="converged",
                message=str(result.message),
                kkt_residual=kkt,
                domain_binding_rate=binding_rate,
            )
        if not bool(np.all(final_all.valid)):
            status = "domain_error"
        elif not np.isfinite(kkt) or kkt > threshold:
            status = "kkt_failure"
        else:
            status = "optimization_failure"
        return _DensityRatioFit(
            beta=None,
            success=False,
            status=status,
            message=str(result.message),
            kkt_residual=kkt,
            domain_binding_rate=binding_rate,
        )


def fit_density_ratio(
    X_num: ArrayLike,
    X_den: ArrayLike,
    *,
    # Feature map / basis
    basis: Basis | Callable | None = None,
    n_centers: int = 200,
    sigma: float | None = 1.0,
    standardize: bool = True,
    # Generator specification (mirrors grr_functional)
    generator: BregmanGenerator | str | None = None,
    g: Callable | None = None,
    grad_g: Callable | None = None,
    inv_grad_g: Callable | None = None,
    grad2_g: Callable | None = None,
    # Regularization
    penalty: str | None = 'l2',
    lam: float = 1e-2,
    p_norm: float | None = None,
    # Cross-validation (Gaussian RKHS basis only)
    cv: bool = False,
    folds: int = 5,
    sigma_grid: Iterable[float] | None = None,
    lam_grid: Iterable[float] | None = None,
    random_state: int | None = 0,
    # Optimizer
    max_iter: int = 500,
    tol: float = 1e-8,
    verbose: bool = False,
) -> DensityRatioResult:
    """Estimate a density ratio under covariate shift.

    Parameters
    ----------
    X_num, X_den:
        Samples from p (numerator) and q (denominator), respectively.
    basis:
        Feature map ``phi(X)`` used for the linear predictor. If None, a
        Gaussian-kernel RKHS basis is used.
    n_centers, sigma, standardize:
        Parameters of the default Gaussian RKHS basis.
    generator:
        Either a :class:`~genriesz.generators.BregmanGenerator` instance or one
        of the built-in names: ``'sq'``, ``'ukl'``, ``'bkl'``, ``'bp'``, ``'pu'``.
        If None and ``g`` is also None, defaults to ``'sq'``.
    g, grad_g, inv_grad_g, grad2_g:
        Custom generator specification (same conventions as :func:`grr_functional`).
    penalty, lam, p_norm:
        Regularization on ``beta``.
    cv, folds, sigma_grid, lam_grid:
        If ``cv=True``, choose (sigma, lam) by cross validation.
        This is currently implemented only for the default Gaussian RKHS basis.
    max_iter, tol, verbose:
        L-BFGS-B controls for the general (non-squared) case.

    Returns
    -------
    DensityRatioResult
        A fitted model with :meth:`~DensityRatioResult.predict_ratio`.
    """

    Xn = as_2d(X_num, name='X_num')
    Xd = as_2d(X_den, name='X_den')

    if Xn.shape[1] != Xd.shape[1]:
        raise ValueError('X_num and X_den must have the same number of columns')

    # Coerce generator
    gen = _coerce_generator(
        generator=generator, g=g, grad_g=grad_g, inv_grad_g=inv_grad_g, grad2_g=grad2_g
    )

    # Regularization
    pen = _Penalty(penalty, lam=float(lam), p_norm=p_norm)

    # Default basis (Gaussian RKHS with centers chosen from the combined sample)
    centers: NDArray[np.float64] | None = None
    sigma_used: float | None = None
    if basis is None:
        if sigma is None:
            raise ValueError('sigma must be provided when basis is None')
        if n_centers <= 0:
            raise ValueError('n_centers must be positive')
        rng = np.random.default_rng(random_state)
        X_all = np.vstack([Xn, Xd])
        n_all = X_all.shape[0]
        m = min(int(n_centers), int(n_all))
        idx = rng.choice(n_all, size=m, replace=False)
        centers = np.asarray(X_all[idx], dtype=float)
        sigma_used = float(sigma)
        basis_obj: Basis = GaussianRKHSBasis(
            centers=centers,
            sigma=sigma_used,
            standardize=standardize,
            include_bias=True,
            random_state=random_state,
        ).fit(X_all)
    else:
        # ``copy`` belongs to the Basis protocol, so coerce_basis guarantees it.
        # Fit on the combined sample by default.
        basis_obj = coerce_basis(basis).copy().fit(np.vstack([Xn, Xd]))

    def solve_beta(
        b,
        Xn_fit: NDArray[np.float64],
        Xd_fit: NDArray[np.float64],
        lam_: float,
        *,
        verbose_: bool,
    ) -> _DensityRatioFit:
        """Dispatch to the solver for the fitted basis and return its status."""

        Phi_num = np.asarray(b(Xn_fit), dtype=float)
        Phi_den = np.asarray(b(Xd_fit), dtype=float)
        pen_local = _Penalty(penalty, lam=float(lam_), p_norm=p_norm)

        # Closed-form only for the squared generator + L2 (or no) penalty.
        if isinstance(gen, SquaredGenerator) and (
            pen_local.penalty is None or pen_local.p_norm == 2.0
        ):
            return _solve_squared_closed_form_status(
                Phi_num=Phi_num, Phi_den=Phi_den, C=gen.C, penalty=pen_local
            )
        if isinstance(gen, BKLGenerator):
            return _fit_bkl_classification(
                Phi_num=Phi_num,
                Phi_den=Phi_den,
                penalty=pen_local,
                max_iter=max_iter,
                tol=tol,
                verbose=verbose_,
            )
        return _fit_numeric(
            X_num=Xn_fit,
            X_den=Xd_fit,
            Phi_num=Phi_num,
            Phi_den=Phi_den,
            generator=gen,
            penalty=pen_local,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose_,
        )

    def fit_for_params(sig: float, lam_: float):
        # Rebuild basis for this sigma if we are using the default RKHS basis.
        if basis is None:
            assert centers is not None
            b = GaussianRKHSBasis(
                centers=centers,
                sigma=float(sig),
                standardize=standardize,
                include_bias=True,
                random_state=random_state,
            ).fit(np.vstack([Xn, Xd]))
        else:
            # For a custom basis, we do not support sigma tuning.
            b = basis_obj

        fit = solve_beta(b, Xn, Xd, lam_, verbose_=verbose)
        return b, fit

    route = "logistic_classification" if isinstance(gen, BKLGenerator) else "bregman"

    if not cv:
        if sigma_used is None and basis is None:
            raise ValueError('sigma must be provided when cv=False and basis is None')
        b, fit = fit_for_params(
            float(sigma_used) if sigma_used is not None else 1.0, float(lam)
        )
        if not fit.success or fit.beta is None:
            raise RuntimeError(
                f"Density-ratio fitting failed with status '{fit.status}': {fit.message}"
            )
        return DensityRatioResult(
            basis=b,
            generator=gen,
            beta=np.asarray(fit.beta, dtype=float),
            penalty=None if penalty is None else str(penalty),
            lam=float(lam),
            p_norm=float(pen.p_norm),
            centers=centers,
            sigma=sigma_used,
            standardize=bool(standardize) if basis is None else None,
            class_prior_ratio=(
                float(len(Xd)) / float(len(Xn)) if isinstance(gen, BKLGenerator) else None
            ),
            route=route,
        )

    # Cross-validation
    if basis is not None:
        raise ValueError('cv=True is currently supported only when basis is None (Gaussian RKHS).')

    if sigma_grid is None:
        sigma_grid = [0.1, 0.3, 1.0, 3.0]
    if lam_grid is None:
        lam_grid = [1e-3, 1e-2, 1e-1]

    sigma_grid = [float(s) for s in sigma_grid]
    lam_grid = [float(lam_val) for lam_val in lam_grid]

    if folds <= 1:
        raise ValueError('folds must be >= 2 when cv=True')

    splits_num = list(kfold_splits(len(Xn), folds=folds, random_state=random_state))
    splits_den = list(
        kfold_splits(
            len(Xd),
            folds=folds,
            random_state=(None if random_state is None else random_state + 1),
        )
    )

    # Fold-local kernel centers: selecting centers from the full sample before
    # splitting would leak validation points into the candidate bases.
    fold_centers: list[NDArray[np.float64]] = []
    for f in range(folds):
        X_tr_all = np.vstack([Xn[splits_num[f].train], Xd[splits_den[f].train]])
        rng_f = np.random.default_rng(
            None if random_state is None else random_state + 7919 * (f + 1)
        )
        m_f = min(int(n_centers), int(X_tr_all.shape[0]))
        idx_f = rng_f.choice(X_tr_all.shape[0], size=m_f, replace=False)
        fold_centers.append(np.asarray(X_tr_all[idx_f], dtype=float))

    best: tuple[float, float] | None = None
    best_score = float("inf")
    cv_path: list[dict[str, object]] = []

    for sig in sigma_grid:
        for lam_ in lam_grid:
            scores: list[float] = []
            fold_status: list[str] = []
            fold_messages: list[str] = []
            for f in range(folds):
                tr_n, te_n = splits_num[f].train, splits_num[f].test
                tr_d, te_d = splits_den[f].train, splits_den[f].test
                b = GaussianRKHSBasis(
                    centers=fold_centers[f],
                    sigma=float(sig),
                    standardize=standardize,
                    include_bias=True,
                    random_state=random_state,
                ).fit(np.vstack([Xn[tr_n], Xd[tr_d]]))
                fit = solve_beta(b, Xn[tr_n], Xd[tr_d], lam_, verbose_=False)
                if not fit.success or fit.beta is None:
                    scores.append(float("nan"))
                    fold_status.append(fit.status)
                    fold_messages.append(fit.message)
                    continue

                v_d = np.asarray(b(Xd[te_d]) @ fit.beta, dtype=float).reshape(-1)
                v_n = np.asarray(b(Xn[te_n]) @ fit.beta, dtype=float).reshape(-1)
                if isinstance(gen, BKLGenerator):
                    score = float(
                        np.mean(np.logaddexp(0.0, v_n) - v_n)
                        + np.mean(np.logaddexp(0.0, v_d))
                    )
                    score_valid = np.isfinite(score)
                else:
                    evaluation = gen.conjugate_status(Xd[te_d], v_d)
                    score_valid = bool(np.all(evaluation.valid))
                    score = (
                        float(np.mean(evaluation.conjugate) - np.mean(v_n))
                        if score_valid
                        else float("nan")
                    )
                if not score_valid or not np.isfinite(score):
                    scores.append(float("nan"))
                    fold_status.append("validation_domain_error")
                    fold_messages.append(
                        "The fitted dual coordinate was invalid on the validation fold."
                    )
                    continue
                scores.append(score)
                fold_status.append("success")
                fold_messages.append("")

            candidate_success = bool(np.all(np.isfinite(scores)))
            average_score = float(np.mean(scores)) if candidate_success else float("nan")
            cv_path.append(
                {
                    "sigma": float(sig),
                    "lam": float(lam_),
                    "success": candidate_success,
                    "score": average_score,
                    "fold_status": tuple(fold_status),
                    "fold_messages": tuple(fold_messages),
                }
            )
            if candidate_success and average_score < best_score:
                best_score = average_score
                best = (sig, lam_)

    if best is None:
        failed = sum(not bool(row["success"]) for row in cv_path)
        raise RuntimeError(
            "Cross-validation did not produce a finite score for any candidate "
            f"({failed} candidate specifications failed)."
        )

    sig_star, lam_star = best
    b, fit = fit_for_params(sig_star, lam_star)
    if not fit.success or fit.beta is None:
        raise RuntimeError(
            f"The selected density-ratio specification failed when refit on the "
            f"full sample with status '{fit.status}': {fit.message}"
        )

    return DensityRatioResult(
        basis=b,
        generator=gen,
        beta=np.asarray(fit.beta, dtype=float),
        penalty=None if penalty is None else str(penalty),
        lam=float(lam_star),
        p_norm=float(_Penalty(penalty, lam=float(lam_star), p_norm=p_norm).p_norm),
        centers=centers,
        sigma=float(sig_star),
        standardize=bool(standardize),
        class_prior_ratio=(
            float(len(Xd)) / float(len(Xn)) if isinstance(gen, BKLGenerator) else None
        ),
        route=route,
        cv_path=tuple(cv_path),
        n_failed_candidates=sum(not bool(row["success"]) for row in cv_path),
    )
