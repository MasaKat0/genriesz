r"""Density ratio estimation via generalized Bregman divergence minimization.

This module provides :func:`fit_density_ratio`, a lightweight estimator for the
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

import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

from .basis import BaseBasis, CallableBasis, GaussianRKHSBasis
from .generators import (
    BKLGenerator,
    BPGenerator,
    BregmanGenerator,
    PUGenerator,
    SquaredGenerator,
    UKLGenerator,
)
from .glm import DomainError, _branch_cache_of, _Penalty
from .utils import kfold_splits


def _as_2d(X: ArrayLike, *, name: str) -> NDArray[np.float64]:
    X_ = np.asarray(X, dtype=float)
    if X_.ndim != 2:
        raise ValueError(f"{name} must be 2D. Got shape {X_.shape}.")
    return X_


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    expz = np.exp(z[~pos])
    out[~pos] = expz / (1.0 + expz)
    return out


def _coerce_basis(basis: BaseBasis | CallableBasis | Callable) -> BaseBasis | CallableBasis:
    if isinstance(basis, (BaseBasis, CallableBasis)):
        return basis
    if callable(basis):
        return CallableBasis(basis)
    raise TypeError('basis must be a BaseBasis instance or a callable basis(X)->Phi')


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

    if isinstance(generator, str):
        key = generator.strip().lower()
        # Density ratios are nonnegative, so by default we use the positive branch.
        def pos_branch(_x):
            return 1
        if key in {'sq', 'squared', 'lsif'}:
            return SquaredGenerator(C=0.0)
        if key in {'ukl'}:
            return UKLGenerator(C=0.0, branch_fn=pos_branch)
        if key in {'bkl'}:
            return BKLGenerator(C=1.0, branch_fn=pos_branch)
        if key in {'bp', 'power'}:
            return BPGenerator(C=0.0, omega=0.5, branch_fn=pos_branch)
        if key in {'pu'}:
            return PUGenerator(C=1.0, branch_fn=pos_branch)
        raise ValueError(
            "Unknown generator name. Use a generator instance or one of: "
            "'sq', 'ukl', 'bkl', 'bp', 'pu'."
        )

    if isinstance(generator, BregmanGenerator):
        return generator

    raise TypeError('generator must be a BregmanGenerator instance, a supported name, or None')


@dataclass(frozen=True)
class DensityRatioResult:
    """Result of :func:`fit_density_ratio`.

    Attributes
    ----------
    basis:
        Fitted basis object used for the linear predictor.
    generator:
        Bregman generator defining the loss and the link.
    beta:
        Coefficients of the linear predictor ``v(x) = phi(x)^T beta``.
    penalty, lam, p_norm:
        Regularization specification for ``beta``.
    centers, sigma, standardize:
        Convenience metadata when the default Gaussian RKHS basis is used.
        These are ``None`` when a custom basis is provided.
    route:
        ``"bregman"`` when the ratio is predicted through the generator's
        canonical link ``generator.inv_grad(v)``. ``"logistic_classification"``
        when the BKL generator was requested: the model is then fit as a
        probabilistic classifier and predictions use
        ``class_prior_ratio * exp(v)`` -- ``generator.inv_grad`` is NOT used.
    """

    basis: BaseBasis | CallableBasis
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

    def predict_v(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict the linear score v(x) = phi(x)^T beta."""

        X_ = _as_2d(X, name='X')
        Phi = np.asarray(self.basis(X_), dtype=float)
        return Phi @ self.beta

    def predict_ratio(self, X: ArrayLike, *, clip_nonnegative: bool = True) -> NDArray[np.float64]:
        """Predict the density ratio r_hat(x).

        Parameters
        ----------
        X:
            Points at which to evaluate the ratio.
        clip_nonnegative:
            If True, clip predictions at 0. This is often used for squared-loss
            density ratio estimators.
        """

        X_ = _as_2d(X, name='X')
        v = self.predict_v(X_)
        if self.class_prior_ratio is not None:
            z = np.clip(v, -700.0, 700.0)
            r = float(self.class_prior_ratio) * np.exp(z)
        else:
            r = self.generator.inv_grad(X_, v)
        r = np.asarray(r, dtype=float).reshape(-1)
        if clip_nonnegative:
            r = np.maximum(r, 0.0)
        return r


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

    return np.linalg.solve(A, b)


def _fit_bkl_classification(
    *,
    Phi_num: NDArray[np.float64],
    Phi_den: NDArray[np.float64],
    penalty: _Penalty,
    max_iter: int,
    tol: float,
    verbose: bool,
) -> NDArray[np.float64]:
    """Fit the probabilistic classification model for BKL density-ratio estimation."""

    Phi_num = np.asarray(Phi_num, dtype=float)
    Phi_den = np.asarray(Phi_den, dtype=float)
    Phi = np.vstack([Phi_num, Phi_den])
    y = np.concatenate(
        [np.ones(Phi_num.shape[0], dtype=float), np.zeros(Phi_den.shape[0], dtype=float)]
    )

    p = Phi.shape[1]
    beta0 = np.zeros(p, dtype=float)
    n = float(Phi.shape[0])

    def fun(beta: NDArray[np.float64]) -> float:
        eta = Phi @ beta
        loss = float(np.mean(np.logaddexp(0.0, eta) - y * eta))
        return loss + penalty.value(beta)

    def jac(beta: NDArray[np.float64]) -> NDArray[np.float64]:
        eta = Phi @ beta
        phat = _sigmoid(eta)
        grad = (Phi.T @ (phat - y)) / n
        return grad + penalty.grad(beta)

    opts_bkl: dict = {'maxiter': int(max_iter), 'ftol': float(tol)}
    if verbose:
        opts_bkl['iprint'] = 1
    res = optimize.minimize(fun=fun, x0=beta0, jac=jac, method='L-BFGS-B', options=opts_bkl)

    if not bool(res.success):
        raise RuntimeError(f"Density-ratio optimization failed: {res.message}")

    return np.asarray(res.x, dtype=float)


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
) -> NDArray[np.float64]:
    """Solve the general density-ratio objective by L-BFGS-B."""

    Phi_num = np.asarray(Phi_num, dtype=float)
    Phi_den = np.asarray(Phi_den, dtype=float)

    p = Phi_den.shape[1]
    beta0 = np.zeros(p, dtype=float)

    Phi_num_mean = Phi_num.mean(axis=0)

    # Generator failures are raised (and reported as an explicit optimization
    # failure below), never converted into a huge objective value with a zero
    # gradient.
    def fun(beta: NDArray[np.float64]) -> float:
        v_den = Phi_den @ beta
        try:
            g_star, _ = generator.conjugate(X_den, v_den)
        except Exception as exc:
            raise DomainError(
                f"generator '{generator.name}' failed to evaluate its conjugate "
                f"during density-ratio optimization: {exc}"
            ) from exc

        v_num = Phi_num @ beta
        loss = float(np.mean(g_star) - np.mean(v_num))
        return loss + penalty.value(beta)

    def jac(beta: NDArray[np.float64]) -> NDArray[np.float64]:
        v_den = Phi_den @ beta
        try:
            _, alpha_den = generator.conjugate(X_den, v_den)
        except Exception as exc:
            raise DomainError(
                f"generator '{generator.name}' failed to evaluate its link "
                f"during density-ratio optimization: {exc}"
            ) from exc

        grad = (alpha_den[:, None] * Phi_den).mean(axis=0) - Phi_num_mean
        return grad + penalty.grad(beta)

    opts_num: dict = {'maxiter': int(max_iter), 'ftol': float(tol)}
    if verbose:
        opts_num['iprint'] = 1
    try:
        # Branch signs depend on X_den only; fun/jac re-evaluate them every step.
        with _branch_cache_of(generator):
            res = optimize.minimize(
                fun=fun, x0=beta0, jac=jac, method='L-BFGS-B', options=opts_num
            )
    except DomainError as exc:
        raise RuntimeError(f"Density-ratio optimization failed: {exc}") from exc

    if not bool(res.success):
        raise RuntimeError(f"Density-ratio optimization failed: {res.message}")

    return np.asarray(res.x, dtype=float)


def fit_density_ratio(
    X_num: ArrayLike,
    X_den: ArrayLike,
    *,
    # Feature map / basis
    basis: BaseBasis | CallableBasis | Callable | None = None,
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

    Xn = _as_2d(X_num, name='X_num')
    Xd = _as_2d(X_den, name='X_den')

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
        basis_obj: BaseBasis | CallableBasis = GaussianRKHSBasis(
            centers=centers,
            sigma=sigma_used,
            standardize=standardize,
            include_bias=True,
            random_state=random_state,
        ).fit(X_all)
    else:
        basis_obj = _coerce_basis(basis)
        # Fit on the combined sample by default.
        try:
            basis_obj = basis_obj.copy().fit(np.vstack([Xn, Xd]))  # type: ignore[attr-defined]
        except Exception:
            # Some user bases may not implement copy(); fall back to in-place fit.
            basis_obj.fit(np.vstack([Xn, Xd]))  # type: ignore[attr-defined]

    def solve_beta(
        b,
        Xn_fit: NDArray[np.float64],
        Xd_fit: NDArray[np.float64],
        lam_: float,
        *,
        verbose_: bool,
    ) -> NDArray[np.float64]:
        """Dispatch to the right solver for the (already fitted) basis ``b``."""

        Phi_num = np.asarray(b(Xn_fit), dtype=float)
        Phi_den = np.asarray(b(Xd_fit), dtype=float)
        pen_local = _Penalty(penalty, lam=float(lam_), p_norm=p_norm)

        # Closed-form only for the squared generator + L2 (or no) penalty.
        if isinstance(gen, SquaredGenerator) and (
            pen_local.penalty is None or pen_local.p_norm == 2.0
        ):
            return _solve_squared_closed_form(
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

        beta_hat = solve_beta(b, Xn, Xd, lam_, verbose_=verbose)
        return b, beta_hat

    route = "logistic_classification" if isinstance(gen, BKLGenerator) else "bregman"

    if not cv:
        if sigma_used is None and basis is None:
            raise ValueError('sigma must be provided when cv=False and basis is None')
        b, beta_hat = fit_for_params(
            float(sigma_used) if sigma_used is not None else 1.0, float(lam)
        )
        return DensityRatioResult(
            basis=b,
            generator=gen,
            beta=np.asarray(beta_hat, dtype=float),
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
    best_score = float('inf')
    n_failed_fits = 0
    last_failure: str | None = None

    for sig in sigma_grid:
        for lam_ in lam_grid:
            scores = []
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

                # A single failing candidate must not abort the whole CV; it is
                # excluded (score = inf) and counted.
                try:
                    beta = solve_beta(b, Xn[tr_n], Xd[tr_d], lam_, verbose_=False)
                except RuntimeError as exc:
                    n_failed_fits += 1
                    last_failure = str(exc)
                    scores.append(float('inf'))
                    continue

                # Validation score: unpenalized objective
                v_d = np.asarray(b(Xd[te_d]) @ beta, dtype=float).reshape(-1)
                v_n = np.asarray(b(Xn[te_n]) @ beta, dtype=float).reshape(-1)
                if isinstance(gen, BKLGenerator):
                    score = float(
                        np.mean(np.logaddexp(0.0, v_n) - v_n)
                        + np.mean(np.logaddexp(0.0, v_d))
                    )
                else:
                    g_star, _ = gen.conjugate(Xd[te_d], v_d)
                    score = float(np.mean(g_star) - np.mean(v_n))
                if not np.isfinite(score):
                    score = float('inf')
                scores.append(score)

            avg = float(np.mean(scores))
            if np.isfinite(avg) and avg < best_score:
                best_score = avg
                best = (sig, lam_)

    if n_failed_fits > 0:
        warnings.warn(
            f"{n_failed_fits} candidate fit(s) failed during density-ratio CV "
            f"and were excluded (last failure: {last_failure}).",
            UserWarning,
            stacklevel=2,
        )
    if best is None:
        raise RuntimeError(
            "Cross-validation failed to find a finite score "
            f"({n_failed_fits} candidate fits failed; last failure: {last_failure})"
        )

    sig_star, lam_star = best
    b, beta_hat = fit_for_params(sig_star, lam_star)

    return DensityRatioResult(
        basis=b,
        generator=gen,
        beta=np.asarray(beta_hat, dtype=float),
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
    )
