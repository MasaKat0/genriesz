"""Density ratio estimation via the uLSIF objective.

This module provides :func:`grr_density_ratio`, a convenient estimator for the
covariate-shift density ratio

    r(x) = p(x) / q(x),

given two samples:

- ``X_num`` ~ p (numerator)
- ``X_den`` ~ q (denominator)

We implement the unnormalized LSIF (uLSIF) estimator in a Gaussian-kernel RKHS
feature space. This is a special case of the generalized Riesz-regression
framework, but it is more efficient to solve as a quadratic problem.

The objective is

    0.5 * E_q[r(x)^2] - E_p[r(x)] + 0.5 * lam * ||beta||_2^2,

with a linear model r(x) = phi(x)^T beta.

The implementation is intentionally lightweight and is designed for experiments
and reproducible baselines (e.g., covariate shift examples).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .basis import GaussianRKHSBasis
from .utils import kfold_splits


@dataclass(frozen=True)
class DensityRatioResult:
    """Result of :func:`grr_density_ratio`.

    Attributes
    ----------
    centers:
        Kernel centers used by the RKHS feature map.
    sigma:
        Gaussian kernel bandwidth.
    lam:
        Ridge parameter.
    beta:
        Linear coefficients such that r_hat(x) = phi(x)^T beta.
    standardize:
        Whether standardization was used.
    """

    centers: NDArray[np.float64]
    sigma: float
    lam: float
    beta: NDArray[np.float64]
    standardize: bool

    def predict_ratio(self, X: ArrayLike, *, clip_nonnegative: bool = True) -> NDArray[np.float64]:
        """Predict r_hat(X).

        Parameters
        ----------
        X:
            Points at which to evaluate the ratio.
        clip_nonnegative:
            If True, clip the predictions at 0. This is often used in LSIF
            implementations to enforce nonnegativity.
        """

        basis = GaussianRKHSBasis(
            centers=self.centers,
            sigma=self.sigma,
            standardize=self.standardize,
            include_bias=True,
        ).fit(self.centers)
        Phi = basis(X)
        r = Phi @ self.beta
        r = np.asarray(r, dtype=float).reshape(-1)
        if clip_nonnegative:
            r = np.maximum(r, 0.0)
        return r


def _ulsif_fit(
    *,
    Phi_num: NDArray[np.float64],
    Phi_den: NDArray[np.float64],
    lam: float,
) -> NDArray[np.float64]:
    """Solve the uLSIF normal equations for beta."""

    Phi_num = np.asarray(Phi_num, dtype=float)
    Phi_den = np.asarray(Phi_den, dtype=float)

    n_den = Phi_den.shape[0]
    if n_den <= 0:
        raise ValueError("Empty denominator sample")

    H = (Phi_den.T @ Phi_den) / float(n_den)
    H = H + float(lam) * np.eye(H.shape[0])
    h = np.mean(Phi_num, axis=0)

    # Solve H beta = h
    return np.linalg.solve(H, h)


def grr_density_ratio(
    X_num: ArrayLike,
    X_den: ArrayLike,
    *,
    n_centers: int = 200,
    sigma: float | None = 1.0,
    lam: float | None = 1e-2,
    standardize: bool = True,
    cv: bool = False,
    folds: int = 5,
    sigma_grid: Iterable[float] | None = None,
    lam_grid: Iterable[float] | None = None,
    random_state: int | None = 0,
    max_iter: int | None = None,
    tol: float | None = None,
) -> DensityRatioResult:
    """Estimate a density ratio using a Gaussian-kernel RKHS basis.

    Parameters
    ----------
    X_num, X_den:
        Samples from p (numerator) and q (denominator), respectively.
    n_centers:
        Number of kernel centers.
    sigma:
        Gaussian kernel bandwidth.
    lam:
        Ridge parameter.
    standardize:
        If True, standardize X by the center-sample mean/std.
    cv:
        If True, choose (sigma, lam) by K-fold cross validation.
    folds:
        Number of folds for CV.
    sigma_grid, lam_grid:
        Candidate grids used when ``cv=True``.
    random_state:
        Seed for center selection and CV splits.

    Notes
    -----
    ``max_iter`` and ``tol`` are accepted for API compatibility but are unused,
    since the estimator is solved in closed form.
    """

    Xn = np.asarray(X_num, dtype=float)
    Xd = np.asarray(X_den, dtype=float)
    if Xn.ndim != 2 or Xd.ndim != 2:
        raise ValueError("X_num and X_den must be 2D arrays")
    if Xn.shape[1] != Xd.shape[1]:
        raise ValueError("X_num and X_den must have the same number of columns")

    if n_centers <= 0:
        raise ValueError("n_centers must be positive")

    # Choose centers from the combined sample (simple and robust).
    rng = np.random.default_rng(random_state)
    X_all = np.vstack([Xn, Xd])
    n_all = X_all.shape[0]
    m = min(int(n_centers), int(n_all))
    idx = rng.choice(n_all, size=m, replace=False)
    centers = X_all[idx]

    def fit_for_params(sig: float, lam_: float):
        basis = GaussianRKHSBasis(
            centers=centers,
            sigma=float(sig),
            standardize=standardize,
            include_bias=True,
        ).fit(centers)
        Phi_num = basis(Xn)
        Phi_den = basis(Xd)
        beta = _ulsif_fit(Phi_num=Phi_num, Phi_den=Phi_den, lam=float(lam_))
        return basis, beta

    if not cv:
        if sigma is None or lam is None:
            raise ValueError("sigma and lam must be provided when cv=False")
        _, beta = fit_for_params(float(sigma), float(lam))
        return DensityRatioResult(
            centers=np.asarray(centers, dtype=float),
            sigma=float(sigma),
            lam=float(lam),
            beta=np.asarray(beta, dtype=float),
            standardize=bool(standardize),
        )

    # Cross-validation
    if sigma_grid is None:
        sigma_grid = [0.1, 0.3, 1.0, 3.0]
    if lam_grid is None:
        lam_grid = [1e-3, 1e-2, 1e-1]

    sigma_grid = [float(s) for s in sigma_grid]
    lam_grid = [float(l) for l in lam_grid]

    if folds <= 1:
        raise ValueError("folds must be >= 2 when cv=True")

    # Separate splits for numerator and denominator samples.
    splits_num = list(kfold_splits(len(Xn), folds=folds, random_state=random_state))
    splits_den = list(kfold_splits(len(Xd), folds=folds, random_state=(None if random_state is None else random_state + 1)))

    best = None
    best_score = float('inf')

    for sig in sigma_grid:
        for lam_ in lam_grid:
            scores = []
            for f in range(folds):
                tr_n, te_n = splits_num[f].train, splits_num[f].test
                tr_d, te_d = splits_den[f].train, splits_den[f].test

                basis = GaussianRKHSBasis(
                    centers=centers,
                    sigma=float(sig),
                    standardize=standardize,
                    include_bias=True,
                ).fit(centers)

                Phi_n_tr = basis(Xn[tr_n])
                Phi_d_tr = basis(Xd[tr_d])
                beta = _ulsif_fit(Phi_num=Phi_n_tr, Phi_den=Phi_d_tr, lam=float(lam_))

                # Validation objective: 0.5 E_q[r^2] - E_p[r]
                r_d = (basis(Xd[te_d]) @ beta).reshape(-1)
                r_n = (basis(Xn[te_n]) @ beta).reshape(-1)
                score = 0.5 * float(np.mean(r_d * r_d)) - float(np.mean(r_n))
                if not np.isfinite(score):
                    score = float('inf')
                scores.append(score)

            avg = float(np.mean(scores))
            if avg < best_score:
                best_score = avg
                best = (sig, lam_)

    if best is None:
        raise RuntimeError("Cross-validation failed to find finite score")

    sig_star, lam_star = best
    _, beta = fit_for_params(sig_star, lam_star)

    return DensityRatioResult(
        centers=np.asarray(centers, dtype=float),
        sigma=float(sig_star),
        lam=float(lam_star),
        beta=np.asarray(beta, dtype=float),
        standardize=bool(standardize),
    )
