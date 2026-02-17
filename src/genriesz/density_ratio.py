"""Density-ratio estimation utilities built on top of GRR.

This module provides a small, self-contained density-ratio estimator that uses
the core :class:`genriesz.GRR` solver together with a linear functional that
encodes the numerator distribution.

Background
----------
Given samples

- ``X_num`` ~ p(x)   (numerator distribution)
- ``X_den`` ~ q(x)   (denominator distribution)

the (plain) density ratio is

    r(x) = p(x) / q(x).

A common least-squares objective (uLSIF) estimates r by minimizing

    1/2 E_q[r(x)^2] - E_p[r(x)],

up to an additive constant independent of r. This objective can be expressed
in the GRR framework by:

- fitting on the denominator sample ``X_den`` (so the quadratic term is an
  empirical approximation to ``E_q``), and
- using a functional that evaluates ``E_p`` empirically via the numerator
  sample ``X_num``.

The resulting fitted Riesz representer ``alpha(x)`` can be interpreted as a
density ratio (up to modeling error / regularization).

Notes
-----
* This implementation is intentionally minimal and does not add any hidden
  stabilization heuristics.
* Hyperparameters (e.g., RBF bandwidth and ridge strength) can be chosen via
  standard K-fold cross validation over the uLSIF objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .bregman import BregmanGenerator, SquaredGenerator
from .basis import GaussianRKHSBasis
from .grr import GRR, BasisFn


Array = np.ndarray


def _as_2d(X: Array) -> Array:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array. Got shape {X.shape}.")
    return X


@dataclass(frozen=True)
class DensityRatioFunctional:
    """Functional that injects the numerator sample into the GRR objective.

    For any test function ``gamma``, we define

        m(x, gamma) = E_p[gamma(X)] ≈ mean_{j in X_num} gamma(X_num[j]).

    This does not depend on the particular denominator point ``x``.
    """

    X_num: Array

    def __post_init__(self) -> None:
        object.__setattr__(self, "X_num", _as_2d(self.X_num))

    def __call__(self, _x: Array, gamma) -> float:
        Xn = self.X_num
        vals = np.empty(len(Xn), dtype=float)
        for i in range(len(Xn)):
            vals[i] = float(gamma(Xn[i]))
        return float(np.mean(vals))

    def basis_matrix(self, X_den: Array, basis: BasisFn) -> Array:
        """Vectorized computation of m(X_i, basis_j).

        Since m does not depend on X_i, the resulting matrix has identical rows:

            M_{ij} = mean_{k in X_num} basis_j(X_num[k]).
        """
        Xd = _as_2d(X_den)
        Xn = self.X_num
        Phi_n = np.asarray(basis(Xn), dtype=float)
        if Phi_n.ndim == 1:
            Phi_n = Phi_n.reshape(len(Xn), -1)
        mu = Phi_n.mean(axis=0, keepdims=True)  # (1,p)
        return np.repeat(mu, repeats=len(Xd), axis=0)


@dataclass
class DensityRatioResult:
    """Return object for :func:`genriesz.grr_density_ratio`."""

    model: GRR
    sigma: float
    lam: float
    centers: Array

    def predict_ratio(self, X: Array) -> Array:
        """Predict the density ratio r(x) at covariates X."""
        return self.model.predict_alpha(X)


def _default_grid() -> Array:
    # Match the wide defaults used in several LSIF/RuLSIF implementations.
    return 10.0 ** np.linspace(-3.0, 9.0, 13)


def _kfold_ids(n: int, folds: int, rng: np.random.Generator) -> Array:
    if folds <= 1:
        raise ValueError("folds must be >= 2 for cross validation.")
    ids = np.arange(n)
    rng.shuffle(ids)
    fold_id = np.empty(n, dtype=int)
    for k, chunk in enumerate(np.array_split(ids, folds)):
        fold_id[chunk] = k
    return fold_id


def _ulsif_score(r_den: Array, r_num: Array) -> float:
    """Validation score for uLSIF objective: 1/2 E_q[r^2] - E_p[r]."""
    r_den = np.asarray(r_den, dtype=float).reshape(-1)
    r_num = np.asarray(r_num, dtype=float).reshape(-1)
    return float(0.5 * np.mean(r_den * r_den) - np.mean(r_num))


def grr_density_ratio(
    X_num: Array,
    X_den: Array,
    *,
    n_centers: int = 200,
    centers: Optional[Array] = None,
    sigma: Optional[float] = None,
    lam: Optional[float] = None,
    sigma_grid: Optional[Sequence[float]] = None,
    lam_grid: Optional[Sequence[float]] = None,
    cv: bool = True,
    folds: int = 5,
    random_state: Optional[int] = None,
    standardize: bool = False,
    generator: Optional[BregmanGenerator] = None,
    max_iter: int = 500,
    tol: float = 1e-8,
    verbose: bool = False,
) -> DensityRatioResult:
    """Estimate a density ratio using GRR with a Gaussian-kernel RKHS basis.

    Parameters
    ----------
    X_num:
        Samples from the numerator distribution p(x).
    X_den:
        Samples from the denominator distribution q(x).
    n_centers:
        Number of Gaussian-kernel centers (ignored if ``centers`` is provided).
    centers:
        Optional centers array of shape (m, d). If ``None``, we sample
        ``min(n_centers, len(X_num))`` points from ``X_num``.
    sigma:
        RBF bandwidth. If ``None`` and ``cv=True``, chosen by cross validation.
    lam:
        Ridge regularization strength. If ``None`` and ``cv=True``, chosen by CV.
    sigma_grid, lam_grid:
        Candidate grids for CV. If omitted, a default logarithmic grid is used.
    cv:
        If True, run K-fold cross validation to select ``sigma`` and ``lam``
        when they are not provided.
    folds:
        Number of CV folds.
    random_state:
        Random seed used for center sampling and fold splitting.
    standardize:
        If True, standardize inputs using pooled mean/std of ``[X_num; X_den]``.
        This mirrors the option used in :mod:`genriesz.nnlsif`.
    generator:
        Bregman generator used to fit the Riesz representer model. If ``None``,
        we use :class:`genriesz.SquaredGenerator` with ``C=0``.
    max_iter, tol:
        Optimization controls passed to :meth:`genriesz.GRR.fit`.
    verbose:
        If True, print CV progress.

    Returns
    -------
    DensityRatioResult
        Contains the fitted :class:`genriesz.GRR` model and selected hyperparameters.
    """

    Xn = _as_2d(X_num)
    Xd = _as_2d(X_den)
    rng = np.random.default_rng(random_state)

    if standardize:
        Z = np.vstack([Xn, Xd])
        mean = Z.mean(axis=0)
        scale = Z.std(axis=0, ddof=0)
        scale[scale == 0] = 1.0
        Xn = (Xn - mean) / scale
        Xd = (Xd - mean) / scale
        if centers is not None:
            centers = (_as_2d(centers) - mean) / scale

    if centers is None:
        m = min(int(n_centers), len(Xn))
        idx = rng.choice(len(Xn), size=m, replace=False)
        centers = Xn[idx]
    else:
        centers = _as_2d(centers)

    if generator is None:
        generator = SquaredGenerator(C=0.0).as_generator()

    # Prepare candidate grids.
    sigma_candidates = np.asarray(sigma_grid if sigma_grid is not None else _default_grid(), dtype=float)
    lam_candidates = np.asarray(lam_grid if lam_grid is not None else _default_grid(), dtype=float)
    if sigma is not None:
        sigma_candidates = np.asarray([float(sigma)], dtype=float)
    if lam is not None:
        lam_candidates = np.asarray([float(lam)], dtype=float)

    if (sigma is None or lam is None) and not cv:
        raise ValueError("If cv=False, you must provide both sigma and lam.")

    # Cross validation (only if needed).
    if cv and (sigma is None or lam is None):
        fold_id_n = _kfold_ids(len(Xn), folds=int(folds), rng=rng)
        fold_id_d = _kfold_ids(len(Xd), folds=int(folds), rng=rng)

        best_score = np.inf
        best_sigma = None
        best_lam = None

        for s in sigma_candidates:
            for l in lam_candidates:
                scores: list[float] = []
                for k in range(int(folds)):
                    n_tr = fold_id_n != k
                    n_va = fold_id_n == k
                    d_tr = fold_id_d != k
                    d_va = fold_id_d == k

                    basis = GaussianRKHSBasis(
                        centers=centers,
                        sigma=float(s),
                        include_bias=False,
                        standardize=False,  # we already standardized externally if requested
                    )
                    mfun = DensityRatioFunctional(X_num=Xn[n_tr])
                    model = GRR(
                        basis=basis,
                        m=mfun,
                        generator=generator,
                        penalty="l2",
                        lam=float(l),
                    ).fit(Xd[d_tr], max_iter=max_iter, tol=tol, verbose=False)

                    r_den = model.predict_alpha(Xd[d_va])
                    r_num = model.predict_alpha(Xn[n_va])
                    scores.append(_ulsif_score(r_den=r_den, r_num=r_num))

                score = float(np.mean(scores))
                if verbose:
                    print(f"[grr_density_ratio][CV] sigma={s:.3g} lam={l:.3g} score={score:.6g}")
                if score < best_score:
                    best_score = score
                    best_sigma = float(s)
                    best_lam = float(l)

        assert best_sigma is not None and best_lam is not None
        sigma = best_sigma
        lam = best_lam

    assert sigma is not None and lam is not None

    # Final fit on the full samples.
    basis_final = GaussianRKHSBasis(
        centers=centers,
        sigma=float(sigma),
        include_bias=False,
        standardize=False,
    )
    mfun_final = DensityRatioFunctional(X_num=Xn)
    model_final = GRR(
        basis=basis_final,
        m=mfun_final,
        generator=generator,
        penalty="l2",
        lam=float(lam),
    ).fit(Xd, max_iter=max_iter, tol=tol, verbose=verbose)

    return DensityRatioResult(model=model_final, sigma=float(sigma), lam=float(lam), centers=centers)
