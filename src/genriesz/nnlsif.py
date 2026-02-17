"""Nearest-neighbor matching weights and local-polynomial NN-LSIF.

This module provides:

* ``nn_matching_weights``: exact M-nearest-neighbor (NN) matching weights for the ATE
  (the "matching-as-weighting" identity).

* ``LocalPolynomialNNLSIF``: a *local-polynomial* extension of the NN-LSIF density
  ratio estimator discussed in

    "Nearest Neighbor Matching as Least Squares Density Ratio Estimation and Riesz
    Regression".

  Two neighborhood constructions are supported:

  - ``kernel='knn_ball'``:
        Uses the M-NN ball around the evaluation point with radius equal to the
        M-th NN distance in the denominator sample.

  - ``kernel='catchment'``:
        Uses a *matching-kernel convention* that is compatible with NN matching.
        The numerator neighborhood is the usual "matched-times" set (reverse kNN)
        and the denominator neighborhood is the M nearest neighbors in the
        denominator sample.

Notes
-----
This implementation intentionally avoids any hidden stabilization heuristics
(e.g. adaptive ridge growth, robust fallbacks, or automatic degree reduction).
If a local normal equation is singular, we use the Moore--Penrose pseudo-inverse,
which corresponds to the minimum-norm solution among minimizers.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

Array = np.ndarray


def _as_2d(X: Array) -> Array:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("Expected a 2D array.")
    return X


def _treatment_and_covariates(X: Array, treatment_index: int) -> Tuple[Array, Array]:
    X = _as_2d(X)
    if not (0 <= int(treatment_index) < X.shape[1]):
        raise ValueError("treatment_index is out of bounds.")
    D = np.asarray(X[:, int(treatment_index)], dtype=int)
    Z = np.delete(X, int(treatment_index), axis=1)
    return D, Z


def _standardize_fit(X: Array) -> Tuple[Array, Array]:
    mean = np.mean(X, axis=0)
    scale = np.std(X, axis=0, ddof=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return mean, scale


def _standardize_transform(X: Array, mean: Array, scale: Array) -> Array:
    return (X - mean) / scale


def nn_matching_weights(
    X: Array,
    treatment_index: int = 0,
    M: int = 1,
    *,
    metric: str = "euclidean",
    standardize: bool = True,
) -> Array:
    """Compute exact NN-matching weights for the ATE.

    Parameters
    ----------
    X:
        Covariates with a binary treatment indicator column. By default we assume
        ``X = [D, Z...]``.
    treatment_index:
        Column index of the treatment indicator ``D``.
    M:
        Number of nearest neighbors used in matching.
    metric:
        Distance metric passed to :class:`scipy.spatial.cKDTree`.
    standardize:
        If True, standardize the covariates ``Z`` (using pooled mean/std) before
        neighbor search.

    Returns
    -------
    w:
        Weights ``w_i`` such that the matching estimator can be written as a
        weighting estimator.

    Notes
    -----
    This follows the same "matched-times" counting logic used in the Lin et al.
    replication code:

    * For each treated unit, match to its ``M`` nearest controls.
    * For each control unit, match to its ``M`` nearest treated units.

    Let ``K0(i)`` be the number of times control unit ``i`` is used as a match for
    treated units, and ``K1(i)`` be the number of times treated unit ``i`` is used
    as a match for control units.

    Then

        w_i = 1 + K1(i)/M  for treated units,
        w_i = 1 + K0(i)/M  for control units.

    """
    X = _as_2d(X)
    if int(M) <= 0:
        raise ValueError("M must be a positive integer.")

    D, Z = _treatment_and_covariates(X, treatment_index=treatment_index)
    if not np.array_equal(np.unique(D), np.array([0, 1])):
        raise ValueError("Treatment indicator must contain both 0 and 1.")

    if standardize:
        mean, scale = _standardize_fit(Z)
        Zs = _standardize_transform(Z, mean, scale)
    else:
        Zs = Z

    Zt = Zs[D == 1]
    Zc = Zs[D == 0]

    tree_c = cKDTree(Zc)
    tree_t = cKDTree(Zt)

    # treated -> control
    _, idx_tc = tree_c.query(Zt, k=int(M), p=2 if metric == "euclidean" else 2)
    idx_tc = np.atleast_2d(idx_tc)
    K0 = np.bincount(idx_tc.ravel(), minlength=Zc.shape[0]).astype(float)

    # control -> treated
    _, idx_ct = tree_t.query(Zc, k=int(M), p=2 if metric == "euclidean" else 2)
    idx_ct = np.atleast_2d(idx_ct)
    K1 = np.bincount(idx_ct.ravel(), minlength=Zt.shape[0]).astype(float)

    w = np.empty(X.shape[0], dtype=float)
    w[D == 0] = 1.0 + K0 / float(M)
    w[D == 1] = 1.0 + K1 / float(M)
    return w


def _poly_multi_indices(d: int, degree: int) -> List[Tuple[int, ...]]:
    """Multi-indices for monomials up to total degree ``degree``.

    We represent a multi-index ``u`` with total degree ``k`` as a length-``k``
    tuple of coordinate indices with repetition. For example ``(0, 0, 2)``
    corresponds to ``t_0^2 t_2``.

    The resulting feature map matches the normalized monomials ``t^u / u!``.
    """
    if d <= 0:
        raise ValueError("d must be positive.")
    if degree < 0:
        raise ValueError("degree must be nonnegative.")

    import itertools

    out: List[Tuple[int, ...]] = [tuple()]  # intercept
    for k in range(1, int(degree) + 1):
        out.extend(itertools.combinations_with_replacement(range(d), k))
    return out


def _poly_features(U: Array, degree: int) -> Array:
    """Compute local-polynomial features up to total degree ``degree``.

    Parameters
    ----------
    U:
        Array of scaled displacements ``(z - x) / rho`` of shape ``(n, d)``.
    degree:
        Polynomial degree.

    Returns
    -------
    Psi:
        Feature matrix of shape ``(n, q)`` where ``q = binom(d+degree, degree)``.
        The first column is 1 (intercept).
    """
    U = _as_2d(U)
    n, d = U.shape
    idxs = _poly_multi_indices(d, degree)

    Psi = np.empty((n, len(idxs)), dtype=float)
    Psi[:, 0] = 1.0

    from math import factorial

    for j, mi in enumerate(idxs[1:], start=1):
        # product of coordinates
        mon = np.ones(n, dtype=float)
        if len(mi) > 0:
            for k in mi:
                mon *= U[:, k]

        # divide by u! (product of factorials of per-coordinate powers)
        denom = 1.0
        if len(mi) > 1:
            counts = np.bincount(np.asarray(mi, dtype=int), minlength=d)
            for c in counts:
                if c > 1:
                    denom *= factorial(int(c))
        Psi[:, j] = mon / denom

    return Psi


@dataclass
class LocalPolynomialNNLSIF:
    """Local-polynomial NN-LSIF density-ratio estimator.

    We estimate a local approximation to the density ratio

        r(x) = f_num(x) / f_den(x)

    where ``X_den`` are samples from the denominator distribution and ``X_num``
    are samples from the numerator distribution.

    Parameters
    ----------
    M:
        Neighborhood size parameter.
    degree:
        Polynomial degree ``p``.
    kernel:
        Either ``'knn_ball'`` or ``'catchment'``.
    standardize:
        If True, standardize covariates using pooled mean/std of
        ``[X_den; X_num]``.
    leafsize:
        Leaf size for :class:`scipy.spatial.cKDTree`.
    clip_min:
        Optional lower clipping for predicted ratios (e.g. 0.0). If ``None``, no
        clipping is applied.

    Notes
    -----
    * For ``kernel='catchment'`` we use the matching-kernel convention:
        - denominator neighborhood: the M nearest neighbors in the denominator sample
        - numerator neighborhood: the reverse-kNN (matched-times) set with respect to
          the denominator sample.

    * No automatic stabilization is performed.
    """

    M: int = 1
    degree: int = 0
    kernel: str = "knn_ball"
    standardize: bool = True
    leafsize: int = 16
    clip_min: Optional[float] = None

    # fitted state
    X_den_: Optional[Array] = None
    X_num_: Optional[Array] = None
    n_den_: int = 0
    n_num_: int = 0
    mean_: Optional[Array] = None
    scale_: Optional[Array] = None
    tree_den_: Optional[cKDTree] = None
    tree_num_: Optional[cKDTree] = None
    # for catchment: reverse neighbor list mapping denominator index -> list of numerator indices
    num_rev_: Optional[List[List[int]]] = None

    def fit(self, X_den: Array, X_num: Array) -> "LocalPolynomialNNLSIF":
        X_den = _as_2d(X_den)
        X_num = _as_2d(X_num)
        if X_den.shape[1] != X_num.shape[1]:
            raise ValueError("X_den and X_num must have the same number of columns.")
        if int(self.M) <= 0:
            raise ValueError("M must be a positive integer.")
        if int(self.degree) < 0:
            raise ValueError("degree must be nonnegative.")
        if self.kernel not in {"knn_ball", "catchment"}:
            raise ValueError("kernel must be either 'knn_ball' or 'catchment'.")

        self.n_den_ = int(X_den.shape[0])
        self.n_num_ = int(X_num.shape[0])
        if self.n_den_ == 0 or self.n_num_ == 0:
            raise ValueError("Both denominator and numerator samples must be non-empty.")

        if self.standardize:
            pooled = np.vstack([X_den, X_num])
            mean, scale = _standardize_fit(pooled)
            self.mean_ = mean
            self.scale_ = scale
            Xd = _standardize_transform(X_den, mean, scale)
            Xn = _standardize_transform(X_num, mean, scale)
        else:
            self.mean_ = None
            self.scale_ = None
            Xd = np.asarray(X_den, dtype=float)
            Xn = np.asarray(X_num, dtype=float)

        self.X_den_ = Xd
        self.X_num_ = Xn
        self.tree_den_ = cKDTree(Xd, leafsize=int(self.leafsize))
        self.tree_num_ = cKDTree(Xn, leafsize=int(self.leafsize))

        if self.kernel == "catchment":
            # reverse kNN: for each numerator point, collect its M nearest denominator points
            _, idx = self.tree_den_.query(Xn, k=int(self.M))
            idx = np.atleast_2d(idx)
            rev: List[List[int]] = [[] for _ in range(self.n_den_)]
            for j in range(self.n_num_):
                for den_i in idx[j]:
                    rev[int(den_i)].append(int(j))
            self.num_rev_ = rev
        else:
            self.num_rev_ = None

        return self

    def _transform_eval(self, X_eval: Array) -> Array:
        X_eval = _as_2d(X_eval)
        if self.standardize:
            assert self.mean_ is not None and self.scale_ is not None
            return _standardize_transform(X_eval, self.mean_, self.scale_)
        return np.asarray(X_eval, dtype=float)

    def predict(self, X_eval: Array) -> Array:
        """Predict the density ratio at evaluation points."""
        if self.X_den_ is None or self.X_num_ is None or self.tree_den_ is None or self.tree_num_ is None:
            raise RuntimeError("Call fit() before predict().")

        Xq = self._transform_eval(X_eval)
        out = np.empty(Xq.shape[0], dtype=float)

        if self.kernel == "knn_ball":
            for i in range(Xq.shape[0]):
                out[i] = self._predict_one_knn_ball(Xq[i])
        else:
            for i in range(Xq.shape[0]):
                out[i] = self._predict_one_catchment(Xq[i])

        if self.clip_min is not None:
            out = np.maximum(out, float(self.clip_min))
        return out

    def _predict_one_knn_ball(self, x: Array) -> float:
        assert self.tree_den_ is not None and self.tree_num_ is not None
        assert self.X_den_ is not None and self.X_num_ is not None

        # Use k=M+1 and drop a potential self-match (distance ~0) to avoid rho=0.
        k = int(self.M) + 1
        dists, idxs = self.tree_den_.query(x, k=k)
        dists = np.asarray(dists, dtype=float).ravel()
        idxs = np.asarray(idxs, dtype=int).ravel()

        if dists[0] == 0.0:
            den_idx = idxs[1 : int(self.M) + 1]
            rho = float(dists[int(self.M)])
        else:
            den_idx = idxs[: int(self.M)]
            rho = float(dists[int(self.M) - 1])

        if rho <= 0.0:
            raise ValueError("Non-positive neighborhood radius encountered.")

        # numerator points within the ball
        num_idx = self.tree_num_.query_ball_point(x, r=rho)

        if len(num_idx) == 0:
            return 0.0

        U_den = (self.X_den_[den_idx] - x[None, :]) / rho
        U_num = (self.X_num_[np.asarray(num_idx, dtype=int)] - x[None, :]) / rho

        Psi_d = _poly_features(U_den, degree=int(self.degree))
        Psi_n = _poly_features(U_num, degree=int(self.degree))

        H = Psi_d.T @ Psi_d
        h = Psi_n.sum(axis=0)

        beta = np.linalg.pinv(H) @ h
        return float((self.n_den_ / self.n_num_) * beta[0])

    def _predict_one_catchment(self, x: Array) -> float:
        assert self.tree_den_ is not None and self.X_den_ is not None
        assert self.X_num_ is not None and self.num_rev_ is not None

        # Map x to the closest denominator point (catchment convention evaluates at denom points).
        dist0, den_center = self.tree_den_.query(x, k=1)
        den_center = int(np.asarray(den_center).item())

        # Denominator neighborhood: M nearest denominator points around x (excluding a self-match).
        k = int(self.M) + 1
        dists, idxs = self.tree_den_.query(x, k=k)
        dists = np.asarray(dists, dtype=float).ravel()
        idxs = np.asarray(idxs, dtype=int).ravel()

        if dists[0] == 0.0:
            den_idx = idxs[1 : int(self.M) + 1]
            max_den_dist = float(dists[int(self.M)])
        else:
            den_idx = idxs[: int(self.M)]
            max_den_dist = float(dists[int(self.M) - 1])

        # Numerator neighborhood: reverse-kNN set for the nearest denominator anchor.
        num_idx = self.num_rev_[den_center]

        if len(num_idx) == 0:
            return 0.0

        # Bandwidth: max distance among the points that actually enter the local moment conditions.
        # This matches the idea of rho_M(x) as the size of the local neighborhood.
        x_center = self.X_den_[den_center]
        max_num_dist = float(np.max(np.linalg.norm(self.X_num_[np.asarray(num_idx, dtype=int)] - x_center[None, :], axis=1)))
        rho = max(max_den_dist, max_num_dist)
        if rho <= 0.0:
            raise ValueError("Non-positive neighborhood radius encountered.")

        U_den = (self.X_den_[den_idx] - x_center[None, :]) / rho
        U_num = (self.X_num_[np.asarray(num_idx, dtype=int)] - x_center[None, :]) / rho

        Psi_d = _poly_features(U_den, degree=int(self.degree))
        Psi_n = _poly_features(U_num, degree=int(self.degree))

        H = Psi_d.T @ Psi_d
        h = Psi_n.sum(axis=0)

        beta = np.linalg.pinv(H) @ h
        return float((self.n_den_ / self.n_num_) * beta[0])


def local_polynomial_nnlsif_weights(
    X: Array,
    treatment_index: int = 0,
    M: int = 1,
    degree: int = 0,
    kernel: str = "knn_ball",
    *,
    standardize: bool = True,
    leafsize: int = 16,
    clip_min: Optional[float] = None,
) -> Array:
    """Construct inverse-propensity weights via local-polynomial NN-LSIF.

    Given covariates ``X = [D, Z...]`` where ``D`` is a binary treatment indicator,
    this estimates the conditional density ratios

        r10(z) = f(z | D=1) / f(z | D=0),
        r01(z) = f(z | D=0) / f(z | D=1),

    and returns the stabilized inverse-propensity weights

        w_i = D_i * 1/e(Z_i) + (1-D_i) * 1/(1-e(Z_i))

    using

        1/(1-e(z)) = 1 + (p1/p0) * r10(z),
        1/e(z)     = 1 + (p0/p1) * r01(z),

    where ``p1 = P(D=1)`` and ``p0 = P(D=0)``.
    """
    X = _as_2d(X)
    if int(M) <= 0:
        raise ValueError("M must be a positive integer.")

    D, Z = _treatment_and_covariates(X, treatment_index=treatment_index)
    p1 = float(np.mean(D))
    p0 = 1.0 - p1
    if p1 <= 0.0 or p0 <= 0.0:
        raise ValueError("Both treatment groups must be present in the sample.")

    Zt = Z[D == 1]
    Zc = Z[D == 0]

    # r10: treated / control evaluated at control points
    r10 = LocalPolynomialNNLSIF(
        M=int(M),
        degree=int(degree),
        kernel=str(kernel),
        standardize=bool(standardize),
        leafsize=int(leafsize),
        clip_min=clip_min,
    ).fit(Zc, Zt)

    # r01: control / treated evaluated at treated points
    r01 = LocalPolynomialNNLSIF(
        M=int(M),
        degree=int(degree),
        kernel=str(kernel),
        standardize=bool(standardize),
        leafsize=int(leafsize),
        clip_min=clip_min,
    ).fit(Zt, Zc)

    r10_c = r10.predict(Zc)
    r01_t = r01.predict(Zt)

    w = np.empty_like(D, dtype=float)
    w[D == 0] = 1.0 + (p1 / p0) * r10_c
    w[D == 1] = 1.0 + (p0 / p1) * r01_t
    return w
