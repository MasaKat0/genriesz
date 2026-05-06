"""Nearest-neighbor matching as LSIF and Riesz regression.

This module implements:

1) The *matching weights* used by bias-corrected kNN matching:

    w_hat(d, x_i) = 1 + K_{d,M}(i) / M,

where K_{d,M}(i) is the number of times unit i (in group d) is used as one of the M nearest
neighbors for a unit in the opposite group.

2) A local-polynomial extension of the one-step NN–LSIF estimator on the matching kernel.

The local-polynomial construction follows Section 6.3 of

    "Nearest Neighbor Matching as Least Squares Density Ratio Estimation and Riesz Regression".

In particular, for a point of interest x, the model class is

    r_beta(z) = K_{M,x}(z) * beta^T * Psi_p((z-x)/rho_M(x)),

where K_{M,x} is either a catchment-area kernel or an M-NN ball kernel.
We implement the *M-NN ball kernel* version.

References:
- Lin, Ding, Han (Econometrica): matching via density ratio estimation.
- Kato (2026): local-polynomial NN–LSIF extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import factorial
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_2d(x: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array of shape (n, d). Got shape {arr.shape}.")
    return arr


def _as_1d_binary(x: ArrayLike, name: str) -> NDArray[np.int64]:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array of shape (n,). Got shape {arr.shape}.")
    uniq = np.unique(arr)
    if not np.all(np.isin(uniq, [0, 1, 0.0, 1.0])):
        raise ValueError(f"{name} must be binary (0/1). Got unique values: {uniq}.")
    return arr.astype(int)


def _standardize(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Standardize columns of X using mean/std computed on X.

    This matches the `scale(X)` call in Lin et al.'s R replication code.
    """

    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd <= 0, 1.0, sd)
    return (X - mu) / sd


@dataclass(frozen=True)
class NNMatchingWeights:
    """Output of :func:`nn_matching_inverse_propensity_weights`."""

    w: NDArray[np.float64]
    w1: NDArray[np.float64]
    w0: NDArray[np.float64]
    K1_over_M: NDArray[np.float64]
    K0_over_M: NDArray[np.float64]


def nn_matching_inverse_propensity_weights(
    X: ArrayLike,
    D: ArrayLike,
    M: int,
    *,
    standardize: bool = True,
    metric: str = "euclidean",
    algorithm: str = "auto",
    n_jobs: int | None = None,
) -> NNMatchingWeights:
    """Compute NN-matching inverse-propensity weights for ATE.

    This reproduces the weights used by bias-corrected kNN matching:

        w_hat(1, X_i) = 1 + K1M(i)
        w_hat(0, X_i) = 1 + K0M(i)

    where K1M(i) (resp. K0M(i)) is the number of times treated (resp. control) unit i is used
    as one of the M nearest neighbors for a unit in the opposite group, divided by M.

    Parameters
    ----------
    X:
        Covariates of shape (n, d). Do **not** include the treatment indicator.
    D:
        Treatment indicator of shape (n,), with values 0/1.
    M:
        Number of nearest neighbors.
    standardize:
        If True, standardize X column-wise using the full sample (recommended).
    metric, algorithm, n_jobs:
        Parameters for the kNN search. Without scikit-learn, only
        ``metric="euclidean"`` is supported and ``algorithm`` is ignored.
        ``n_jobs`` is passed to SciPy's cKDTree as ``workers``.

    Returns
    -------
    NNMatchingWeights
        Contains full-sample weights w (aligned with rows of X), plus group-specific pieces.

    Notes
    -----
    This matches the logic in Lin et al.'s R code:

        Index1 = knnx.index(X1, X0, M)
        K1M = tabulate(Index1) / M

    and similarly for controls.
    """

    X_ = _as_2d(X, "X")
    D_ = _as_1d_binary(D, "D")

    if M <= 0:
        raise ValueError("M must be a positive integer.")

    if standardize:
        X_ = _standardize(X_)

    X1 = X_[D_ == 1]
    X0 = X_[D_ == 0]
    n1 = len(X1)
    n0 = len(X0)

    if n1 == 0 or n0 == 0:
        raise ValueError("Both treatment groups must be non-empty.")
    if M > min(n0, n1):
        raise ValueError(f"M={M} is larger than min(n0, n1)={min(n0, n1)}.")

    if metric != "euclidean":
        raise NotImplementedError(
            "Only metric='euclidean' is supported without scikit-learn. "
            "Install genriesz[sklearn] if you need alternative metrics."
        )

    # SciPy's cKDTree supports Euclidean kNN queries efficiently.
    from scipy.spatial import cKDTree

    workers = 1
    if n_jobs is not None:
        workers = int(n_jobs)

    # Treated weights: how many times each treated unit is matched to a control unit.
    tree1 = cKDTree(X1)
    _, idx1 = tree1.query(X0, k=M, workers=workers)
    idx1 = np.asarray(idx1)
    counts1 = np.bincount(idx1.reshape(-1), minlength=n1).astype(float)
    K1_over_M = counts1 / float(M)

    # Control weights: how many times each control unit is matched to a treated unit.
    tree0 = cKDTree(X0)
    _, idx0 = tree0.query(X1, k=M, workers=workers)
    idx0 = np.asarray(idx0)
    counts0 = np.bincount(idx0.reshape(-1), minlength=n0).astype(float)
    K0_over_M = counts0 / float(M)

    w1 = 1.0 + K1_over_M
    w0 = 1.0 + K0_over_M

    w = np.empty(len(X_), dtype=float)
    w[D_ == 1] = w1
    w[D_ == 0] = w0

    return NNMatchingWeights(w=w, w1=w1, w0=w0, K1_over_M=K1_over_M, K0_over_M=K0_over_M)


@lru_cache(maxsize=64)
def _multi_indices(d: int, degree: int) -> tuple[tuple[int, ...], ...]:
    """Return all multi-indices u in Z^d_{>=0} with |u|<=degree.

    The ordering is by total degree, then lexicographic within each total degree.
    The first index is always (0, ..., 0) (the intercept).
    """

    if d <= 0:
        raise ValueError("d must be positive.")
    if degree < 0:
        raise ValueError("degree must be >= 0.")

    out: list[tuple[int, ...]] = []

    def rec(pos: int, remaining: int, cur: list[int]) -> None:
        if pos == d - 1:
            cur[pos] = remaining
            out.append(tuple(cur))
            return
        for e in range(remaining + 1):
            cur[pos] = e
            rec(pos + 1, remaining - e, cur)

    cur = [0] * d
    for total in range(degree + 1):
        rec(0, total, cur)

    return tuple(out)


@lru_cache(maxsize=64)
def _multi_index_factorials(indices: tuple[tuple[int, ...], ...]) -> NDArray[np.float64]:
    """Compute u! = prod_k u_k! for each multi-index u."""

    fac = np.empty(len(indices), dtype=float)
    for j, u in enumerate(indices):
        prod = 1
        for uk in u:
            prod *= factorial(int(uk))
        fac[j] = float(prod)
    return fac


def _poly_features(t: NDArray[np.float64], degree: int) -> NDArray[np.float64]:
    """Compute the scaled polynomial feature map Psi_p(t) used in the paper.

    Psi_p(t) := ( t^u / u! )_{u in U_p}, where U_p = {u: |u| <= p}.

    Parameters
    ----------
    t:
        Array of shape (n, d).
    degree:
        Polynomial degree p.

    Returns
    -------
    (n, q_p) feature matrix.
    """

    t = _as_2d(t, "t")
    d = t.shape[1]
    indices = _multi_indices(d, degree)
    fac = _multi_index_factorials(indices)

    Phi = np.empty((len(t), len(indices)), dtype=float)
    for j, u in enumerate(indices):
        # t^u = prod_k t_k^{u_k}
        # For u=(0,...,0), this is 1.
        col = np.ones(len(t), dtype=float)
        for k, uk in enumerate(u):
            if uk:
                col *= t[:, k] ** float(uk)
        Phi[:, j] = col / fac[j]
    return Phi


KernelKind = Literal["ball"]


def local_polynomial_nn_lsif_density_ratio(
    numerator: ArrayLike,
    denominator: ArrayLike,
    eval_points: ArrayLike,
    M: int,
    *,
    degree: int = 0,
    kernel: KernelKind = "ball",
    exclude_self: bool = False,
    ridge: float = 1e-8,
    metric: str = "euclidean",
    algorithm: str = "auto",
    n_jobs: int | None = None,
    verbose: bool = False,
) -> NDArray[np.float64]:
    """Local-polynomial NN–LSIF density-ratio estimation.

    Implements equations (8) in Section 6.3:

        beta_hat(x) = H_hat(x)^{-1} h_hat(x)
        r_hat(x) = e0^T beta_hat(x)

    with the M-NN ball kernel.

    Parameters
    ----------
    numerator:
        Sample from f(1), shape (N1, d).
    denominator:
        Sample from f(0), shape (N0, d).
    eval_points:
        Points x at which to estimate r0(x) = f(1)(x)/f(0)(x), shape (n_eval, d).
    M:
        Neighborhood size.
    degree:
        Polynomial degree p.
    kernel:
        Currently only "ball" (M-NN ball kernel) is implemented.
    exclude_self:
        If True, then when an eval point equals a denominator sample point, we attempt to
        exclude that identical point when defining the M-th neighbor radius.
        This is mostly relevant for in-sample evaluation.
    ridge:
        A small ridge term added to H_hat(x) for numerical stability.
    metric, algorithm, n_jobs:
        Parameters for the kNN search. Without scikit-learn, only
        ``metric="euclidean"`` is supported and ``algorithm`` is ignored.
        ``n_jobs`` is passed to SciPy's cKDTree as ``workers``.
    verbose:
        If True, prints progress every ~500 eval points.

    Returns
    -------
    r_hat:
        Array of shape (n_eval,).

    Notes
    -----
    This implementation is intentionally explicit (a loop over eval points) to keep the logic
    faithful to the definition. For large-scale simulations you may want to vectorize and/or
    parallelize externally.
    """

    if kernel != "ball":
        raise NotImplementedError("Only the M-NN ball kernel is implemented (kernel='ball').")

    Z = _as_2d(numerator, "numerator")
    X = _as_2d(denominator, "denominator")
    x_eval = _as_2d(eval_points, "eval_points")

    if M <= 0:
        raise ValueError("M must be a positive integer.")
    if M > len(X):
        raise ValueError(f"M={M} exceeds denominator sample size N0={len(X)}.")
    if degree < 0:
        raise ValueError("degree must be >= 0.")

    N1 = float(len(Z))
    N0 = float(len(X))

    if metric != "euclidean":
        raise NotImplementedError(
            "Only metric='euclidean' is supported without scikit-learn. "
            "Install genriesz[sklearn] if you need alternative metrics."
        )

    from scipy.spatial import cKDTree

    workers = 1
    if n_jobs is not None:
        workers = int(n_jobs)

    # Trees for neighborhood search.
    tree_denom = cKDTree(X)
    tree_num = cKDTree(Z)

    # Precompute the M-th neighbor radius for each eval point.
    # If exclude_self=True, we query M+1 neighbors and (heuristically) drop exact matches.
    k_for_radius = M + 1 if exclude_self else M
    dist, _ = tree_denom.query(x_eval, k=k_for_radius, workers=workers)
    dist = np.asarray(dist)
    if dist.ndim == 1:
        dist = dist.reshape(-1, 1)

    r_hat = np.empty(len(x_eval), dtype=float)

    for i in range(len(x_eval)):
        if verbose and (i % 500 == 0):
            print(f"[local_polynomial_nn_lsif] eval {i}/{len(x_eval)}")

        dists_i = dist[i]

        # Heuristic: if the nearest neighbor is an exact match (distance ~ 0), treat it as 'self'
        # and use the next M neighbors to define the radius.
        if exclude_self and len(dists_i) == M + 1 and dists_i[0] <= 1e-12:
            rho = float(dists_i[M])
        else:
            rho = float(dists_i[-1])

        # Guard against rho==0 (duplicate points).
        if rho <= 0:
            rho = 1e-12

        x0 = x_eval[i]

        # Denominator points within radius rho.
        denom_idx = tree_denom.query_ball_point(x0, r=rho, workers=workers)
        X_loc = X[denom_idx]

        # Numerator points within radius rho.
        num_idx = tree_num.query_ball_point(x0, r=rho, workers=workers)
        Z_loc = Z[num_idx]

        # Feature maps at scaled coordinates.
        Psi_X = _poly_features((X_loc - x0) / rho, degree=degree)
        Psi_Z = _poly_features((Z_loc - x0) / rho, degree=degree)

        # Empirical H and h (note the normalization by N0 and N1, not by local counts).
        H = (Psi_X.T @ Psi_X) / N0
        # h is (1/N1) sum Psi over local numerator.
        h = Psi_Z.sum(axis=0) / N1 if len(Z_loc) > 0 else np.zeros(Psi_X.shape[1], dtype=float)

        # Solve (H + ridge I) beta = h.
        # Ridge scaling: interpret `ridge` as a multiple of trace(H)/q to be robust across scales.
        q = H.shape[0]
        ridge_scale = ridge
        if ridge_scale > 0:
            ridge_scale = ridge * (np.trace(H) / max(q, 1)) if np.isfinite(np.trace(H)) else ridge
        H_reg = H + ridge_scale * np.eye(q)

        try:
            beta = np.linalg.solve(H_reg, h)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(H_reg, h, rcond=None)[0]

        r_hat[i] = float(beta[0])  # intercept

    return r_hat


@dataclass(frozen=True)
class LocalPolynomialLSIFWeights:
    """Output of :func:`local_polynomial_nn_lsif_inverse_propensity_weights`."""

    w: NDArray[np.float64]
    w1: NDArray[np.float64]
    w0: NDArray[np.float64]


def local_polynomial_nn_lsif_inverse_propensity_weights(
    X: ArrayLike,
    D: ArrayLike,
    M: int,
    *,
    degree: int = 1,
    standardize: bool = True,
    kernel: KernelKind = "ball",
    exclude_self: bool = True,
    ridge: float = 1e-8,
    clip_min: float | None = 1e-8,
    metric: str = "euclidean",
    algorithm: str = "auto",
    n_jobs: int | None = None,
    verbose: bool = False,
) -> LocalPolynomialLSIFWeights:
    """Local-polynomial NN–LSIF inverse-propensity weight estimation for ATE.

    This uses the density ratio representation of the ATE Riesz representer:

    - numerator sample: f(X)   (all units)
    - denominator sample: f(X | D=d)

    For each d in {0,1}, local-polynomial NN–LSIF estimates f(x)/f(x | D=d).

    Parameters
    ----------
    X:
        Covariates of shape (n, d). Do **not** include the treatment indicator.
    D:
        Treatment indicator of shape (n,), with values 0/1.
    M:
        Neighborhood size.
    degree:
        Polynomial degree p (p=0: local constant, p=1: local linear, ...).
    standardize:
        If True, standardize X using the full sample.
    clip_min:
        Optional lower bound for weights. If not None, we set w = max(w, clip_min).
        This is a pragmatic safeguard because unconstrained LSIF may yield negative estimates.

    Returns
    -------
    LocalPolynomialLSIFWeights
        w aligns with the original sample order.
    """

    X_ = _as_2d(X, "X")
    D_ = _as_1d_binary(D, "D")

    if standardize:
        X_ = _standardize(X_)

    Z = X_  # numerator sample is the full covariate sample

    X1 = X_[D_ == 1]
    X0 = X_[D_ == 0]

    if len(X1) == 0 or len(X0) == 0:
        raise ValueError("Both treatment groups must be non-empty.")

    w = np.empty(len(X_), dtype=float)

    # d=1
    w1 = local_polynomial_nn_lsif_density_ratio(
        numerator=Z,
        denominator=X1,
        eval_points=X1,
        M=M,
        degree=degree,
        kernel=kernel,
        exclude_self=exclude_self,
        ridge=ridge,
        metric=metric,
        algorithm=algorithm,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    # d=0
    w0 = local_polynomial_nn_lsif_density_ratio(
        numerator=Z,
        denominator=X0,
        eval_points=X0,
        M=M,
        degree=degree,
        kernel=kernel,
        exclude_self=exclude_self,
        ridge=ridge,
        metric=metric,
        algorithm=algorithm,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    pi1 = float(len(X1) / len(X_))
    pi0 = float(len(X0) / len(X_))
    w1 = w1 / pi1
    w0 = w0 / pi0

    if clip_min is not None:
        w1 = np.maximum(w1, float(clip_min))
        w0 = np.maximum(w0, float(clip_min))

    w[D_ == 1] = w1
    w[D_ == 0] = w0

    return LocalPolynomialLSIFWeights(w=w, w1=w1, w0=w0)
