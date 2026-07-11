"""Small utility helpers used across *genriesz*.

The library tries to keep the core dependency footprint small. This module
therefore uses only NumPy and SciPy.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm


def as_2d(x: ArrayLike, *, name: str = "X") -> NDArray[np.float64]:
    """Convert an array-like object to a 2D float64 NumPy array."""

    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array of shape (n, d). Got shape {arr.shape}.")
    return arr


def as_1d(x: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Convert an array-like object to a 1D float64 NumPy array."""

    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array of shape (n,). Got shape {arr.shape}.")
    return arr


def as_1d_of_length(x: ArrayLike, *, n: int, name: str) -> NDArray[np.float64]:
    """Flatten ``x`` to 1D float64 and require it to have exactly ``n`` entries."""

    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"{name} must have length {n}. Got shape {arr.shape}.")
    return arr


def sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable logistic sigmoid, evaluated branch-wise on the sign of ``z``.

    Splitting on ``z >= 0`` keeps ``exp`` away from overflow on either tail:
    the positive branch exponentiates ``-z`` and the negative branch ``z``.
    """

    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    expz = np.exp(z[~pos])
    out[~pos] = expz / (1.0 + expz)
    return out


def standardize_columns(
    X: NDArray[np.float64], *, enabled: bool = True
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Center and scale the columns of ``X``; return ``(Xs, mean, std)``.

    Columns with zero (or non-positive) spread are left unscaled -- their ``std``
    entry is set to 1.0 -- so that a constant column maps to zeros rather than to
    NaN. When ``enabled`` is False the transform is the identity, but ``mean`` and
    ``std`` are still returned so callers can store them unconditionally.
    """

    d = X.shape[1]
    if enabled:
        mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        std = np.where(std > 0, std, 1.0)
    else:
        mean = np.zeros(d)
        std = np.ones(d)
    return (X - mean) / std, mean, std


def is_binary_y(y: NDArray[np.float64]) -> bool:
    """Return True iff ``y`` contains only values in {0, 1}."""

    if y.size == 0:
        return False
    uniq = np.unique(y)
    if uniq.size > 2:
        return False
    return bool(np.all(np.isin(uniq, [0.0, 1.0])))


@dataclass(frozen=True)
class Fold:
    """A single cross-fitting fold."""

    train: NDArray[np.int64]
    test: NDArray[np.int64]


def kfold_splits(
    n: int,
    *,
    folds: int,
    random_state: int | None = None,
    shuffle: bool = True,
) -> Iterator[Fold]:
    """Yield K-fold train/test splits.

    Parameters
    ----------
    n:
        Number of observations.
    folds:
        Number of folds (K).
    random_state:
        Random seed used when ``shuffle=True``.
    shuffle:
        Whether to shuffle indices before splitting.
    """

    n = int(n)
    folds = int(folds)
    if n < 2:
        raise ValueError("n must be >= 2 for cross-fitting")
    if folds <= 1:
        raise ValueError("folds must be >= 2 for cross-fitting")
    if folds > n:
        raise ValueError("folds must be <= n for cross-fitting")
    idx = np.arange(n, dtype=int)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)

    # Split into approximately equal folds
    parts = np.array_split(idx, folds)
    for k in range(folds):
        test = parts[k]
        train = np.concatenate([parts[j] for j in range(folds) if j != k])
        yield Fold(train=train.astype(int), test=test.astype(int))


def se_ci_pvalue(
    est: float,
    psi: NDArray[np.float64],
    *,
    alpha: float,
    null: float,
) -> tuple[float, float, float, float]:
    """Compute standard error, (1-alpha) CI and a two-sided p-value.

    We use the usual normal approximation and the empirical variance of the
    provided influence function (or score) values.
    """

    def _nan_result(reason: str) -> tuple[float, float, float, float]:
        warnings.warn(
            f"se_ci_pvalue returned NaN: {reason}. Downstream tables will show "
            "NaN standard errors and confidence intervals.",
            RuntimeWarning,
            stacklevel=3,
        )
        return float("nan"), float("nan"), float("nan"), float("nan")

    n = int(len(psi))
    if n <= 1:
        return _nan_result("need at least two score values")

    var = float(np.var(psi, ddof=1))
    if not np.isfinite(var) or var < 0:
        return _nan_result("score variance is not finite")

    se = float(np.sqrt(var / n))
    if not np.isfinite(se) or se <= 0:
        return _nan_result("standard error is not finite or not positive")

    z = float(norm.ppf(1.0 - alpha / 2.0))
    ci_low = float(est - z * se)
    ci_high = float(est + z * se)

    z_stat = float((est - null) / se)
    p_value = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))
    return se, ci_low, ci_high, p_value


def bias_proxy(held_out_imbalance: float, outcome_coef_norm: float) -> float:
    """Worst-case first-order bias proxy of a DML/ARW functional estimate.

    The augmented (ARW/one-step) estimator has a second-order remainder that is
    controlled by the product of the Riesz-representer imbalance on the working
    span and the size of the outcome regression on that span. Concretely, for a
    linear outcome model ``gamma(x) = phi(x)^T theta`` the remainder is bounded by::

        |E[(alpha_hat - alpha0)(gamma0 - gamma_hat)]|
            <= max_j |Delta_j| * ||theta||_1  (Hoelder),

    and a convenient scale-free surrogate uses the held-out working-span
    imbalance ``max_j |Delta_j|`` (see :func:`genriesz.estimation.grr_functional`)
    together with the outcome coefficient budget ``||theta||_2``.

    This is a *diagnostic* proxy, not an exact bias. It is used to flag cells
    where the reported interval may under-cover; it is never used to select
    hyper-parameters.

    Parameters
    ----------
    held_out_imbalance:
        The held-out (out-of-fold) working-span imbalance ``max_j |Delta_j|``.
    outcome_coef_norm:
        A norm of the outcome regression coefficients on the same span
        (the "coefficient budget").
    """

    b = float(np.abs(held_out_imbalance) * np.abs(outcome_coef_norm))
    return b if np.isfinite(b) else float("nan")


def coverage_decomposition(
    *,
    estimate: float,
    se: float,
    n: int,
    b_hat: float,
    truth: float | None = None,
) -> dict[str, float]:
    """Assemble a bias / variance / standardized-bias decomposition for a cell.

    Returns a dictionary with the pieces needed for the coverage-failure tables
    (revision plan section 5.2/5.3): the point estimate, its standard error, the
    implied score variance ``V_hat = n * se^2``, the bias proxy ``b_hat`` and the
    standardized bias ``sqrt(n) * b_hat / sqrt(V_hat) = b_hat / se``.

    When ``truth`` is provided (simulation only), the realized signed bias and a
    nominal coverage indicator for the Wald interval are added. These oracle
    quantities are for *evaluation only* and must not drive selection.
    """

    n = int(n)
    se = float(se)
    v_hat = float(n * se * se) if np.isfinite(se) else float("nan")
    std_bias = float(b_hat / se) if np.isfinite(se) and se > 0 else float("nan")

    out: dict[str, float] = {
        "estimate": float(estimate),
        "se": se,
        "v_hat": v_hat,
        "b_hat": float(b_hat),
        "std_bias": std_bias,
    }
    if truth is not None:
        from scipy.stats import norm as _norm

        z = float(_norm.ppf(0.975))
        bias = float(estimate) - float(truth)
        out["bias"] = bias
        out["covered"] = float(abs(bias) <= z * se) if np.isfinite(se) else float("nan")
    return out


def oracle_decomposition(
    *,
    y: ArrayLike,
    alpha_hat: ArrayLike,
    alpha0: ArrayLike,
    gamma_hat: ArrayLike,
    gamma0: ArrayLike,
    m_gamma_hat: ArrayLike,
    m_gamma0: ArrayLike,
) -> dict[str, float]:
    """Simulation-only decomposition of a coverage failure into its sources.

    Coverage collapse in the augmented (ARW/one-step) estimator is driven by the
    *product* of the Riesz error ``alpha_hat - alpha0`` and the outcome error
    ``gamma_hat - gamma0``. Given both true nuisances (available only in
    simulation), this returns the RMS nuisance errors, the empirical product
    drift ``E_n[(alpha0 - alpha_hat)(gamma_hat - gamma0)]`` (the leading bias
    term), and the one-step estimators obtained by substituting each true
    nuisance in turn -- which isolates whether the failure is on the Riesz side,
    the outcome side, or their interaction.

    These quantities are for **evaluation only** and must never be used to select
    hyper-parameters (that would be oracle selection).

    Parameters
    ----------
    y:
        Observed outcome.
    alpha_hat, alpha0:
        Estimated and true Riesz representer values at the sample points.
    gamma_hat, gamma0:
        Estimated and true outcome regression values at the sample points.
    m_gamma_hat, m_gamma0:
        The functional applied to the estimated and true outcome regressions,
        ``m(X_i, gamma_hat)`` and ``m(X_i, gamma0)`` (e.g. from
        ``functional.m_from_function``).
    """

    y = as_1d(np.asarray(y, dtype=float).reshape(-1), name="y")
    a_hat = np.asarray(alpha_hat, dtype=float).reshape(-1)
    a0 = np.asarray(alpha0, dtype=float).reshape(-1)
    g_hat = np.asarray(gamma_hat, dtype=float).reshape(-1)
    g0 = np.asarray(gamma0, dtype=float).reshape(-1)
    mg_hat = np.asarray(m_gamma_hat, dtype=float).reshape(-1)
    mg0 = np.asarray(m_gamma0, dtype=float).reshape(-1)

    n = y.shape[0]
    if not (a_hat.shape[0] == a0.shape[0] == g_hat.shape[0] == g0.shape[0] == n):
        raise ValueError("all nuisance arrays must have the same length as y")

    alpha_rmse = float(np.sqrt(np.mean((a_hat - a0) ** 2)))
    gamma_rmse = float(np.sqrt(np.mean((g_hat - g0) ** 2)))
    product_drift = float(np.mean((a0 - a_hat) * (g_hat - g0)))

    # One-step estimators substituting each true nuisance in turn.
    theta_true_alpha = float(np.mean(mg_hat + a0 * (y - g_hat)))
    theta_true_gamma = float(np.mean(mg0 + a_hat * (y - g0)))
    theta_true_both = float(np.mean(mg0 + a0 * (y - g0)))

    return {
        "alpha_rmse": alpha_rmse,
        "gamma_rmse": gamma_rmse,
        "product_drift": product_drift,
        "theta_true_alpha": theta_true_alpha,
        "theta_true_gamma": theta_true_gamma,
        "theta_true_both": theta_true_both,
    }
