"""Small utility helpers used across *genriesz*.

The library tries to keep the core dependency footprint small. This module
therefore uses only NumPy and SciPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm


def as_2d(x: ArrayLike, *, name: str) -> NDArray[np.float64]:
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

    if folds <= 1:
        raise ValueError("folds must be >= 2 for cross-fitting")
    idx = np.arange(n, dtype=int)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)

    # Split into approximately equal folds
    parts = np.array_split(idx, int(folds))
    for k in range(int(folds)):
        test = parts[k]
        train = np.concatenate([parts[j] for j in range(int(folds)) if j != k])
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

    n = int(len(psi))
    if n <= 1:
        return float("nan"), float("nan"), float("nan"), float("nan")

    var = float(np.var(psi, ddof=1))
    if not np.isfinite(var) or var < 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    se = float(np.sqrt(var / n))
    if not np.isfinite(se) or se <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    z = float(norm.ppf(1.0 - alpha / 2.0))
    ci_low = float(est - z * se)
    ci_high = float(est + z * se)

    z_stat = float((est - null) / se)
    p_value = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))
    return se, ci_low, ci_high, p_value
