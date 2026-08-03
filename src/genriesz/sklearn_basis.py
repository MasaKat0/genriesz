"""scikit-learn based bases.

*genriesz* optionally integrates with scikit-learn. This module
provides additional wrappers that turn fitted scikit-learn models into linear
feature maps usable by GRR.

Notes
-----
This file is imported only when you explicitly use it. Keeping these wrappers
in a separate module makes the optional integrations easy to discover.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.preprocessing import OneHotEncoder

from .basis import BaseBasis


def _as_2d_allow_1d(X: ArrayLike) -> tuple[NDArray[np.float64], bool]:
    X_ = np.asarray(X, dtype=float)
    if X_.ndim == 1:
        return X_.reshape(1, -1), True
    if X_.ndim != 2:
        raise ValueError(f"X must be 1D or 2D. Got shape {X_.shape}.")
    return X_, False


@dataclass
class RandomForestLeafBasis(BaseBasis):
    """Leaf encodings from a fitted RandomForest.

    Parameters
    ----------
    model:
        A scikit-learn estimator with a ``fit`` method and an ``apply`` method
        (e.g., :class:`sklearn.ensemble.RandomForestRegressor`).
    include_bias:
        If True, prepend a constant-1 column.
    normalize:
        If True (default), divide each leaf encoding by sqrt(n_estimators) so
        that the row L2-norm stays O(1) regardless of forest size.  Without
        this, the norm grows as sqrt(T) and makes the effective regularisation
        scale T-dependent.

    Attributes
    ----------
    n_output_:
        Alias for the number of output features (including bias if enabled).
        This attribute is provided for compatibility with older example code.
    """

    model: Any
    include_bias: bool = False
    normalize: bool = True

    def __post_init__(self) -> None:
        self._encoder = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None):
        X2, _ = _as_2d_allow_1d(X)
        if y is not None:
            self.model.fit(X2, np.asarray(y))

        leaves = self.model.apply(X2)
        # Some sklearn versions return (n, n_estimators, 1)
        if leaves.ndim == 3 and leaves.shape[-1] == 1:
            leaves = leaves[:, :, 0]

        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        enc.fit(leaves)
        self._encoder = enc
        return self

    @property
    def n_features(self) -> int:
        if self._encoder is None:
            raise RuntimeError("RandomForestLeafBasis must be fit() before use")
        n = int(sum(len(cats) for cats in self._encoder.categories_))
        return n + (1 if self.include_bias else 0)

    @property
    def n_output_(self) -> int:
        return self.n_features

    def __call__(self, X: ArrayLike) -> NDArray[np.float64]:
        if self._encoder is None:
            raise RuntimeError("RandomForestLeafBasis must be fit() before use")

        X2, single = _as_2d_allow_1d(X)
        leaves = self.model.apply(X2)
        if leaves.ndim == 3 and leaves.shape[-1] == 1:
            leaves = leaves[:, :, 0]

        F = self._encoder.transform(leaves).astype(float)
        if self.normalize:
            n_trees = int(self._encoder.n_features_in_)
            if n_trees > 0:
                F /= np.sqrt(n_trees)
        if self.include_bias:
            F = np.concatenate([np.ones((F.shape[0], 1), dtype=float), F], axis=1)
        return F[0] if single else F
