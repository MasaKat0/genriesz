"""Optional scikit-learn based bases.

The core *genriesz* package does not require scikit-learn for the GRR solvers,
but many users will want to use tree-based feature maps.

This module provides a wrapper that turns RandomForest leaf indices into a
one-hot feature map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .basis import BaseBasis


@dataclass
class RandomForestLeafBasis(BaseBasis):
    """One-hot encoding of leaf indices from a fitted RandomForest.

    Parameters
    ----------
    model:
        A scikit-learn estimator with a ``fit`` method and an ``apply`` method
        (e.g., :class:`sklearn.ensemble.RandomForestRegressor`).
    include_bias:
        If True, prepend a constant-1 column.
    """

    model: object
    include_bias: bool = True

    def __post_init__(self) -> None:
        self._encoder = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None):
        try:
            from sklearn.preprocessing import OneHotEncoder
        except Exception as e:  # pragma: no cover
            raise ImportError("RandomForestLeafBasis requires scikit-learn") from e

        X_ = np.asarray(X, dtype=float)
        if y is not None:
            self.model.fit(X_, np.asarray(y))

        leaves = self.model.apply(X_)
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
            raise RuntimeError("RandomForestLeafBasis must be fit before use")
        n = int(self._encoder.transform([[0] * self._encoder.n_features_in_]).shape[1])
        return n + (1 if self.include_bias else 0)

    def __call__(self, X: ArrayLike) -> NDArray[np.float64]:
        if self._encoder is None:
            raise RuntimeError("RandomForestLeafBasis must be fit before use")
        X_ = np.asarray(X, dtype=float)
        leaves = self.model.apply(X_)
        if leaves.ndim == 3 and leaves.shape[-1] == 1:
            leaves = leaves[:, :, 0]

        F = self._encoder.transform(leaves).astype(float)
        if self.include_bias:
            F = np.concatenate([np.ones((F.shape[0], 1), dtype=float), F], axis=1)
        return F
