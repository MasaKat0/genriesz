"""Optional scikit-learn based basis functions.

This module adds tree-based feature maps that can be used as bases for
:class:`genriesz.glm.GRRGLM`.

Why tree bases?
--------------
Random forests induce a partition of the covariate space. A common way to turn
this into a linear-in-parameters model is to use **leaf indicators** as
features. Concretely, for each tree we compute the leaf index that an input
falls into and one-hot encode those indices across all trees.

This can be useful as a flexible, nonparametric basis (similar in spirit to
random features for RKHS) while still keeping :class:`genriesz.glm.GRRGLM` convex in
its parameters `beta`.

Notes
-----
This module requires scikit-learn. Install with:

    pip install genriesz[sklearn]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


try:
    from sklearn.preprocessing import OneHotEncoder
except Exception as e:  # pragma: no cover
    raise ImportError(
        "scikit-learn is required for genriesz.sklearn_basis. Install with: pip install genriesz[sklearn]"
    ) from e


Array = np.ndarray


def _as_2d(X: Array) -> Array:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array. Got shape {X.shape}.")
    return X


def _dense_one_hot_encoder() -> OneHotEncoder:
    """Create a dense-output OneHotEncoder across scikit-learn versions."""
    try:
        # Newer scikit-learn
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        # Older scikit-learn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@dataclass
class RandomForestLeafBasis:
    """Leaf-indicator basis from a tree ensemble.

    Parameters
    ----------
    model:
        A scikit-learn estimator with an ``apply(X)`` method such as
        :class:`sklearn.ensemble.RandomForestClassifier` or
        :class:`sklearn.ensemble.RandomForestRegressor`.
    include_bias:
        If True, append a constant feature 1.

    Notes
    -----
    Call :meth:`fit` before using the instance as a basis.

    The output feature dimension equals the total number of distinct leaves
    observed in the fit data across all trees. For stability, consider limiting
    tree depth (e.g., ``max_depth`` or ``max_leaf_nodes``).
    """

    model: Any
    include_bias: bool = False

    encoder_: Optional[OneHotEncoder] = None
    n_output_: Optional[int] = None

    def fit(self, X: Array, y: Optional[Array] = None, **fit_kwargs: Any) -> "RandomForestLeafBasis":
        """Fit the underlying model (if needed) and the leaf one-hot encoder."""
        X2 = _as_2d(X)

        # If the model looks unfitted, try fitting it.
        # Many sklearn estimators expose `n_features_in_` after fit.
        if not hasattr(self.model, "n_features_in_"):
            if y is None:
                raise ValueError(
                    "y must be provided to fit the RandomForestLeafBasis when the model is not fitted."
                )
            self.model.fit(X2, np.asarray(y).reshape(-1), **fit_kwargs)

        leaves = self.model.apply(X2)
        leaves = np.asarray(leaves)
        if leaves.ndim != 2:
            raise ValueError(
                "model.apply(X) must return a 2D array of shape (n_samples, n_trees). "
                f"Got shape {leaves.shape}."
            )

        enc = _dense_one_hot_encoder()
        enc.fit(leaves)
        self.encoder_ = enc
        Z = enc.transform(leaves)
        self.n_output_ = int(Z.shape[1] + (1 if self.include_bias else 0))
        return self

    def __call__(self, X: Array) -> Array:
        if self.encoder_ is None:
            raise RuntimeError("RandomForestLeafBasis is not fitted. Call fit(X, y) first.")

        X_in = np.asarray(X, dtype=float)
        X2 = _as_2d(X_in)
        leaves = np.asarray(self.model.apply(X2))
        Z = np.asarray(self.encoder_.transform(leaves), dtype=float)

        if self.include_bias:
            Z = np.concatenate([Z, np.ones((len(Z), 1))], axis=1)

        if X_in.ndim == 1:
            return Z.reshape(-1)
        return Z
