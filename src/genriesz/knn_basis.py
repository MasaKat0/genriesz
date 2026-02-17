"""k-nearest-neighbor (kNN) catchment-area basis functions.

This module implements the *catchment-area* indicator basis that appears in the
"nearest-neighbor matching as density-ratio estimation / LSIF" discussion in the
paper.

Definition (catchment areas)
----------------------------
Fix a set of *centers* ``C = {c_j}`` in the covariate space. For a query point
``z``, let ``NN_k(z)`` be the set of its ``k`` nearest neighbors in ``C``.

For each center ``c_j``, define the indicator basis function

    phi_j(z) = 1[ c_j \in NN_k(z) ].

Equivalently, ``z`` belongs to the *catchment area* of ``c_j`` (as defined via
the kNN radius) if and only if ``c_j`` is among the ``k`` nearest neighbors of
``z``.

With this basis, every query row has exactly ``k`` ones (up to ties), and the
column sums correspond to the "matched-times" counts often used in
nearest-neighbor matching and kNN density-ratio estimation.

Notes
-----
* This basis is **data dependent**: you must call :meth:`KNNCatchmentBasis.fit`
  to choose the centers.
* The output is a **dense** NumPy array. For large ``n_centers`` this can be
  memory intensive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from scipy.spatial import cKDTree


Array = np.ndarray


def _as_2d(X: Array) -> Array:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array. Got shape {X.shape}.")
    return X


@dataclass
class KNNCatchmentBasis:
    """Catchment-area indicator basis built from kNN relationships.

    Parameters
    ----------
    n_neighbors:
        Number of nearest neighbors ``k``.
    include_bias:
        If True, append a constant feature 1.
    leafsize:
        Leaf size passed to :class:`scipy.spatial.cKDTree`.

    Examples
    --------
    >>> import numpy as np
    >>> from genriesz.knn_basis import KNNCatchmentBasis
    >>> rng = np.random.default_rng(0)
    >>> centers = rng.normal(size=(10, 2))
    >>> queries = rng.normal(size=(3, 2))
    >>> basis = KNNCatchmentBasis(n_neighbors=2).fit(centers)
    >>> Phi = basis(queries)
    >>> Phi.shape
    (3, 10)
    """

    n_neighbors: int = 1
    include_bias: bool = False
    leafsize: int = 16

    centers_: Optional[Array] = None
    tree_: Optional[cKDTree] = None
    n_output_: Optional[int] = None

    def fit(self, X_centers: Array) -> "KNNCatchmentBasis":
        Xc = _as_2d(X_centers)
        if len(Xc) == 0:
            raise ValueError("X_centers must have at least one row.")

        k = int(self.n_neighbors)
        if k <= 0:
            raise ValueError("n_neighbors must be >= 1")
        if k > len(Xc):
            raise ValueError(
                f"n_neighbors={k} is larger than the number of centers ({len(Xc)})."
            )

        self.centers_ = np.asarray(Xc, dtype=float)
        self.tree_ = cKDTree(self.centers_, leafsize=int(self.leafsize))
        self.n_output_ = int(len(self.centers_) + (1 if self.include_bias else 0))
        return self

    def _check_is_fitted(self) -> None:
        if self.tree_ is None or self.centers_ is None or self.n_output_ is None:
            raise RuntimeError("KNNCatchmentBasis is not fitted. Call fit(X_centers) first.")

    def __call__(self, X: Array) -> Array:
        self._check_is_fitted()

        X_in = np.asarray(X, dtype=float)
        Xq = _as_2d(X_in)
        assert self.tree_ is not None
        assert self.centers_ is not None

        k = int(self.n_neighbors)
        # Query indices of k nearest centers.
        # For k=1, SciPy returns shape (n,). We coerce to (n,1).
        _dist, idx = self.tree_.query(Xq, k=k)
        idx = np.asarray(idx)
        if idx.ndim == 1:
            idx = idx.reshape(-1, 1)

        n = len(Xq)
        p = len(self.centers_)
        out = np.zeros((n, p), dtype=float)

        # Multi-hot assignment.
        rows = np.repeat(np.arange(n), idx.shape[1])
        cols = idx.reshape(-1)
        out[rows, cols] = 1.0

        if self.include_bias:
            out = np.concatenate([out, np.ones((n, 1), dtype=float)], axis=1)

        if X_in.ndim == 1:
            return out.reshape(-1)
        return out
