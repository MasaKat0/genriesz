"""Basis / feature-map utilities.

In *genriesz*, Riesz representers and nuisance regressions are typically fit
in a (possibly high-dimensional) linear model on top of a **basis** / feature
map ``phi(x)``.

The API is intentionally lightweight:

- ``basis.fit(X, y=None)`` (optional)
- ``basis(X) -> (n, p)`` feature matrix
- ``basis.derivative(X, coordinate) -> (n, p)`` (optional; required for AME)

All docstrings and comments are in English as requested.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Basis(Protocol):
    """Protocol for basis objects."""

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "Basis":
        ...

    def __call__(self, X: ArrayLike) -> NDArray[np.float64]:
        ...

    def derivative(self, X: ArrayLike, coordinate: int) -> NDArray[np.float64]:
        ...

    @property
    def n_features(self) -> int:
        ...

    def copy(self) -> "Basis":
        ...


class BaseBasis:
    """Convenience base class implementing ``copy``."""

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "BaseBasis":
        return self

    def copy(self):
        return copy.deepcopy(self)

    @property
    def n_features(self) -> int:
        raise NotImplementedError

    def derivative(self, X: ArrayLike, coordinate: int) -> NDArray[np.float64]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement derivative()."
        )


class PolynomialBasis(BaseBasis):
    """Full polynomial features up to a given total degree.

    This is a small wrapper around ``sklearn.preprocessing.PolynomialFeatures``
    because it provides a convenient and deterministic enumeration of monomials.

    Derivatives are implemented analytically via the monomial exponent table.
    """

    def __init__(self, degree: int = 2, *, include_bias: bool = True):
        if int(degree) < 0:
            raise ValueError("degree must be >= 0")
        self.degree = int(degree)
        self.include_bias = bool(include_bias)

        self._poly = None
        self._powers: NDArray[np.int64] | None = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "PolynomialBasis":
        import sklearn.preprocessing

        X_ = np.asarray(X, dtype=float)
        if X_.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape {X_.shape}.")

        poly = sklearn.preprocessing.PolynomialFeatures(
            degree=self.degree,
            include_bias=self.include_bias,
            interaction_only=False,
            order="C",
        )
        poly.fit(X_)
        self._poly = poly
        self._powers = np.asarray(poly.powers_, dtype=int)
        return self

    @property
    def n_features(self) -> int:
        if self._powers is None:
            raise RuntimeError("PolynomialBasis must be fit() before use.")
        return int(self._powers.shape[0])

    def __call__(self, X: ArrayLike) -> NDArray[np.float64]:
        if self._poly is None:
            # Allow stateless usage by fitting on the fly.
            self.fit(X)
        X_ = np.asarray(X, dtype=float)
        return np.asarray(self._poly.transform(X_), dtype=float)

    def derivative(self, X: ArrayLike, coordinate: int) -> NDArray[np.float64]:
        """Derivative of the feature map wrt ``X[:, coordinate]``.

        Returns a matrix of shape (n, p).
        """

        if self._poly is None or self._powers is None:
            self.fit(X)

        X_ = np.asarray(X, dtype=float)
        if X_.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape {X_.shape}.")

        n, d = X_.shape
        if coordinate < 0 or coordinate >= d:
            raise ValueError(f"coordinate must be in [0, {d-1}]. Got {coordinate}.")

        powers = self._powers  # (p, d)
        p = powers.shape[0]
        pk = powers[:, coordinate].astype(int)  # (p,)

        # Base monomials
        Phi = self.__call__(X_)  # (n, p)

        # Default formula: d/dx_k prod x^p = p_k * prod x^p / x_k.
        xk = X_[:, coordinate].reshape(n, 1)
        xk_safe = np.where(xk != 0.0, xk, 1.0)

        der = Phi * pk.reshape(1, p) / xk_safe

        # Fix the special case: p_k == 1 and x_k == 0.
        # In that case, monomial is x_k * rest, derivative is rest.
        mask_feat = pk == 1
        if np.any(mask_feat):
            mask_obs = (xk.reshape(-1) == 0.0)
            if np.any(mask_obs):
                # Compute rest = prod_{j!=k} x_j^{p_j} for those features.
                other_powers = powers[mask_feat].copy()
                other_powers[:, coordinate] = 0
                # Compute rest via a stable multiplication.
                rest = np.ones((mask_obs.sum(), other_powers.shape[0]), dtype=float)
                X_sub = X_[mask_obs]
                for j in range(d):
                    pj = other_powers[:, j]
                    if np.all(pj == 0):
                        continue
                    rest *= np.power(X_sub[:, [j]], pj.reshape(1, -1))

                der[np.ix_(mask_obs, np.where(mask_feat)[0])] = rest

        # Features with pk == 0 should be exactly 0.
        der[:, pk == 0] = 0.0
        return der


class TreatmentInteractionBasis(BaseBasis):
    """Interaction basis for binary-treatment functionals.

    Given a base basis on covariates ``Z`` (excluding the treatment), this basis
    maps ``X = [D, Z]`` to

        phi(X) = [ D * psi(Z) , (1 - D) * psi(Z) ].

    This is a convenient default for ATE/ATT/DID-style functionals.
    """

    def __init__(self, *, base_basis: BaseBasis, treatment_index: int = 0):
        self.base_basis = base_basis
        self.treatment_index = int(treatment_index)
        self._base_dim: int | None = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "TreatmentInteractionBasis":
        X_ = np.asarray(X, dtype=float)
        if X_.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape {X_.shape}.")
        if self.treatment_index < 0 or self.treatment_index >= X_.shape[1]:
            raise ValueError("treatment_index is out of bounds")

        Z = np.delete(X_, self.treatment_index, axis=1)
        self.base_basis.fit(Z, y=None)
        self._base_dim = self.base_basis.n_features
        return self

    @property
    def n_features(self) -> int:
        if self._base_dim is None:
            raise RuntimeError("TreatmentInteractionBasis must be fit() before use.")
        return 2 * int(self._base_dim)

    def __call__(self, X: ArrayLike) -> NDArray[np.float64]:
        X_ = np.asarray(X, dtype=float)
        if X_.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape {X_.shape}.")
        if self._base_dim is None:
            self.fit(X_)

        D = X_[:, self.treatment_index].reshape(-1, 1)
        if not np.all(np.isin(np.unique(D), [0.0, 1.0])):
            raise ValueError("Treatment column must be binary (0/1).")

        Z = np.delete(X_, self.treatment_index, axis=1)
        Psi = np.asarray(self.base_basis(Z), dtype=float)

        return np.concatenate([D * Psi, (1.0 - D) * Psi], axis=1)

    def derivative(self, X: ArrayLike, coordinate: int) -> NDArray[np.float64]:
        X_ = np.asarray(X, dtype=float)
        if X_.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape {X_.shape}.")
        if self._base_dim is None:
            self.fit(X_)

        if coordinate == self.treatment_index:
            raise ValueError(
                "Derivative w.r.t. the treatment indicator is not supported (binary variable)."
            )

        d = X_.shape[1]
        if coordinate < 0 or coordinate >= d:
            raise ValueError(f"coordinate must be in [0, {d-1}]. Got {coordinate}.")

        # Map full coordinate index -> Z coordinate index
        z_coord = coordinate - 1 if coordinate > self.treatment_index else coordinate

        D = X_[:, self.treatment_index].reshape(-1, 1)
        Z = np.delete(X_, self.treatment_index, axis=1)
        dPsi = self.base_basis.derivative(Z, z_coord)

        return np.concatenate([D * dPsi, (1.0 - D) * dPsi], axis=1)


class RBFRandomFourierBasis(BaseBasis):
    """RBF random Fourier features (Rahimi-Recht) with optional standardization."""

    def __init__(
        self,
        *,
        n_features: int = 500,
        sigma: float = 1.0,
        include_bias: bool = True,
        standardize: bool = True,
        random_state: int | None = None,
    ):
        if int(n_features) <= 0:
            raise ValueError("n_features must be positive")
        if float(sigma) <= 0:
            raise ValueError("sigma must be positive")
        self.n_features_rff = int(n_features)
        self.sigma = float(sigma)
        self.include_bias = bool(include_bias)
        self.standardize = bool(standardize)
        self.random_state = random_state

        self._mean: NDArray[np.float64] | None = None
        self._std: NDArray[np.float64] | None = None
        self._W: NDArray[np.float64] | None = None
        self._b: NDArray[np.float64] | None = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "RBFRandomFourierBasis":
        X_ = np.asarray(X, dtype=float)
        if X_.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape {X_.shape}.")
        n, d = X_.shape

        if self.standardize:
            mean = X_.mean(axis=0)
            std = X_.std(axis=0, ddof=0)
            std = np.where(std > 0, std, 1.0)
        else:
            mean = np.zeros(d)
            std = np.ones(d)

        rng = np.random.default_rng(self.random_state)
        W = rng.normal(loc=0.0, scale=1.0 / self.sigma, size=(d, self.n_features_rff))
        b = rng.uniform(0.0, 2.0 * np.pi, size=self.n_features_rff)

        self._mean = mean.astype(float)
        self._std = std.astype(float)
        self._W = W.astype(float)
        self._b = b.astype(float)
        return self

    @property
    def n_features(self) -> int:
        if self._W is None:
            raise RuntimeError("RBFRandomFourierBasis must be fit() before use.")
        return int(self.n_features_rff + (1 if self.include_bias else 0))

    def __call__(self, X: ArrayLike) -> NDArray[np.float64]:
        if self._W is None or self._b is None or self._mean is None or self._std is None:
            self.fit(X)
        X_ = np.asarray(X, dtype=float)
        Z = (X_ - self._mean) / self._std
        proj = Z @ self._W + self._b
        feats = np.sqrt(2.0 / self.n_features_rff) * np.cos(proj)
        if self.include_bias:
            feats = np.column_stack([np.ones(len(X_), dtype=float), feats])
        return feats.astype(float)


class KNNCatchmentBasis(BaseBasis):
    """kNN catchment (Voronoi) basis.

    After fitting on a set of *centers*, evaluating on query points returns a
    (dense) indicator matrix whose columns correspond to centers.

    Notes
    -----
    - This is mainly intended for small-to-medium center sets used in notebooks.
    - For large-scale matching, prefer the dedicated NN/LSIF utilities.
    """

    def __init__(
        self,
        *,
        n_neighbors: int = 1,
        standardize: bool = True,
        random_state: int | None = None,
    ):
        if int(n_neighbors) <= 0:
            raise ValueError("n_neighbors must be positive")
        self.n_neighbors = int(n_neighbors)
        self.standardize = bool(standardize)
        self.random_state = random_state

        self._centers: NDArray[np.float64] | None = None
        self._mean: NDArray[np.float64] | None = None
        self._std: NDArray[np.float64] | None = None
        self._nn = None

    def fit(self, centers: ArrayLike, y: ArrayLike | None = None) -> "KNNCatchmentBasis":
        import sklearn.neighbors

        C = np.asarray(centers, dtype=float)
        if C.ndim != 2:
            raise ValueError(f"centers must be 2D. Got shape {C.shape}.")

        if self.standardize:
            mean = C.mean(axis=0)
            std = C.std(axis=0, ddof=0)
            std = np.where(std > 0, std, 1.0)
        else:
            mean = np.zeros(C.shape[1])
            std = np.ones(C.shape[1])

        C_std = (C - mean) / std

        nn = sklearn.neighbors.NearestNeighbors(n_neighbors=self.n_neighbors, algorithm="auto")
        nn.fit(C_std)

        self._centers = C
        self._mean = mean
        self._std = std
        self._nn = nn
        return self

    @property
    def n_features(self) -> int:
        if self._centers is None:
            raise RuntimeError("KNNCatchmentBasis must be fit() before use.")
        return int(self._centers.shape[0])

    def __call__(self, X: ArrayLike) -> NDArray[np.float64]:
        if self._nn is None or self._centers is None or self._mean is None or self._std is None:
            raise RuntimeError("KNNCatchmentBasis must be fit() before use.")

        Q = np.asarray(X, dtype=float)
        if Q.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape {Q.shape}.")

        Q_std = (Q - self._mean) / self._std
        _, ind = self._nn.kneighbors(Q_std, return_distance=True)

        n = len(Q)
        m = self.n_features
        Phi = np.zeros((n, m), dtype=float)

        # Mark membership in the selected neighbor set.
        for i in range(n):
            Phi[i, ind[i]] = 1.0
        return Phi
