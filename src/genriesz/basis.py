"""Basis-function utilities for :class:`genriesz.glm.GRRGLM`.

In :class:`genriesz.glm.GRRGLM`, the user supplies a basis function ``phi(W)``.
This module provides a few convenient basis builders:

- :class:`PolynomialBasis` for polynomial feature expansions.
- :class:`RBFRandomFourierBasis` for an RBF-kernel (RKHS) approximation via
  random Fourier features.
- :class:`RBFNystromBasis` for an RBF-kernel (RKHS) approximation via
  Nyström features.
- :class:`TreatmentInteractionBasis` for the common structure
  ``[1, D, psi(Z), D*psi(Z)]`` in binary-treatment problems.

These classes are intentionally lightweight and depend only on NumPy.

Notes
-----
For very high-dimensional bases, prefer using linear functionals that provide
a vectorized ``basis_matrix(W, basis)`` implementation (see
:class:`genriesz.functionals.ATEFunctional`, etc.). This avoids an expensive
``O(n p^2)`` fallback computation of ``m(W_i, basis_j)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Callable, Optional

import numpy as np


Array = np.ndarray


def _as_2d(X: Array) -> Array:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array. Got shape {X.shape}.")
    return X


@dataclass
class PolynomialBasis:
    """Polynomial feature expansion.

    This implements a small subset of ``sklearn.preprocessing.PolynomialFeatures``
    but without any dependency on scikit-learn.

    Parameters
    ----------
    degree:
        Maximum total degree.
    include_bias:
        If True, include the constant feature 1.

    Notes
    -----
    The number of output features grows quickly with the input dimension and
    degree.
    """

    degree: int = 2
    include_bias: bool = True

    # Internal state built lazily
    _tuples: Optional[list[tuple[int, ...]]] = None
    n_input_: Optional[int] = None

    def _build(self, n_input: int) -> None:
        if self.degree < 0:
            raise ValueError("degree must be >= 0")
        tuples: list[tuple[int, ...]] = []
        if self.include_bias:
            tuples.append(())

        for deg in range(1, int(self.degree) + 1):
            tuples.extend(combinations_with_replacement(range(n_input), deg))

        self._tuples = tuples
        self.n_input_ = int(n_input)

    @property
    def n_output_(self) -> int:
        if self._tuples is None:
            raise AttributeError("PolynomialBasis is not initialized yet. Call it once to infer n_input.")
        return len(self._tuples)

    def __call__(self, X: Array) -> Array:
        X_in = np.asarray(X, dtype=float)
        X2 = _as_2d(X_in)

        if self._tuples is None:
            self._build(X2.shape[1])
        assert self._tuples is not None
        assert self.n_input_ is not None

        if X2.shape[1] != self.n_input_:
            raise ValueError(
                f"PolynomialBasis expected input dim {self.n_input_}, got {X2.shape[1]}."
            )

        n = len(X2)
        out = np.empty((n, len(self._tuples)), dtype=float)
        for j, idxs in enumerate(self._tuples):
            if len(idxs) == 0:
                out[:, j] = 1.0
            else:
                out[:, j] = np.prod(X2[:, idxs], axis=1)

        if X_in.ndim == 1:
            return out.reshape(-1)
        return out


@dataclass
class RBFRandomFourierBasis:
    """Random Fourier features for the RBF kernel.

    Approximates the RBF kernel

        k(x,y) = exp(-||x-y||^2 / (2*sigma^2)).

    Parameters
    ----------
    n_features:
        Number of random features.
    sigma:
        RBF bandwidth.
    include_bias:
        If True, append a constant feature 1.
    standardize:
        If True, standardize inputs using mean/std estimated on first call.
        This is a convenience option; for cross-fitting you may prefer to
        standardize externally to avoid leakage.
    random_state:
        Seed for reproducibility.
    """

    n_features: int = 200
    sigma: float = 1.0
    include_bias: bool = False
    standardize: bool = True
    random_state: Optional[int] = None

    # Learned / sampled parameters
    omega_: Optional[Array] = None  # shape (d, n_features)
    b_: Optional[Array] = None      # shape (n_features,)
    mean_: Optional[Array] = None
    scale_: Optional[Array] = None

    def _init_params(self, d: int) -> None:
        if self.n_features <= 0:
            raise ValueError("n_features must be > 0")
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")

        rng = np.random.default_rng(self.random_state)
        self.omega_ = rng.normal(loc=0.0, scale=1.0 / float(self.sigma), size=(d, self.n_features))
        self.b_ = rng.uniform(low=0.0, high=2.0 * np.pi, size=(self.n_features,))

    def _maybe_fit_standardizer(self, X2: Array) -> None:
        if not self.standardize:
            return
        if self.mean_ is None or self.scale_ is None:
            self.mean_ = X2.mean(axis=0)
            scale = X2.std(axis=0)
            scale[scale == 0] = 1.0
            self.scale_ = scale

    def _standardize(self, X2: Array) -> Array:
        if not self.standardize:
            return X2
        assert self.mean_ is not None and self.scale_ is not None
        return (X2 - self.mean_) / self.scale_

    def __call__(self, X: Array) -> Array:
        X_in = np.asarray(X, dtype=float)
        X2 = _as_2d(X_in)
        d = X2.shape[1]

        if self.omega_ is None or self.b_ is None:
            self._init_params(d)

        self._maybe_fit_standardizer(X2)
        Xs = self._standardize(X2)

        assert self.omega_ is not None and self.b_ is not None
        Z = Xs @ self.omega_ + self.b_[None, :]
        out = np.sqrt(2.0 / float(self.n_features)) * np.cos(Z)

        if self.include_bias:
            out = np.concatenate([out, np.ones((len(out), 1))], axis=1)

        if X_in.ndim == 1:
            return out.reshape(-1)
        return out


@dataclass
class RBFNystromBasis:
    """Nyström basis features for the RBF kernel.

    This returns features

        phi(x) = [k(x, c_1), ..., k(x, c_m)]

    where the centers c_j are sampled from the data on first call (or you can
    call :meth:`fit` explicitly).

    Parameters
    ----------
    n_centers:
        Number of Nyström centers.
    sigma:
        RBF bandwidth.
    include_bias:
        If True, append a constant feature 1.
    standardize:
        If True, standardize inputs using mean/std estimated in :meth:`fit`.
    random_state:
        Seed for reproducibility.
    """

    n_centers: int = 200
    sigma: float = 1.0
    include_bias: bool = False
    standardize: bool = True
    random_state: Optional[int] = None

    centers_: Optional[Array] = None
    mean_: Optional[Array] = None
    scale_: Optional[Array] = None

    def fit(self, X: Array) -> "RBFNystromBasis":
        X2 = _as_2d(np.asarray(X, dtype=float))
        if self.n_centers <= 0:
            raise ValueError("n_centers must be > 0")
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")

        if self.standardize:
            self.mean_ = X2.mean(axis=0)
            scale = X2.std(axis=0)
            scale[scale == 0] = 1.0
            self.scale_ = scale
            X2 = (X2 - self.mean_) / self.scale_

        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(X2), size=min(self.n_centers, len(X2)), replace=False)
        self.centers_ = X2[idx]
        return self

    def _standardize(self, X2: Array) -> Array:
        if not self.standardize:
            return X2
        if self.mean_ is None or self.scale_ is None:
            # Fit on first call.
            self.mean_ = X2.mean(axis=0)
            scale = X2.std(axis=0)
            scale[scale == 0] = 1.0
            self.scale_ = scale
        return (X2 - self.mean_) / self.scale_

    @staticmethod
    def _sq_dists(X: Array, C: Array) -> Array:
        # ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x c^T
        x2 = np.sum(X * X, axis=1, keepdims=True)  # (n,1)
        c2 = np.sum(C * C, axis=1, keepdims=True).T  # (1,m)
        return x2 + c2 - 2.0 * (X @ C.T)

    def __call__(self, X: Array) -> Array:
        X_in = np.asarray(X, dtype=float)
        X2 = _as_2d(X_in)
        if self.centers_ is None:
            # Fit on the *raw* inputs; fit() handles standardization.
            self.fit(X2)

        Xs = self._standardize(X2)
        assert self.centers_ is not None

        d2 = self._sq_dists(Xs, self.centers_)
        out = np.exp(-d2 / (2.0 * float(self.sigma) ** 2))

        if self.include_bias:
            out = np.concatenate([out, np.ones((len(out), 1))], axis=1)

        if X_in.ndim == 1:
            return out.reshape(-1)
        return out


@dataclass
class TreatmentInteractionBasis:
    """Build a (binary) treatment-interaction basis from covariate features.

    Many binary-treatment causal estimands parameterize the regression function
    as

        gamma(D,Z) = gamma0(Z) + D * (gamma1(Z) - gamma0(Z)).

    A convenient basis for this structure is

        phi(D,Z) = [1, D, psi(Z), D * psi(Z)].

    Parameters
    ----------
    base_basis:
        Callable ``psi(Z)`` returning (p,) or (n,p).
    include_intercept:
        If True, include the constant 1.
    include_treatment:
        If True, include the raw treatment indicator D as a feature.
    """

    base_basis: Callable[[Array], Array]
    include_intercept: bool = True
    include_treatment: bool = True

    def __call__(self, W: Array) -> Array:
        W_in = np.asarray(W, dtype=float)
        if W_in.ndim == 1:
            d = float(W_in[0])
            z = W_in[1:]
            psi = np.asarray(self.base_basis(z), dtype=float).reshape(-1)
            parts: list[Array] = []
            if self.include_intercept:
                parts.append(np.array([1.0]))
            if self.include_treatment:
                parts.append(np.array([d]))
            parts.append(psi)
            parts.append(d * psi)
            return np.concatenate(parts)

        d = W_in[:, [0]]
        z = W_in[:, 1:]
        psi = np.asarray(self.base_basis(z), dtype=float)
        if psi.ndim == 1:
            psi = psi.reshape(len(W_in), -1)

        parts2: list[Array] = []
        if self.include_intercept:
            parts2.append(np.ones((len(W_in), 1)))
        if self.include_treatment:
            parts2.append(d)
        parts2.append(psi)
        parts2.append(d * psi)
        return np.concatenate(parts2, axis=1)
