"""Basis and feature-map utilities.

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
from typing import Callable, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_2d(X: ArrayLike, *, name: str = "X") -> NDArray[np.float64]:
    X_ = np.asarray(X, dtype=float)
    if X_.ndim != 2:
        raise ValueError(f"{name} must be a 2D array. Got shape {X_.shape}.")
    return X_


def _as_2d_allow_1d(X: ArrayLike, *, name: str = "X") -> tuple[NDArray[np.float64], bool]:
    """Return (X2d, is_single).

    This helper lets bases support both batch inputs (n, d) and single-row
    inputs (d,). For single-row inputs, we reshape to (1, d).
    """

    X_ = np.asarray(X, dtype=float)
    if X_.ndim == 1:
        return X_.reshape(1, -1), True
    if X_.ndim == 2:
        return X_, False
    raise ValueError(f"{name} must be 1D or 2D. Got shape {X_.shape}.")


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
    """Convenience base class implementing ``copy`` and a no-op ``fit``."""

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "BaseBasis":
        return self

    def copy(self):
        return copy.deepcopy(self)

    @property
    def n_features(self) -> int:
        raise NotImplementedError

    def derivative(self, X: ArrayLike, coordinate: int) -> NDArray[np.float64]:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement derivative().")


class CallableBasis(BaseBasis):
    """Wrap a Python callable as a basis.

    This is useful when you want to define a custom feature map
    without creating a full class.

    Parameters
    ----------
    func:
        A callable returning a feature matrix. It may accept either a 2D array
        ``X`` or a 1D array ``x``.
    derivative:
        Optional callable implementing the derivative feature map required by
        AME-type functionals.

    Notes
    -----
    We infer ``n_features`` during ``fit`` (or the first call).
    """

    def __init__(
        self,
        func: Callable[[ArrayLike], ArrayLike],
        *,
        derivative: Callable[[ArrayLike, int], ArrayLike] | None = None,
    ):
        self.func = func
        self._derivative = derivative
        self._n_features: int | None = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "CallableBasis":
        Phi = np.asarray(self.__call__(X), dtype=float)
        if Phi.ndim == 1:
            self._n_features = int(Phi.shape[0])
        elif Phi.ndim == 2:
            self._n_features = int(Phi.shape[1])
        else:
            raise ValueError("Callable basis must return 1D or 2D array")
        return self

    @property
    def n_features(self) -> int:
        if self._n_features is None:
            raise RuntimeError("CallableBasis must be fit() before use.")
        return int(self._n_features)

    def __call__(self, X: ArrayLike) -> NDArray[np.float64]:
        X2, single = _as_2d_allow_1d(X)
        out = self.func(X if not single else X2[0])
        Phi = np.asarray(out, dtype=float)

        if single:
            if Phi.ndim == 2 and Phi.shape[0] == 1:
                Phi = Phi[0]
            if Phi.ndim != 1:
                raise ValueError("CallableBasis(func) returned an invalid shape for single-row input")
            if self._n_features is None:
                self._n_features = int(Phi.shape[0])
            return Phi

        if Phi.ndim == 1:
            # If the callable returns (p,) for batch input, treat it as one row.
            Phi = Phi.reshape(1, -1)
        if Phi.ndim != 2 or Phi.shape[0] != X2.shape[0]:
            raise ValueError("CallableBasis(func) must return an array of shape (n, p)")
        if self._n_features is None:
            self._n_features = int(Phi.shape[1])
        return Phi

    def derivative(self, X: ArrayLike, coordinate: int) -> NDArray[np.float64]:
        if self._derivative is None:
            return super().derivative(X, coordinate)
        out = self._derivative(X, int(coordinate))
        return np.asarray(out, dtype=float)


class PolynomialBasis(BaseBasis):
    """Full polynomial features up to a given total degree.

    This class intentionally implements polynomial features without relying on
    scikit-learn so that the *core* package can depend only on NumPy and SciPy.
    The enumeration of monomials is deterministic and ordered by total degree,
    then lexicographic within each total degree.

    Derivatives are implemented analytically via the monomial exponent table.
    """

    def __init__(self, degree: int = 2, *, include_bias: bool = True):
        if int(degree) < 0:
            raise ValueError("degree must be >= 0")
        self.degree = int(degree)
        self.include_bias = bool(include_bias)

        self._powers: NDArray[np.int64] | None = None
        self._n_input_features: int | None = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "PolynomialBasis":
        X2, _ = _as_2d_allow_1d(X)

        d = int(X2.shape[1])
        self._n_input_features = d

        # Enumerate multi-indices u in Z^d_{>=0} with |u|<=degree.
        # Ordering: total degree, then lexicographic.
        powers: list[list[int]] = []

        def rec(pos: int, remaining: int, cur: list[int]) -> None:
            if pos == d - 1:
                cur[pos] = remaining
                powers.append(cur.copy())
                return
            for e in range(remaining + 1):
                cur[pos] = e
                rec(pos + 1, remaining - e, cur)

        cur = [0] * d
        for total in range(self.degree + 1):
            rec(0, total, cur)

        P = np.asarray(powers, dtype=int)
        if not self.include_bias:
            # Drop the intercept term u=(0,...,0), which is first by construction.
            P = P[1:, :]
        self._powers = P
        return self

    @property
    def n_features(self) -> int:
        if self._powers is None:
            raise RuntimeError("PolynomialBasis must be fit() before use.")
        return int(self._powers.shape[0])

    def __call__(self, X: ArrayLike) -> NDArray[np.float64]:
        X2, single = _as_2d_allow_1d(X)
        if self._powers is None or self._n_input_features is None:
            # Allow stateless usage by fitting on the fly.
            self.fit(X2)
        if self._n_input_features is None or int(X2.shape[1]) != int(self._n_input_features):
            raise ValueError("PolynomialBasis received X with a different number of columns than at fit time")

        powers = self._powers
        n, d = X2.shape
        p = int(powers.shape[0])

        Phi = np.ones((n, p), dtype=float)
        for k in range(d):
            pk = powers[:, k]
            if np.all(pk == 0):
                continue
            Phi *= np.power(X2[:, [k]], pk.reshape(1, -1))

        return Phi[0] if single else Phi

    def derivative(self, X: ArrayLike, coordinate: int) -> NDArray[np.float64]:
        """Derivative of the feature map wrt ``X[:, coordinate]``."""

        X2, single = _as_2d_allow_1d(X)
        if self._powers is None or self._n_input_features is None:
            self.fit(X2)
        if self._n_input_features is None or int(X2.shape[1]) != int(self._n_input_features):
            raise ValueError("PolynomialBasis received X with a different number of columns than at fit time")

        n, d = X2.shape
        coordinate = int(coordinate)
        if coordinate < 0 or coordinate >= d:
            raise ValueError(f"coordinate must be in [0, {d-1}]. Got {coordinate}.")

        powers = self._powers  # (p, d)
        p = powers.shape[0]
        pk = powers[:, coordinate].astype(int)  # (p,)

        Phi = np.asarray(self.__call__(X2), dtype=float)  # (n, p)

        # Default formula: d/dx_k prod x^p = p_k * prod x^p / x_k.
        xk = X2[:, coordinate].reshape(n, 1)
        xk_safe = np.where(xk != 0.0, xk, 1.0)

        der = Phi * pk.reshape(1, p) / xk_safe

        # Fix special case: p_k == 1 and x_k == 0. Derivative is the product of other factors.
        mask_feat = pk == 1
        if np.any(mask_feat):
            mask_obs = (xk.reshape(-1) == 0.0)
            if np.any(mask_obs):
                other_powers = powers[mask_feat].copy()
                other_powers[:, coordinate] = 0
                rest = np.ones((mask_obs.sum(), other_powers.shape[0]), dtype=float)
                X_sub = X2[mask_obs]
                for j in range(d):
                    pj = other_powers[:, j]
                    if np.all(pj == 0):
                        continue
                    rest *= np.power(X_sub[:, [j]], pj.reshape(1, -1))
                der[np.ix_(mask_obs, np.where(mask_feat)[0])] = rest

        der[:, pk == 0] = 0.0

        return der[0] if single else der


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
        X2, _ = _as_2d_allow_1d(X)
        if self.treatment_index < 0 or self.treatment_index >= X2.shape[1]:
            raise ValueError("treatment_index is out of bounds")
        D = X2[:, self.treatment_index]
        Z = np.delete(X2, self.treatment_index, axis=1)
        self.base_basis.fit(Z, y=D)
        self._base_dim = int(self.base_basis.n_features)
        return self

    @property
    def n_features(self) -> int:
        if self._base_dim is None:
            raise RuntimeError("TreatmentInteractionBasis must be fit() before use.")
        return 2 * int(self._base_dim)

    def __call__(self, X: ArrayLike) -> NDArray[np.float64]:
        X2, single = _as_2d_allow_1d(X)
        if self._base_dim is None:
            self.fit(X2)

        D = X2[:, self.treatment_index].reshape(-1, 1)
        if not np.all(np.isin(np.unique(D), [0.0, 1.0])):
            raise ValueError("Treatment column must be binary (0/1).")

        Z = np.delete(X2, self.treatment_index, axis=1)
        Psi = np.asarray(self.base_basis(Z), dtype=float)
        out = np.concatenate([D * Psi, (1.0 - D) * Psi], axis=1)
        return out[0] if single else out

    def derivative(self, X: ArrayLike, coordinate: int) -> NDArray[np.float64]:
        X2, single = _as_2d_allow_1d(X)
        if self._base_dim is None:
            self.fit(X2)

        coordinate = int(coordinate)
        if coordinate == self.treatment_index:
            raise ValueError(
                "Derivative w.r.t. the treatment indicator is not supported (binary variable)."
            )

        d = X2.shape[1]
        if coordinate < 0 or coordinate >= d:
            raise ValueError(f"coordinate must be in [0, {d-1}]. Got {coordinate}.")

        # Map full coordinate index -> Z coordinate index
        z_coord = coordinate - 1 if coordinate > self.treatment_index else coordinate

        D = X2[:, self.treatment_index].reshape(-1, 1)
        Z = np.delete(X2, self.treatment_index, axis=1)
        dPsi = self.base_basis.derivative(Z, z_coord)

        out = np.concatenate([D * dPsi, (1.0 - D) * dPsi], axis=1)
        return out[0] if single else out


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
        X2, _ = _as_2d_allow_1d(X)
        n, d = X2.shape

        if self.standardize:
            mean = X2.mean(axis=0)
            std = X2.std(axis=0, ddof=0)
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
        X2, single = _as_2d_allow_1d(X)
        if self._W is None or self._b is None or self._mean is None or self._std is None:
            self.fit(X2)

        Z = (X2 - self._mean) / self._std
        proj = Z @ self._W + self._b
        feats = np.sqrt(2.0 / self.n_features_rff) * np.cos(proj)
        if self.include_bias:
            feats = np.column_stack([np.ones(len(X2), dtype=float), feats])
        feats = feats.astype(float)
        return feats[0] if single else feats

    def derivative(self, X: ArrayLike, coordinate: int) -> NDArray[np.float64]:
        """Derivative of the feature map wrt ``X[:, coordinate]``.

        d/dx_k [ sqrt(2/D) cos(W[:,j]^T (x-mu)/std + b_j) ]
            = -sqrt(2/D) * W[k,j]/std[k] * sin(W[:,j]^T (x-mu)/std + b_j)
        """
        X2, single = _as_2d_allow_1d(X)
        if self._W is None or self._b is None or self._mean is None or self._std is None:
            self.fit(X2)

        n, d = X2.shape
        coordinate = int(coordinate)
        if coordinate < 0 or coordinate >= d:
            raise ValueError(f"coordinate must be in [0, {d - 1}]. Got {coordinate}.")

        Z = (X2 - self._mean) / self._std
        proj = Z @ self._W + self._b  # (n, D)
        scale = self._W[coordinate, :] / self._std[coordinate]  # (D,)
        dfeats = np.sqrt(2.0 / self.n_features_rff) * (-np.sin(proj)) * scale  # (n, D)
        if self.include_bias:
            dfeats = np.column_stack([np.zeros(n, dtype=float), dfeats])
        dfeats = dfeats.astype(float)
        return dfeats[0] if single else dfeats


def _rbf_kernel(
    X: NDArray[np.float64],
    C: NDArray[np.float64],
    *,
    sigma: float,
) -> NDArray[np.float64]:
    """Compute the Gaussian (RBF) kernel matrix K(X, C)."""

    if float(sigma) <= 0:
        raise ValueError("sigma must be positive")
    X = np.asarray(X, dtype=float)
    C = np.asarray(C, dtype=float)

    x2 = np.sum(X * X, axis=1).reshape(-1, 1)
    c2 = np.sum(C * C, axis=1).reshape(1, -1)
    dist2 = x2 + c2 - 2.0 * (X @ C.T)
    dist2 = np.maximum(dist2, 0.0)
    return np.exp(-dist2 / (2.0 * sigma * sigma))


class GaussianRKHSBasis(BaseBasis):
    """Gaussian-kernel RKHS basis using kernel evaluations at fitted center points.

    The feature map is

        phi_j(x) = K(x, c_j),

    where {c_j} are fitted centers (one basis function per center).

    This is a convenient default for RKHS regression in moderate dimensions.
    """

    def __init__(
        self,
        *,
        n_centers: int = 300,
        sigma: float = 1.0,
        include_bias: bool = True,
        standardize: bool = True,
        random_state: int | None = None,
        centers: ArrayLike | None = None,
    ):
        if int(n_centers) <= 0:
            raise ValueError("n_centers must be positive")
        if float(sigma) <= 0:
            raise ValueError("sigma must be positive")
        self.n_centers = int(n_centers)
        self.sigma = float(sigma)
        self.include_bias = bool(include_bias)
        self.standardize = bool(standardize)
        self.random_state = random_state
        self._centers_input = centers

        self._centers: NDArray[np.float64] | None = None
        self._mean: NDArray[np.float64] | None = None
        self._std: NDArray[np.float64] | None = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "GaussianRKHSBasis":
        X2, _ = _as_2d_allow_1d(X)
        n, d = X2.shape

        if self.standardize:
            mean = X2.mean(axis=0)
            std = X2.std(axis=0, ddof=0)
            std = np.where(std > 0, std, 1.0)
        else:
            mean = np.zeros(d)
            std = np.ones(d)

        Xs = (X2 - mean) / std

        if self._centers_input is not None:
            C = np.asarray(self._centers_input, dtype=float)
            if C.ndim != 2 or C.shape[1] != d:
                raise ValueError("centers must be 2D with the same number of columns as X")
            Cs = (C - mean) / std
        else:
            m = min(self.n_centers, n)
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(n, size=m, replace=False)
            Cs = Xs[idx]

        self._mean = mean.astype(float)
        self._std = std.astype(float)
        self._centers = Cs.astype(float)
        return self

    @property
    def centers(self) -> NDArray[np.float64]:
        if self._centers is None:
            raise RuntimeError("GaussianRKHSBasis must be fit() before use.")
        return self._centers

    @property
    def n_features(self) -> int:
        if self._centers is None:
            raise RuntimeError("GaussianRKHSBasis must be fit() before use.")
        m = int(self._centers.shape[0])
        return m + (1 if self.include_bias else 0)

    def __call__(self, X: ArrayLike) -> NDArray[np.float64]:
        X2, single = _as_2d_allow_1d(X)
        if self._centers is None or self._mean is None or self._std is None:
            self.fit(X2)

        Xs = (X2 - self._mean) / self._std
        K = _rbf_kernel(Xs, self._centers, sigma=self.sigma)
        if self.include_bias:
            K = np.column_stack([np.ones(len(X2), dtype=float), K])
        K = K.astype(float)
        return K[0] if single else K


class RBFNystromBasis(BaseBasis):
    """Nyström feature map for the RBF kernel.

    This class computes a Nyström approximation to the implicit RBF-RKHS feature
    map. Compared to :class:`GaussianRKHSBasis`, Nyström features apply an
    additional whitening transform based on the kernel matrix among centers.

    The feature map is

        Phi(x) = K(x, C) (K(C, C) + jitter I)^{-1/2}.

    Notes
    -----
    - This is an O(m^3) preprocessing step in the number of centers ``m``.
    - For large ``m``, consider :class:`RBFRandomFourierBasis`.
    """

    def __init__(
        self,
        *,
        n_centers: int = 300,
        sigma: float = 1.0,
        include_bias: bool = True,
        standardize: bool = True,
        random_state: int | None = None,
        centers: ArrayLike | None = None,
        jitter: float = 1e-8,
    ):
        if int(n_centers) <= 0:
            raise ValueError("n_centers must be positive")
        if float(sigma) <= 0:
            raise ValueError("sigma must be positive")
        if float(jitter) <= 0:
            raise ValueError("jitter must be positive")

        self.n_centers = int(n_centers)
        self.sigma = float(sigma)
        self.include_bias = bool(include_bias)
        self.standardize = bool(standardize)
        self.random_state = random_state
        self._centers_input = centers
        self.jitter = float(jitter)

        self._centers: NDArray[np.float64] | None = None
        self._mean: NDArray[np.float64] | None = None
        self._std: NDArray[np.float64] | None = None
        self._inv_sqrt: NDArray[np.float64] | None = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "RBFNystromBasis":
        X2, _ = _as_2d_allow_1d(X)
        n, d = X2.shape

        if self.standardize:
            mean = X2.mean(axis=0)
            std = X2.std(axis=0, ddof=0)
            std = np.where(std > 0, std, 1.0)
        else:
            mean = np.zeros(d)
            std = np.ones(d)

        Xs = (X2 - mean) / std

        if self._centers_input is not None:
            C = np.asarray(self._centers_input, dtype=float)
            if C.ndim != 2 or C.shape[1] != d:
                raise ValueError("centers must be 2D with the same number of columns as X")
            Cs = (C - mean) / std
        else:
            m = min(self.n_centers, n)
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(n, size=m, replace=False)
            Cs = Xs[idx]

        Kmm = _rbf_kernel(Cs, Cs, sigma=self.sigma)
        Kmm = Kmm + self.jitter * np.eye(Kmm.shape[0])

        # Symmetric eigendecomposition
        evals, evecs = np.linalg.eigh(Kmm)
        evals = np.maximum(evals, self.jitter)
        inv_sqrt = evecs @ (np.diag(1.0 / np.sqrt(evals))) @ evecs.T

        self._mean = mean.astype(float)
        self._std = std.astype(float)
        self._centers = Cs.astype(float)
        self._inv_sqrt = inv_sqrt.astype(float)
        return self

    @property
    def centers(self) -> NDArray[np.float64]:
        if self._centers is None:
            raise RuntimeError("RBFNystromBasis must be fit() before use.")
        return self._centers

    @property
    def n_features(self) -> int:
        if self._centers is None:
            raise RuntimeError("RBFNystromBasis must be fit() before use.")
        m = int(self._centers.shape[0])
        return m + (1 if self.include_bias else 0)

    def __call__(self, X: ArrayLike) -> NDArray[np.float64]:
        X2, single = _as_2d_allow_1d(X)
        if self._centers is None or self._mean is None or self._std is None or self._inv_sqrt is None:
            self.fit(X2)

        Xs = (X2 - self._mean) / self._std
        Knm = _rbf_kernel(Xs, self._centers, sigma=self.sigma)
        Phi = Knm @ self._inv_sqrt
        if self.include_bias:
            Phi = np.column_stack([np.ones(len(X2), dtype=float), Phi])
        Phi = Phi.astype(float)
        return Phi[0] if single else Phi


class KNNCatchmentBasis(BaseBasis):
    """kNN nearest-neighbor indicator basis.

    After fitting on a set of *centers*, evaluating on query points returns a
    (dense) indicator matrix whose entry (i, j) is 1 if center j is among the
    k nearest neighbors of query i, and 0 otherwise.

    Notes
    -----
    - This is mainly intended for small-to-medium center sets used in notebooks.
    - For large-scale matching, prefer the dedicated NN/LSIF utilities.
    """

    def __init__(
        self,
        *,
        n_neighbors: int = 1,
        include_bias: bool = False,
        standardize: bool = True,
        random_state: int | None = None,
    ):
        if int(n_neighbors) <= 0:
            raise ValueError("n_neighbors must be positive")
        self.n_neighbors = int(n_neighbors)
        self.include_bias = bool(include_bias)
        self.standardize = bool(standardize)
        self.random_state = random_state

        self._centers: NDArray[np.float64] | None = None
        self._mean: NDArray[np.float64] | None = None
        self._std: NDArray[np.float64] | None = None
        self._nn = None

    def fit(self, centers: ArrayLike, y: ArrayLike | None = None) -> "KNNCatchmentBasis":
        from scipy.spatial import cKDTree

        C = np.asarray(centers, dtype=float)
        if C.ndim != 2:
            raise ValueError(f"centers must be 2D. Got shape {C.shape}.")
        if C.shape[0] == 0:
            raise ValueError("centers must contain at least one row.")
        if self.n_neighbors > C.shape[0]:
            raise ValueError(
                f"n_neighbors={self.n_neighbors} exceeds the number of centers={C.shape[0]}."
            )

        if self.standardize:
            mean = C.mean(axis=0)
            std = C.std(axis=0, ddof=0)
            std = np.where(std > 0, std, 1.0)
        else:
            mean = np.zeros(C.shape[1])
            std = np.ones(C.shape[1])

        C_std = (C - mean) / std

        tree = cKDTree(C_std)

        self._centers = C
        self._mean = mean
        self._std = std
        self._nn = tree
        return self

    @property
    def n_features(self) -> int:
        if self._centers is None:
            raise RuntimeError("KNNCatchmentBasis must be fit() before use.")
        m = int(self._centers.shape[0])
        return m + (1 if self.include_bias else 0)

    def __call__(self, X: ArrayLike) -> NDArray[np.float64]:
        if self._nn is None or self._centers is None or self._mean is None or self._std is None:
            raise RuntimeError("KNNCatchmentBasis must be fit() before use.")

        Q2, single = _as_2d_allow_1d(X)

        Q_std = (Q2 - self._mean) / self._std

        # cKDTree.query returns shape (n,) when k=1 and (n,k) when k>1.
        _, ind = self._nn.query(Q_std, k=self.n_neighbors)
        ind = np.asarray(ind)
        if ind.ndim == 1:
            ind = ind.reshape(-1, 1)

        n = len(Q2)
        m = int(self._centers.shape[0])
        Phi = np.zeros((n, m), dtype=float)

        for i in range(n):
            Phi[i, ind[i]] = 1.0

        if self.include_bias:
            Phi = np.column_stack([np.ones(n, dtype=float), Phi])

        return Phi[0] if single else Phi
