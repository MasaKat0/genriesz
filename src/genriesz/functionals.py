"""Common linear functionals (estimands) for GRR.

The GLM-style estimator :class:`genriesz.genriesz.GRR` is written to accept a generic
callable ``m(X_i, gamma)`` as in the paper.

For many estimands, we can compute the matrix

    M_{ij} = m(X_i, phi_j)

in a *vectorized* way, given a basis function ``phi(X)``.

Why this matters
----------------
The fully generic fallback implementation (treating ``m`` as a black box)
requires calling ``m`` for every (i,j) pair and may end up doing expensive
repeated basis evaluations. That is fine for tiny bases, but becomes
prohibitively slow for RKHS / random-feature / neural-network bases.

If you use one of the classes below, :class:`genriesz.genriesz.GRR` will automatically
use its vectorized :meth:`basis_matrix` implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


Array = np.ndarray


class SupportsBasisMatrix(Protocol):
    """Protocol for linear functionals that support vectorized M-matrix computation."""

    def __call__(self, w: Array, gamma: Callable[[Array], float]) -> float:  # pragma: no cover
        ...

    def basis_matrix(self, X: Array, basis: Callable[[Array], Array]) -> Array:  # pragma: no cover
        ...


@dataclass
class ATEFunctional:
    """Average Treatment Effect (ATE).

For binary treatment D and covariates Z, with W = [D, Z], the ATE target is

    theta = E[ gamma(1, Z) - gamma(0, Z) ].

This corresponds to the linear functional

    m(W, gamma) = gamma(1, Z) - gamma(0, Z).

The vectorized implementation computes

    M = basis(X1) - basis(X0)

where X1 sets D=1 and X0 sets D=0.
"""

    treatment_index: int = 0
    treat_value_1: float = 1.0
    treat_value_0: float = 0.0

    def __call__(self, w: Array, gamma: Callable[[Array], float]) -> float:
        w = np.asarray(w, dtype=float).reshape(-1)
        w1 = w.copy()
        w0 = w.copy()
        w1[self.treatment_index] = self.treat_value_1
        w0[self.treatment_index] = self.treat_value_0
        return float(gamma(w1) - gamma(w0))

    def basis_matrix(self, X: Array, basis: Callable[[Array], Array]) -> Array:
        X2 = np.asarray(X, dtype=float)
        if X2.ndim != 2:
            raise ValueError("X must be 2D.")
        X1 = X2.copy()
        X0 = X2.copy()
        X1[:, self.treatment_index] = self.treat_value_1
        X0[:, self.treatment_index] = self.treat_value_0
        return np.asarray(basis(X1), dtype=float) - np.asarray(basis(X0), dtype=float)


@dataclass
class PolicyEffectFunctional:
    """Average policy effect between two (possibly state-dependent) policies.

Assume W = [D, Z] where D is a (binary or continuous) treatment and Z are
covariates.

Let pi1(z) and pi0(z) be two policies mapping Z to a treatment value.
The policy effect target is

    theta = E[ gamma(pi1(Z), Z) - gamma(pi0(Z), Z) ].

This reduces to ATE when pi1(z) == 1 and pi0(z) == 0.
"""

    policy_1: Callable[[Array], float]
    policy_0: Callable[[Array], float]
    treatment_index: int = 0

    def __call__(self, w: Array, gamma: Callable[[Array], float]) -> float:
        w = np.asarray(w, dtype=float).reshape(-1)
        z = np.delete(w, self.treatment_index)
        w1 = w.copy()
        w0 = w.copy()
        w1[self.treatment_index] = float(self.policy_1(z))
        w0[self.treatment_index] = float(self.policy_0(z))
        return float(gamma(w1) - gamma(w0))

    def basis_matrix(self, X: Array, basis: Callable[[Array], Array]) -> Array:
        X2 = np.asarray(X, dtype=float)
        if X2.ndim != 2:
            raise ValueError("X must be 2D.")

        z = np.delete(X2, self.treatment_index, axis=1)
        # Vectorize policy application rowwise.
        d1 = np.array([float(self.policy_1(z[i])) for i in range(len(z))])
        d0 = np.array([float(self.policy_0(z[i])) for i in range(len(z))])

        X1 = X2.copy()
        X0 = X2.copy()
        X1[:, self.treatment_index] = d1
        X0[:, self.treatment_index] = d0
        return np.asarray(basis(X1), dtype=float) - np.asarray(basis(X0), dtype=float)


@dataclass
class AverageDerivativeFunctional:
    """Average derivative (average marginal effect) with respect to one coordinate.

Given W in R^d and a target

    theta = E[ d/dw_k gamma(W) ],

the linear functional is

    m(W, gamma) = d/dw_k gamma(W).

We implement this via finite differences.
"""

    coordinate: int = 0
    eps: float = 1e-4

    def __call__(self, w: Array, gamma: Callable[[Array], float]) -> float:
        w = np.asarray(w, dtype=float).reshape(-1)
        wp = w.copy()
        wm = w.copy()
        wp[self.coordinate] += self.eps
        wm[self.coordinate] -= self.eps
        return float((gamma(wp) - gamma(wm)) / (2.0 * self.eps))

    def basis_matrix(self, X: Array, basis: Callable[[Array], Array]) -> Array:
        X2 = np.asarray(X, dtype=float)
        if X2.ndim != 2:
            raise ValueError("X must be 2D.")

        Xp = X2.copy()
        Xm = X2.copy()
        Xp[:, self.coordinate] += self.eps
        Xm[:, self.coordinate] -= self.eps
        return (np.asarray(basis(Xp), dtype=float) - np.asarray(basis(Xm), dtype=float)) / (
            2.0 * self.eps
        )
