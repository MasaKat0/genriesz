"""Common linear functionals (estimands) for GRR.

The core solver :class:`genriesz.GRR` accepts an estimand written as a linear
functional of an unknown regression function

    gamma(x) = E[Y | X=x].

Concretely, the solver expects a callable

    m(x, gamma) -> float,

where ``x`` is a single regressor row and ``gamma`` is a callable that maps a
single row to a scalar.

Vectorized basis-matrix support
-------------------------------
For many standard estimands, we can compute the matrix

    M_{ij} = m(X_i, phi_j)

in a vectorized way given a basis function ``phi(X)``. This avoids calling ``m``
for every (i, j) pair and is essential for large bases (RKHS random features,
tree-leaf bases, neural embeddings, ...).

If you use one of the classes below, :class:`genriesz.GRR` automatically uses its
vectorized :meth:`basis_matrix` implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

Array = np.ndarray


class SupportsBasisMatrix(Protocol):
    """Protocol for functionals that support vectorized M-matrix computation."""

    def __call__(self, x: Array, gamma: Callable[[Array], float]) -> float:  # pragma: no cover
        ...

    def basis_matrix(self, X: Array, basis: Callable[[Array], Array]) -> Array:  # pragma: no cover
        ...


@dataclass
class ATEFunctional:
    r"""Average Treatment Effect (ATE) for a binary treatment.

    Assume the regressor has the form ``X = [D, Z]``, where ``D`` is a binary
    treatment indicator and ``Z`` are covariates.

    The ATE target is

    .. math::

        \theta = \mathbb{E}\bigl[\gamma(1, Z) - \gamma(0, Z)\bigr],

    corresponding to the linear functional

    .. math::

        m(X, \gamma) = \gamma(1, Z) - \gamma(0, Z).

    The vectorized implementation computes

    ``M = basis(X1) - basis(X0)``,

    where ``X1`` sets ``D=1`` and ``X0`` sets ``D=0``.
    """

    treatment_index: int = 0
    treat_value_1: float = 1.0
    treat_value_0: float = 0.0

    def __call__(self, x: Array, gamma: Callable[[Array], float]) -> float:
        x = np.asarray(x, dtype=float).reshape(-1)
        x1 = x.copy()
        x0 = x.copy()
        x1[self.treatment_index] = self.treat_value_1
        x0[self.treatment_index] = self.treat_value_0
        return float(gamma(x1) - gamma(x0))

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
class ATTFunctional:
    r"""Average Treatment Effect on the Treated (ATT) for a binary treatment.

    Assume the regressor has the form ``X = [D, Z]``, where ``D`` is a binary
    treatment indicator.

    Let :math:`\pi = \mathbb{P}(D=1)`. The ATT target can be written as

    .. math::

        \theta = \mathbb{E}\bigl[\gamma(1,Z) - \gamma(0,Z) \mid D=1\bigr]
        = \frac{1}{\pi} \mathbb{E}\bigl[D\,(\gamma(1,Z) - \gamma(0,Z))\bigr].

    This corresponds to the linear functional

    .. math::

        m(X, \gamma) = \frac{D}{\pi}\bigl(\gamma(1,Z) - \gamma(0,Z)\bigr).

    Notes
    -----
    In finite samples, a plug-in estimate of :math:`\pi` is often used. The
    high-level wrapper :func:`genriesz.grr_att` sets ``pi`` to the sample mean
    of ``D`` (or of ``D == treat_value_1``).
    """

    treatment_index: int = 0
    treat_value_1: float = 1.0
    treat_value_0: float = 0.0
    pi: float | None = None

    def _check_pi(self) -> float:
        if self.pi is None:
            raise ValueError(
                "ATTFunctional requires 'pi' to be set (pi = P[D=1]). "
                "Use genriesz.grr_att(...) or set pi explicitly."
            )
        pi = float(self.pi)
        if not np.isfinite(pi) or pi <= 0.0:
            raise ValueError(f"ATTFunctional requires pi > 0 and finite. Got pi={self.pi}.")
        return pi

    def __call__(self, x: Array, gamma: Callable[[Array], float]) -> float:
        pi = self._check_pi()
        x = np.asarray(x, dtype=float).reshape(-1)
        d = float(x[self.treatment_index])
        x1 = x.copy()
        x0 = x.copy()
        x1[self.treatment_index] = self.treat_value_1
        x0[self.treatment_index] = self.treat_value_0
        return float((d / pi) * (gamma(x1) - gamma(x0)))

    def basis_matrix(self, X: Array, basis: Callable[[Array], Array]) -> Array:
        pi = self._check_pi()
        X2 = np.asarray(X, dtype=float)
        if X2.ndim != 2:
            raise ValueError("X must be 2D.")
        d = X2[:, self.treatment_index].astype(float)
        X1 = X2.copy()
        X0 = X2.copy()
        X1[:, self.treatment_index] = self.treat_value_1
        X0[:, self.treatment_index] = self.treat_value_0
        M = np.asarray(basis(X1), dtype=float) - np.asarray(basis(X0), dtype=float)
        return (d[:, None] / float(pi)) * M


@dataclass
class AverageDerivativeFunctional:
    r"""Average derivative (average marginal effect) with respect to one coordinate.

    Given a target

    .. math::

        \theta = \mathbb{E}\left[\frac{\partial}{\partial x_k}\gamma(X)\right],

    the linear functional is

    .. math::

        m(X, \gamma) = \frac{\partial}{\partial x_k}\gamma(X).

    This implementation uses a symmetric finite difference with step ``eps``.
    """

    coordinate: int = 0
    eps: float = 1e-4

    def __call__(self, x: Array, gamma: Callable[[Array], float]) -> float:
        x = np.asarray(x, dtype=float).reshape(-1)
        xp = x.copy()
        xm = x.copy()
        xp[self.coordinate] += self.eps
        xm[self.coordinate] -= self.eps
        return float((gamma(xp) - gamma(xm)) / (2.0 * self.eps))

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
