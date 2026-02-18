"""Built-in linear functionals (estimands).

The central object in *genriesz* is a (typically) linear functional

    theta = E[ m(X, gamma0) ],

where ``gamma0(x) = E[Y | X=x]`` is the outcome regression and ``m`` is a
user-specified linear operator acting on functions.

For models ``v(x) = phi(x)^T beta`` (where ``phi`` is a basis), linearity means

    m(X_i, v) = M_i^T beta

for some row vector ``M_i`` that depends on ``X_i`` and the basis.

This module provides built-in functionals used in the notebooks:

- ATE, ATT, and DID (as ATT on delta outcomes)
- AME (average marginal effect / average derivative)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .basis import Basis


def _as_2d(X: ArrayLike) -> NDArray[np.float64]:
    X_ = np.asarray(X, dtype=float)
    if X_.ndim != 2:
        raise ValueError(f"X must be 2D. Got shape {X_.shape}.")
    return X_


def _toggle_treatment(
    X: NDArray[np.float64], *, treatment_index: int, value: float
) -> NDArray[np.float64]:
    X_cf = X.copy()
    X_cf[:, treatment_index] = float(value)
    return X_cf


PredictFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]
DerivFn = Callable[[NDArray[np.float64], int], NDArray[np.float64]]


@dataclass(frozen=True)
class LinearFunctional:
    """Base class for linear functionals used by GRR."""

    name: str

    def m_basis_matrix(self, X: ArrayLike, basis: Basis) -> NDArray[np.float64]:  # pragma: no cover
        raise NotImplementedError

    def m_from_predictor(self, X: ArrayLike, predict: PredictFn) -> NDArray[np.float64]:  # pragma: no cover
        raise NotImplementedError

    def m_from_function(
        self,
        X: ArrayLike,
        *,
        predict: PredictFn,
        derivative: DerivFn | None = None,
    ) -> NDArray[np.float64]:
        """Apply the functional to a generic function.

        This is mostly used for TMLE updates where we need ``m(X, alpha_hat)``.
        """

        return self.m_from_predictor(X, predict)


@dataclass(frozen=True)
class ATEFunctional(LinearFunctional):
    """Average treatment effect: E[ gamma(1,Z) - gamma(0,Z) ]."""

    treatment_index: int = 0

    def __init__(self, treatment_index: int = 0):
        super().__init__(name="ATE")
        object.__setattr__(self, "treatment_index", int(treatment_index))

    def m_basis_matrix(self, X: ArrayLike, basis: Basis) -> NDArray[np.float64]:
        X_ = _as_2d(X)
        X1 = _toggle_treatment(X_, treatment_index=self.treatment_index, value=1.0)
        X0 = _toggle_treatment(X_, treatment_index=self.treatment_index, value=0.0)
        return np.asarray(basis(X1) - basis(X0), dtype=float)

    def m_from_predictor(self, X: ArrayLike, predict: PredictFn) -> NDArray[np.float64]:
        X_ = _as_2d(X)
        X1 = _toggle_treatment(X_, treatment_index=self.treatment_index, value=1.0)
        X0 = _toggle_treatment(X_, treatment_index=self.treatment_index, value=0.0)
        return np.asarray(predict(X1) - predict(X0), dtype=float).reshape(-1)


@dataclass(frozen=True)
class ATTFunctional(LinearFunctional):
    """Average treatment effect on the treated.

    theta = E[ Y(1) - Y(0) | D=1 ]
          = E[ D * (gamma(1,Z) - gamma(0,Z)) ] / E[D].

    We treat this as a *plug-in linear functional* given a fixed value of
    ``pi = E[D]`` (estimated from the sample in the wrapper).
    """

    treatment_index: int = 0
    pi: float = 0.5

    def __init__(self, *, treatment_index: int = 0, pi: float):
        if not np.isfinite(pi) or pi <= 0.0:
            raise ValueError("pi must be positive")
        super().__init__(name="ATT")
        object.__setattr__(self, "treatment_index", int(treatment_index))
        object.__setattr__(self, "pi", float(pi))

    def m_basis_matrix(self, X: ArrayLike, basis: Basis) -> NDArray[np.float64]:
        X_ = _as_2d(X)
        D = X_[:, self.treatment_index].reshape(-1, 1)
        X1 = _toggle_treatment(X_, treatment_index=self.treatment_index, value=1.0)
        X0 = _toggle_treatment(X_, treatment_index=self.treatment_index, value=0.0)
        return (D / self.pi) * (basis(X1) - basis(X0))

    def m_from_predictor(self, X: ArrayLike, predict: PredictFn) -> NDArray[np.float64]:
        X_ = _as_2d(X)
        D = X_[:, self.treatment_index].reshape(-1)
        X1 = _toggle_treatment(X_, treatment_index=self.treatment_index, value=1.0)
        X0 = _toggle_treatment(X_, treatment_index=self.treatment_index, value=0.0)
        return (D / self.pi) * (predict(X1) - predict(X0))


@dataclass(frozen=True)
class AMEFunctional(LinearFunctional):
    """Average marginal effect (average derivative) of gamma wrt x_k."""

    coordinate: int = 0

    def __init__(self, coordinate: int = 0):
        super().__init__(name=f"AME(coord={int(coordinate)})")
        object.__setattr__(self, "coordinate", int(coordinate))

    def m_basis_matrix(self, X: ArrayLike, basis: Basis) -> NDArray[np.float64]:
        return np.asarray(basis.derivative(X, self.coordinate), dtype=float)

    def m_from_predictor(self, X: ArrayLike, predict: PredictFn) -> NDArray[np.float64]:
        raise NotImplementedError(
            "AME requires a derivative-capable predictor; use m_from_function(..., derivative=...)"
        )

    def m_from_function(
        self,
        X: ArrayLike,
        *,
        predict: PredictFn,
        derivative: DerivFn | None = None,
    ) -> NDArray[np.float64]:
        if derivative is None:
            raise ValueError("AME requires derivative()")
        return np.asarray(derivative(_as_2d(X), self.coordinate), dtype=float).reshape(-1)


@dataclass(frozen=True)
class DIDFunctional(ATTFunctional):
    """Difference-in-differences as ATT on delta outcomes.

    In the notebooks we treat DID as an ATT estimand on the panel difference

        ΔY = Y_post - Y_pre.

    The functional form is identical to ATT (with the same ``pi``), but we keep
    a separate name for clarity.
    """

    def __init__(self, *, treatment_index: int = 0, pi: float):
        super().__init__(treatment_index=treatment_index, pi=pi)
        object.__setattr__(self, "name", "DID")
