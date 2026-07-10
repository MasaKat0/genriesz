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

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .basis import Basis
from .utils import as_2d


def _toggle_treatment(
    X: NDArray[np.float64], *, treatment_index: int, value: float
) -> NDArray[np.float64]:
    X_cf = X.copy()
    X_cf[:, treatment_index] = float(value)
    return X_cf


PredictFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]
DerivFn = Callable[[NDArray[np.float64], int], NDArray[np.float64]]


# A scalar-valued function on a single regressor row.
GammaFn = Callable[[NDArray[np.float64]], float]


def _as_1d_row(x: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(x, dtype=float).reshape(-1)


@dataclass(frozen=True)
class LinearFunctional:
    """Base class for linear functionals used by GRR."""

    name: str

    def m_basis_matrix(self, X: ArrayLike, basis: Basis) -> NDArray[np.float64]:  # pragma: no cover
        raise NotImplementedError

    def m_from_predictor(  # pragma: no cover
        self, X: ArrayLike, predict: PredictFn
    ) -> NDArray[np.float64]:
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


class CallableFunctional(LinearFunctional):
    """Wrap a Python callable ``m(x, gamma)`` as a :class:`LinearFunctional`.

    This adapter is meant for quick experiments and README-style usage.
    The callable must implement a **linear** map in ``gamma``.

    Parameters
    ----------
    m:
        A callable of the form ``m(x_row, gamma) -> float`` where:

        - ``x_row`` is a 1D regressor vector,
        - ``gamma`` is a callable that maps a 1D regressor vector to a scalar.

        The callable is assumed to be linear in the function argument ``gamma``.
        If it is not, the GRR objective will generally be invalid.
    name:
        Name used in results.

    Notes
    -----
    To solve GRR in a finite-dimensional model ``alpha(x)=phi(x)^T beta``, the solver
    needs the matrix ``M`` such that ``m(X_i, phi(\\cdot)^T beta)=M_i^T beta``.
    For a callable functional we construct ``M`` by repeatedly applying ``m`` to each
    basis coordinate function.

    This is flexible but can be slower than implementing a dedicated
    :class:`~genriesz.functionals.LinearFunctional`.
    """

    m: Callable[[NDArray[np.float64], GammaFn], float]

    def __init__(self, m: Callable[[NDArray[np.float64], GammaFn], float], *, name: str = "custom"):
        super().__init__(name=str(name))
        object.__setattr__(self, "m", m)

    def m_basis_matrix(self, X: ArrayLike, basis: Basis) -> NDArray[np.float64]:
        X_ = as_2d(X)
        n = X_.shape[0]

        # Determine the basis dimension without touching unfitted properties.
        Phi0 = np.asarray(basis(X_[:1]), dtype=float)
        p = int(Phi0.shape[0] if Phi0.ndim == 1 else Phi0.shape[1])

        M = np.empty((n, p), dtype=float)

        # Build M_{i,j} = m(X_i, phi_j). The n*p calls to a black-box `m` are
        # unavoidable, but for a fixed row every coordinate function phi_j
        # evaluates the *same* basis at the same points that `m(X_i, .)` probes.
        # Caching the basis vector per point (per row) collapses the p redundant
        # full-basis evaluations into one per distinct point. The result is
        # identical because a fitted basis is a pure function of its input.
        for i in range(n):
            x_i = _as_1d_row(X_[i])
            cache: dict[bytes, NDArray[np.float64]] = {}

            def phi_vec(
                x_row: NDArray[np.float64], *, _cache=cache, _basis=basis
            ) -> NDArray[np.float64]:
                x1 = _as_1d_row(x_row)
                key = x1.tobytes()
                vec = _cache.get(key)
                if vec is None:
                    Phi = np.asarray(_basis(x1.reshape(1, -1)), dtype=float)
                    vec = Phi if Phi.ndim == 1 else Phi[0]
                    _cache[key] = vec
                return vec

            for j in range(p):
                def phi_j(x_row: NDArray[np.float64], *, _j: int = j, _phi_vec=phi_vec) -> float:
                    return float(_phi_vec(x_row)[_j])

                M[i, j] = float(self.m(x_i, phi_j))

        return M

    def m_from_predictor(self, X: ArrayLike, predict: PredictFn) -> NDArray[np.float64]:
        X_ = as_2d(X)
        out = np.empty(X_.shape[0], dtype=float)

        def gamma(x_row: NDArray[np.float64]) -> float:
            x1 = _as_1d_row(x_row)
            return float(np.asarray(predict(x1.reshape(1, -1)), dtype=float).reshape(-1)[0])

        for i in range(X_.shape[0]):
            out[i] = float(self.m(_as_1d_row(X_[i]), gamma))
        return out


@dataclass(frozen=True)
class ATEFunctional(LinearFunctional):
    """Average treatment effect: E[ gamma(1,Z) - gamma(0,Z) ]."""

    treatment_index: int = 0

    def __init__(self, treatment_index: int = 0):
        super().__init__(name="ATE")
        object.__setattr__(self, "treatment_index", int(treatment_index))

    def m_basis_matrix(self, X: ArrayLike, basis: Basis) -> NDArray[np.float64]:
        X_ = as_2d(X)
        X1 = _toggle_treatment(X_, treatment_index=self.treatment_index, value=1.0)
        X0 = _toggle_treatment(X_, treatment_index=self.treatment_index, value=0.0)
        return np.asarray(basis(X1) - basis(X0), dtype=float)

    def m_from_predictor(self, X: ArrayLike, predict: PredictFn) -> NDArray[np.float64]:
        X_ = as_2d(X)
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

    Parameters
    ----------
    pi:
        Treatment probability E[D] used inside the functional.
    pi_is_estimated:
        Set to True when ``pi`` was estimated as the sample mean of ``D``
        (the ``grr_att``/``grr_did`` wrappers do this). Estimation then adds
        the first-step correction ``-(theta/pi) * (D - pi)`` to the influence
        function so that standard errors account for the estimation of ``pi``.
        Leave False when ``pi`` is externally known.
    """

    treatment_index: int = 0
    pi: float = 0.5
    pi_is_estimated: bool = False

    def __init__(self, *, treatment_index: int = 0, pi: float, pi_is_estimated: bool = False):
        if not np.isfinite(pi) or pi <= 0.0:
            raise ValueError("pi must be positive")
        super().__init__(name="ATT")
        object.__setattr__(self, "treatment_index", int(treatment_index))
        object.__setattr__(self, "pi", float(pi))
        object.__setattr__(self, "pi_is_estimated", bool(pi_is_estimated))

    def m_basis_matrix(self, X: ArrayLike, basis: Basis) -> NDArray[np.float64]:
        X_ = as_2d(X)
        D = X_[:, self.treatment_index].reshape(-1, 1)
        X1 = _toggle_treatment(X_, treatment_index=self.treatment_index, value=1.0)
        X0 = _toggle_treatment(X_, treatment_index=self.treatment_index, value=0.0)
        return (D / self.pi) * (basis(X1) - basis(X0))

    def m_from_predictor(self, X: ArrayLike, predict: PredictFn) -> NDArray[np.float64]:
        X_ = as_2d(X)
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
        return np.asarray(derivative(as_2d(X), self.coordinate), dtype=float).reshape(-1)


@dataclass(frozen=True)
class DIDFunctional(ATTFunctional):
    """Difference-in-differences as ATT on delta outcomes.

    In the notebooks we treat DID as an ATT estimand on the panel difference

        ΔY = Y_post - Y_pre.

    The functional form is identical to ATT (with the same ``pi``), but we keep
    a separate name for clarity.
    """

    def __init__(self, *, treatment_index: int = 0, pi: float, pi_is_estimated: bool = False):
        super().__init__(treatment_index=treatment_index, pi=pi, pi_is_estimated=pi_is_estimated)
        object.__setattr__(self, "name", "DID")
