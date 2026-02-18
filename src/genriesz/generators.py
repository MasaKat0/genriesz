"""Bregman generators used in Generalized Riesz Regression.

The GRR objective is expressed in terms of a convex generator ``g`` and its
convex conjugate ``g*``. For the built-in families we implement a convenient
interface that provides:

- ``alpha = inv_grad(x, v)`` where ``v`` is the linear predictor ``phi(x)^T beta``
- ``g(alpha)``
- ``g*(v)`` via ``g*(v) = v*alpha - g(alpha)`` (with ``alpha = inv_grad(v)``)

A **branch function** ``branch_fn`` can be supplied for generators whose link is
branch-wise (e.g., UKL/BP). It must map a single regressor row ``x`` to either:

- ``1`` (positive branch), or
- ``0`` (negative branch).

If ``branch_fn`` is not provided, the generator defaults to a data-independent
sign choice based on the linear predictor (``sign(v)``), which is useful for
functionals whose Riesz representer can take both signs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

BranchFn = Callable[[NDArray[np.float64]], int]


def _as_2d(X: ArrayLike) -> NDArray[np.float64]:
    X_ = np.asarray(X, dtype=float)
    if X_.ndim != 2:
        raise ValueError(f"X must be 2D. Got shape {X_.shape}.")
    return X_


def _as_1d(v: ArrayLike, n: int) -> NDArray[np.float64]:
    v_ = np.asarray(v, dtype=float).reshape(-1)
    if v_.shape[0] != n:
        raise ValueError(f"v must have length {n}. Got {v_.shape}.")
    return v_


@dataclass
class BregmanGenerator:
    """A concrete Bregman generator with a branch-wise inverse gradient."""

    name: str
    C: float
    branch_fn: BranchFn | None

    def as_generator(self) -> "BregmanGenerator":
        """For API compatibility with earlier drafts."""

        return self

    def _sign(self, X: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return an array of signs (+1/-1) for branch-wise generators."""

        if self.branch_fn is None:
            # Default: choose sign by the linear predictor.
            return np.where(v >= 0.0, 1.0, -1.0)
        s = np.empty(len(v), dtype=float)
        for i in range(len(v)):
            s[i] = 1.0 if int(self.branch_fn(X[i])) == 1 else -1.0
        return s

    # The following methods are overridden by subclasses.
    def inv_grad(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:  # pragma: no cover
        raise NotImplementedError

    def g(self, alpha: ArrayLike) -> NDArray[np.float64]:  # pragma: no cover
        raise NotImplementedError

    def grad2(self, alpha: ArrayLike) -> NDArray[np.float64]:  # pragma: no cover
        """Second derivative of g (elementwise). Needed for derivatives of alpha."""

        raise NotImplementedError

    def conjugate(self, X: ArrayLike, v: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (g*(v), alpha) evaluated row-wise."""

        X_ = _as_2d(X)
        v_ = _as_1d(v, n=len(X_))
        alpha = self.inv_grad(X_, v_)
        g_val = self.g(alpha)
        g_star = v_ * alpha - g_val
        return g_star, alpha


class SquaredGenerator(BregmanGenerator):
    """Squared generator: g(alpha) = (alpha - C)^2.

    This generator has no domain constraints and uses an identity-like link.
    """

    def __init__(self, C: float = 0.0):
        super().__init__(name="SQ", C=float(C), branch_fn=None)

    def inv_grad(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
        X_ = _as_2d(X)
        v_ = _as_1d(v, n=len(X_))
        return self.C + 0.5 * v_

    def g(self, alpha: ArrayLike) -> NDArray[np.float64]:
        a = np.asarray(alpha, dtype=float).reshape(-1)
        return np.square(a - self.C)

    def grad2(self, alpha: ArrayLike) -> NDArray[np.float64]:
        a = np.asarray(alpha, dtype=float).reshape(-1)
        return np.full_like(a, 2.0, dtype=float)


class UKLGenerator(BregmanGenerator):
    """Unnormalized KL generator.

    g(alpha) = (|alpha| - C) log(|alpha| - C) - |alpha|,  with |alpha| > C.

    The inverse gradient is branch-wise:

    - positive branch (sign +1): alpha =  C + exp(v)
    - negative branch (sign -1): alpha = -C - exp(-v)

    If ``branch_fn`` is provided, it determines which branch is used for each
    observation.
    """

    def __init__(self, C: float = 1.0, *, branch_fn: BranchFn | None = None):
        if float(C) < 0:
            raise ValueError("C must be >= 0")
        super().__init__(name="UKL", C=float(C), branch_fn=branch_fn)

    def inv_grad(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
        """Inverse gradient map alpha = (g')^{-1}(v).

        Notes
        -----
        For numerical robustness we clip the exponential term away from zero.
        Extremely negative linear predictors can underflow ``exp`` to 0 in
        float64, which would otherwise violate the strict domain constraint
        ``|alpha| > C``.
        """

        X_ = _as_2d(X)
        v_ = _as_1d(v, n=len(X_))
        s = self._sign(X_, v_)

        # exp can underflow to 0 for large negative inputs; clip and floor.
        z = np.clip(s * v_, -700.0, 700.0)
        exp_term = np.exp(z)
        exp_term = np.maximum(exp_term, 1e-12)

        return s * (self.C + exp_term)

    def g(self, alpha: ArrayLike) -> NDArray[np.float64]:
        a = np.asarray(alpha, dtype=float).reshape(-1)
        t = np.abs(a) - self.C
        # Numerical guard: treat tiny/negative values as boundary.
        t = np.maximum(t, 1e-12)
        return t * np.log(t) - np.abs(a)

    def grad2(self, alpha: ArrayLike) -> NDArray[np.float64]:
        a = np.asarray(alpha, dtype=float).reshape(-1)
        t = np.abs(a) - self.C
        t = np.maximum(t, 1e-12)
        return 1.0 / t

class BPGenerator(BregmanGenerator):
    """Box-Power generator.

    A smooth family interpolating between UKL-like (small power) and squared-like
    (power near 1) behavior.

    We use the parametrization:

        g(alpha) = (|alpha|-C)^{1+omega} - (|alpha|-C)^omega - |alpha|,

    with domain |alpha| > C and omega > 0.

    The inverse gradient (branch-wise) is:

        k = 1 + 1/omega
        t = 1 + sign * v / k   (must be > 0)
        alpha = sign * ( C + t^{1/omega} ).

    As with UKL, ``branch_fn`` can be supplied to select the sign.
    """

    def __init__(
        self,
        C: float = 1.0,
        *,
        omega: float = 0.5,
        power: float | None = None,
        branch_fn: BranchFn | None = None,
    ):
        if power is not None:
            omega = float(power)
        if float(C) < 0:
            raise ValueError("C must be >= 0")
        if float(omega) <= 0:
            raise ValueError("omega must be > 0")
        self.omega = float(omega)
        super().__init__(name=f"BP(omega={self.omega:g})", C=float(C), branch_fn=branch_fn)

    def inv_grad(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
        """Inverse gradient map for BP.

        The theoretical domain restriction is ``t = 1 + sign*v/k > 0``. In finite
        samples (and especially out-of-sample, e.g. in cross-fitting) the linear
        predictor can violate this constraint. Instead of raising an exception,
        we **clip** ``t`` to a small positive value. This keeps evaluation of
        alpha well-defined and avoids hard failures in notebooks.

        This clipping only activates when the predictor is far outside the
        domain; in typical use it is inactive.
        """

        X_ = _as_2d(X)
        v_ = _as_1d(v, n=len(X_))
        s = self._sign(X_, v_)
        k = 1.0 + 1.0 / self.omega

        t = 1.0 + s * v_ / k
        t = np.maximum(t, 1e-6)

        return s * (self.C + np.power(t, 1.0 / self.omega))

    def g(self, alpha: ArrayLike) -> NDArray[np.float64]:
        a = np.asarray(alpha, dtype=float).reshape(-1)
        t = np.abs(a) - self.C
        t = np.maximum(t, 1e-12)
        return np.power(t, 1.0 + self.omega) - np.power(t, self.omega) - np.abs(a)

    def grad2(self, alpha: ArrayLike) -> NDArray[np.float64]:
        a = np.asarray(alpha, dtype=float).reshape(-1)
        t = np.abs(a) - self.C
        t = np.maximum(t, 1e-12)
        k = 1.0 + 1.0 / self.omega
        return k * self.omega * np.power(t, self.omega - 1.0)
