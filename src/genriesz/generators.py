"""Bregman generators for Generalized Riesz Regression.

In *genriesz*, generalized Riesz regression (GRR) fits a finite-dimensional model

    v(x) = phi(x)^T beta,

and uses a generator-induced **link** to map the linear predictor ``v`` to a
Riesz representer ``alpha``.

A (possibly regressor-dependent) **Bregman generator** is a convex function

    g(x, alpha),

with derivative (wrt ``alpha``) ``∂g(x, alpha)/∂alpha``. GRR uses the *canonical*
(automatic) link

    alpha(x) = (∂g(x, ·))^{-1}( v(x) ),

which is the key mechanism behind **Automatic Regressor Balancing (ARB)**.

This module provides:

- Built-in generator families:
  - :class:`SquaredGenerator` ("SQ")
  - :class:`UKLGenerator` ("UKL")
  - :class:`BKLGenerator` ("BKL")
  - :class:`BPGenerator` ("BP")
  - :class:`PUGenerator` ("PU")

- A flexible :class:`BregmanGenerator` that lets users specify an arbitrary
  generator ``g`` and (optionally) its derivatives. If derivatives are omitted,
  they are approximated numerically.

Notes
-----
The public interface expected by the GRR solvers is:

- ``alpha = inv_grad(X, v)``
- ``g_val = g(X, alpha)``
- ``g2 = grad2(X, alpha)`` (second derivative wrt ``alpha``; elementwise)
- ``(g_star, alpha) = conjugate(X, v)``

All evaluations are row-wise: ``X`` is (n, d) and outputs are 1D arrays of
length n.
"""

from __future__ import annotations

import inspect
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

BranchFn = Callable[[NDArray[np.float64]], int]


def _as_2d(X: ArrayLike, *, name: str = "X") -> NDArray[np.float64]:
    X_ = np.asarray(X, dtype=float)
    if X_.ndim != 2:
        raise ValueError(f"{name} must be 2D. Got shape {X_.shape}.")
    return X_


def _as_1d(v: ArrayLike, *, n: int, name: str = "v") -> NDArray[np.float64]:
    v_ = np.asarray(v, dtype=float).reshape(-1)
    if v_.shape[0] != n:
        raise ValueError(f"{name} must have length {n}. Got shape {v_.shape}.")
    return v_


class _RowwiseScalarFn:
    """Wrap a scalar function and provide vectorized / rowwise evaluation.

    The wrapped callable can have signature:

    - ``f(alpha)``
    - ``f(x, alpha)`` where ``x`` is a 1D regressor row
    - a vectorized form: ``f(alpha_array)`` or ``f(X, alpha_array)``

    The wrapper tries the vectorized call once and caches the outcome.
    """

    def __init__(self, func: Callable):
        self.func = func

        # Determine whether the function expects 1 or 2 positional args.
        try:
            sig = inspect.signature(func)
            n_pos = 0
            for p in sig.parameters.values():
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                    n_pos += 1
            self._arity = 1 if n_pos <= 1 else 2
        except Exception:
            # If we cannot inspect, assume 2-arg form.
            self._arity = 2

        # None = unknown, True = vectorized works, False = rowwise only
        self._vectorized: bool | None = None

    def __call__(self, X: NDArray[np.float64], a: NDArray[np.float64]) -> NDArray[np.float64]:
        X = _as_2d(X)
        a = _as_1d(a, n=len(X), name="a")

        if self._vectorized is not False:
            try:
                if self._arity == 1:
                    out = self.func(a)
                else:
                    out = self.func(X, a)
                out_arr = np.asarray(out, dtype=float).reshape(-1)
                if out_arr.shape[0] == len(a):
                    self._vectorized = True
                    return out_arr
            except Exception:
                self._vectorized = False

        out = np.empty(len(a), dtype=float)
        if self._arity == 1:
            for i in range(len(a)):
                out[i] = float(self.func(float(a[i])))
        else:
            for i in range(len(a)):
                out[i] = float(self.func(np.asarray(X[i], dtype=float), float(a[i])))
        return out


class BregmanGenerator:
    """A Bregman generator with an (optional) automatic link.

    Parameters
    ----------
    g:
        Generator function ``g(x, alpha)`` or ``g(alpha)``.
    grad:
        First derivative wrt alpha, ``∂g(x, alpha)/∂alpha``.
        If omitted, it is approximated by finite differences.
    inv_grad:
        Inverse derivative (link) ``alpha = (∂g)^{-1}(x, v)``.
        If omitted, it is computed by Newton iterations using ``grad`` and
        ``grad2``.
    grad2:
        Second derivative wrt alpha (elementwise). If omitted, it is
        approximated from ``g`` via a second-order finite difference.
    name:
        Display name.
    C:
        Optional domain parameter used by some generator families.
        The generic implementation uses it only as a soft domain guard.
    branch_fn:
        Optional branch selector returning 1 (positive) or 0 (negative).
        Built-in UKL/BP generators use this to choose the sign branch.

    Notes
    -----
    The generic (user-specified) generator supports regressor-dependent
    generators via ``g(x, alpha)``. For performance and numerical stability,
    providing analytic ``grad`` and especially ``inv_grad`` is strongly
    recommended.
    """

    def __init__(
        self,
        *,
        g: Callable | None = None,
        grad: Callable | None = None,
        inv_grad: Callable | None = None,
        grad2: Callable | None = None,
        name: str = "Custom",
        C: float = 0.0,
        branch_fn: BranchFn | None = None,
        finite_diff_eps: float = 1e-6,
        newton_max_iter: int = 60,
        newton_tol: float = 1e-10,
    ):
        self.name = str(name)
        self.C = float(C)
        self.branch_fn = branch_fn

        self._g = None if g is None else _RowwiseScalarFn(g)
        self._grad = None if grad is None else _RowwiseScalarFn(grad)
        self._inv_grad = None if inv_grad is None else _RowwiseScalarFn(inv_grad)
        self._grad2 = None if grad2 is None else _RowwiseScalarFn(grad2)

        self._eps = float(finite_diff_eps)
        self._newton_max_iter = int(newton_max_iter)
        self._newton_tol = float(newton_tol)

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------
    def as_generator(self) -> "BregmanGenerator":
        """For API compatibility with earlier drafts."""

        return self

    def evaluate_g(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        return self.g(X, alpha)

    def evaluate_grad(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        return self.grad(X, alpha)

    def evaluate_inv_grad(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
        return self.inv_grad(X, v)

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _sign(self, X: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return +1/-1 sign array for branch-wise generators."""

        if self.branch_fn is None:
            return np.where(v >= 0.0, 1.0, -1.0)
        s = np.empty(len(v), dtype=float)
        for i in range(len(v)):
            s[i] = 1.0 if int(self.branch_fn(X[i])) == 1 else -1.0
        return s

    def _require_g(self) -> _RowwiseScalarFn:
        if self._g is None:
            raise ValueError("This generator does not define g().")
        return self._g

    # ------------------------------------------------------------------
    # Public interface required by GRR solvers
    # ------------------------------------------------------------------
    def g(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        """Evaluate g(x, alpha) row-wise."""

        X_ = _as_2d(X)
        a_ = _as_1d(alpha, n=len(X_), name="alpha")
        gfn = self._require_g()
        return gfn(X_, a_)

    def grad(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the derivative ∂g/∂alpha row-wise."""

        X_ = _as_2d(X)
        a_ = _as_1d(alpha, n=len(X_), name="alpha")

        if self._grad is not None:
            return self._grad(X_, a_)

        # Finite differences on g
        eps = self._eps
        gfn = self._require_g()
        return (gfn(X_, a_ + eps) - gfn(X_, a_ - eps)) / (2.0 * eps)

    def grad2(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the second derivative ∂²g/∂alpha² row-wise."""

        X_ = _as_2d(X)
        a_ = _as_1d(alpha, n=len(X_), name="alpha")

        if self._grad2 is not None:
            return self._grad2(X_, a_)

        # Second-order finite difference on g
        eps = self._eps
        gfn = self._require_g()
        g_p = gfn(X_, a_ + eps)
        g_0 = gfn(X_, a_)
        g_m = gfn(X_, a_ - eps)
        return (g_p - 2.0 * g_0 + g_m) / (eps * eps)

    def inv_grad(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
        """Inverse derivative map alpha = (∂g)^{-1}(x, v)."""

        X_ = _as_2d(X)
        v_ = _as_1d(v, n=len(X_), name="v")

        if self._inv_grad is not None:
            return self._inv_grad(X_, v_)

        # Automatic inversion via Newton iterations.
        # This is a generic fallback and may be slow / unstable.
        s = self._sign(X_, v_)

        # Heuristic initialization.
        alpha = v_.copy()
        if self.branch_fn is not None:
            alpha = s * np.abs(alpha)

        # Soft domain guard used by UKL/BP-like generators.
        if self.C > 0:
            alpha = s * np.maximum(np.abs(alpha), self.C + 1e-6)

        tol = self._newton_tol
        for _ in range(self._newton_max_iter):
            g1 = self.grad(X_, alpha)
            diff = g1 - v_
            max_abs = float(np.max(np.abs(diff)))
            if not np.isfinite(max_abs):
                break
            if max_abs < tol:
                return alpha

            g2 = self.grad2(X_, alpha)
            g2 = np.asarray(g2, dtype=float)
            # Guard against non-positive curvature (should not happen for strictly convex g).
            g2 = np.where(np.isfinite(g2) & (g2 > 1e-12), g2, 1e-12)

            step = diff / g2
            step = np.clip(step, -50.0, 50.0)
            alpha = alpha - step

            if self.branch_fn is not None:
                alpha = s * np.abs(alpha)
            if self.C > 0:
                alpha = s * np.maximum(np.abs(alpha), self.C + 1e-6)

        # If we reach here, Newton did not converge reliably.
        raise RuntimeError(
            "Failed to numerically invert grad. Provide inv_grad (and preferably grad/grad2) "
            "for this generator."
        )

    def conjugate(self, X: ArrayLike, v: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (g*(v), alpha) evaluated row-wise."""

        X_ = _as_2d(X)
        v_ = _as_1d(v, n=len(X_), name="v")
        alpha = self.inv_grad(X_, v_)
        g_val = self.g(X_, alpha)
        g_star = v_ * alpha - g_val
        return g_star, alpha


class SquaredGenerator(BregmanGenerator):
    """Squared generator (SQ-Riesz).

    g(alpha) = (alpha - C)^2.

    This generator has no strict domain constraints and induces a linear link

        alpha = C + 0.5 * v.
    """

    def __init__(self, C: float = 0.0):
        super().__init__(name="SQ", C=float(C), branch_fn=None)

    def inv_grad(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
        X_ = _as_2d(X)
        v_ = _as_1d(v, n=len(X_), name="v")
        return self.C + 0.5 * v_

    def g(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = _as_2d(X)
        a = _as_1d(alpha, n=len(X_), name="alpha")
        return np.square(a - self.C)

    def grad(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = _as_2d(X)
        a = _as_1d(alpha, n=len(X_), name="alpha")
        return 2.0 * (a - self.C)

    def grad2(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = _as_2d(X)
        a = _as_1d(alpha, n=len(X_), name="alpha")
        return np.full_like(a, 2.0, dtype=float)


class UKLGenerator(BregmanGenerator):
    """Unnormalized KL generator (UKL-Riesz).

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
        """Branch-wise inverse gradient alpha = (g')^{-1}(v)."""

        X_ = _as_2d(X)
        v_ = _as_1d(v, n=len(X_), name="v")
        s = self._sign(X_, v_)

        # exp can underflow to 0 for large negative inputs; clip and floor.
        z = np.clip(s * v_, -700.0, 700.0)
        exp_term = np.exp(z)
        exp_term = np.maximum(exp_term, 1e-12)
        return s * (self.C + exp_term)

    def g(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = _as_2d(X)
        a = _as_1d(alpha, n=len(X_), name="alpha")
        t = np.abs(a) - self.C
        t = np.maximum(t, 1e-12)
        return t * np.log(t) - np.abs(a)

    def grad(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = _as_2d(X)
        a = _as_1d(alpha, n=len(X_), name="alpha")
        t = np.abs(a) - self.C
        t = np.maximum(t, 1e-12)
        return np.sign(a) * np.log(t)

    def grad2(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = _as_2d(X)
        a = _as_1d(alpha, n=len(X_), name="alpha")
        t = np.abs(a) - self.C
        t = np.maximum(t, 1e-12)
        return 1.0 / t


class BPGenerator(BregmanGenerator):
    """Box-Power generator (BP-Riesz).

    A smooth family interpolating between UKL-like (small power) and squared-like
    (power near 1) behavior.

    We use the parametrization, for ``t = |alpha| - C``,

        g(alpha) = ( t^{1+omega} - (1+omega) t ) / omega,

    with domain ``|alpha| > C`` and ``omega > 0``.

    The derivative is

        g'(alpha) = sign(alpha) * (1+omega)/omega * ( t^omega - 1 ),

    so the inverse gradient (branch-wise) can be written as

        k = (1+omega)/omega
        u = 1 + sign * v / k   (must be > 0)
        alpha = sign * ( C + u^{1/omega} ).

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
        """Branch-wise inverse gradient map for BP.

        The theoretical domain restriction is ``t = 1 + sign*v/k > 0``. In finite
        samples (and especially out-of-sample, e.g. in cross-fitting) the linear
        predictor can violate this constraint. Instead of raising an exception,
        we **clip** ``t`` to a small positive value.
        """

        X_ = _as_2d(X)
        v_ = _as_1d(v, n=len(X_), name="v")
        s = self._sign(X_, v_)
        k = 1.0 + 1.0 / self.omega

        t = 1.0 + s * v_ / k
        t = np.maximum(t, 1e-6)
        return s * (self.C + np.power(t, 1.0 / self.omega))

    def g(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = _as_2d(X)
        a = _as_1d(alpha, n=len(X_), name="alpha")
        t = np.abs(a) - self.C
        t = np.maximum(t, 1e-12)
        return (np.power(t, 1.0 + self.omega) - (1.0 + self.omega) * t) / self.omega

    def grad(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = _as_2d(X)
        a = _as_1d(alpha, n=len(X_), name="alpha")
        t = np.abs(a) - self.C
        t = np.maximum(t, 1e-12)
        k = 1.0 + 1.0 / self.omega
        return np.sign(a) * k * (np.power(t, self.omega) - 1.0)

    def grad2(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = _as_2d(X)
        a = _as_1d(alpha, n=len(X_), name="alpha")
        t = np.abs(a) - self.C
        t = np.maximum(t, 1e-12)
        k = 1.0 + 1.0 / self.omega
        return k * self.omega * np.power(t, self.omega - 1.0)



class BKLGenerator(BregmanGenerator):
    """Binary KL generator (BKL-Riesz).

    The generator is

        g(alpha) = (|alpha| - C) log(|alpha| - C) - (|alpha| + C) log(|alpha| + C),

    with domain ``|alpha| > C`` and ``C > 0``.

    Its derivative is

        g'(alpha) = sign(alpha) * log( (|alpha|-C) / (|alpha|+C) ).

    The inverse gradient is branch-wise. Let ``s`` be the desired sign branch
    (+1 or -1) and let ``u = s * v``. Since the log-ratio is always negative,
    the theoretical domain is ``u < 0``. In finite samples (and especially
    out-of-sample, e.g. under cross-fitting), ``u`` may violate this.
    Instead of raising an exception, we **clip** ``u`` to a small negative
    value to keep the link well-defined.

    If ``branch_fn`` is provided, it selects the sign branch.
    """

    def __init__(self, C: float = 1.0, *, branch_fn: BranchFn | None = None):
        if float(C) <= 0:
            raise ValueError("C must be > 0 for BKLGenerator")
        super().__init__(name="BKL", C=float(C), branch_fn=branch_fn)

    def inv_grad(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
        X_ = _as_2d(X)
        v_ = _as_1d(v, n=len(X_), name="v")

        # For BKL, the positive branch corresponds to v <= 0.
        if self.branch_fn is None:
            s = np.where(v_ <= 0.0, 1.0, -1.0)
        else:
            s = self._sign(X_, v_)

        # Enforce u = s*v < 0.
        u = s * v_
        u = np.clip(u, -700.0, -1e-8)

        t = np.exp(u)  # in (0, 1)
        denom = 1.0 - t
        denom = np.maximum(denom, 1e-12)
        a = self.C * (1.0 + t) / denom
        return s * a

    def g(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = _as_2d(X)
        a = _as_1d(alpha, n=len(X_), name="alpha")
        t1 = np.abs(a) - self.C
        t2 = np.abs(a) + self.C
        t1 = np.maximum(t1, 1e-12)
        t2 = np.maximum(t2, 1e-12)
        return t1 * np.log(t1) - t2 * np.log(t2)

    def grad(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = _as_2d(X)
        a = _as_1d(alpha, n=len(X_), name="alpha")
        t1 = np.abs(a) - self.C
        t2 = np.abs(a) + self.C
        t1 = np.maximum(t1, 1e-12)
        t2 = np.maximum(t2, 1e-12)
        return np.sign(a) * (np.log(t1) - np.log(t2))

    def grad2(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = _as_2d(X)
        a = _as_1d(alpha, n=len(X_), name="alpha")
        abs_a = np.abs(a)
        denom = abs_a * abs_a - self.C * self.C
        denom = np.maximum(denom, 1e-12)
        return (2.0 * self.C) / denom


class PUGenerator(BregmanGenerator):
    """PU generator (PU-Riesz).

    This generator is based on the binary-entropy potential

        g(alpha) = C * [ |alpha| log|alpha| + (1-|alpha|) log(1-|alpha|) ],

    with domain ``|alpha| in (0, 1)`` and ``C > 0``.

    The derivative is

        g'(alpha) = sign(alpha) * C * log( |alpha| / (1-|alpha|) ).

    The inverse gradient is a (scaled) logistic map.

    Notes
    -----
    This generator is primarily useful when you want the representer to be
    bounded (in absolute value) by 1.
    """

    def __init__(self, C: float = 1.0, *, branch_fn: BranchFn | None = None):
        if float(C) <= 0:
            raise ValueError("C must be > 0 for PUGenerator")
        super().__init__(name="PU", C=float(C), branch_fn=branch_fn)

    def inv_grad(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
        X_ = _as_2d(X)
        v_ = _as_1d(v, n=len(X_), name="v")
        s = self._sign(X_, v_)

        z = np.clip(s * v_ / self.C, -700.0, 700.0)
        a = 1.0 / (1.0 + np.exp(-z))
        a = np.clip(a, 1e-10, 1.0 - 1e-10)
        return s * a

    def g(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = _as_2d(X)
        a = _as_1d(alpha, n=len(X_), name="alpha")
        t = np.clip(np.abs(a), 1e-10, 1.0 - 1e-10)
        return self.C * (t * np.log(t) + (1.0 - t) * np.log(1.0 - t))

    def grad(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = _as_2d(X)
        a = _as_1d(alpha, n=len(X_), name="alpha")
        t = np.clip(np.abs(a), 1e-10, 1.0 - 1e-10)
        return np.sign(a) * self.C * (np.log(t) - np.log(1.0 - t))

    def grad2(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = _as_2d(X)
        a = _as_1d(alpha, n=len(X_), name="alpha")
        t = np.clip(np.abs(a), 1e-10, 1.0 - 1e-10)
        return self.C / (t * (1.0 - t))
