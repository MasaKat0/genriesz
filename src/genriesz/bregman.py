r"""Bregman generators used in generalized Riesz regression.

In generalized Riesz regression (GRR), the empirical objective is

.. math::

    \widehat{\mathrm{BD}}_g(\alpha)
    = \frac{1}{n}\sum_{i=1}^n\Bigl[-g(\alpha(X_i)) + \partial g(\alpha(X_i))\,\alpha(X_i)
      - m\bigl(W_i,(\partial g)\circ\alpha\bigr)\Bigr],

where ``g`` is a differentiable and strictly convex *Bregman generator*.

Automatic covariate balancing (ACB) can be enforced by choosing a link function
such that

.. math::

    (\partial g)\circ\alpha_\beta(x) = \phi(x)^\top\beta,

for a chosen basis function :math:`\phi`. One convenient way to achieve this is

.. math::

    \alpha_\beta(x) = (\partial g_x)^{-1}(\phi(x)^\top\beta),

where :math:`(\partial g_x)^{-1}` is an (optionally branchwise) inverse of the
derivative with respect to the scalar argument.

This module provides several generators with closed-form derivatives and
inverse-derivatives, plus a fallback class for user-supplied generators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy import optimize

ArrayLike = Union[np.ndarray, float, int]

ConjugateFn = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


def _as_2d(W: ArrayLike) -> np.ndarray:
    W_arr = np.asarray(W)
    if W_arr.ndim == 1:
        return W_arr[None, :]
    if W_arr.ndim != 2:
        raise ValueError(f"W must be 1D or 2D. Got shape {W_arr.shape}.")
    return W_arr


def _as_1d(x: ArrayLike, n: Optional[int] = None) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 0:
        x_arr = x_arr.reshape(1)
    elif x_arr.ndim == 2 and x_arr.shape[1] == 1:
        x_arr = x_arr.reshape(-1)
    elif x_arr.ndim != 1:
        x_arr = x_arr.reshape(-1)

    if n is not None and len(x_arr) not in (1, n):
        raise ValueError(f"Expected len(x) to be 1 or {n}, got {len(x_arr)}.")
    if n is not None and len(x_arr) == 1:
        x_arr = np.repeat(x_arr, n)
    return x_arr


def _try_vectorized_call(
    func: Callable[[np.ndarray, np.ndarray], ArrayLike],
    W: np.ndarray,
    a: np.ndarray,
) -> Optional[np.ndarray]:
    """Try calling ``func(W, a)`` assuming it supports vectorization.

    Returns ``None`` if it fails or returns an unexpected shape.
    """
    try:
        out = func(W, a)
        out_arr = np.asarray(out, dtype=float)
        if out_arr.shape == a.shape:
            return out_arr
        if out_arr.shape == (len(a), 1):
            return out_arr.reshape(-1)
    except Exception:
        return None
    return None


def _rowwise_call(
    func: Callable[[np.ndarray, float], float],
    W: np.ndarray,
    a: np.ndarray,
) -> np.ndarray:
    out = np.empty(len(a), dtype=float)
    for i in range(len(a)):
        out[i] = float(func(W[i], float(a[i])))
    return out


@dataclass
class BregmanGenerator:
    """A Bregman generator g(·) and (optionally branchwise) derivative inversion.

    Parameters
    ----------
    g:
        Callable ``g(W_i, alpha_i) -> float`` or a vectorized callable
        ``g(W, alpha) -> array``.
    grad:
        Derivative with respect to the scalar argument ``alpha``.
        If ``None``, a finite-difference approximation is used.
    inv_grad:
        Inverse of the derivative (the inverse link). If ``None``, a numerical
        root finder is used.
    finite_diff_eps:
        Step size used when approximating the derivative.
    inv_grad_bounds:
        Bracketing interval used for numerical inversion. If ``None``, the code
        will attempt to find a bracket automatically.

    Notes
    -----
    The derivative is understood as ``∂g(W, alpha) / ∂alpha``.

    For some generators (e.g. UKL) the derivative is invertible only after
    selecting a branch. For those cases, use the dedicated generator classes
    (e.g. :class:`UKLGenerator`) rather than the generic fallback.
    """

    g: Callable[[np.ndarray, ArrayLike], ArrayLike]
    grad: Optional[Callable[[np.ndarray, ArrayLike], ArrayLike]] = None
    inv_grad: Optional[Callable[[np.ndarray, ArrayLike], ArrayLike]] = None
    conjugate_fn: Optional[ConjugateFn] = None
    finite_diff_eps: float = 1e-6
    inv_grad_bounds: Optional[Tuple[float, float]] = None

    def evaluate_g(self, W: ArrayLike, alpha: ArrayLike) -> np.ndarray:
        W2 = _as_2d(W)
        a1 = _as_1d(alpha, n=len(W2))

        vec = _try_vectorized_call(self.g, W2, a1)
        if vec is not None:
            return vec
        return _rowwise_call(self.g, W2, a1)

    def evaluate_grad(self, W: ArrayLike, alpha: ArrayLike) -> np.ndarray:
        W2 = _as_2d(W)
        a1 = _as_1d(alpha, n=len(W2))

        if self.grad is None:
            eps = float(self.finite_diff_eps)
            return (self.evaluate_g(W2, a1 + eps) - self.evaluate_g(W2, a1 - eps)) / (2.0 * eps)

        vec = _try_vectorized_call(self.grad, W2, a1)
        if vec is not None:
            return vec
        return _rowwise_call(self.grad, W2, a1)

    def evaluate_inv_grad(self, W: ArrayLike, v: ArrayLike) -> np.ndarray:
        """Compute alpha satisfying grad(W, alpha) = v."""
        W2 = _as_2d(W)
        v1 = _as_1d(v, n=len(W2))

        if self.inv_grad is not None:
            vec = _try_vectorized_call(self.inv_grad, W2, v1)
            if vec is not None:
                return vec
            return _rowwise_call(self.inv_grad, W2, v1)

        # Numerical inversion by root finding.
        bounds = self.inv_grad_bounds
        alpha = np.empty(len(v1), dtype=float)

        for i in range(len(v1)):
            wi = W2[i]
            target = float(v1[i])

            def f(a: float) -> float:
                return float(self.evaluate_grad(wi, a)[0] - target)

            if bounds is not None:
                lo, hi = map(float, bounds)
                try:
                    res = optimize.root_scalar(f, bracket=[lo, hi], method="brentq")
                except ValueError as e:
                    raise RuntimeError(
                        "Failed to invert grad() on the provided bracket. "
                        "Consider supplying inv_grad or a wider inv_grad_bounds."
                    ) from e
            else:
                # Adaptive bracketing around an initial guess.
                x0 = 0.0 if not np.isfinite(target) else float(np.clip(target, -10.0, 10.0))
                step = 1.0
                lo, hi = x0 - step, x0 + step
                flo, fhi = f(lo), f(hi)
                for _ in range(60):
                    if np.sign(flo) != np.sign(fhi):
                        break
                    step *= 2.0
                    lo, hi = x0 - step, x0 + step
                    flo, fhi = f(lo), f(hi)
                else:
                    raise RuntimeError(
                        "Could not bracket the inverse derivative. "
                        "Supply inv_grad_bounds or inv_grad explicitly."
                    )
                res = optimize.root_scalar(f, bracket=[lo, hi], method="brentq")

            if not res.converged:
                raise RuntimeError("Root finding failed to converge for inv_grad.")
            alpha[i] = float(res.root)

        return alpha

    def conjugate(self, W: ArrayLike, v: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """Compute g*(v) = sup_alpha { alpha*v - g(alpha) } and the argmax alpha.

        Returns
        -------
        g_star:
            Array of shape (n,).
        alpha:
            Array of shape (n,), where alpha = (grad)^{-1}(v).
        """
        W2 = _as_2d(W)
        v1 = _as_1d(v, n=len(W2))
        if self.conjugate_fn is not None:
            return self.conjugate_fn(W2, v1)

        alpha = self.evaluate_inv_grad(W2, v1)
        g_val = self.evaluate_g(W2, alpha)
        return alpha * v1 - g_val, alpha


@dataclass
class SquaredGenerator:
    """Squared-distance generator g(α) = (α - C)^2."""

    C: float = 0.0

    def as_generator(self) -> BregmanGenerator:
        C = float(self.C)

        def g(_W: np.ndarray, alpha: ArrayLike) -> np.ndarray:
            a = np.asarray(alpha, dtype=float)
            return (a - C) ** 2

        def grad(_W: np.ndarray, alpha: ArrayLike) -> np.ndarray:
            a = np.asarray(alpha, dtype=float)
            return 2.0 * (a - C)

        def inv_grad(_W: np.ndarray, v: ArrayLike) -> np.ndarray:
            vv = np.asarray(v, dtype=float)
            return vv / 2.0 + C

        def conjugate_fn(W: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Closed-form conjugate for g(α)=(α-C)^2."""

            W2 = _as_2d(W)
            v1 = _as_1d(v, n=len(W2))
            alpha = v1 / 2.0 + C
            g_star = (v1 * v1) / 4.0 + C * v1
            return g_star, alpha

        return BregmanGenerator(g=g, grad=grad, inv_grad=inv_grad, conjugate_fn=conjugate_fn)


BranchFn = Callable[[np.ndarray], int]


@dataclass
class UKLGenerator:
    """Unnormalized KL generator.

    g(α) = (abs(alpha) - C) log(abs(alpha) - C) - abs(alpha),  domain: abs(alpha) > C.

    The derivative is

        ∂g(α) = sign(α) * log(abs(alpha) - C).

    The inverse of the derivative is two-valued; to make it single-valued, this
    implementation uses a user-provided branch function ξ(W) in {0,1}.

    - If ξ(W)=1 ("positive branch"), it returns α = C + exp(v) (> C).
    - If ξ(W)=0 ("negative branch"), it returns α = -C - exp(-v) (< -C).

    This matches the branchwise inverses used for automatic covariate balancing
    in the paper.
    """

    C: float = 1.0
    branch_fn: BranchFn = lambda w: 1

    def as_generator(self) -> BregmanGenerator:
        C = float(self.C)
        branch_fn = self.branch_fn

        def g(_W: np.ndarray, alpha: ArrayLike) -> np.ndarray:
            a = np.asarray(alpha, dtype=float)
            u = np.abs(a) - C
            if np.any(u <= 0):
                raise ValueError("UKLGenerator requires abs(alpha) > C.")
            return u * np.log(u) - np.abs(a)

        def grad(_W: np.ndarray, alpha: ArrayLike) -> np.ndarray:
            a = np.asarray(alpha, dtype=float)
            u = np.abs(a) - C
            if np.any(u <= 0):
                raise ValueError("UKLGenerator requires abs(alpha) > C.")
            return np.sign(a) * np.log(u)

        def inv_grad(W: np.ndarray, v: ArrayLike) -> np.ndarray:
            W2 = _as_2d(W)
            v1 = _as_1d(v, n=len(W2))
            out = np.empty(len(v1), dtype=float)
            for i in range(len(v1)):
                if int(branch_fn(W2[i])) == 1:
                    out[i] = C + float(np.exp(np.clip(v1[i], -745.0, 700.0)))
                else:
                    out[i] = -C - float(np.exp(np.clip(-v1[i], -745.0, 700.0)))
            return out

        def conjugate_fn(W: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Closed-form conjugate for numerical stability.

            This avoids evaluating g(alpha) when alpha is extremely close to the
            boundary abs(alpha) = C, which can happen due to exp underflow.
            """

            W2 = _as_2d(W)
            v1 = _as_1d(v, n=len(W2))

            branch = np.array([int(branch_fn(W2[i])) == 1 for i in range(len(W2))], dtype=bool)

            alpha = np.empty(len(v1), dtype=float)
            g_star = np.empty(len(v1), dtype=float)

            vmax = 700.0
            vmin = -745.0

            if np.any(branch):
                vp = v1[branch]
                u = np.exp(np.clip(vp, vmin, vmax))
                alpha[branch] = C + u
                g_star[branch] = C * vp + C + u

            if np.any(~branch):
                vn = v1[~branch]
                u = np.exp(np.clip(-vn, vmin, vmax))
                alpha[~branch] = -C - u
                g_star[~branch] = -C * vn + C + u

            return g_star, alpha

        return BregmanGenerator(g=g, grad=grad, inv_grad=inv_grad, conjugate_fn=conjugate_fn)


@dataclass
class BPGenerator:
    """Basu's power (BP) generator.

    g(α) = ((abs(alpha)-C)^{1+ω} - (abs(alpha)-C)) / ω - abs(alpha),  domain: abs(alpha) > C.

    The derivative is

        ∂g(α) = k * sign(α) * ( (abs(alpha) - C)^ω - 1 ),  k = 1 + 1/ω.

    As with UKL, the inverse derivative is two-valued. We use a branch function
    ξ(W) in {0,1} to make it single-valued.

    - If ξ(W)=1 (positive branch): α = C + (1 + v/k)^{1/ω}, requiring v >= -k.
    - If ξ(W)=0 (negative branch): α = -C - (1 - v/k)^{1/ω}, requiring v <= k.
    """

    omega: float = 0.5
    C: float = 1.0
    branch_fn: BranchFn = lambda w: 1

    def as_generator(self) -> BregmanGenerator:
        omega = float(self.omega)
        if omega <= 0:
            raise ValueError("omega must be > 0 for BPGenerator.")
        C = float(self.C)
        k = 1.0 + 1.0 / omega
        branch_fn = self.branch_fn

        def g(_W: np.ndarray, alpha: ArrayLike) -> np.ndarray:
            a = np.asarray(alpha, dtype=float)
            u = np.abs(a) - C
            if np.any(u <= 0):
                raise ValueError("BPGenerator requires abs(alpha) > C.")
            return (u ** (1.0 + omega) - u) / omega - np.abs(a)

        def grad(_W: np.ndarray, alpha: ArrayLike) -> np.ndarray:
            a = np.asarray(alpha, dtype=float)
            u = np.abs(a) - C
            if np.any(u <= 0):
                raise ValueError("BPGenerator requires abs(alpha) > C.")
            return k * np.sign(a) * (u**omega - 1.0)

        def inv_grad(W: np.ndarray, v: ArrayLike) -> np.ndarray:
            W2 = _as_2d(W)
            v1 = _as_1d(v, n=len(W2))
            out = np.empty(len(v1), dtype=float)
            for i in range(len(v1)):
                vi = float(v1[i])
                if int(branch_fn(W2[i])) == 1:
                    # Positive branch
                    if vi < -k:
                        raise ValueError(f"BPGenerator positive branch requires v >= {-k}. Got {vi}.")
                    out[i] = C + float((1.0 + vi / k) ** (1.0 / omega))
                else:
                    # Negative branch
                    if vi > k:
                        raise ValueError(f"BPGenerator negative branch requires v <= {k}. Got {vi}.")
                    out[i] = -C - float((1.0 - vi / k) ** (1.0 / omega))
            return out

        def conjugate_fn(W: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Closed-form conjugate for numerical stability.

            This avoids evaluating g(alpha) when alpha is extremely close to the
            boundary abs(alpha) = C, which can happen due to exp underflow.
            """

            W2 = _as_2d(W)
            v1 = _as_1d(v, n=len(W2))

            branch = np.array([int(branch_fn(W2[i])) == 1 for i in range(len(W2))], dtype=bool)

            alpha = np.empty(len(v1), dtype=float)
            g_star = np.empty(len(v1), dtype=float)

            vmax = 700.0
            vmin = -745.0

            if np.any(branch):
                vp = v1[branch]
                u = np.exp(np.clip(vp, vmin, vmax))
                alpha[branch] = C + u
                g_star[branch] = C * vp + C + u

            if np.any(~branch):
                vn = v1[~branch]
                u = np.exp(np.clip(-vn, vmin, vmax))
                alpha[~branch] = -C - u
                g_star[~branch] = -C * vn + C + u

            return g_star, alpha

        return BregmanGenerator(g=g, grad=grad, inv_grad=inv_grad, conjugate_fn=conjugate_fn)


@dataclass
class BKLGenerator:
    """Binary KL (BKL) generator.

    g(α) = (abs(alpha)-C) log(abs(alpha)-C) - (abs(alpha)+C) log(abs(alpha)+C),  domain: abs(alpha) > C.

    ∂g(α) = sign(α) * log((abs(alpha)-C)/(abs(alpha)+C)).

    Notes
    -----
    The inverse derivative can be written in closed form, but the mapping has a
    restricted range on each branch. In many causal applications, UKL or SQ are
    more convenient.

    We provide an inverse that uses ``branch_fn`` to choose the sign of α. The
    user is responsible for ensuring that the linear index v lies in the valid
    range of the selected branch:

    - positive branch: v < 0
    - negative branch: v > 0

    If you do not provide ``branch_fn``, this implementation will choose the
    branch by the sign of v.
    """

    C: float = 1.0
    branch_fn: Optional[BranchFn] = None

    def as_generator(self) -> BregmanGenerator:
        C = float(self.C)
        branch_fn = self.branch_fn

        def g(_W: np.ndarray, alpha: ArrayLike) -> np.ndarray:
            a = np.asarray(alpha, dtype=float)
            u = np.abs(a)
            if np.any(u <= C):
                raise ValueError("BKLGenerator requires abs(alpha) > C.")
            return (u - C) * np.log(u - C) - (u + C) * np.log(u + C)

        def grad(_W: np.ndarray, alpha: ArrayLike) -> np.ndarray:
            a = np.asarray(alpha, dtype=float)
            u = np.abs(a)
            if np.any(u <= C):
                raise ValueError("BKLGenerator requires abs(alpha) > C.")
            return np.sign(a) * np.log((u - C) / (u + C))

        def inv_grad(W: np.ndarray, v: ArrayLike) -> np.ndarray:
            W2 = _as_2d(W)
            v1 = _as_1d(v, n=len(W2))
            out = np.empty(len(v1), dtype=float)
            for i in range(len(v1)):
                vi = float(v1[i])

                if branch_fn is None:
                    # Choose branch by the sign of v.
                    # For α > C, v is negative. For α < -C, v is positive.
                    choose_positive = vi < 0
                else:
                    choose_positive = int(branch_fn(W2[i])) == 1

                t = float(np.exp(vi))
                if choose_positive:
                    if vi >= 0:
                        raise ValueError(
                            "BKL positive branch requires v < 0. "
                            "Either change the basis, or omit branch_fn and let the sign of v choose the branch."
                        )
                    out[i] = C * (1.0 + t) / (1.0 - t)
                else:
                    if vi <= 0:
                        raise ValueError(
                            "BKL negative branch requires v > 0. "
                            "Either change the basis, or omit branch_fn and let the sign of v choose the branch."
                        )
                    out[i] = -C * (1.0 + t) / (t - 1.0)

            return out

        def conjugate_fn(W: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Closed-form conjugate for numerical stability.

            This avoids evaluating g(alpha) when alpha is extremely close to the
            boundary abs(alpha) = C, which can happen due to exp underflow.
            """

            W2 = _as_2d(W)
            v1 = _as_1d(v, n=len(W2))

            branch = np.array([int(branch_fn(W2[i])) == 1 for i in range(len(W2))], dtype=bool)

            alpha = np.empty(len(v1), dtype=float)
            g_star = np.empty(len(v1), dtype=float)

            vmax = 700.0
            vmin = -745.0

            if np.any(branch):
                vp = v1[branch]
                u = np.exp(np.clip(vp, vmin, vmax))
                alpha[branch] = C + u
                g_star[branch] = C * vp + C + u

            if np.any(~branch):
                vn = v1[~branch]
                u = np.exp(np.clip(-vn, vmin, vmax))
                alpha[~branch] = -C - u
                g_star[~branch] = -C * vn + C + u

            return g_star, alpha

        return BregmanGenerator(g=g, grad=grad, inv_grad=inv_grad, conjugate_fn=conjugate_fn)
