"""Bregman generators for Generalized Riesz Regression.

In *genriesz*, generalized Riesz regression (GRR) fits a finite-dimensional model

    v(x) = phi(x)^T beta,

and uses a **link function** to map the linear predictor ``v`` to a
Riesz representer ``alpha``.

A (possibly regressor-dependent) **Bregman generator** is a convex function

    g(x, alpha),

with derivative (wrt ``alpha``) ``∂g(x, alpha)/∂alpha``. GRR uses the *canonical*
(automatic) link

    alpha(x) = (∂g(x, ·))^{-1}( v(x) ),

which is the key mechanism behind **automatic regressor balancing**.

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
The public interface expected by the generalized Riesz regression solvers is:

- ``alpha = inv_grad(X, v)``
- ``g_val = g(X, alpha)``
- ``g2 = grad2(X, alpha)`` (second derivative wrt ``alpha``; elementwise)
- ``(g_star, alpha) = conjugate(X, v)``

All evaluations are row-wise: ``X`` is (n, d) and outputs are 1D arrays of
length n.
"""

from __future__ import annotations

import contextlib
import inspect
import warnings
from collections.abc import Callable, Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit, xlogy

from .utils import as_1d_of_length, as_2d

BranchFn = Callable[[NDArray[np.float64]], int]

#: Upper bound on the number of arrays memoized inside ``branch_cache()``.
#: Solvers see one array per fit; the bound only exists so a caller that hands a
#: fresh array to every call degrades in speed rather than in memory.
_BRANCH_CACHE_MAX_ENTRIES = 8


class DomainError(RuntimeError):
    """Raised when a generator cannot evaluate its link/conjugate at a point.

    This replaces the previous behavior of silently returning a huge objective
    value and a zero gradient (or silently clipping the pre-image to a value
    that makes the returned ``alpha`` explode), which could make the optimizer
    stop at a broken point with ``success=True``. Callers (e.g. ``GRRGLM.fit``)
    catch this and record an explicit ``status="domain_error"`` failure.
    """


class _RowwiseScalarFn:
    """Wrap a scalar function and provide vectorized or rowwise evaluation.

    The wrapped callable can have signature:

    - ``f(alpha)``
    - ``f(x, alpha)`` where ``x`` is a 1D regressor row
    - a vectorized form: ``f(alpha_array)`` or ``f(X, alpha_array)``

    Whether the callable is vectorized is decided once, by probing it. The probe
    must not confuse *"this function cannot take arrays"* with *"this data is
    outside the function's domain"*: a generator that raises
    :class:`DomainError` on a bad ``alpha`` is still vectorized, and permanently
    demoting it to a Python loop would silently cost O(n) per objective
    evaluation for the rest of the fit. So when the vectorized probe raises, the
    same inputs are retried rowwise:

    - rowwise succeeds -> the failure was the signature; use rowwise from now on.
    - rowwise fails too -> the failure was the data; propagate it and leave the
      vectorization verdict undecided so the next call can probe again.

    Once the verdict is ``True`` the vectorized call is made directly, with no
    exception handling, so domain errors surface unchanged.
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

    def _call_vectorized(
        self, X: NDArray[np.float64], a: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        out = self.func(a) if self._arity == 1 else self.func(X, a)
        return np.asarray(out, dtype=float).reshape(-1)

    def _call_rowwise(
        self, X: NDArray[np.float64], a: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        out = np.empty(len(a), dtype=float)
        if self._arity == 1:
            for i in range(len(a)):
                out[i] = float(self.func(float(a[i])))
        else:
            for i in range(len(a)):
                out[i] = float(self.func(np.asarray(X[i], dtype=float), float(a[i])))
        return out

    def _probe(self, X: NDArray[np.float64], a: NDArray[np.float64]) -> bool:
        # Two rows: one cannot separate "returns a scalar" from "returns one
        # value per row".
        k = 2
        Xk, ak = X[:k], a[:k]
        try:
            out = self._call_vectorized(Xk, ak)
        except Exception:
            # Signature problem, or bad data? Ask the rowwise path.
            self._call_rowwise(Xk, ak)  # raises the data error, if that is what it was
            return False
        return bool(out.shape[0] == k)

    def _try_both(self, X: NDArray[np.float64], a: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate without recording a verdict (fewer than two rows).

        Vectorized first, exactly as before this module learned to probe: a
        vectorized-only callable must keep working when it is handed a single
        row. A failure falls through to rowwise, which re-raises if the cause was
        the data rather than the signature.
        """

        try:
            out = self._call_vectorized(X, a)
        except Exception:
            return self._call_rowwise(X, a)
        if out.shape[0] == len(a):
            return out
        return self._call_rowwise(X, a)

    def __call__(self, X: NDArray[np.float64], a: NDArray[np.float64]) -> NDArray[np.float64]:
        X = as_2d(X)
        a = as_1d_of_length(a, n=len(X), name="a")

        if self._vectorized is None:
            if len(a) < 2:
                return self._try_both(X, a)
            self._vectorized = self._probe(X, a)

        if self._vectorized:
            out = self._call_vectorized(X, a)
            if out.shape[0] != len(a):
                raise ValueError(
                    f"vectorized callable returned {out.shape[0]} values for "
                    f"{len(a)} observations"
                )
            return out

        return self._call_rowwise(X, a)


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
        Built-in UKL/BP generators use this to choose the sign branch. It is
        called on one row at a time, unless it has the attribute
        ``vectorized = True``: it is then called once on the ``(n, d)`` array
        of rows and must return ``n`` values. Marking a function vectorized is
        the caller's guarantee that value ``i`` depends on row ``i`` alone and
        equals what the function returns for that row on its own; the result
        is then identical to the row-by-row evaluation.

    Notes
    -----
    The generic (user-specified) generator supports regressor-dependent
    generators via ``g(x, alpha)``. For performance and numerical stability,
    providing analytic ``grad`` and especially ``inv_grad`` is strongly
    recommended.
    """

    #: Whether this generator's link intentionally bounds/caps the representer
    #: and therefore targets a *modified* estimand. Model selection uses this
    #: (together with ``domain_binding``) to keep such variants out of the
    #: admissible set and treat them as target-sensitivity candidates (§9-4).
    modifies_estimand: bool = False

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
        # Active only inside branch_cache(): id(X) -> (X, signs). The array is
        # kept alive by the tuple so a stale id can never alias a new array.
        self._branch_cache: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] | None = (
            None
        )

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
    def as_generator(self) -> BregmanGenerator:
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
    @contextlib.contextmanager
    def branch_cache(self) -> Iterator[None]:
        """Memoize the branch signs of each ``X`` seen inside the block.

        ``branch_fn`` is a function of ``x`` alone, so within a fit the sign
        array is constant while ``v`` changes on every objective and gradient
        evaluation. Without this, an L-BFGS fit calls ``branch_fn`` once per row
        per evaluation -- tens of thousands of Python calls for a few hundred
        rows of real work.

        Entries are keyed by array identity (a strong reference is held, so ids
        cannot be recycled). **Do not mutate ``X`` in place inside the block**;
        the cache cannot see that.

        Use it as a ``with`` block. Nesting is safe -- the previous cache is
        restored on exit -- but the save/restore is LIFO, so a generator instance
        must not be shared between concurrently running fits (threads, or manual
        out-of-order ``__enter__``/``__exit__``). Generators already carry other
        mutable probe state and were never thread-safe; the library's solvers are
        sequential.

        A caller that hands a *fresh* array to every call -- e.g. an ``int`` or
        ``float32`` ``X``, which ``np.asarray(..., dtype=float)`` must copy --
        would never hit the cache and would otherwise accumulate one full array
        per call. The cache is therefore capped at
        :data:`_BRANCH_CACHE_MAX_ENTRIES`; past that it is cleared rather than
        grown, degrading to the uncached cost instead of leaking memory. The
        solvers normalize ``X`` once before fitting, so they keep a single entry.
        """

        prev = self._branch_cache
        self._branch_cache = {}
        try:
            yield
        finally:
            self._branch_cache = prev

    def _branch_signs(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if getattr(self.branch_fn, "vectorized", False):
            # One call on all rows; the same signs as the row-by-row loop.
            b = np.asarray(self.branch_fn(X)).reshape(-1)  # type: ignore[misc]
            if b.shape != (len(X),):
                raise ValueError(
                    f"a vectorized branch_fn must return one value per row, got {b.shape}"
                )
            return np.where(b.astype(int) == 1, 1.0, -1.0)
        s = np.empty(len(X), dtype=float)
        for i in range(len(X)):
            s[i] = 1.0 if int(self.branch_fn(X[i])) == 1 else -1.0  # type: ignore[misc]
        return s

    def _sign(self, X: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return +1/-1 sign array for branch-wise generators."""

        if self.branch_fn is None:
            return np.where(v >= 0.0, 1.0, -1.0)

        cache = self._branch_cache
        if cache is None:
            return self._branch_signs(X)

        hit = cache.get(id(X))
        # The identity re-check is unreachable while the tuple holds X alive, and
        # is kept so that weakening that reference cannot silently alias arrays.
        if hit is not None and hit[0] is X:
            return hit[1]
        s = self._branch_signs(X)
        if len(cache) >= _BRANCH_CACHE_MAX_ENTRIES:
            # A caller that never reuses an array must not accumulate copies of
            # it. Drop the cache and start over rather than grow without bound.
            cache.clear()
        cache[id(X)] = (X, s)
        return s

    def _require_g(self) -> _RowwiseScalarFn:
        if self._g is None:
            raise ValueError("This generator does not define g().")
        return self._g

    # ------------------------------------------------------------------
    # Public interface required by generalized Riesz regression solvers
    # ------------------------------------------------------------------
    def g(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        """Evaluate g(x, alpha) row-wise."""

        X_ = as_2d(X)
        a_ = as_1d_of_length(alpha, n=len(X_), name="alpha")
        gfn = self._require_g()
        return gfn(X_, a_)

    def grad(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the derivative ∂g/∂alpha row-wise."""

        X_ = as_2d(X)
        a_ = as_1d_of_length(alpha, n=len(X_), name="alpha")

        if self._grad is not None:
            return self._grad(X_, a_)

        # Finite differences on g
        eps = self._eps
        gfn = self._require_g()
        return (gfn(X_, a_ + eps) - gfn(X_, a_ - eps)) / (2.0 * eps)

    def grad2(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the second derivative ∂²g/∂alpha² row-wise."""

        X_ = as_2d(X)
        a_ = as_1d_of_length(alpha, n=len(X_), name="alpha")

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

        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")

        if self._inv_grad is not None:
            return self._inv_grad(X_, v_)

        # Automatic inversion via Newton iterations.
        # This is a generic fallback and may be slow or unstable.
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

    def conjugate(
        self, X: ArrayLike, v: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (g*(v), alpha) evaluated row-wise."""

        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        alpha = self.inv_grad(X_, v_)
        g_val = self.g(X_, alpha)
        g_star = v_ * alpha - g_val
        return g_star, alpha

    def domain_binding(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        """Return a boolean mask of observations where an internal domain clip binds.

        The base implementation reports no binding. Built-in generators with
        numerical clips in ``inv_grad`` override this so that solvers and
        diagnostics can surface the clip binding rate instead of hiding it.
        """

        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        return np.zeros(v_.shape[0], dtype=bool)

    # ------------------------------------------------------------------
    # Interface used by the strict solvers (genriesz.solvers)
    # ------------------------------------------------------------------
    def link_domain(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        """Mask of rows whose dual coordinate ``v`` lies in the open range of ``g'``.

        The range is taken on the branch (sign component) allowed at each row.
        The generic implementation knows no domain and only requires ``v`` to be
        finite; built-in generators override it with their exact ranges.
        """

        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        return np.isfinite(v_)

    def dual_eval(
        self, X: ArrayLike, v: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(g*(v), alpha, d alpha / d v)`` row-wise, without clipping.

        ``alpha = (g')^{-1}(v)`` is the link and ``d alpha / d v = 1 / g''(alpha)``
        its derivative, so the Hessian of ``mean g*(v)`` in ``beta`` is
        ``Phi' diag(dalpha) Phi / n``. Rows outside :meth:`link_domain` are NaN in
        all three outputs; callers must check the domain (the solvers do).

        The generic implementation evaluates the user's ``inv_grad``, ``g`` and
        ``grad2`` on the in-domain rows.
        """

        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        g_star = np.full(v_.shape[0], np.nan)
        alpha = np.full(v_.shape[0], np.nan)
        dalpha = np.full(v_.shape[0], np.nan)
        ok = self.link_domain(X_, v_)
        if np.any(ok):
            Xo, vo = X_[ok], v_[ok]
            a = self.inv_grad(Xo, vo)
            g_star[ok] = vo * a - self.g(Xo, a)
            alpha[ok] = a
            dalpha[ok] = 1.0 / np.asarray(self.grad2(Xo, a), dtype=float)
        return g_star, alpha, dalpha

    def alpha_domain(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.bool_]:
        """Mask of rows whose representer value lies in the open domain of ``g``.

        Used by :class:`~genriesz.general_link.GRRGeneralLink`, whose link is not
        the generator's own. The generic implementation only requires finiteness.
        """

        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        return np.isfinite(a)

    def boundary_mask(
        self,
        X: ArrayLike,
        v: ArrayLike,
        alpha: ArrayLike,
        *,
        tol: float = 1e-8,
    ) -> NDArray[np.bool_]:
        """Mask of rows numerically at the boundary of the domain of ``g``.

        A fit whose final iterate has any such row is reported with status
        ``"uncertified_numerical_boundary"`` rather than ``"ok"``. The generic
        implementation knows no boundary (as for SQ, whose domain is the whole real line).
        """

        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        return np.zeros(a.shape[0], dtype=bool)

    def boundary_margin(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        """Distance of ``alpha`` to the finite boundary of the domain of ``g``.

        ``+inf`` when the domain has no finite boundary (SQ, generic). The
        solvers use it for a fraction-to-the-boundary rule: one step may not
        shrink any row's margin below ``margin_shrink`` (default ``0.01``) times
        its current value, so that a single long step cannot land on the
        boundary when the optimum is interior.
        """

        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        return np.full(a.shape[0], np.inf)

    def dual_margin(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
        """Distance of ``v`` to the finite end of the range of ``g'`` on its branch.

        ``+inf`` when the range is unbounded on the relevant side. Used with
        :meth:`boundary_margin` by the fraction-to-the-boundary rule.
        """

        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        return np.full(v_.shape[0], np.inf)

    def grad3(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        """Third derivative of ``g`` in ``alpha`` (central differences of ``grad2``).

        Built-in generators override this analytically. It is needed for the
        Hessian of the arbitrary-link objective (:class:`GRRGeneralLink`).
        """

        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        eps = self._eps
        return (self.grad2(X_, a + eps) - self.grad2(X_, a - eps)) / (2.0 * eps)

    def _alpha_sign_ok(self, X: NDArray[np.float64], a: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Whether ``alpha`` lies on the branch selected by ``branch_fn`` (if any)."""

        if self.branch_fn is None:
            return np.ones(a.shape[0], dtype=bool)
        return np.sign(a) == self._sign(X, a)


def _xlogx(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """``t log t`` with its continuous extension ``0`` at ``t = 0`` (NaN for ``t < 0``)."""

    with np.errstate(invalid="ignore"):
        return np.where(t >= 0.0, xlogy(t, t), np.nan)


class SquaredGenerator(BregmanGenerator):
    """Squared generator (SQ-Riesz).

    g(alpha) = (alpha - C)^2.

    This generator has no domain constraints (its domain is the whole real
    line, so no boundary rule applies) and induces the linear link

        alpha = C + 0.5 * v,    g*(v) = C v + v^2 / 4.
    """

    def __init__(self, C: float = 0.0):
        super().__init__(name="SQ", C=float(C), branch_fn=None)

    def inv_grad(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        return self.C + 0.5 * v_

    def g(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        return np.square(a - self.C)

    def grad(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        return 2.0 * (a - self.C)

    def grad2(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        return np.full_like(a, 2.0, dtype=float)

    def grad3(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        return np.zeros_like(a, dtype=float)

    def dual_eval(
        self, X: ArrayLike, v: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        g_star = self.C * v_ + 0.25 * v_ * v_
        return g_star, self.C + 0.5 * v_, np.full_like(v_, 0.5, dtype=float)




class UKLGenerator(BregmanGenerator):
    """Unnormalized KL generator (UKL-Riesz).

    The generator is::

        g(alpha) = (|alpha| - C) log(|alpha| - C) - |alpha|,  with |alpha| > C >= 0.

    The inverse gradient is branch-wise. With ``s`` the branch sign and
    ``u = s * v``::

        alpha = s (C + exp(u)),   g*(v) = C u + C + exp(u),   d alpha/d v = exp(u).

    If ``branch_fn`` is provided, it determines which branch is used for each
    observation.

    Parameters
    ----------
    C:
        Shift, ``C >= 0``.
    branch_fn:
        Branch selector returning 1 (positive) or 0 (negative).
    legacy_clip:
        ``False`` (default): the link, ``g`` and its derivatives are evaluated
        exactly. Outside the domain ``g``/``grad``/``grad2`` return NaN and the
        link raises :class:`DomainError`; nothing is clipped. ``True`` restores
        the clipping of releases <= 0.2.6 (floors on ``|alpha| - C`` and on the
        link argument), kept only so that old notebooks reproduce. A clipped
        link is not ``(g')^{-1}`` where the clip binds, so the fitted
        representer then targets a modified estimand.
    """

    def __init__(
        self,
        C: float = 1.0,
        *,
        branch_fn: BranchFn | None = None,
        legacy_clip: bool = False,
    ):
        if float(C) < 0:
            raise ValueError("C must be >= 0")
        if branch_fn is None:
            warnings.warn(
                "UKLGenerator without branch_fn uses sign(v) to select the alpha branch. "
                "This is correct only when |alpha| > C + 1. "
                "For GRR with functionals that require negative alpha (e.g. ATE/ATT), "
                "provide branch_fn or use SquaredGenerator instead.",
                UserWarning,
                stacklevel=2,
            )
        self.legacy_clip = bool(legacy_clip)
        super().__init__(name="UKL", C=float(C), branch_fn=branch_fn)

    def inv_grad(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
        """Branch-wise inverse gradient alpha = (g')^{-1}(v)."""

        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        s = self._sign(X_, v_)
        if self.legacy_clip:
            z = np.clip(s * v_, -700.0, 700.0)
            exp_term = np.maximum(np.exp(z), 1e-12)
            return s * (self.C + exp_term)
        # Exact: exp overflows to +inf for u > ~709, which the solvers report as
        # "nonfinite" (it is not clipped to a finite stand-in).
        with np.errstate(over="ignore"):
            return s * (self.C + np.exp(s * v_))

    def domain_binding(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        z = self._sign(X_, v_) * v_
        if self.legacy_clip:
            return (z <= np.log(1e-12)) | (z >= 700.0)
        return ~np.isfinite(z)

    def _t(self, a: NDArray[np.float64]) -> NDArray[np.float64]:
        t = np.abs(a) - self.C
        if self.legacy_clip:
            return np.maximum(t, 1e-12)
        return np.where(t >= 0.0, t, np.nan)

    def g(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = self._t(a)
        return _xlogx(t) - np.abs(a)

    def grad(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = self._t(a)
        with np.errstate(divide="ignore"):
            return np.sign(a) * np.log(t)

    def grad2(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = self._t(a)
        with np.errstate(divide="ignore"):
            return 1.0 / t

    def grad3(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = self._t(a)
        with np.errstate(divide="ignore"):
            return -np.sign(a) / (t * t)

    def link_domain(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        return np.isfinite(v_)

    def dual_eval(
        self, X: ArrayLike, v: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        s = self._sign(X_, v_)
        u = s * v_
        with np.errstate(over="ignore"):
            t = np.exp(u)
        return self.C * u + self.C + t, s * (self.C + t), t

    def alpha_domain(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.bool_]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        return np.isfinite(a) & (np.abs(a) > self.C) & self._alpha_sign_ok(X_, a)

    def boundary_mask(
        self, X: ArrayLike, v: ArrayLike, alpha: ArrayLike, *, tol: float = 1e-8
    ) -> NDArray[np.bool_]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        return (np.abs(a) - self.C) <= tol

    def boundary_margin(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        return np.abs(a) - self.C


class BPGenerator(BregmanGenerator):
    """Basu-power generator (BP-Riesz).

    A smooth family interpolating between UKL-like (small power) and squared-like
    (power near 1) behavior.

    We use the parametrization, for ``t = |alpha| - C``,

        g(alpha) = ( t^{1+omega} - (1+omega) t ) / omega,

    with domain ``|alpha| > C >= 0`` and ``omega > 0``. This equals the
    manuscript's ``((|alpha|-C)^{1+omega} - (|alpha|-C))/omega - |alpha|`` up to
    the constant ``-C``.

    The derivative is

        g'(alpha) = sign(alpha) * k * ( t^omega - 1 ),   k = (1+omega)/omega,

    whose range on each branch is the open half-line ``s * v > -k``. With
    ``u = s * v`` and ``w = 1 + u / k > 0``::

        alpha = s ( C + w^{1/omega} ),   g*(v) = C u + w^k,
        d alpha / d v = w^{1/omega - 1} / (1 + omega).

    The derivative of ``g`` stays bounded (``-> -k``) as ``|alpha| -> C``, so a
    fitted representer can approach the boundary at a finite dual coordinate;
    the solvers report this as status ``"uncertified_numerical_boundary"``.

    As with UKL, ``branch_fn`` can be supplied to select the sign.

    Parameters
    ----------
    C:
        Shift, ``C >= 0``.
    omega:
        Power, ``omega > 0`` (``power`` is an alias).
    branch_fn:
        Branch selector returning 1 (positive) or 0 (negative).
    legacy_clip:
        ``False`` (default): the link, ``g`` and its derivatives are evaluated
        exactly. Outside the domain ``g``/``grad``/``grad2`` return NaN and the
        link raises :class:`DomainError`; nothing is clipped. ``True`` restores
        the clipping of releases <= 0.2.6 (floors on ``|alpha| - C`` and on the
        link argument), kept only so that old notebooks reproduce. A clipped
        link is not ``(g')^{-1}`` where the clip binds, so the fitted
        representer then targets a modified estimand.
    """

    def __init__(
        self,
        C: float = 1.0,
        *,
        omega: float = 0.5,
        power: float | None = None,
        branch_fn: BranchFn | None = None,
        legacy_clip: bool = False,
    ):
        if power is not None:
            omega = float(power)
        if float(C) < 0:
            raise ValueError("C must be >= 0")
        if float(omega) <= 0:
            raise ValueError("omega must be > 0")
        self.omega = float(omega)
        self.legacy_clip = bool(legacy_clip)
        if branch_fn is None:
            warnings.warn(
                "BPGenerator without branch_fn uses sign(v) to select the alpha branch. "
                "This is correct only when |alpha| - C > 1. "
                "For GRR with functionals that require negative alpha (e.g. ATE/ATT), "
                "provide branch_fn or use SquaredGenerator instead.",
                UserWarning,
                stacklevel=2,
            )
        super().__init__(name=f"BP(omega={self.omega:g})", C=float(C), branch_fn=branch_fn)

    @property
    def _k(self) -> float:
        return 1.0 + 1.0 / self.omega

    def inv_grad(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
        """Branch-wise inverse gradient map for BP.

        The link is defined for ``w = 1 + sign*v/k > 0``. A violation raises
        :class:`DomainError` (with ``legacy_clip=True`` it is instead clipped to
        ``w = 1e-6``, the behavior of releases <= 0.2.6).
        """

        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        s = self._sign(X_, v_)
        w = 1.0 + s * v_ / self._k
        if self.legacy_clip:
            w = np.maximum(w, 1e-6)
        elif not np.all(w > 0.0):
            n_bad = int(np.sum(~(w > 0.0)))
            raise DomainError(
                f"BPGenerator domain violation: {n_bad}/{w.shape[0]} observation(s) "
                f"have 1 + s*v/k <= 0, outside the range of g' on their branch."
            )
        return s * (self.C + np.power(w, 1.0 / self.omega))

    def domain_binding(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        s = self._sign(X_, v_)
        w = 1.0 + s * v_ / self._k
        if self.legacy_clip:
            return w <= 1e-6
        return ~(w > 0.0)

    def _t(self, a: NDArray[np.float64]) -> NDArray[np.float64]:
        t = np.abs(a) - self.C
        if self.legacy_clip:
            return np.maximum(t, 1e-12)
        return np.where(t >= 0.0, t, np.nan)

    def g(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = self._t(a)
        return (np.power(t, 1.0 + self.omega) - (1.0 + self.omega) * t) / self.omega

    def grad(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = self._t(a)
        return np.sign(a) * self._k * (np.power(t, self.omega) - 1.0)

    def grad2(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = self._t(a)
        with np.errstate(divide="ignore"):
            return (1.0 + self.omega) * np.power(t, self.omega - 1.0)

    def grad3(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = self._t(a)
        om = self.omega
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sign(a) * (1.0 + om) * (om - 1.0) * np.power(t, om - 2.0)

    def dual_margin(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        return self._k + self._sign(X_, v_) * v_

    def link_domain(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        s = self._sign(X_, v_)
        return np.isfinite(v_) & (1.0 + s * v_ / self._k > 0.0)

    def dual_eval(
        self, X: ArrayLike, v: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        s = self._sign(X_, v_)
        u = s * v_
        w = 1.0 + u / self._k
        w = np.where(w > 0.0, w, np.nan)
        om = self.omega
        with np.errstate(over="ignore"):
            g_star = self.C * u + np.power(w, self._k)
            alpha = s * (self.C + np.power(w, 1.0 / om))
            dalpha = np.power(w, 1.0 / om - 1.0) / (1.0 + om)
        return g_star, alpha, dalpha

    def alpha_domain(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.bool_]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        return np.isfinite(a) & (np.abs(a) > self.C) & self._alpha_sign_ok(X_, a)

    def boundary_mask(
        self, X: ArrayLike, v: ArrayLike, alpha: ArrayLike, *, tol: float = 1e-8
    ) -> NDArray[np.bool_]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        return (np.abs(a) - self.C) <= tol

    def boundary_margin(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        return np.abs(a) - self.C


# ---------------------------------------------------------------------------
# Shared BKL math (used by both BKLGenerator and BoundedBKLGenerator so the two
# variants cannot drift apart). The generator function g is identical for both;
# only the inverse-link (inv_grad) differs: exact-and-raising vs bounded.
# ---------------------------------------------------------------------------
def _bkl_g(a: NDArray[np.float64], C: float) -> NDArray[np.float64]:
    """g(alpha) = t1 log t1 - t2 log t2, evaluated without cancellation (floored).

    Written literally, the two terms are each O(|alpha| log|alpha|) while their
    difference is only O(C log|alpha|), so the leading digits cancel: in float64
    the naive form loses all precision once |alpha| ~ 1e17 and returns exactly
    0. Substituting ``t2 = t1 + 2C`` gives

        g = -t1 log1p(2C / t1) - 2C log(t2),

    where ``t1 log1p(2C/t1) -> 2C`` smoothly as ``t1 -> inf``. Both terms are
    then O(C log|alpha|) and no cancellation occurs.

    This floored variant (``t1 >= 1e-12``) is used by :class:`BoundedBKLGenerator`
    and by ``BKLGenerator(legacy_clip=True)``; :func:`_bkl_g_exact` is the
    unfloored version.
    """

    t1 = np.maximum(np.abs(a) - C, 1e-12)
    t2 = np.maximum(np.abs(a) + C, 1e-12)
    return -t1 * np.log1p(2.0 * C / t1) - 2.0 * C * np.log(t2)


def _bkl_grad(a: NDArray[np.float64], C: float) -> NDArray[np.float64]:
    t1 = np.maximum(np.abs(a) - C, 1e-12)
    t2 = np.maximum(np.abs(a) + C, 1e-12)
    return np.sign(a) * (np.log(t1) - np.log(t2))


def _bkl_grad2(a: NDArray[np.float64], C: float) -> NDArray[np.float64]:
    denom = np.maximum(np.abs(a) * np.abs(a) - C * C, 1e-12)
    return (2.0 * C) / denom


def _bkl_g_exact(a: NDArray[np.float64], C: float) -> NDArray[np.float64]:
    """Unfloored BKL generator: closure value ``-2C log(2C)`` at ``|alpha| = C``, NaN below."""

    A = np.abs(a)
    t1 = A - C
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = -t1 * np.log1p(2.0 * C / t1) - 2.0 * C * np.log(A + C)
    out = np.where(t1 > 0.0, inner, np.nan)
    return np.where(t1 == 0.0, -2.0 * C * np.log(2.0 * C), out)


def _bkl_grad_exact(a: NDArray[np.float64], C: float) -> NDArray[np.float64]:
    A = np.abs(a)
    t1 = np.where(A - C >= 0.0, A - C, np.nan)
    with np.errstate(divide="ignore"):
        return np.sign(a) * (np.log(t1) - np.log(A + C))


def _bkl_grad2_exact(a: NDArray[np.float64], C: float) -> NDArray[np.float64]:
    A = np.abs(a)
    denom = np.where(A >= C, (A - C) * (A + C), np.nan)
    with np.errstate(divide="ignore"):
        return (2.0 * C) / denom


def _bkl_grad3_exact(a: NDArray[np.float64], C: float) -> NDArray[np.float64]:
    A = np.abs(a)
    denom = np.where(A >= C, (A - C) * (A + C), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.sign(a) * 4.0 * C * A / (denom * denom)


def _bkl_abs_alpha_from_u(u: NDArray[np.float64], C: float) -> NDArray[np.float64]:
    """|alpha| = C (1 + e^u) / (1 - e^u) for the BKL link, valid for u < 0.

    Callers must guarantee ``u < 0`` (``u`` bounded away from 0); this routine
    does not itself guard the ``u -> 0`` blow-up.

    The denominator is computed as ``-expm1(u)`` rather than ``1 - exp(u)``:
    the latter cancels as ``u -> 0-`` and collapses to 0 for ``|u| < 5.6e-17``,
    where the 1e-300 floor would then report ``|alpha| ~ 2e300`` instead of the
    correct ``~2C/|u|``.
    """

    t = np.exp(u)  # in (0, 1) for u < 0
    denom = np.maximum(-np.expm1(u), 1e-300)  # = 1 - e^u, exact as u -> 0-
    return C * (1.0 + t) / denom


def _bkl_dual(
    u: NDArray[np.float64], C: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """``(g*, |alpha|, d|alpha|/du)`` of the exact BKL link for ``u < 0`` (NaN elsewhere).

    With ``e = exp(u)``: ``|alpha| = C (1+e)/(1-e)``,
    ``g* = C u + 2C log(2C) - 2C log(1-e)`` and ``d|alpha|/du = 2C e/(1-e)^2``.
    ``g*`` diverges to ``+inf`` as ``u -> 0-``.
    """

    uu = np.where(u < 0.0, u, np.nan)
    e = np.exp(uu)
    one_minus_e = -np.expm1(uu)
    with np.errstate(divide="ignore", over="ignore"):
        A = C * (1.0 + e) / one_minus_e
        g_star = C * uu + 2.0 * C * np.log(2.0 * C) - 2.0 * C * np.log(one_minus_e)
        dA = 2.0 * C * e / (one_minus_e * one_minus_e)
    return g_star, A, dA


class BKLGenerator(BregmanGenerator):
    """Binary KL generator (BKL-Riesz).

    The generator is::

        g(alpha) = (|alpha| - C) log(|alpha| - C) - (|alpha| + C) log(|alpha| + C),

    with domain ``|alpha| > C`` and ``C > 0``.

    Its derivative is::

        g'(alpha) = sign(alpha) * log( (|alpha|-C) / (|alpha|+C) ).

    The inverse gradient is branch-wise. Let ``s`` be the desired sign branch
    (+1 or -1) and let ``u = s * v``. Since the log-ratio is always negative,
    the theoretical domain is ``u < 0`` and ``alpha`` diverges as ``u -> 0``.
    In particular ``v = 0`` is never admissible, so a BKL fit needs a starting
    point (or an offset ``u_ref``) with ``s * u_ref < 0``; the strict solvers
    report ``"infeasible_start"`` otherwise.

    This is the **uncapped** (mathematically exact) link: a domain violation
    (``u >= 0``) raises :class:`DomainError` instead of being silently clipped.
    The previous clip mapped ``u`` to ``-1e-8``, which produced
    ``alpha ~ 2C / 1e-8 ~ 2e8``. That value is not ``(g')^{-1}(v)``, so it broke
    the conjugate identity ``d g*(v)/dv = alpha`` and destroyed the GRR weights.

    ``L-BFGS-B`` (``GRRGLM(solver="lbfgs")``) cannot optimize this uncapped
    objective directly: its unconstrained line search steps out of the domain.
    The default damped-Newton/FISTA solvers backtrack into the domain instead.
    :class:`BoundedBKLGenerator` is a bounded variant that targets a modified
    estimand.

    If ``branch_fn`` is provided, it selects the sign branch.

    Parameters
    ----------
    C:
        Shift, ``C > 0``.
    branch_fn:
        Branch selector returning 1 (positive) or 0 (negative).
    legacy_clip:
        ``False`` (default): ``g`` and its derivatives are exact, with NaN below
        ``|alpha| = C``. ``True`` restores the ``1e-12`` floors of releases
        <= 0.2.6 in ``g``/``grad``/``grad2``. The link is exact in both cases.
    """

    def __init__(
        self,
        C: float = 1.0,
        *,
        branch_fn: BranchFn | None = None,
        legacy_clip: bool = False,
    ):
        if float(C) <= 0:
            raise ValueError("C must be > 0 for BKLGenerator")
        if branch_fn is None:
            warnings.warn(
                "BKLGenerator without branch_fn selects the alpha branch from "
                "sign(v) (positive branch for v <= 0). "
                "For GRR with functionals that require a fixed sign per "
                "observation (e.g. ATE/ATT), provide branch_fn or use "
                "SquaredGenerator instead.",
                UserWarning,
                stacklevel=2,
            )
        self.legacy_clip = bool(legacy_clip)
        super().__init__(name="BKL", C=float(C), branch_fn=branch_fn)

    def _branch_sign(
        self, X: NDArray[np.float64], v: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For BKL, the positive branch corresponds to v <= 0.
        if self.branch_fn is None:
            return np.where(v <= 0.0, 1.0, -1.0)
        return self._sign(X, v)

    def inv_grad(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        s = self._branch_sign(X_, v_)

        # The theoretical domain is u = s*v < 0. A violation has no finite
        # inverse image, so raise instead of clipping to a value that would
        # make alpha explode and break the conjugate identity (方針A).
        u = s * v_
        if np.any(u >= 0.0):
            n_bad = int(np.sum(u >= 0.0))
            raise DomainError(
                f"BKLGenerator domain violation: {n_bad}/{u.shape[0]} observation(s) "
                f"have u = s*v >= 0, where the exact link alpha = (g')^{{-1}}(v) is "
                f"undefined (alpha -> +inf). Use an offset (or a starting point) with "
                f"s*v < 0, or BoundedBKLGenerator for a bounded variant."
            )

        if self.legacy_clip:
            u = np.maximum(u, -700.0)
        return s * _bkl_abs_alpha_from_u(u, self.C)

    def domain_binding(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        """Mask of observations that violate the BKL domain (``u = s*v >= 0``).

        The exact link has no finite value there; :meth:`inv_grad` raises
        :class:`DomainError` rather than clipping. This mask lets callers count
        the violation rate before attempting a fit (or after catching the
        failure).
        """

        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        u = self._branch_sign(X_, v_) * v_
        return u >= 0.0

    def g(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        return _bkl_g(a, self.C) if self.legacy_clip else _bkl_g_exact(a, self.C)

    def grad(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        return _bkl_grad(a, self.C) if self.legacy_clip else _bkl_grad_exact(a, self.C)

    def grad2(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        return _bkl_grad2(a, self.C) if self.legacy_clip else _bkl_grad2_exact(a, self.C)

    def grad3(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        return _bkl_grad3_exact(a, self.C)

    def dual_margin(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        return -self._branch_sign(X_, v_) * v_

    def link_domain(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        return np.isfinite(v_) & (self._branch_sign(X_, v_) * v_ < 0.0)

    def dual_eval(
        self, X: ArrayLike, v: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        s = self._branch_sign(X_, v_)
        g_star, A, dA = _bkl_dual(s * v_, self.C)
        return g_star, s * A, dA

    def alpha_domain(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.bool_]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        return np.isfinite(a) & (np.abs(a) > self.C) & self._alpha_sign_ok(X_, a)

    def boundary_mask(
        self, X: ArrayLike, v: ArrayLike, alpha: ArrayLike, *, tol: float = 1e-8
    ) -> NDArray[np.bool_]:
        """Rows with ``|alpha| - C <= tol`` or ``s*v >= -1e-12`` (``|alpha|`` diverging)."""

        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        u = self._branch_sign(X_, v_) * v_
        return ((np.abs(a) - self.C) <= tol) | (u >= -1e-12)

    def boundary_margin(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        return np.abs(a) - self.C


class BoundedBKLGenerator(BregmanGenerator):
    """Bounded-representer BKL variant (方針B; target-sensitivity candidate).

    The BKL generator function ``g`` is the same as :class:`BKLGenerator`, but
    the inverse link is **bounded and smooth** instead of exact-and-raising::

        |alpha| = C (1 + e^u) / (1 - e^u),  with u = s * v clamped to u <= u_min,

    where ``u_min = log((alpha_max - C) / (alpha_max + C)) < 0`` is chosen so
    that ``|alpha| <= alpha_max``. On the dangerous side (``u -> 0``, where the
    exact link diverges) ``u`` is clamped to ``u_min`` and ``alpha`` is pinned at
    ``alpha_max``. Because ``alpha`` is then constant in ``v`` there,
    ``d alpha / d v = 0`` and the envelope identity ``d g*(v)/d v = alpha`` still
    holds exactly (unlike the old BKL clip, which clamped the pre-image to a
    near-zero ``u`` and produced a huge, inconsistent ``alpha``). The objective
    and gradient therefore stay mutually consistent everywhere, so this variant
    is optimizable with ``L-BFGS-B``.

    The price is that where the bound binds the estimator no longer targets the
    exact BKL-Riesz representer: it targets a **modified (bounded) estimand**.
    Report :meth:`domain_binding` (the bound-binding rate) and treat this as a
    *target-sensitivity* candidate, not an admissible one (design §9-4).

    Parameters
    ----------
    C:
        Positive generator shift (domain ``|alpha| > C``).
    alpha_max:
        Bound on ``|alpha|`` (must be ``> C``). This is a sensitivity knob: sweep
        it and report how the estimate moves. Defaults to ``50.0``.
    branch_fn:
        Optional branch selector (as in :class:`BKLGenerator`).
    """

    modifies_estimand = True

    def __init__(
        self,
        C: float = 1.0,
        *,
        alpha_max: float = 50.0,
        branch_fn: BranchFn | None = None,
    ):
        if float(C) <= 0:
            raise ValueError("C must be > 0 for BoundedBKLGenerator")
        if float(alpha_max) <= float(C):
            raise ValueError(
                f"alpha_max must be > C. Got alpha_max={alpha_max}, C={C}."
            )
        if branch_fn is None:
            warnings.warn(
                "BoundedBKLGenerator without branch_fn selects the alpha branch "
                "from sign(v) (positive branch for v <= 0). For GRR with "
                "functionals that require a fixed sign per observation (e.g. "
                "ATE/ATT), provide branch_fn.",
                UserWarning,
                stacklevel=2,
            )
        self.alpha_max = float(alpha_max)
        # u_min < 0 is the pre-image of alpha_max under the BKL link.
        self._u_min = float(
            np.log((self.alpha_max - float(C)) / (self.alpha_max + float(C)))
        )
        super().__init__(
            name=f"BoundedBKL(alpha_max={self.alpha_max:g})",
            C=float(C),
            branch_fn=branch_fn,
        )

    def _branch_sign(
        self, X: NDArray[np.float64], v: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # Same convention as BKLGenerator: positive branch corresponds to v <= 0.
        if self.branch_fn is None:
            return np.where(v <= 0.0, 1.0, -1.0)
        return self._sign(X, v)

    def inv_grad(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        s = self._branch_sign(X_, v_)
        # Clamp the dangerous side (u -> 0) to u_min so |alpha| <= alpha_max, and
        # the exp-underflow tail to -700. Both sides keep u strictly negative.
        u = np.clip(s * v_, -700.0, self._u_min)
        return s * _bkl_abs_alpha_from_u(u, self.C)

    def domain_binding(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        """Mask where the bound binds (``u = s*v > u_min`` -> ``|alpha| = alpha_max``).

        These are the observations at which the bounded variant departs from the
        exact BKL-Riesz representer, i.e. the modified-estimand region.
        """

        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        u = self._branch_sign(X_, v_) * v_
        return u > self._u_min

    def g(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        return _bkl_g(a, self.C)

    def grad(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        return _bkl_grad(a, self.C)

    def grad2(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        return _bkl_grad2(a, self.C)



    def dual_eval(
        self, X: ArrayLike, v: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        # Where the bound binds, alpha is constant in v (d alpha/d v = 0) and
        # g*(v) = v alpha - g(alpha) is affine in v, as the envelope identity
        # requires.
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        s = self._branch_sign(X_, v_)
        u = s * v_
        uc = np.clip(u, -700.0, self._u_min)
        A = _bkl_abs_alpha_from_u(uc, self.C)
        alpha = s * A
        e = np.exp(uc)
        dA = 2.0 * self.C * e / np.square(np.expm1(uc))
        dA = np.where((u > self._u_min) | (u < -700.0), 0.0, dA)
        g_star = v_ * alpha - _bkl_g(alpha, self.C)
        return g_star, alpha, dA

    def link_domain(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        return np.isfinite(v_)


class PUGenerator(BregmanGenerator):
    """PU generator (PU-Riesz).

    This generator is based on the binary-entropy potential::

        g(alpha) = C * [ |alpha| log|alpha| + (1-|alpha|) log(1-|alpha|) ],

    with domain ``|alpha| in (0, 1)`` and ``C > 0``.

    The derivative is::

        g'(alpha) = sign(alpha) * C * log( |alpha| / (1-|alpha|) ).

    The inverse gradient is a (scaled) logistic map: with ``u = s * v``,
    ``alpha = s * sigmoid(u / C)``, ``g*(v) = C log(1 + exp(u / C))``.

    Notes
    -----
    This generator is primarily useful when you want the representer to be
    bounded (in absolute value) by 1.

    Parameters
    ----------
    C:
        Scale, ``C > 0``.
    branch_fn:
        Branch selector returning 1 (positive) or 0 (negative).
    legacy_clip:
        ``False`` (default): the link, ``g`` and its derivatives are evaluated
        exactly. Outside the domain ``g``/``grad``/``grad2`` return NaN and the
        link raises :class:`DomainError`; nothing is clipped. ``True`` restores
        the clipping of releases <= 0.2.6 (floors on ``|alpha| - C`` and on the
        link argument), kept only so that old notebooks reproduce. A clipped
        link is not ``(g')^{-1}`` where the clip binds, so the fitted
        representer then targets a modified estimand.
    """

    def __init__(
        self,
        C: float = 1.0,
        *,
        branch_fn: BranchFn | None = None,
        legacy_clip: bool = False,
    ):
        if float(C) <= 0:
            raise ValueError("C must be > 0 for PUGenerator")
        if branch_fn is None:
            warnings.warn(
                "PUGenerator without branch_fn uses sign(v) to select the alpha "
                "branch. For GRR with functionals that require a fixed sign per "
                "observation (e.g. ATE/ATT), provide branch_fn.",
                UserWarning,
                stacklevel=2,
            )
        self.legacy_clip = bool(legacy_clip)
        super().__init__(name="PU", C=float(C), branch_fn=branch_fn)

    def inv_grad(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        s = self._sign(X_, v_)
        if self.legacy_clip:
            z = np.clip(s * v_ / self.C, -700.0, 700.0)
            a = 1.0 / (1.0 + np.exp(-z))
            a = np.clip(a, 1e-10, 1.0 - 1e-10)
            return s * a
        return s * expit(s * v_ / self.C)

    def domain_binding(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        z = self._sign(X_, v_) * v_ / self.C
        if self.legacy_clip:
            return np.abs(z) >= np.log(1e10)
        return ~np.isfinite(z)

    def _t(self, a: NDArray[np.float64]) -> NDArray[np.float64]:
        t = np.abs(a)
        if self.legacy_clip:
            return np.clip(t, 1e-10, 1.0 - 1e-10)
        return np.where(t <= 1.0, t, np.nan)

    def g(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = self._t(a)
        return self.C * (_xlogx(t) + _xlogx(1.0 - t))

    def grad(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = self._t(a)
        with np.errstate(divide="ignore"):
            return np.sign(a) * self.C * (np.log(t) - np.log(1.0 - t))

    def grad2(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = self._t(a)
        with np.errstate(divide="ignore"):
            return self.C / (t * (1.0 - t))

    def grad3(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = self._t(a)
        q = t * (1.0 - t)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sign(a) * self.C * (2.0 * t - 1.0) / (q * q)

    def link_domain(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        return np.isfinite(v_)

    def dual_eval(
        self, X: ArrayLike, v: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        s = self._sign(X_, v_)
        z = s * v_ / self.C
        p = expit(z)
        return self.C * np.logaddexp(0.0, z), s * p, p * (1.0 - p) / self.C

    def alpha_domain(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.bool_]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = np.abs(a)
        return np.isfinite(a) & (t > 0.0) & (t < 1.0) & self._alpha_sign_ok(X_, a)

    def boundary_mask(
        self, X: ArrayLike, v: ArrayLike, alpha: ArrayLike, *, tol: float = 1e-8
    ) -> NDArray[np.bool_]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = np.abs(a)
        return (t <= tol) | (1.0 - t <= tol)

    def boundary_margin(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = np.abs(a)
        return np.minimum(t, 1.0 - t)


_SQUARED_NAMES = frozenset({"sq", "squared", "lsif"})

# Names whose generator is branch-wise: the sign of alpha decides which branch
# of (g')^{-1} applies, so they cannot be built from a name alone.
_BRANCHWISE_NAMES = frozenset({"ukl", "bkl", "bp", "power", "pu"})


def coerce_generator(
    generator: BregmanGenerator | str,
    *,
    branch_fn: BranchFn | None = None,
    allow_branchwise_names: bool = True,
) -> BregmanGenerator:
    """Resolve a generator instance or a supported name to a generator.

    Parameters
    ----------
    generator:
        A :class:`BregmanGenerator`, or one of ``'sq'``, ``'ukl'``, ``'bkl'``,
        ``'bp'``, ``'pu'`` (with the aliases ``'squared'``, ``'lsif'`` and
        ``'power'``).
    branch_fn:
        Branch selector applied to the branch-wise generators built by name.
        It is **not** applied to a generator passed as an instance.
    allow_branchwise_names:
        When False, only the squared names resolve. A branch-wise name raises,
        because its branch depends on the estimand: a density ratio is
        nonnegative and always takes the positive branch, whereas an ATE Riesz
        representer is negative on the control units. Building one from a name
        would silently impose the wrong branch.
    """

    if isinstance(generator, BregmanGenerator):
        return generator

    if isinstance(generator, str):
        key = generator.strip().lower()
        if key in _SQUARED_NAMES:
            return SquaredGenerator(C=0.0)
        if key in _BRANCHWISE_NAMES:
            if not allow_branchwise_names:
                raise ValueError(
                    f"generator={generator!r} names a branch-wise generator, whose branch "
                    "depends on the estimand and cannot be inferred from the name. Pass an "
                    "instance with an explicit branch_fn, e.g. "
                    "BKLGenerator(C=1.0, branch_fn=lambda x: int(x[0] == 1.0)). "
                    "Only the squared-generator names may be given by name here: "
                    + ", ".join(repr(name) for name in sorted(_SQUARED_NAMES))
                    + "."
                )
            if key == "ukl":
                return UKLGenerator(C=0.0, branch_fn=branch_fn)
            if key == "bkl":
                return BKLGenerator(C=1.0, branch_fn=branch_fn)
            if key in {"bp", "power"}:
                return BPGenerator(C=0.0, omega=0.5, branch_fn=branch_fn)
            return PUGenerator(C=1.0, branch_fn=branch_fn)
        raise ValueError(
            "Unknown generator name. Use a generator instance or one of: "
            "'sq', 'ukl', 'bkl', 'bp', 'pu'."
        )

    raise TypeError(
        "generator must be a BregmanGenerator instance or a supported name, "
        f"got {type(generator).__name__}"
    )
