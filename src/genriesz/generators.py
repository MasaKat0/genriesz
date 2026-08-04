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

- A flexible :class:`BregmanGenerator` for a user-specified generator ``g``.
  The first derivative, inverse derivative, and second derivative must be
  supplied explicitly.

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

import inspect
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .utils import as_1d_of_length, as_2d

BranchFn = Callable[[NDArray[np.float64]], int]


@dataclass(frozen=True)
class GeneratorEvaluation:
    """Values and validity indicators for an exact inverse-link evaluation."""

    values: NDArray[np.float64]
    valid: NDArray[np.bool_]


@dataclass(frozen=True)
class ConjugateEvaluation:
    """Conjugate values, inverse-link values, and their validity indicators."""

    conjugate: NDArray[np.float64]
    alpha: NDArray[np.float64]
    valid: NDArray[np.bool_]

#: Upper bound on the number of arrays memoized inside ``branch_cache()``.
#: Solvers see one array per fit; the bound only exists so a caller that hands a
#: fresh array to every call degrades in speed rather than in memory.
_BRANCH_CACHE_MAX_ENTRIES = 8


class DomainError(RuntimeError):
    """Raised when a generator cannot evaluate its link/conjugate at a point.

    This replaces the previous behavior of silently returning a large objective
    value and a zero gradient or changing the inverse-link value. Status-returning
    generator methods let solvers record a failed fit without substituting a
    different representer value. Direct calls to an exact link raise this error.
    """


class _RowwiseScalarFn:
    """Evaluate a scalar callable once per observation.

    A custom generator callable must have signature ``f(alpha)`` or
    ``f(x, alpha)`` and must return one scalar for one observation. The
    convention is explicit: the library does not probe a callable, catch an
    exception, and silently switch between vectorized and rowwise evaluation.
    """

    def __init__(self, func: Callable):
        self.func = func
        sig = inspect.signature(func)
        positional = [
            parameter
            for parameter in sig.parameters.values()
            if parameter.kind
            in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) not in (1, 2):
            raise TypeError(
                "A custom generator callable must accept f(alpha) or f(x, alpha)."
            )
        self._arity = len(positional)
        self._vectorized = False

    def __call__(
        self, X: NDArray[np.float64], a: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a_ = as_1d_of_length(a, n=len(X_), name="a")
        out = np.empty(len(a_), dtype=float)
        if self._arity == 1:
            for i in range(len(a_)):
                out[i] = float(self.func(float(a_[i])))
        else:
            for i in range(len(a_)):
                out[i] = float(
                    self.func(np.asarray(X_[i], dtype=float), float(a_[i]))
                )
        return out


class _BranchCacheContext(AbstractContextManager[None]):
    """Install and restore one generator branch-sign cache."""

    def __init__(self, generator: BregmanGenerator):
        self.generator = generator
        self.previous: dict[
            int, tuple[NDArray[np.float64], NDArray[np.float64]]
        ] | None = None

    def __enter__(self) -> None:
        self.previous = self.generator._branch_cache
        self.generator._branch_cache = {}
        return None

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.generator._branch_cache = self.previous
        return False


class BregmanGenerator:
    """A Bregman generator with an explicit fitting link.

    Parameters
    ----------
    g:
        Generator function ``g(x, alpha)`` or ``g(alpha)``.
    grad:
        First derivative wrt alpha, ``∂g(x, alpha)/∂alpha``.
    inv_grad:
        Inverse derivative (link) ``alpha = (∂g)^{-1}(x, v)``. A custom
        generator used for fitting must supply this function explicitly.
    grad2:
        Second derivative wrt alpha (elementwise).
    name:
        Display name.
    C:
        Optional domain parameter used by some generator families.
    branch_fn:
        Optional branch selector returning 1 (positive) or 0 (negative).
        Built-in UKL/BP generators use this to choose the sign branch.

    Notes
    -----
    The generic generator supports regressor-dependent functions through
    ``g(x, alpha)``. The solver never infers a link from failed numerical
    inversion; ``inv_grad`` defines the fitted model.
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
    ):
        self.name = str(name)
        self.C = float(C)
        self.branch_fn = branch_fn
        # Active only inside branch_cache(): id(X) -> (X, signs). The array is
        # kept alive by the tuple so a stale id can never alias a new array.
        self._branch_cache: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] | None = (
            None
        )

        if type(self) is BregmanGenerator:
            missing = [
                name
                for name, value in (
                    ("g", g),
                    ("grad", grad),
                    ("inv_grad", inv_grad),
                    ("grad2", grad2),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "A custom BregmanGenerator requires explicit "
                    + ", ".join(missing)
                    + "."
                )

        self._g = None if g is None else _RowwiseScalarFn(g)
        self._grad = None if grad is None else _RowwiseScalarFn(grad)
        self._inv_grad = None if inv_grad is None else _RowwiseScalarFn(inv_grad)
        self._grad2 = None if grad2 is None else _RowwiseScalarFn(grad2)

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

    def dual_domain_mask(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        """Return where the exact inverse derivative is defined."""

        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        return np.isfinite(v_)

    def signed_dual_interval(
        self, *, margin: float
    ) -> tuple[float, float, float] | None:
        """Return bounds for ``u = s v`` when the exact domain is linear."""

        _ = margin
        return None

    def inv_grad_status(self, X: ArrayLike, v: ArrayLike) -> GeneratorEvaluation:
        """Evaluate the exact inverse link without clipping or substitution."""

        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        valid = np.asarray(self.dual_domain_mask(X_, v_), dtype=bool)
        values = np.full(v_.shape, np.nan, dtype=float)
        if bool(np.all(valid)):
            # Keep the original X object so a branch selector that depends only
            # on X is evaluated once inside branch_cache().
            values[:] = np.asarray(self.inv_grad(X_, v_), dtype=float)
        elif np.any(valid):
            values[valid] = np.asarray(self.inv_grad(X_[valid], v_[valid]), dtype=float)
        valid = valid & np.isfinite(values)
        values[~valid] = np.nan
        return GeneratorEvaluation(values=values, valid=valid)

    def conjugate_status(self, X: ArrayLike, v: ArrayLike) -> ConjugateEvaluation:
        """Evaluate the exact convex conjugate and report invalid rows."""

        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        link = self.inv_grad_status(X_, v_)
        conjugate = np.full(v_.shape, np.nan, dtype=float)
        if np.any(link.valid):
            alpha = link.values[link.valid]
            g_value = self.g(X_[link.valid], alpha)
            extended = (
                np.asarray(v_[link.valid], dtype=np.longdouble)
                * np.asarray(alpha, dtype=np.longdouble)
                - np.asarray(g_value, dtype=np.longdouble)
            )
            representable = np.isfinite(extended) & (
                np.abs(extended) <= np.finfo(float).max
            )
            valid_locations = np.flatnonzero(link.valid)
            conjugate[valid_locations[representable]] = np.asarray(
                extended[representable], dtype=float
            )
        valid = link.valid & np.isfinite(conjugate)
        conjugate[~valid] = np.nan
        alpha = link.values.copy()
        alpha[~valid] = np.nan
        return ConjugateEvaluation(conjugate=conjugate, alpha=alpha, valid=valid)

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def branch_cache(self) -> AbstractContextManager[None]:
        """Memoize branch signs during one sequential fit.

        The selector is a function of ``X`` alone. Reusing its values avoids
        repeated Python calls while the optimizer changes the dual coordinate.
        """

        return _BranchCacheContext(self)

    def _branch_signs(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        s = np.empty(len(X), dtype=float)
        for i in range(len(X)):
            s[i] = 1.0 if int(self.branch_fn(X[i])) == 1 else -1.0  # type: ignore[misc]
        return s

    def _sign(self, X: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return +1/-1 sign array for branch-wise generators."""

        if self.branch_fn is None:
            raise RuntimeError(
                "A branch-wise generator requires an explicit branch_fn. "
                "The branch cannot be inferred from the fitted dual coordinate."
            )

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

        if self._grad is None:
            raise RuntimeError("This generator does not define grad().")
        return self._grad(X_, a_)

    def grad2(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the second derivative ∂²g/∂alpha² row-wise."""

        X_ = as_2d(X)
        a_ = as_1d_of_length(alpha, n=len(X_), name="alpha")

        if self._grad2 is None:
            raise RuntimeError("This generator does not define grad2().")
        return self._grad2(X_, a_)

    def inv_grad(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the inverse derivative supplied by the user."""

        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        if self._inv_grad is None:
            raise RuntimeError(
                "A custom BregmanGenerator used for fitting must provide inv_grad. "
                "No numerical inverse or substitute link is used."
            )
        return self._inv_grad(X_, v_)

    def conjugate(
        self, X: ArrayLike, v: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (g*(v), alpha) evaluated row-wise."""

        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        alpha = self.inv_grad(X_, v_)
        g_val = self.g(X_, alpha)
        extended = (
            np.asarray(v_, dtype=np.longdouble) * np.asarray(alpha, dtype=np.longdouble)
            - np.asarray(g_val, dtype=np.longdouble)
        )
        representable = np.isfinite(extended) & (np.abs(extended) <= np.finfo(float).max)
        if not np.all(representable):
            n_bad = int(np.sum(~representable))
            raise DomainError(
                f"Generator '{self.name}' produced {n_bad}/{len(v_)} convex-conjugate "
                "value(s) outside the finite float64 range."
            )
        g_star = np.asarray(extended, dtype=float)
        return g_star, alpha

    def domain_binding(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        """Return where an explicit bound or an exact-domain margin is active.

        The base implementation reports no binding. A generator with a
        saturating (truncated) link or a restricted exact domain overrides
        this method to report where its stated bound or domain margin is
        active, so fits can surface the rate as a diagnostic.
        """

        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        return np.zeros(v_.shape[0], dtype=bool)


class SquaredGenerator(BregmanGenerator):
    """Squared generator (SQ-Riesz).

    g(alpha) = (alpha - C)^2.

    This generator has no strict domain constraints and induces a linear link

        alpha = C + 0.5 * v.
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


def _ukl_positive_distance(
    alpha: NDArray[np.float64], C: float, *, name: str
) -> NDArray[np.float64]:
    t = np.abs(alpha) - C
    valid = np.isfinite(alpha) & np.isfinite(t) & (t > 0.0)
    if not np.all(valid):
        n_bad = int(np.sum(~valid))
        raise DomainError(
            f"{name} alpha-domain violation for {n_bad}/{len(alpha)} "
            "observation(s): |alpha| must be strictly greater than C."
        )
    return t


class UKLGenerator(BregmanGenerator):
    """Unnormalized KL generator (UKL-Riesz).

    The generator is::

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
        if branch_fn is None:
            raise ValueError(
                "UKLGenerator requires branch_fn because the representer branch "
                "must be fixed by the estimand rather than inferred from v."
            )
        super().__init__(name="UKL", C=float(C), branch_fn=branch_fn)

    def _representable_link(
        self, X: NDArray[np.float64], v: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        s = self._sign(X, v)
        z = s * v
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            exp_term = np.exp(z)
            alpha_abs = self.C + exp_term
        valid = (
            np.isfinite(v)
            & np.isfinite(exp_term)
            & (exp_term > 0.0)
            & np.isfinite(alpha_abs)
            & (alpha_abs > self.C)
        )
        return s, alpha_abs, valid

    def signed_dual_interval(
        self, *, margin: float
    ) -> tuple[float, float, float]:
        inward = max(float(margin), np.finfo(float).eps)
        min_positive = np.nextafter(0.0, 1.0)
        max_float = np.finfo(float).max
        minimum_distance = (
            np.nextafter(self.C, np.inf) - self.C if self.C > 0.0 else min_positive
        )
        maximum_distance = np.nextafter(max_float - self.C, 0.0)
        return (
            float(np.log(minimum_distance) + inward),
            float(np.log(maximum_distance) - inward),
            0.0,
        )

    def inv_grad(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
        """Branch-wise inverse gradient alpha = (g')^{-1}(v)."""

        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        s, alpha_abs, valid = self._representable_link(X_, v_)
        if not np.all(valid):
            n_bad = int(np.sum(~valid))
            raise DomainError(
                f"UKLGenerator cannot represent the exact inverse link for "
                f"{n_bad}/{len(v_)} observation(s) in float64."
            )
        return s * alpha_abs

    def dual_domain_mask(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        _, _, valid = self._representable_link(X_, v_)
        return valid

    def domain_binding(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        return ~self.dual_domain_mask(X, v)

    def _positive_distance(self, alpha: NDArray[np.float64]) -> NDArray[np.float64]:
        return _ukl_positive_distance(alpha, self.C, name="UKLGenerator")

    def g(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = self._positive_distance(a)
        return t * np.log(t) - np.abs(a)

    def grad(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = self._positive_distance(a)
        return np.sign(a) * np.log(t)

    def grad2(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = self._positive_distance(a)
        return 1.0 / t


class BPGenerator(BregmanGenerator):
    """Basu-power generator (BP-Riesz).

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
        if branch_fn is None:
            raise ValueError(
                "BPGenerator requires branch_fn because the representer branch "
                "must be fixed by the estimand rather than inferred from v."
            )
        super().__init__(name=f"BP(omega={self.omega:g})", C=float(C), branch_fn=branch_fn)

    def signed_dual_interval(
        self, *, margin: float
    ) -> tuple[float, float, float]:
        inward = max(float(margin), np.finfo(float).eps)
        min_positive = np.nextafter(0.0, 1.0)
        max_float = np.finfo(float).max
        k = 1.0 + 1.0 / self.omega
        minimum_power = (
            np.nextafter(self.C, np.inf) - self.C if self.C > 0.0 else min_positive
        )
        log_t_min = self.omega * np.log(minimum_power)
        t_min = min_positive if log_t_min <= np.log(min_positive) else float(np.exp(log_t_min))
        lower = float(k * (max(t_min, min_positive) - 1.0) + inward)
        maximum_power = np.nextafter(max_float - self.C, 0.0)
        log_t_max = self.omega * np.log(maximum_power)
        upper = np.inf
        if log_t_max < np.log(max_float):
            upper = float(k * (float(np.exp(log_t_max)) - 1.0) - inward)
        return lower, upper, 0.0

    def inv_grad(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
        """Branch-wise inverse gradient map for BP.

        The theoretical domain restriction is ``t = 1 + sign*v/k > 0``. A
        violation has no real inverse image, so the method raises
        :class:`DomainError`. ``GRRGLM`` enforces the corresponding linear
        constraints when the branch selector is fixed by the regressors.
        """

        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        s = self._sign(X_, v_)
        k = 1.0 + 1.0 / self.omega

        t = 1.0 + s * v_ / k
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            power = np.power(t, 1.0 / self.omega)
            alpha_abs = self.C + power
        valid = (
            np.isfinite(v_)
            & np.isfinite(t)
            & (t > 0.0)
            & np.isfinite(power)
            & (power > 0.0)
            & np.isfinite(alpha_abs)
            & (alpha_abs > self.C)
        )
        if not np.all(valid):
            n_bad = int(np.sum(~valid))
            raise DomainError(
                f"BPGenerator exact-link failure for {n_bad}/{t.shape[0]} "
                "observation(s): 1 + s*v/k must be positive and the resulting "
                "alpha must be representable in float64."
            )
        return s * alpha_abs

    def dual_domain_mask(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        s = self._sign(X_, v_)
        k = 1.0 + 1.0 / self.omega
        t = 1.0 + s * v_ / k
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            power = np.power(t, 1.0 / self.omega)
            alpha_abs = self.C + power
        return (
            np.isfinite(v_)
            & np.isfinite(t)
            & (t > 0.0)
            & np.isfinite(power)
            & (power > 0.0)
            & np.isfinite(alpha_abs)
            & (alpha_abs > self.C)
        )

    def domain_binding(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        s = self._sign(X_, v_)
        k = 1.0 + 1.0 / self.omega
        return (1.0 + s * v_ / k) <= 1e-6

    def _positive_distance(self, alpha: NDArray[np.float64]) -> NDArray[np.float64]:
        t = np.abs(alpha) - self.C
        valid = np.isfinite(alpha) & np.isfinite(t) & (t > 0.0)
        if not np.all(valid):
            n_bad = int(np.sum(~valid))
            raise DomainError(
                f"BPGenerator alpha-domain violation for {n_bad}/{len(alpha)} "
                "observation(s): |alpha| must be strictly greater than C."
            )
        return t

    def g(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = self._positive_distance(a)
        return (np.power(t, 1.0 + self.omega) - (1.0 + self.omega) * t) / self.omega

    def grad(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = self._positive_distance(a)
        k = 1.0 + 1.0 / self.omega
        return np.sign(a) * k * (np.power(t, self.omega) - 1.0)

    def grad2(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = self._positive_distance(a)
        k = 1.0 + 1.0 / self.omega
        return k * self.omega * np.power(t, self.omega - 1.0)


# ---------------------------------------------------------------------------
# Shared BKL math (used by both BKLGenerator and BoundedBKLGenerator so the two
# variants cannot drift apart). The generator function g is identical for both;
# only the inverse-link (inv_grad) differs: exact-and-raising vs bounded.
# ---------------------------------------------------------------------------
def _bkl_g(a: NDArray[np.float64], C: float) -> NDArray[np.float64]:
    """g(alpha) = t1 log t1 - t2 log t2, evaluated without cancellation.

    Written literally, the two terms are each O(|alpha| log|alpha|) while their
    difference is only O(C log|alpha|), so the leading digits cancel: in float64
    the naive form loses all precision once |alpha| ~ 1e17 and returns exactly
    0. Substituting ``t2 = t1 + 2C`` gives

        g = -t1 log1p(2C / t1) - 2C log(t2),

    where ``t1 log1p(2C/t1) -> 2C`` smoothly as ``t1 -> inf``. Both terms are
    then O(C log|alpha|) and no cancellation occurs.

    The rewrite is an identity throughout the open domain ``|alpha| > C``.
    Inputs outside that domain raise :class:`DomainError`; no floor is used.
    """

    t1 = np.abs(a) - C
    t2 = np.abs(a) + C
    valid = np.isfinite(a) & np.isfinite(t1) & np.isfinite(t2) & (t1 > 0.0)
    if not np.all(valid):
        n_bad = int(np.sum(~valid))
        raise DomainError(
            f"BKLGenerator alpha-domain violation for {n_bad}/{len(a)} "
            "observation(s): |alpha| must be strictly greater than C."
        )
    return -t1 * np.log1p(2.0 * C / t1) - 2.0 * C * np.log(t2)


def _bkl_grad(a: NDArray[np.float64], C: float) -> NDArray[np.float64]:
    t1 = np.abs(a) - C
    t2 = np.abs(a) + C
    valid = np.isfinite(a) & np.isfinite(t1) & np.isfinite(t2) & (t1 > 0.0)
    if not np.all(valid):
        n_bad = int(np.sum(~valid))
        raise DomainError(
            f"BKLGenerator alpha-domain violation for {n_bad}/{len(a)} "
            "observation(s): |alpha| must be strictly greater than C."
        )
    return np.sign(a) * (np.log(t1) - np.log(t2))


def _bkl_grad2(a: NDArray[np.float64], C: float) -> NDArray[np.float64]:
    denom = np.abs(a) * np.abs(a) - C * C
    valid = np.isfinite(a) & np.isfinite(denom) & (denom > 0.0)
    if not np.all(valid):
        n_bad = int(np.sum(~valid))
        raise DomainError(
            f"BKLGenerator alpha-domain violation for {n_bad}/{len(a)} "
            "observation(s): |alpha| must be strictly greater than C."
        )
    return (2.0 * C) / denom


def _bkl_abs_alpha_from_u(u: NDArray[np.float64], C: float) -> NDArray[np.float64]:
    """|alpha| = C (1 + e^u) / (1 - e^u) for the BKL link, valid for u < 0.

    Callers must guarantee ``u < 0`` (``u`` bounded away from 0); this routine
    does not itself guard the ``u -> 0`` blow-up.

    The denominator is computed as ``-expm1(u)`` rather than ``1 - exp(u)``:
    the latter cancels as ``u -> 0-`` and collapses to 0 for ``|u| < 5.6e-17``,
    where the 1e-300 floor would then report ``|alpha| ~ 2e300`` instead of the
    correct ``~2C/|u|``.
    """

    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        t = np.exp(u)  # in (0, 1) for u < 0
        denom = -np.expm1(u)  # = 1 - e^u, accurate as u -> 0-
        return C * (1.0 + t) / denom


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

    This is the **uncapped** (mathematically exact) link: a domain violation
    (``u >= 0``) raises :class:`DomainError` instead of being silently clipped.
    The previous clip mapped ``u`` to ``-1e-8``, which produced
    ``alpha ~ 2C / 1e-8 ~ 2e8``. That value is not ``(g')^{-1}(v)``, so it broke
    the conjugate identity ``d g*(v)/dv = alpha`` and destroyed the GRR weights.
    Raising instead lets ``GRRGLM.fit`` report an explicit
    ``status="domain_error"`` failure (see design item E / coverage-design
    "KL系lossとcapの修正設計", 方針A).

    An unconstrained line search cannot optimize this objective with a fixed
    branch because it may leave the dual domain. ``GRRGLM`` therefore imposes
    the exact observationwise linear constraints when ``branch_fn`` is supplied.
    :class:`BoundedBKLGenerator` is the truncated variant of this model: its
    link saturates at stated representer bounds, so it stays defined for every
    finite dual coordinate and is optimizable unconstrained.

    If ``branch_fn`` is provided, it selects the sign branch.
    """

    def __init__(self, C: float = 1.0, *, branch_fn: BranchFn | None = None):
        if float(C) <= 0:
            raise ValueError("C must be > 0 for BKLGenerator")
        if branch_fn is None:
            raise ValueError(
                "BKLGenerator requires branch_fn because the representer branch "
                "must be fixed by the estimand rather than inferred from v."
            )
        super().__init__(name="BKL", C=float(C), branch_fn=branch_fn)

    def _branch_sign(
        self, X: NDArray[np.float64], v: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For BKL, the positive branch corresponds to v <= 0.
        return self._sign(X, v)

    def signed_dual_interval(
        self, *, margin: float
    ) -> tuple[float, float, float]:
        inward = max(float(margin), np.finfo(float).eps)
        alpha_next = np.nextafter(self.C, np.inf)
        delta = alpha_next - self.C
        lower = float(np.log(delta) - np.log(2.0 * self.C + delta) + inward)
        return lower, -inward, -1.0

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
                f"undefined (alpha -> +inf). Use BoundedBKLGenerator for a bounded, "
                f"optimizable variant."
            )

        alpha_abs = _bkl_abs_alpha_from_u(u, self.C)
        representable = np.isfinite(alpha_abs) & (alpha_abs > self.C)
        if not np.all(representable):
            n_bad = int(np.sum(~representable))
            raise DomainError(
                f"BKLGenerator cannot represent the exact inverse link for "
                f"{n_bad}/{len(v_)} observation(s) in float64."
            )
        return s * alpha_abs

    def dual_domain_mask(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        u = self._branch_sign(X_, v_) * v_
        alpha_abs = _bkl_abs_alpha_from_u(u, self.C)
        return np.isfinite(v_) & (u < 0.0) & np.isfinite(alpha_abs) & (alpha_abs > self.C)

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
        return _bkl_g(a, self.C)

    def grad(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        return _bkl_grad(a, self.C)

    def grad2(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        return _bkl_grad2(a, self.C)


class BoundedBKLGenerator(BregmanGenerator):
    """Truncated BKL model with a bounded representer link.

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

    The truncation absorbs the numerical instability of the exact link: the
    stated bounds are part of the fitted model, not a post-hoc rewrite of its
    values. Where a bound is active the fit reports it through
    :meth:`domain_binding` and the per-side binding rates, which are ordinary
    diagnostics to read alongside the estimate.

    Parameters
    ----------
    C:
        Positive generator shift (domain ``|alpha| > C``).
    alpha_max:
        Bound on ``|alpha|`` (must be ``> C``). Defaults to ``50.0``.
    branch_fn:
        Optional branch selector (as in :class:`BKLGenerator`).
    """

    #: The link is pinned at the bound where ``domain_binding`` is True, so
    #: ``d alpha / d v = 0`` there. ``GRRGLM.derivative_alpha`` uses this.
    link_is_constant_where_binding = True

    def __init__(
        self,
        C: float = 1.0,
        *,
        alpha_max: float = 50.0,
        branch_fn: BranchFn | None = None,
    ):
        if not (np.isfinite(float(C)) and float(C) > 0):
            raise ValueError("C must be finite and > 0 for BoundedBKLGenerator")
        if not np.isfinite(float(alpha_max)):
            raise ValueError(
                f"alpha_max must be finite. Got alpha_max={alpha_max}."
            )
        if float(alpha_max) <= float(C):
            raise ValueError(
                f"alpha_max must be > C. Got alpha_max={alpha_max}, C={C}."
            )
        if branch_fn is None:
            raise ValueError(
                "BoundedBKLGenerator requires branch_fn because the representer "
                "branch must be fixed by the estimand rather than inferred from v."
            )
        self.alpha_max = float(alpha_max)
        # u_min < 0 is the pre-image of alpha_max under the BKL link. The
        # lower dual bound is the pre-image of the smallest float64 value that
        # remains strictly above C. Both bounds are part of the stated
        # truncated model and are reported by domain_binding.
        self._u_min = float(
            np.log((self.alpha_max - float(C)) / (self.alpha_max + float(C)))
        )
        self.alpha_floor = float(np.nextafter(float(C), np.inf))
        self._u_floor = float(
            np.log(self.alpha_floor - float(C))
            - np.log(self.alpha_floor + float(C))
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
        return self._sign(X, v)

    def inv_grad(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        s = self._branch_sign(X_, v_)
        raw_u = s * v_
        lower_binding = raw_u < self._u_floor
        upper_binding = raw_u > self._u_min
        u = np.maximum(np.minimum(raw_u, self._u_min), self._u_floor)
        alpha_abs = _bkl_abs_alpha_from_u(u, self.C)
        alpha_abs[lower_binding] = self.alpha_floor
        alpha_abs[upper_binding] = self.alpha_max
        representable = np.isfinite(alpha_abs) & (alpha_abs > self.C)
        if not np.all(representable):
            n_bad = int(np.sum(~representable))
            raise DomainError(
                f"BoundedBKLGenerator cannot represent the bounded link for "
                f"{n_bad}/{len(v_)} observation(s) in float64."
            )
        return s * alpha_abs

    def dual_domain_mask(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        return np.isfinite(v_)

    def _signed_u(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        return self._branch_sign(X_, v_) * v_

    def lower_binding(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        """Mask where ``|alpha|`` is pinned at the representability floor."""

        return self._signed_u(X, v) < self._u_floor

    def upper_binding(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        """Mask where ``|alpha|`` is pinned at ``alpha_max``."""

        return self._signed_u(X, v) > self._u_min

    def domain_binding(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        """Mask where a stated representer bound is active.

        A point exactly on a pre-image boundary is not binding: the clamp is
        inactive there and the interior derivative applies.
        """

        return self.lower_binding(X, v) | self.upper_binding(X, v)

    def binding_diagnostics(self, X: ArrayLike, v: ArrayLike) -> dict[str, float]:
        """Stated bounds and per-side binding counts (audit GEN-07)."""

        return {
            "alpha_lower_bound": self.alpha_floor,
            "alpha_upper_bound": self.alpha_max,
            "n_lower_binding": int(np.sum(self.lower_binding(X, v))),
            "n_upper_binding": int(np.sum(self.upper_binding(X, v))),
        }

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


class BoundedUKLGenerator(BregmanGenerator):
    """Truncated UKL model with a bounded representer link.

    The UKL generator function is unchanged, but the fitted model is
    truncated: with ``z = s v`` the inverse link ``|alpha| = C + e^z`` is
    clamped to the pre-image interval ``[log(alpha_min - C),
    log(alpha_max - C)]``, so the representer satisfies
    ``alpha_min <= |alpha| <= alpha_max`` by construction. Where the clamp
    binds, ``alpha`` is pinned at the stated bound; ``d alpha / d v = 0``
    there, so the envelope identity ``d g*(v)/d v = alpha`` holds exactly and
    the objective and gradient stay mutually consistent everywhere. Values
    are never rewritten after fitting.

    The truncation absorbs the numerical instability of the exact link: the
    stated bounds are part of the fitted model, not a post-hoc rewrite of its
    values. Where a bound is active the fit reports it through
    :meth:`domain_binding` and the per-side binding rates, which are ordinary
    diagnostics to read alongside the estimate.

    Parameters
    ----------
    C:
        Generator shift (domain ``|alpha| > C``), ``C >= 0``.
    alpha_max:
        Finite upper bound on ``|alpha|`` (must exceed ``alpha_min``).
    alpha_min:
        Optional lower bound on ``|alpha|`` (must exceed ``C``). ``None``
        uses the smallest float64 value strictly above ``C``, which makes the
        lower clamp a pure representability floor rather than a model choice.
    branch_fn:
        Branch selector fixed by the estimand (required).
    """

    #: The link is pinned at the bound where ``domain_binding`` is True, so
    #: ``d alpha / d v = 0`` there. ``GRRGLM.derivative_alpha`` uses this.
    link_is_constant_where_binding = True

    def __init__(
        self,
        C: float = 1.0,
        *,
        alpha_max: float,
        alpha_min: float | None = None,
        branch_fn: BranchFn | None = None,
    ):
        if not (np.isfinite(float(C)) and float(C) >= 0):
            raise ValueError("C must be finite and >= 0 for BoundedUKLGenerator")
        if not np.isfinite(float(alpha_max)):
            raise ValueError(
                f"alpha_max must be finite. Got alpha_max={alpha_max}."
            )
        if alpha_min is None:
            alpha_min_ = float(np.nextafter(float(C), np.inf))
        else:
            if not np.isfinite(float(alpha_min)):
                raise ValueError(
                    f"alpha_min must be finite. Got alpha_min={alpha_min}."
                )
            alpha_min_ = float(alpha_min)
        if alpha_min_ <= float(C):
            raise ValueError(
                f"alpha_min must be > C. Got alpha_min={alpha_min_}, C={C}."
            )
        if not float(alpha_max) > alpha_min_:
            raise ValueError(
                "alpha_max must be > alpha_min. "
                f"Got alpha_max={alpha_max}, alpha_min={alpha_min_}."
            )
        if branch_fn is None:
            raise ValueError(
                "BoundedUKLGenerator requires branch_fn because the representer "
                "branch must be fixed by the estimand rather than inferred from v."
            )
        self.alpha_max = float(alpha_max)
        self.alpha_min = alpha_min_
        # Pre-images of the representer bounds under the UKL link.
        self._z_hi = float(np.log(self.alpha_max - float(C)))
        self._z_lo = float(np.log(self.alpha_min - float(C)))
        super().__init__(
            name=f"BoundedUKL(alpha_min={self.alpha_min:g}, alpha_max={self.alpha_max:g})",
            C=float(C),
            branch_fn=branch_fn,
        )

    @classmethod
    def from_propensity_bounds(
        cls,
        e_min: float,
        e_max: float,
        *,
        C: float = 1.0,
        branch_fn: BranchFn | None = None,
    ) -> BoundedUKLGenerator:
        """Build the ATE parameterization from propensity bounds.

        The treated arm has representer magnitude ``1/e`` and the control arm
        ``1/(1-e)``. For a symmetric window (``e_max = 1 - e_min``) both arms
        share the magnitude interval ``[1/e_max, 1/e_min]``, which is what
        this constructor states; an asymmetric window is rejected because a
        single magnitude interval cannot represent it -- state the representer
        bounds directly via ``alpha_min`` and ``alpha_max`` instead. Other estimands map
        differently; supply ``alpha_min``/``alpha_max`` directly there. There
        is no default range: the bounds are part of the model and must be
        stated explicitly.
        """

        e_min_ = float(e_min)
        e_max_ = float(e_max)
        if not 0.0 < e_min_ < 1.0:
            raise ValueError(
                f"e_min must be strictly inside (0, 1). Got e_min={e_min}."
            )
        if not 0.0 < e_max_ < 1.0:
            raise ValueError(
                f"e_max must be strictly inside (0, 1). Got e_max={e_max}."
            )
        if e_min_ >= e_max_:
            raise ValueError(
                f"e_min must be smaller than e_max. Got e_min={e_min}, e_max={e_max}."
            )
        if abs((1.0 - e_min_) - e_max_) > 1e-12:
            raise ValueError(
                "e_max must equal 1 - e_min: the treated arm has |alpha| = 1/e "
                "and the control arm has |alpha| = 1/(1 - e), so one magnitude "
                "interval covers both arms only for a symmetric propensity "
                f"window. Got e_min={e_min}, e_max={e_max}. For an asymmetric "
                "window, do not use this classmethod; state the representer "
                "bounds directly via alpha_min and alpha_max."
            )
        return cls(
            C=C,
            alpha_max=1.0 / e_min_,
            alpha_min=1.0 / e_max_,
            branch_fn=branch_fn,
        )

    def _branch_sign(
        self, X: NDArray[np.float64], v: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self._sign(X, v)

    def inv_grad(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        s = self._branch_sign(X_, v_)
        raw_z = s * v_
        lower_binding = raw_z < self._z_lo
        upper_binding = raw_z > self._z_hi
        z = np.minimum(np.maximum(raw_z, self._z_lo), self._z_hi)
        alpha_abs = self.C + np.exp(z)
        alpha_abs[lower_binding] = self.alpha_min
        alpha_abs[upper_binding] = self.alpha_max
        representable = np.isfinite(alpha_abs) & (alpha_abs > self.C)
        if not np.all(representable):
            n_bad = int(np.sum(~representable))
            raise DomainError(
                f"BoundedUKLGenerator cannot represent the bounded link for "
                f"{n_bad}/{len(v_)} observation(s) in float64."
            )
        return s * alpha_abs

    def dual_domain_mask(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        return np.isfinite(v_)

    def _signed_z(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        return self._branch_sign(X_, v_) * v_

    def lower_binding(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        """Mask where ``|alpha|`` is pinned at ``alpha_min``."""

        return self._signed_z(X, v) < self._z_lo

    def upper_binding(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        """Mask where ``|alpha|`` is pinned at ``alpha_max``."""

        return self._signed_z(X, v) > self._z_hi

    def domain_binding(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        """Mask where a stated representer bound is active.

        A point exactly on a pre-image boundary is not binding: the clamp is
        inactive there and the interior derivative applies.
        """

        return self.lower_binding(X, v) | self.upper_binding(X, v)

    def binding_diagnostics(self, X: ArrayLike, v: ArrayLike) -> dict[str, float]:
        """Stated bounds and per-side binding counts (audit GEN-07)."""

        return {
            "alpha_lower_bound": self.alpha_min,
            "alpha_upper_bound": self.alpha_max,
            "n_lower_binding": int(np.sum(self.lower_binding(X, v))),
            "n_upper_binding": int(np.sum(self.upper_binding(X, v))),
        }

    def g(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = _ukl_positive_distance(a, self.C, name="BoundedUKLGenerator")
        return t * np.log(t) - np.abs(a)

    def grad(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = _ukl_positive_distance(a, self.C, name="BoundedUKLGenerator")
        return np.sign(a) * np.log(t)

    def grad2(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = _ukl_positive_distance(a, self.C, name="BoundedUKLGenerator")
        return 1.0 / t


class PUGenerator(BregmanGenerator):
    """Binary-entropy generator with ``|alpha|`` in ``(0, 1)``.

    The derivative is ``sign(alpha) * C * log(|alpha| / (1-|alpha|))``.
    ``branch_fn`` fixes the sign branch. The inverse derivative is evaluated
    exactly whenever its value is representable in float64.
    """

    def __init__(self, C: float = 1.0, *, branch_fn: BranchFn | None = None):
        if float(C) <= 0:
            raise ValueError("C must be > 0 for PUGenerator")
        if branch_fn is None:
            raise ValueError(
                "PUGenerator requires branch_fn because the representer branch "
                "must be fixed by the estimand rather than inferred from v."
            )
        super().__init__(name="PU", C=float(C), branch_fn=branch_fn)

    def signed_dual_interval(
        self, *, margin: float
    ) -> tuple[float, float, float]:
        inward = max(float(margin), np.finfo(float).eps)
        min_positive = np.nextafter(0.0, 1.0)
        one_below = np.nextafter(1.0, 0.0)
        lower_logit = float(np.log(min_positive) - np.log1p(-min_positive))
        upper_logit = float(np.log(one_below) - np.log1p(-one_below))
        return (
            float(self.C * lower_logit + inward),
            float(self.C * upper_logit - inward),
            0.0,
        )

    def _representable_link(
        self, X: NDArray[np.float64], v: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        s = self._sign(X, v)
        z = s * v / self.C
        magnitude = np.empty_like(z, dtype=float)
        positive = z >= 0.0
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            magnitude[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
            exp_z = np.exp(z[~positive])
            magnitude[~positive] = exp_z / (1.0 + exp_z)
        valid = (
            np.isfinite(v)
            & np.isfinite(z)
            & np.isfinite(magnitude)
            & (magnitude > 0.0)
            & (magnitude < 1.0)
        )
        return s, magnitude, valid

    def inv_grad(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        s, magnitude, valid = self._representable_link(X_, v_)
        if not np.all(valid):
            n_bad = int(np.sum(~valid))
            raise DomainError(
                "PUGenerator cannot represent the exact inverse link for "
                f"{n_bad}/{len(v_)} observation(s) in float64."
            )
        return s * magnitude

    def dual_domain_mask(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        X_ = as_2d(X)
        v_ = as_1d_of_length(v, n=len(X_), name="v")
        _, _, valid = self._representable_link(X_, v_)
        return valid

    def domain_binding(self, X: ArrayLike, v: ArrayLike) -> NDArray[np.bool_]:
        return ~self.dual_domain_mask(X, v)

    def _probability_magnitude(self, alpha: NDArray[np.float64]) -> NDArray[np.float64]:
        magnitude = np.abs(alpha)
        valid = (
            np.isfinite(alpha)
            & np.isfinite(magnitude)
            & (magnitude > 0.0)
            & (magnitude < 1.0)
        )
        if not np.all(valid):
            n_bad = int(np.sum(~valid))
            raise DomainError(
                f"PUGenerator alpha-domain violation for {n_bad}/{len(alpha)} "
                "observation(s): |alpha| must lie strictly between 0 and 1."
            )
        return magnitude

    def g(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = self._probability_magnitude(a)
        return self.C * (t * np.log(t) + (1.0 - t) * np.log1p(-t))

    def grad(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = self._probability_magnitude(a)
        return np.sign(a) * self.C * (np.log(t) - np.log1p(-t))

    def grad2(self, X: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
        X_ = as_2d(X)
        a = as_1d_of_length(alpha, n=len(X_), name="alpha")
        t = self._probability_magnitude(a)
        return self.C / (t * (1.0 - t))


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
