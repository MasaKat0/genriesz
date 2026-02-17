"""Generalized Riesz Regression (GRR) with automatic covariate balancing.

This module implements a *parametric* (linear-in-parameters) GRR solver based on

- a user-provided basis function ``phi(X)`` and
- a Bregman generator ``g``.

Automatic covariate balancing (ACB)
-----------------------------------
The ACB linearity condition holds by construction if we parameterize the Riesz
representer ``alpha`` via the inverse derivative (inverse link) of the Bregman
generator:

.. math::

    \alpha_\beta(x) = (\partial g_x)^{-1}(\phi(x)^\top\beta).

This implies

.. math::

    (\partial g)\circ\alpha_\beta(x) = \phi(x)^\top\beta,

which is the key identity used in Theorem 4.1 of the paper.

Naming convention
-----------------
The original paper uses ``W`` for the regressor. In this package, the user-facing
API uses the more common notation ``X`` for the regressor and ``Y`` for the
outcome.

The regressor ``X`` can contain anything you want (e.g., ``X = [D, Z]`` in causal
settings with treatment ``D`` and covariates ``Z``); it is only interpreted by
``phi`` and by the functional ``m``.
"""

from __future__ import annotations

from dataclasses import dataclass

import re
from typing import Callable, Optional, Tuple

import numpy as np
from scipy import optimize

from .bregman import BregmanGenerator, _as_2d


BasisFn = Callable[[np.ndarray], np.ndarray]
MFunctional = Callable[[np.ndarray, Callable[[np.ndarray], float]], float]


@dataclass(frozen=True)
class ACBLink:
    """Link functions induced by a Bregman generator.

    ``link`` is the (branchwise) derivative of ``g`` with respect to the scalar
    argument, and ``inverse`` is its inverse.

    These are the link / inverse-link used to parameterize a GLM-style Riesz
    representer model that automatically satisfies the ACB linearity condition.
    """

    generator: BregmanGenerator

    def link(self, X: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        return self.generator.evaluate_grad(X, alpha)

    def inverse(self, X: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.generator.evaluate_inv_grad(X, v)


def _ensure_2d_features(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float)
    if phi.ndim == 1:
        return phi.reshape(1, -1)
    if phi.ndim != 2:
        raise ValueError(f"basis(X) must return a 1D or 2D array. Got shape {phi.shape}.")
    return phi


class GRR:
    r"""Generalized Riesz regression for GLM-style models.

    Parameters
    ----------
    basis:
        Basis function ``phi``. Must accept either a single sample ``X_i`` (1D
        array) or a batch ``X`` (2D array). The return shape must be ``(p,)`` or
        ``(n, p)``.
    m:
        Linear functional ``m(X, gamma)`` from the GRR paper.

        The callable ``gamma`` passed to ``m`` must accept a *single* sample
        ``X_i`` and return a float.
    generator:
        A :class:`~genriesz.bregman.BregmanGenerator` (or one of the predefined
        generators via ``...Generator(...).as_generator()``).
    penalty:
        One of ``"l2"``, ``"l1"``, or ``"lp"``.

        - ``"l2"``: ridge penalty ``(lam/2) * ||beta||_2^2``.
        - ``"l1"``: lasso penalty ``lam * ||beta||_1`` (via proximal gradient).
        - ``"lp"``: for any ``p >= 1``, penalty ``(lam/p) * sum_j |beta_j|^p``.
          Set ``penalty_p`` to choose p. You may also use the shorthand
          ``penalty="l<p>"`` (e.g., ``"l1.5"`` or ``"l3"``).
    lam:
        Regularization strength.
    penalty_p:
        Exponent ``p`` for the ``"lp"`` penalty.

    Notes
    -----
    This estimator fits a parameter vector ``beta`` by minimizing

    .. math::

        \frac{1}{n}\sum_{i=1}^n \left[g^*(v_i) - m\bigl(X_i, v(\cdot)\bigr)\right]
        + \lambda\,\Omega(\beta),

    where ``v_i = phi(X_i)^T beta`` and ``g^*`` is the convex conjugate of ``g``.

    After fitting, the estimated Riesz representer on a sample ``X`` is

    .. math::

        \hat\alpha(X) = (\partial g_X)^{-1}(\phi(X)^\top\hat\beta).
    """

    def __init__(
        self,
        *,
        basis: BasisFn,
        m: MFunctional,
        generator: BregmanGenerator,
        penalty: str = "l2",
        lam: float = 0.0,
        penalty_p: float | None = None,
    ) -> None:
        self.basis = basis
        self.m = m
        self.generator = generator
        self.link = ACBLink(generator)
        self.penalty = penalty
        self.lam = float(lam)
        self.penalty_p = penalty_p

        self.beta_: Optional[np.ndarray] = None
        self.p_: Optional[int] = None
        self._m_basis_matrix: Optional[np.ndarray] = None
        self._phi_matrix: Optional[np.ndarray] = None
        self._X_fit: Optional[np.ndarray] = None

    def clone(self) -> "GRR":
        """Create a new (unfitted) estimator with the same configuration."""
        return GRR(
            basis=self.basis,
            m=self.m,
            generator=self.generator,
            penalty=self.penalty,
            lam=float(self.lam),
            penalty_p=self.penalty_p,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        *,
        beta0: Optional[np.ndarray] = None,
        max_iter: int = 500,
        tol: float = 1e-8,
        verbose: bool = False,
    ) -> "GRR":
        X2 = _as_2d(X)
        phi = _ensure_2d_features(self.basis(X2))
        n, p = phi.shape

        self._X_fit = X2
        self._phi_matrix = phi
        self.p_ = p

        if beta0 is None:
            beta0 = np.zeros(p, dtype=float)
        beta0 = np.asarray(beta0, dtype=float).reshape(-1)
        if len(beta0) != p:
            raise ValueError(f"beta0 must have length {p}, got {len(beta0)}.")

        # Precompute M_{ij} = m(X_i, basis_j) for all i,j.
        #
        # For small bases, we can treat m as a black box. For large bases (RKHS,
        # random features, NN embeddings), this can become expensive. To support
        # large bases, m may implement a vectorized `basis_matrix` method
        # (see genriesz.functionals).
        if verbose:
            print("[GRR] Computing m(X_i, basis_j) matrix ...")

        if hasattr(self.m, "basis_matrix"):
            M = getattr(self.m, "basis_matrix")(X2, self.basis)
            M = np.asarray(M, dtype=float)
            if M.shape != (n, p):
                raise ValueError(
                    "m.basis_matrix(X, basis) must return shape (n,p). "
                    f"Expected {(n, p)}, got {M.shape}."
                )
            self._m_basis_matrix = M
        else:
            self._m_basis_matrix = self._compute_m_on_basis(X2, p)

        penalty_kind, p_norm = self._parse_penalty(self.penalty, self.penalty_p)
        if penalty_kind == "l1":
            beta_hat = self._fit_l1(beta0, max_iter=max_iter, tol=tol, verbose=verbose)
        else:
            beta_hat = self._fit_lp(beta0, p=p_norm, max_iter=max_iter, tol=tol, verbose=verbose)

        self.beta_ = beta_hat
        return self

    def predict_alpha(self, X: np.ndarray) -> np.ndarray:
        r"""Evaluate the fitted Riesz representer ``\hat\alpha(X)``."""

        self._check_is_fitted()
        X2 = _as_2d(X)
        phi = _ensure_2d_features(self.basis(X2))
        v = phi @ self.beta_
        return self.generator.evaluate_inv_grad(X2, v)

    def predict_gamma(self, X: np.ndarray) -> np.ndarray:
        r"""Evaluate ``\hat\gamma(X) := (\partial g)\circ\hat\alpha(X)``.

        For ACB-parameterized models, this is simply ``phi(X)^T beta``.
        """

        self._check_is_fitted()
        X2 = _as_2d(X)
        phi = _ensure_2d_features(self.basis(X2))
        return phi @ self.beta_

    def estimate_linear_functional(self, Y: np.ndarray, X: Optional[np.ndarray] = None) -> float:
        r"""Estimate the target parameter via ``E[\hat\alpha(X) Y]``."""

        self._check_is_fitted()
        if X is None:
            if self._X_fit is None:
                raise ValueError("If X is not provided, the estimator must have been fit first.")
            X = self._X_fit
        alpha = self.predict_alpha(X)
        Y1 = np.asarray(Y, dtype=float).reshape(-1)
        if len(alpha) != len(Y1):
            raise ValueError("Y and X must have the same number of rows.")
        return float(np.mean(alpha * Y1))

    def covariate_balance_residual(self) -> np.ndarray:
        """Return the sample balancing residual mean_i[alpha_i * phi_i - m_i]."""

        self._check_is_fitted()
        if self._X_fit is None or self._phi_matrix is None or self._m_basis_matrix is None:
            raise RuntimeError("Internal fit state is missing.")
        alpha = self.predict_alpha(self._X_fit)
        residual = (alpha[:, None] * self._phi_matrix) - self._m_basis_matrix
        return residual.mean(axis=0)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _check_is_fitted(self) -> None:
        if self.beta_ is None:
            raise RuntimeError("This GRR instance is not fitted yet. Call fit() first.")

    def _basis_single(self, x: np.ndarray) -> np.ndarray:
        # Ensure shape (p,)
        x1 = np.asarray(x, dtype=float).reshape(-1)
        try:
            out = np.asarray(self.basis(x1), dtype=float)
        except Exception:
            # Some user-provided bases only support batch inputs.
            out = np.asarray(self.basis(x1[None, :]), dtype=float)
        if out.ndim == 2:
            if out.shape[0] != 1:
                raise ValueError("basis(single_sample) returned a 2D array with more than one row.")
            out = out.reshape(-1)
        if out.ndim != 1:
            raise ValueError("basis(single_sample) must return a 1D vector.")
        return out

    def _compute_m_on_basis(self, X: np.ndarray, p: int) -> np.ndarray:
        """Compute ``M_{ij} = m(X_i, basis_j)``."""

        n = len(X)
        M = np.empty((n, p), dtype=float)

        # We build basis_j as a gamma callable.
        for j in range(p):

            def gamma_j(x: np.ndarray, j=j) -> float:
                return float(self._basis_single(x)[j])

            for i in range(n):
                M[i, j] = float(self.m(X[i], gamma_j))

        return M

    # -----------------------
    # Optimization: smooth lp (includes ridge)
    # -----------------------
    @staticmethod
    def _parse_penalty(penalty: str, penalty_p: Optional[float]) -> tuple[str, float]:
        """Parse a penalty specification.

        Returns
        -------
        kind:
            Either ``"l1"`` or ``"lp"``.
        p:
            The lp exponent (>= 1). For ``kind == "l1"``, this is 1.
        """

        pen = str(penalty).lower().strip()

        if pen in {"l1", "lasso"}:
            return "l1", 1.0
        if pen in {"l2", "ridge"}:
            return "lp", 2.0

        if pen in {"lp", "l_p"}:
            if penalty_p is None:
                raise ValueError("penalty_p must be provided when penalty is 'lp'.")
            p = float(penalty_p)
            if p < 1.0:
                raise ValueError("penalty_p must be >= 1.")
            if np.isclose(p, 1.0):
                return "l1", 1.0
            return "lp", p

        # Allow shorthand like "l1.5", "l3", ...
        m = re.fullmatch(r"l(\d+(?:\.\d+)?)", pen)
        if m is not None:
            p = float(m.group(1))
            if p < 1.0:
                raise ValueError("Penalty exponent must be >= 1.")
            if np.isclose(p, 1.0):
                return "l1", 1.0
            return "lp", p

        raise ValueError("penalty must be one of {'l2', 'l1', 'lp', 'l<p>'}")

    def _objective_and_grad_lp(self, beta: np.ndarray, *, p: float) -> Tuple[float, np.ndarray]:
        """Objective and gradient for a smooth lp penalty with p > 1 (ridge is p=2)."""

        assert self._X_fit is not None
        assert self._phi_matrix is not None
        assert self._m_basis_matrix is not None

        if p <= 1.0:
            raise ValueError("Smooth lp objective requires p > 1. Use the l1 solver for p=1.")

        X = self._X_fit
        Phi = self._phi_matrix
        M = self._m_basis_matrix

        v = Phi @ beta
        g_star, alpha = self.generator.conjugate(X, v)
        loss = float(np.mean(g_star - (M @ beta)))
        grad = (alpha[:, None] * Phi - M).mean(axis=0)

        if self.lam > 0:
            absb = np.abs(beta)
            # Use (lam/p) * sum |beta|^p so that grad is lam * sign(beta) * |beta|^{p-1}.
            loss += (self.lam / float(p)) * float(np.sum(absb**p))
            grad = grad + self.lam * np.sign(beta) * (absb ** (p - 1.0))

        return loss, grad

    def _fit_lp(
        self, beta0: np.ndarray, *, p: float, max_iter: int, tol: float, verbose: bool
    ) -> np.ndarray:
        """Fit the smooth lp-penalized objective for p > 1."""

        if p <= 1.0:
            raise ValueError("p must be > 1 for _fit_lp. Use penalty='l1' for p=1.")

        def fun(beta: np.ndarray) -> float:
            val, _ = self._objective_and_grad_lp(beta, p=p)
            return val

        def jac(beta: np.ndarray) -> np.ndarray:
            _, g = self._objective_and_grad_lp(beta, p=p)
            return g

        res = optimize.minimize(
            fun=fun,
            x0=beta0,
            jac=jac,
            method="L-BFGS-B",
            options={"maxiter": int(max_iter), "ftol": float(tol), "disp": bool(verbose)},
        )
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")
        return np.asarray(res.x, dtype=float)

    # -----------------------
    # Optimization: L1 (proximal gradient)
    # -----------------------
    def _smooth_objective_and_grad(self, beta: np.ndarray) -> Tuple[float, np.ndarray]:
        """Smooth part: mean(g*(v) - M beta)."""

        assert self._X_fit is not None
        assert self._phi_matrix is not None
        assert self._m_basis_matrix is not None

        X = self._X_fit
        Phi = self._phi_matrix
        M = self._m_basis_matrix

        v = Phi @ beta
        g_star, alpha = self.generator.conjugate(X, v)
        f = float(np.mean(g_star - (M @ beta)))
        grad = (alpha[:, None] * Phi - M).mean(axis=0)
        return f, grad

    @staticmethod
    def _soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
        return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)

    def _fit_l1(self, beta0: np.ndarray, *, max_iter: int, tol: float, verbose: bool) -> np.ndarray:
        if self.lam <= 0:
            # Degenerates to unregularized problem.
            return self._fit_lp(beta0, p=2.0, max_iter=max_iter, tol=tol, verbose=verbose)

        beta = beta0.copy()
        step = 1.0

        def objective(beta_vec: np.ndarray) -> float:
            f, _ = self._smooth_objective_and_grad(beta_vec)
            return f + self.lam * float(np.sum(np.abs(beta_vec)))

        prev_obj = objective(beta)

        for it in range(int(max_iter)):
            f, grad = self._smooth_objective_and_grad(beta)

            # Backtracking line search on the smooth part.
            step_k = step
            while True:
                beta_next = self._soft_threshold(beta - step_k * grad, step_k * self.lam)
                f_next, _ = self._smooth_objective_and_grad(beta_next)
                # Sufficient decrease condition (quadratic upper bound)
                diff = beta_next - beta
                if f_next <= f + grad.dot(diff) + (0.5 / step_k) * float(np.dot(diff, diff)):
                    break
                step_k *= 0.5
                if step_k < 1e-12:
                    break

            beta_new = beta_next
            obj = f_next + self.lam * float(np.sum(np.abs(beta_new)))

            rel_impr = abs(prev_obj - obj) / max(1.0, abs(prev_obj))
            if verbose and (it % 50 == 0 or it == max_iter - 1):
                print(f"[GRR:L1] iter={it} obj={obj:.6g} rel_impr={rel_impr:.3e} step={step_k:.3e}")

            beta = beta_new
            prev_obj = obj
            step = step_k

            if rel_impr < tol:
                break

        return beta


def run_grr_glm_acb(
    *,
    X: np.ndarray,
    Y: Optional[np.ndarray],
    basis: BasisFn,
    m: MFunctional,
    g: Callable[[np.ndarray, float], float],
    g_grad: Optional[Callable[[np.ndarray, float], float]] = None,
    g_inv_grad: Optional[Callable[[np.ndarray, float], float]] = None,
    penalty: str = "l2",
    lam: float = 0.0,
    penalty_p: float | None = None,
    beta0: Optional[np.ndarray] = None,
    max_iter: int = 500,
    tol: float = 1e-8,
    verbose: bool = False,
) -> tuple[GRR, np.ndarray, Optional[float]]:
    """Convenience wrapper to fit GLM-style GRR with an ACB link.

    This helper matches the workflow:

    1) user specifies ``m(X, gamma)``
    2) user specifies a basis ``phi(X)``
    3) user specifies a Bregman generator ``g(X, alpha)``
    4) we compute the link via the (inverse) derivative of ``g``
    5) we fit GRR and return ``alpha_hat`` (and optionally ``theta_hat``)

    Parameters
    ----------
    X:
        Regressor array of shape (n, d). The meaning of each column is up to the
        user; it is only interpreted by ``basis`` and by ``m``.
    Y:
        Optional response of shape (n,). If provided, this function returns the
        plug-in estimate ``mean(alpha_hat * Y)``.
    basis, m:
        See :class:`GRR`.
    g:
        Bregman generator g. You can also pass ``g_grad`` / ``g_inv_grad`` for
        speed and numerical stability.

    Returns
    -------
    est:
        Fitted :class:`GRR` instance.
    alpha_hat:
        Estimated Riesz representer evaluated on ``X``.
    theta_hat:
        If ``Y`` is provided, the estimate ``mean(alpha_hat * Y)``. Otherwise ``None``.
    """

    generator = BregmanGenerator(g=g, grad=g_grad, inv_grad=g_inv_grad)
    est = GRR(basis=basis, m=m, generator=generator, penalty=penalty, lam=lam, penalty_p=penalty_p)
    est.fit(X, beta0=beta0, max_iter=max_iter, tol=tol, verbose=verbose)

    alpha_hat = est.predict_alpha(X)
    theta_hat = None
    if Y is not None:
        theta_hat = est.estimate_linear_functional(Y, X)

    return est, alpha_hat, theta_hat


# Backward-compatible aliases.
GRRGLM = GRR
