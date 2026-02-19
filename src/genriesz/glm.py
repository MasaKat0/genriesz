"""GLM-style solvers used by Generalized Riesz Regression.

The library uses a simple finite-dimensional model:

    v(x) = phi(x)^T beta,

and a generator-specific link that maps the linear predictor ``v`` to a Riesz
representer ``alpha(x)``.

The GRR objective for beta can be written as

    min_beta  E[ g*(v(X)) - m(X, v) ] + penalty(beta),

where ``g*`` is the convex conjugate of the generator and ``m(X,v)`` is linear
in ``v``.

We solve the resulting convex (often smooth) problem with L-BFGS-B.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

from .basis import Basis
from .functionals import LinearFunctional
from .generators import BregmanGenerator


def _as_2d(X: ArrayLike) -> NDArray[np.float64]:
    X_ = np.asarray(X, dtype=float)
    if X_.ndim != 2:
        raise ValueError(f"X must be 2D. Got shape {X_.shape}.")
    return X_


def _as_1d(y: ArrayLike, *, n: int, name: str) -> NDArray[np.float64]:
    y_ = np.asarray(y, dtype=float).reshape(-1)
    if y_.shape[0] != n:
        raise ValueError(f"{name} must have length {n}. Got {y_.shape}.")
    return y_


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    # Numerically stable sigmoid
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    expz = np.exp(z[~pos])
    out[~pos] = expz / (1.0 + expz)
    return out


@dataclass
class FitResult:
    beta: NDArray[np.float64]
    success: bool
    message: str
    n_iter: int


class _Penalty:
    def __init__(self, penalty: str | None, lam: float, p_norm: float | None):
        self.penalty = None if penalty is None else str(penalty).lower()
        self.lam = float(lam)
        self.p_norm = 2.0 if p_norm is None else float(p_norm)
        if self.lam < 0:
            raise ValueError("lam must be >= 0")
        if self.penalty in {"l1", "lasso"}:
            self.p_norm = 1.0
        elif self.penalty in {"l2", "ridge"}:
            self.p_norm = 2.0
        elif self.penalty in {"lp", "l_p", "p"}:
            if self.p_norm < 1.0:
                raise ValueError("p_norm must be >= 1")
        elif self.penalty in {None, "none", ""}:
            self.penalty = None
        else:
            raise ValueError(f"Unknown penalty: {penalty}")

        # Smoothing for the l1 gradient (subgradient at 0)
        self._eps = 1e-8

    def value(self, beta: NDArray[np.float64]) -> float:
        if self.penalty is None or self.lam == 0.0:
            return 0.0
        if self.p_norm == 1.0:
            return float(self.lam * np.sum(np.abs(beta)))
        return float((self.lam / self.p_norm) * np.sum(np.abs(beta) ** self.p_norm))

    def grad(self, beta: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.penalty is None or self.lam == 0.0:
            return np.zeros_like(beta)
        if self.p_norm == 1.0:
            return self.lam * beta / np.sqrt(beta * beta + self._eps)
        return self.lam * np.sign(beta) * (np.abs(beta) ** (self.p_norm - 1.0))


class GRRGLM:
    """Finite-dimensional generalized Riesz regression (GLM form)."""

    def __init__(
        self,
        *,
        basis: Basis,
        generator: BregmanGenerator,
        functional: LinearFunctional,
        penalty: str | None = "l2",
        lam: float = 1e-3,
        p_norm: float | None = None,
    ):
        self.basis = basis
        self.generator = generator
        self.functional = functional
        self.penalty = _Penalty(penalty, lam, p_norm)

        self._Phi: NDArray[np.float64] | None = None
        self._M: NDArray[np.float64] | None = None
        self.beta_: NDArray[np.float64] | None = None
        self.fit_result_: FitResult | None = None

    def fit(
        self,
        X: ArrayLike,
        *,
        beta0: ArrayLike | None = None,
        max_iter: int = 500,
        tol: float = 1e-8,
        verbose: bool = False,
    ) -> FitResult:
        X_ = _as_2d(X)
        self._Phi = np.asarray(self.basis(X_), dtype=float)
        self._M = np.asarray(self.functional.m_basis_matrix(X_, self.basis), dtype=float)

        n, p = self._Phi.shape
        if self._M.shape != (n, p):
            raise ValueError(
                f"m_basis_matrix returned shape {self._M.shape}, expected {(n, p)}."
            )

        if beta0 is None:
            beta0_ = np.zeros(p, dtype=float)
        else:
            beta0_ = np.asarray(beta0, dtype=float).reshape(-1)
            if beta0_.shape[0] != p:
                raise ValueError(f"beta0 must have length {p}. Got {beta0_.shape}.")

        Phi = self._Phi
        M = self._M

        # Objective and gradient
        big = 1e20

        def fun(beta: NDArray[np.float64]) -> float:
            v = Phi @ beta
            try:
                g_star, _ = self.generator.conjugate(X_, v)
            except Exception:
                return big
            loss = float(np.mean(g_star - (M @ beta)))
            return loss + self.penalty.value(beta)

        def jac(beta: NDArray[np.float64]) -> NDArray[np.float64]:
            v = Phi @ beta
            try:
                _, alpha = self.generator.conjugate(X_, v)
            except Exception:
                return np.zeros_like(beta)
            grad = (alpha[:, None] * Phi - M).mean(axis=0)
            return grad + self.penalty.grad(beta)

        res = optimize.minimize(
            fun=fun,
            x0=beta0_,
            jac=jac,
            method="L-BFGS-B",
            options={"maxiter": int(max_iter), "ftol": float(tol), "disp": bool(verbose)},
        )

        beta_hat = np.asarray(res.x, dtype=float)
        out = FitResult(
            beta=beta_hat,
            success=bool(res.success),
            message=str(res.message),
            n_iter=int(getattr(res, "nit", -1)),
        )
        self.beta_ = beta_hat
        self.fit_result_ = out
        return out

    def predict_v(self, X: ArrayLike) -> NDArray[np.float64]:
        if self.beta_ is None:
            raise RuntimeError("Model is not fit.")
        Phi = np.asarray(self.basis(_as_2d(X)), dtype=float)
        return Phi @ self.beta_

    def predict_alpha(self, X: ArrayLike) -> NDArray[np.float64]:
        v = self.predict_v(X)
        return self.generator.inv_grad(_as_2d(X), v)

    def derivative_alpha(self, X: ArrayLike, coordinate: int) -> NDArray[np.float64]:
        """Derivative of alpha(x) wrt x_coordinate.

        Uses the identity grad_g(alpha(x)) = v(x) and the inverse function theorem:

            g''(alpha) * d alpha/dx = d v/dx.
        """

        if self.beta_ is None:
            raise RuntimeError("Model is not fit.")
        X_ = _as_2d(X)
        dPhi = self.basis.derivative(X_, coordinate)
        dv = dPhi @ self.beta_
        alpha = self.predict_alpha(X_)
        g2 = self.generator.grad2(X_, alpha)
        return dv / g2


class OutcomeGLM:
    """Simple (penalized) outcome regression on top of a basis."""

    def __init__(
        self,
        *,
        basis: Basis,
        link: str = "identity",
        penalty: str | None = "l2",
        lam: float = 1e-3,
        p_norm: float | None = None,
    ):
        self.basis = basis
        self.link = str(link).lower()
        if self.link not in {"identity", "logit"}:
            raise ValueError("link must be 'identity' or 'logit'")
        self.penalty = _Penalty(penalty, lam, p_norm)

        self.theta_: NDArray[np.float64] | None = None

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        theta0: ArrayLike | None = None,
        max_iter: int = 500,
        tol: float = 1e-8,
        verbose: bool = False,
    ) -> FitResult:
        X_ = _as_2d(X)
        Phi = np.asarray(self.basis(X_), dtype=float)
        n, p = Phi.shape
        y_ = _as_1d(y, n=n, name="y")

        if theta0 is None:
            theta0_ = np.zeros(p, dtype=float)
        else:
            theta0_ = np.asarray(theta0, dtype=float).reshape(-1)
            if theta0_.shape[0] != p:
                raise ValueError(f"theta0 must have length {p}. Got {theta0_.shape}.")

        # Closed form for identity + l2 (ridge)
        if self.link == "identity" and self.penalty.penalty in {"l2", "ridge"}:
            if self.penalty.lam == 0.0:
                # Ordinary least squares (with pseudo-inverse)
                theta = np.linalg.pinv(Phi) @ y_
            else:
                A = (Phi.T @ Phi) / n + self.penalty.lam * np.eye(p)
                b = (Phi.T @ y_) / n
                theta = np.linalg.solve(A, b)
            self.theta_ = np.asarray(theta, dtype=float)
            return FitResult(beta=self.theta_, success=True, message="closed_form", n_iter=1)

        big = 1e20

        def fun(theta: NDArray[np.float64]) -> float:
            eta = Phi @ theta
            if self.link == "identity":
                resid = y_ - eta
                loss = 0.5 * float(np.mean(resid * resid))
            else:
                # Bernoulli negative log-likelihood
                # mean(log(1+exp(eta)) - y*eta)
                loss = float(np.mean(np.logaddexp(0.0, eta) - y_ * eta))
            return loss + self.penalty.value(theta)

        def jac(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            eta = Phi @ theta
            if self.link == "identity":
                resid = y_ - eta
                grad = -(Phi.T @ resid) / n
            else:
                p_hat = _sigmoid(eta)
                grad = (Phi.T @ (p_hat - y_)) / n
            return grad + self.penalty.grad(theta)

        res = optimize.minimize(
            fun=fun,
            x0=theta0_,
            jac=jac,
            method="L-BFGS-B",
            options={"maxiter": int(max_iter), "ftol": float(tol), "disp": bool(verbose)},
        )

        theta_hat = np.asarray(res.x, dtype=float)
        self.theta_ = theta_hat
        return FitResult(
            beta=theta_hat,
            success=bool(res.success),
            message=str(res.message),
            n_iter=int(getattr(res, "nit", -1)),
        )

    def predict_link(self, X: ArrayLike) -> NDArray[np.float64]:
        if self.theta_ is None:
            raise RuntimeError("OutcomeGLM is not fit.")
        Phi = np.asarray(self.basis(_as_2d(X)), dtype=float)
        return Phi @ self.theta_

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        eta = self.predict_link(X)
        if self.link == "identity":
            return eta
        return _sigmoid(eta)

    def derivative(self, X: ArrayLike, coordinate: int) -> NDArray[np.float64]:
        if self.theta_ is None:
            raise RuntimeError("OutcomeGLM is not fit.")
        X_ = _as_2d(X)
        dPhi = self.basis.derivative(X_, coordinate)
        deta = dPhi @ self.theta_
        if self.link == "identity":
            return deta
        mu = self.predict(X_)
        return mu * (1.0 - mu) * deta
