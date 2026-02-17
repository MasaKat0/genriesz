"""High-level estimation of linear functionals using GRR.

The most general entry point is :func:`grr_functional`.

You provide

- data ``(X, Y)``,
- a linear functional ``m(X, gamma)``,
- a basis function ``phi(X)``, and
- a Bregman generator (or equivalently a link via its derivative).

The function fits a Riesz representer model via :class:`genriesz.genriesz.GRR` and returns
common estimators of the target parameter:

- **DM** (direct method / plug-in): ``E[m(X, \hat\gamma)]``
- **IPW** (weighting only): ``E[\hat\alpha(X) Y]``
- **AIPW** (augmented): ``E[\hat\alpha(X)(Y-\hat\gamma(X)) + m(X, \hat\gamma)]``

Cross-fitting is supported via a simple K-fold splitting.

Convenience wrappers
--------------------
For common causal estimands, this module includes thin wrappers that simply set
``m``:

- :func:`grr_ate` — average treatment effect
- :func:`grr_ame` — average marginal effect (average derivative)

All public docstrings and comments are written in English.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Protocol

import numpy as np
from scipy import optimize
from scipy.stats import norm

from .bregman import BregmanGenerator, _as_2d
from .functionals import ATEFunctional, AverageDerivativeFunctional, PolicyEffectFunctional
from .grr import GRR, BasisFn


Array = np.ndarray


class OutcomeModel(Protocol):
    """Protocol for outcome regression models used by DM/AIPW."""

    def fit(self, X: Array, y: Array, **kwargs: Any) -> "OutcomeModel":  # pragma: no cover
        ...

    def predict(self, X: Array) -> Array:  # pragma: no cover
        ...


def _as_1d(a: Array, *, n: Optional[int] = None) -> Array:
    a = np.asarray(a, dtype=float).reshape(-1)
    if n is not None and len(a) != n:
        raise ValueError(f"Expected length {n}, got {len(a)}.")
    return a


def _clone(est: Any) -> Any:
    """Best-effort estimator cloning.

    Uses sklearn.clone if available, otherwise falls back to deepcopy.
    """

    try:
        from sklearn.base import clone as skl_clone  # type: ignore

        return skl_clone(est)
    except Exception:
        return copy.deepcopy(est)


def _make_folds(n: int, folds: int, rng: np.random.Generator) -> Array:
    if folds < 2:
        return np.zeros(n, dtype=int)
    idx = rng.permutation(n)
    fold_id = np.empty(n, dtype=int)
    for k, part in enumerate(np.array_split(idx, folds)):
        fold_id[part] = k
    return fold_id


def _predict_1(model: OutcomeModel, x: Array) -> float:
    x2 = _as_2d(x)
    pred = np.asarray(model.predict(x2), dtype=float).reshape(-1)
    return float(pred[0])


def _functional_values(m: Any, X: Array, outcome_model: OutcomeModel) -> Array:
    """Compute m(X_i, gamma_hat) for all rows X_i.

    For the built-in functional classes, this is vectorized.
    For a generic callable m(x, gamma), we fall back to a Python loop.
    """

    X2 = _as_2d(X)

    if isinstance(m, ATEFunctional):
        X1 = X2.copy()
        X0 = X2.copy()
        X1[:, m.treatment_index] = m.treat_value_1
        X0[:, m.treatment_index] = m.treat_value_0
        return np.asarray(outcome_model.predict(X1), dtype=float).reshape(-1) - np.asarray(
            outcome_model.predict(X0), dtype=float
        ).reshape(-1)

    if isinstance(m, PolicyEffectFunctional):
        z = np.delete(X2, m.treatment_index, axis=1)
        d1 = np.array([float(m.policy_1(z[i])) for i in range(len(z))], dtype=float)
        d0 = np.array([float(m.policy_0(z[i])) for i in range(len(z))], dtype=float)
        X1 = X2.copy()
        X0 = X2.copy()
        X1[:, m.treatment_index] = d1
        X0[:, m.treatment_index] = d0
        return np.asarray(outcome_model.predict(X1), dtype=float).reshape(-1) - np.asarray(
            outcome_model.predict(X0), dtype=float
        ).reshape(-1)

    if isinstance(m, AverageDerivativeFunctional):
        Xp = X2.copy()
        Xm = X2.copy()
        Xp[:, m.coordinate] += m.eps
        Xm[:, m.coordinate] -= m.eps
        return (
            np.asarray(outcome_model.predict(Xp), dtype=float).reshape(-1)
            - np.asarray(outcome_model.predict(Xm), dtype=float).reshape(-1)
        ) / (2.0 * float(m.eps))

    # Generic fallback: evaluate per-row using a callable gamma_hat.
    def gamma_hat(x: Array) -> float:
        return _predict_1(outcome_model, x)

    out = np.empty(len(X2), dtype=float)
    for i in range(len(X2)):
        out[i] = float(m(X2[i], gamma_hat))
    return out


@dataclass
class LinearOutcomeModel:
    """Penalized linear outcome regression on a user-specified basis.

    We fit a coefficient vector ``theta`` by minimizing

        (1/2) * mean_i (y_i - phi(X_i)^T theta)^2  +  (lam/p) * sum_j abs(theta_j)^p,

    where p >= 1. For p=1 we use a proximal-gradient solver; for p>1 we use
    L-BFGS-B.

    Parameters
    ----------
    basis:
        Feature map phi(X).
    penalty:
        One of "l2", "l1", "lp", or the shorthand "l<p>" (e.g., "l1.5").
    lam:
        Regularization strength.
    penalty_p:
        Exponent p when penalty is "lp".
    """

    basis: BasisFn
    penalty: str = "l2"
    lam: float = 0.0
    penalty_p: float | None = None

    theta_: Optional[Array] = None

    @staticmethod
    def _parse_penalty(penalty: str, penalty_p: Optional[float]) -> tuple[str, float]:
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

        m = re.fullmatch(r"l(\d+(?:\.\d+)?)", pen)
        if m is not None:
            p = float(m.group(1))
            if p < 1.0:
                raise ValueError("Penalty exponent must be >= 1.")
            if np.isclose(p, 1.0):
                return "l1", 1.0
            return "lp", p

        raise ValueError("penalty must be one of {'l2', 'l1', 'lp', 'l<p>'}")

    def clone(self) -> "LinearOutcomeModel":
        return LinearOutcomeModel(
            basis=self.basis,
            penalty=self.penalty,
            lam=float(self.lam),
            penalty_p=self.penalty_p,
        )

    def fit(self, X: Array, y: Array) -> "LinearOutcomeModel":
        X2 = _as_2d(X)
        y1 = _as_1d(y, n=len(X2))

        Phi = np.asarray(self.basis(X2), dtype=float)
        if Phi.ndim != 2:
            raise ValueError("basis(X) must return a 2D array for batched input.")

        n, p = Phi.shape
        pen_kind, p_norm = self._parse_penalty(self.penalty, self.penalty_p)

        if pen_kind == "l1":
            theta = np.zeros(p, dtype=float)
            step = 1.0
            # Basic proximal gradient; suitable for small/medium p.
            for _ in range(2000):
                resid = Phi @ theta - y1
                grad = (Phi.T @ resid) / n
                theta_new = theta - step * grad
                # Soft-thresholding
                thresh = step * float(self.lam)
                theta_new = np.sign(theta_new) * np.maximum(0.0, np.abs(theta_new) - thresh)

                if np.linalg.norm(theta_new - theta) <= 1e-10 * max(1.0, np.linalg.norm(theta)):
                    theta = theta_new
                    break
                theta = theta_new
            self.theta_ = theta
            return self

        # Smooth p-norm penalty (p>1) + squared loss: use L-BFGS-B.
        lam = float(self.lam)

        def objective(theta: Array) -> tuple[float, Array]:
            theta = np.asarray(theta, dtype=float)
            resid = Phi @ theta - y1
            loss = 0.5 * float(np.mean(resid**2))
            # (lam/p) * sum |theta|^p
            reg = (lam / p_norm) * float(np.sum(np.abs(theta) ** p_norm))
            val = loss + reg

            grad_loss = (Phi.T @ resid) / n
            grad_reg = lam * np.sign(theta) * (np.abs(theta) ** (p_norm - 1.0))
            grad = grad_loss + grad_reg
            return val, grad

        theta0 = np.zeros(p, dtype=float)
        res = optimize.minimize(
            fun=lambda t: objective(t)[0],
            x0=theta0,
            jac=lambda t: objective(t)[1],
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        if not res.success:
            raise RuntimeError(f"Outcome optimization failed: {res.message}")

        self.theta_ = np.asarray(res.x, dtype=float)
        return self

    def predict(self, X: Array) -> Array:
        if self.theta_ is None:
            raise RuntimeError("Outcome model is not fitted.")
        X2 = _as_2d(X)
        Phi = np.asarray(self.basis(X2), dtype=float)
        return Phi @ self.theta_


@dataclass(frozen=True)
class ScalarEstimate:
    """A scalar estimate with standard error, CI, and p-value."""

    name: str
    theta: float
    stderr: float
    ci_low: float
    ci_high: float
    p_value: float

    @property
    def estimate(self) -> float:
        """Alias for :attr:`theta` (kept for backwards compatibility)."""
        return self.theta


@dataclass
class FunctionalEstimateResult:
    """Container returned by :func:`grr_functional`."""

    n: int
    cross_fit: bool
    folds: int
    alpha: float
    null: float

    estimates: dict[str, ScalarEstimate]

    # Optional nuisance predictions (always length n if present).
    alpha_hat: Optional[Array] = None
    gamma_hat_shared: Optional[Array] = None
    gamma_hat_separate: Optional[Array] = None
    m_hat_shared: Optional[Array] = None
    m_hat_separate: Optional[Array] = None

    def summary_text(self) -> str:
        """Return a human-readable multi-line summary."""

        lines = []
        lines.append(
            f"n={self.n}, cross_fit={self.cross_fit}, folds={self.folds}, alpha={self.alpha}, null={self.null}"
        )

        # Stable ordering
        for key in sorted(self.estimates.keys()):
            e = self.estimates[key]
            lines.append(
                f"{e.name:>14s}: {e.estimate: .6f}  (se={e.stderr:.6f})  "
                f"CI[{1.0-self.alpha:.0%}]={e.ci_low:.6f},{e.ci_high:.6f}  p={e.p_value:.4g}"
            )
        return "\n".join(lines)


def _wald_stats(scores: Array, *, alpha: float, null: float, name: str) -> ScalarEstimate:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    n = len(scores)
    theta = float(np.mean(scores))
    # Influence-function style standard error
    stderr = float(np.std(scores - theta, ddof=1) / np.sqrt(n)) if n >= 2 else float("nan")

    if not np.isfinite(stderr) or stderr <= 0:
        z = float("nan")
        p = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")
    else:
        z = (theta - null) / stderr
        p = float(2.0 * (1.0 - norm.cdf(abs(z))))
        zcrit = float(norm.ppf(1.0 - alpha / 2.0))
        ci_low = theta - zcrit * stderr
        ci_high = theta + zcrit * stderr

    return ScalarEstimate(
        name=name,
        theta=theta,
        stderr=stderr,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p),
    )


def grr_functional(
    *,
    X: Optional[Array] = None,
    Y: Optional[Array] = None,
    # Backward-compatible aliases
    W: Optional[Array] = None,
    outcome: Optional[Array] = None,
    m: Any,
    basis: BasisFn,
    generator: Optional[BregmanGenerator] = None,
    g: Optional[Callable[[Array, float], float]] = None,
    g_grad: Optional[Callable[[Array, float], float]] = None,
    g_inv_grad: Optional[Callable[[Array, float], float]] = None,
    # Riesz regression (GRR) hyperparameters
    riesz_penalty: str = "l2",
    riesz_lam: float = 0.0,
    riesz_p_norm: float | None = None,
    # Outcome model hyperparameters (for default linear model)
    outcome_model: Optional[OutcomeModel] = None,
    outcome_basis: Optional[BasisFn] = None,
    outcome_penalty: Optional[str] = None,
    outcome_lam: Optional[float] = None,
    outcome_p_norm: float | None = None,
    # How to use outcome models for DM/AIPW
    outcome_models: str = "auto",
    # Cross-fitting
    cross_fit: bool = True,
    folds: int = 5,
    random_state: int | None = None,
    # Which estimators to report
    estimators: Iterable[str] = ("dm", "ipw", "aipw"),
    alpha: float = 0.05,
    null: float = 0.0,
    # Optimizer controls
    max_iter: int = 500,
    tol: float = 1e-8,
    verbose: bool = False,
) -> FunctionalEstimateResult:
    """Estimate a linear functional using GRR.

    Parameters
    ----------
    X, Y:
        The regressor and outcome.

        Backward-compatible aliases ``W`` and ``outcome`` are accepted.
    m, basis:
        The functional and basis.
    generator:
        A :class:`~genriesz.bregman.BregmanGenerator`. If omitted, you must provide ``g``.
    g, g_grad, g_inv_grad:
        If ``generator`` is omitted, these are used to build one.
    outcome_models:
        Controls which outcome model variants are fitted (relevant for DM/AIPW):

        - "none": no outcome model; only IPW is returned.
        - "shared": outcome model uses the same basis/penalty as the Riesz model.
        - "separate": use the user-provided ``outcome_model`` (or ``outcome_basis``).
        - "both": fit both shared and separate.
        - "auto" (default): shared is always used when DM/AIPW is requested; if a
          separate outcome model is provided, it is also used.

    Returns
    -------
    FunctionalEstimateResult
        Contains estimates for DM/IPW/AIPW (as requested), each with a standard
        error, confidence interval, and p-value.
    """

    if X is None and W is not None:
        X = W
    if Y is None and outcome is not None:
        Y = outcome

    if X is None or Y is None:
        raise ValueError("X and Y must be provided.")

    X2 = _as_2d(X)
    n = len(X2)
    Y1 = _as_1d(Y, n=n)

    if generator is None:
        if g is None:
            raise ValueError("Either 'generator' or 'g' must be provided.")
        generator = BregmanGenerator(g=g, grad=g_grad, inv_grad=g_inv_grad)

    folds = int(folds)
    cross_fit = bool(cross_fit) and folds >= 2

    rng = (
        np.random.default_rng(int(random_state))
        if random_state is not None
        else np.random.default_rng()
    )

    estimators_set = {str(e).lower().strip() for e in estimators}
    if "all" in estimators_set:
        estimators_set = {"dm", "ipw", "aipw"}

    want_dm = "dm" in estimators_set or "direct" in estimators_set
    want_ipw = "ipw" in estimators_set
    want_aipw = "aipw" in estimators_set

    # Outcome model selection.
    outcome_models = str(outcome_models).lower().strip()
    if outcome_models not in {"auto", "none", "shared", "separate", "both"}:
        raise ValueError("outcome_models must be one of {'auto','none','shared','separate','both'}." )

    use_outcome = want_dm or want_aipw
    if not use_outcome:
        outcome_models = "none"

    has_separate = outcome_model is not None or outcome_basis is not None

    fit_shared = False
    fit_separate = False
    if outcome_models == "none":
        fit_shared = False
        fit_separate = False
    elif outcome_models == "shared":
        fit_shared = True
        fit_separate = False
    elif outcome_models == "separate":
        fit_shared = False
        fit_separate = True
    elif outcome_models == "both":
        fit_shared = True
        fit_separate = True
    elif outcome_models == "auto":
        fit_shared = True
        fit_separate = bool(has_separate)

    if fit_separate and not has_separate:
        raise ValueError(
            "outcome_models requires a separate outcome model, but neither outcome_model nor outcome_basis was provided."
        )

    # Shared outcome model defaults: use Riesz settings unless the user overrides.
    if outcome_penalty is None:
        outcome_penalty = riesz_penalty
    if outcome_lam is None:
        outcome_lam = float(riesz_lam)

    if outcome_basis is None:
        outcome_basis = basis

    # Allocate nuisance arrays.
    alpha_hat = np.full(n, np.nan, dtype=float) if (want_ipw or want_aipw) else None

    gamma_hat_shared = np.full(n, np.nan, dtype=float) if fit_shared else None
    m_hat_shared = np.full(n, np.nan, dtype=float) if fit_shared else None

    gamma_hat_sep = np.full(n, np.nan, dtype=float) if fit_separate else None
    m_hat_sep = np.full(n, np.nan, dtype=float) if fit_separate else None

    # Fold ids.
    fold_id = _make_folds(n, folds, rng) if cross_fit else np.zeros(n, dtype=int)

    # Loop over folds (or a single pseudo-fold).
    unique_folds = np.unique(fold_id)

    for k in unique_folds:
        te = fold_id == k
        tr = ~te

        X_tr = X2[tr]
        Y_tr = Y1[tr]
        X_te = X2[te]

        # Riesz regression model.
        riesz = GRR(
            basis=basis,
            m=m,
            generator=generator,
            penalty=riesz_penalty,
            lam=float(riesz_lam),
            penalty_p=riesz_p_norm,
        )
        riesz.fit(X_tr, max_iter=max_iter, tol=tol, verbose=verbose)

        if alpha_hat is not None:
            alpha_hat[te] = riesz.predict_alpha(X_te)

        # Shared outcome model (default linear regression on the same basis).
        if fit_shared:
            out_shared = LinearOutcomeModel(
                basis=basis,
                penalty=str(outcome_penalty),
                lam=float(outcome_lam),
                penalty_p=outcome_p_norm,
            )
            out_shared.fit(X_tr, Y_tr)
            gamma_hat_shared[te] = np.asarray(out_shared.predict(X_te), dtype=float).reshape(-1)
            m_hat_shared[te] = _functional_values(m, X_te, out_shared)

        # Separate outcome model.
        if fit_separate:
            if outcome_model is not None:
                out_sep = _clone(outcome_model)
            else:
                out_sep = LinearOutcomeModel(
                    basis=outcome_basis,
                    penalty=str(outcome_penalty),
                    lam=float(outcome_lam),
                    penalty_p=outcome_p_norm,
                )
            out_sep.fit(X_tr, Y_tr)
            gamma_hat_sep[te] = np.asarray(out_sep.predict(X_te), dtype=float).reshape(-1)
            m_hat_sep[te] = _functional_values(m, X_te, out_sep)

    # Sanity check: no NaNs for requested nuisances.
    if alpha_hat is not None and not np.all(np.isfinite(alpha_hat)):
        raise RuntimeError("alpha_hat contains non-finite values. Consider stronger regularization.")

    estimates: dict[str, ScalarEstimate] = {}

    if want_ipw:
        if alpha_hat is None:
            raise RuntimeError("Internal error: alpha_hat not computed.")
        scores = alpha_hat * Y1
        estimates["ipw"] = _wald_stats(scores, alpha=alpha, null=null, name="IPW")

    if want_dm and fit_shared:
        if m_hat_shared is None:
            raise RuntimeError("Internal error: m_hat_shared not computed.")
        estimates["dm_shared"] = _wald_stats(m_hat_shared, alpha=alpha, null=null, name="DM (shared)")

    if want_dm and fit_separate:
        if m_hat_sep is None:
            raise RuntimeError("Internal error: m_hat_separate not computed.")
        estimates["dm_separate"] = _wald_stats(m_hat_sep, alpha=alpha, null=null, name="DM (separate)")

    if want_aipw and fit_shared:
        if alpha_hat is None or gamma_hat_shared is None or m_hat_shared is None:
            raise RuntimeError("Internal error: shared nuisances not computed for AIPW.")
        scores = alpha_hat * (Y1 - gamma_hat_shared) + m_hat_shared
        estimates["aipw_shared"] = _wald_stats(scores, alpha=alpha, null=null, name="AIPW (shared)")

    if want_aipw and fit_separate:
        if alpha_hat is None or gamma_hat_sep is None or m_hat_sep is None:
            raise RuntimeError("Internal error: separate nuisances not computed for AIPW.")
        scores = alpha_hat * (Y1 - gamma_hat_sep) + m_hat_sep
        estimates["aipw_separate"] = _wald_stats(scores, alpha=alpha, null=null, name="AIPW (separate)")

    return FunctionalEstimateResult(
        n=n,
        cross_fit=cross_fit,
        folds=folds,
        alpha=float(alpha),
        null=float(null),
        estimates=estimates,
        alpha_hat=alpha_hat,
        gamma_hat_shared=gamma_hat_shared,
        gamma_hat_separate=gamma_hat_sep,
        m_hat_shared=m_hat_shared,
        m_hat_separate=m_hat_sep,
    )


def grr_ate(
    *,
    X: Array,
    Y: Array,
    treatment_index: int = 0,
    treat_value_1: float = 1.0,
    treat_value_0: float = 0.0,
    **kwargs: Any,
) -> FunctionalEstimateResult:
    """Convenience wrapper for the ATE.

    This is equivalent to calling :func:`grr_functional` with
    ``m=ATEFunctional(...)``.

    Parameters
    ----------
    X:
        Regressor, typically ``X=[D,Z]``.
    Y:
        Outcome.
    treatment_index:
        Column index of the treatment variable in X.
    treat_value_1, treat_value_0:
        Counterfactual treatment values.
    kwargs:
        Passed through to :func:`grr_functional`.
    """

    m = ATEFunctional(
        treatment_index=int(treatment_index),
        treat_value_1=float(treat_value_1),
        treat_value_0=float(treat_value_0),
    )
    return grr_functional(X=X, Y=Y, m=m, **kwargs)


def grr_ame(
    *,
    X: Array,
    Y: Array,
    coordinate: int = 0,
    eps: float = 1e-4,
    **kwargs: Any,
) -> FunctionalEstimateResult:
    """Convenience wrapper for an average marginal effect (average derivative)."""

    m = AverageDerivativeFunctional(coordinate=int(coordinate), eps=float(eps))
    return grr_functional(X=X, Y=Y, m=m, **kwargs)


def grr_policy_effect(
    *,
    X: Array,
    Y: Array,
    policy_1: Callable[[Array], float],
    policy_0: Callable[[Array], float],
    treatment_index: int = 0,
    **kwargs: Any,
) -> FunctionalEstimateResult:
    """Convenience wrapper for an average policy effect."""

    m = PolicyEffectFunctional(policy_1=policy_1, policy_0=policy_0, treatment_index=int(treatment_index))
    return grr_functional(X=X, Y=Y, m=m, **kwargs)
