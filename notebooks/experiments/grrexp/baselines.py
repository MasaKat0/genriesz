"""Baseline ATE estimators of registration §1.4 and E-12 (shared by E-12, E-21, E-22, E-24).

Every estimator is cross-fitted with the registered fold ids and returns a dict
with ``estimate``, ``se`` and ``status``; a failure is a status with NaN values,
never an exception that is caught (registration §1.1).

- :func:`logit_aipw`: logistic MLE (no penalty, lbfgs, max_iter 10000, tol 1e-10)
  for the propensity score, OLS for the outcome on the treatment-interaction
  span, no clipping. A ConvergenceWarning or a fitted propensity outside
  ``(1e-12, 1 - 1e-12)`` is a failure.
- :func:`dml_gbm`: HistGradientBoosting for the propensity (features Z) and the
  outcome (features D, Z), no clipping, hyper-parameters chosen beforehand by
  :func:`tune_gbm` on pilot data.
- :func:`eb_dual_newton`: an independently written Newton solver of the
  entropy-balancing dual of the ATE (each arm reweighted to the full-sample mean
  of the regressors), used to check the EB row (= UKL with C = 0).
- :func:`run_recording_warnings`: runs one fit and records its warnings
  (§1.1); a ConvergenceWarning makes that fit a failure.
"""

from __future__ import annotations

import collections
import itertools
import warnings

import numpy as np

GBM_GRID = list(itertools.product((0.05, 0.1), (7, 15, 31), (100, 300)))  # lr, leaves, iters (E-12)
FAILED = {"estimate": float("nan"), "se": float("nan")}
#: Failure statuses of the baselines, beside the genriesz solver statuses (§1.3 B).
BASELINE_STATUSES = ("convergence_warning", "propensity_range")
#: The only warning filter §1.1 allows: Apple Accelerate's spurious matmul warnings.
ACCELERATE_MATMUL = r"(?:divide by zero|overflow|invalid value) encountered in matmul$"


def run_recording_warnings(fit):
    """``(result, warning_counts, converged)`` of the call ``fit()`` (§1.1).

    Every warning except Accelerate's spurious matmul ones is recorded, counted
    by category and message. ``converged`` is False when a scikit-learn or
    torch ConvergenceWarning was raised; the caller then counts the fit as
    failed. Recording is not catching an exception: an exception propagates.
    """
    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message=ACCELERATE_MATMUL, category=RuntimeWarning)
        result = fit()
    counts = collections.Counter(f"{w.category.__name__}: {w.message}" for w in caught)
    converged = not any(
        issubclass(w.category, ConvergenceWarning) or w.category.__name__ == "ConvergenceWarning"
        for w in caught
    )
    return result, dict(sorted(counts.items())), converged


def _aipw(D, Y, e, mu1, mu0):
    psi = mu1 - mu0 + D / e * (Y - mu1) - (1 - D) / (1 - e) * (Y - mu0)
    theta = float(np.mean(psi))
    return theta, float(np.std(psi - theta, ddof=0) / np.sqrt(len(psi)))


def _ols(F, y):
    coef, *_ = np.linalg.lstsq(F, y, rcond=None)
    return coef


def logit_aipw(D, Z_feats, Y, fold_ids):
    """``Z_feats``: the non-constant regressors (n x q). Outcome OLS on (1, Z_feats) per arm.

    Run it through :func:`run_recording_warnings`: a ConvergenceWarning of the
    logistic fit is a failure (``convergence_warning``).
    """
    from sklearn.linear_model import LogisticRegression

    n = len(Y)
    e = np.empty(n)
    mu1 = np.empty(n)
    mu0 = np.empty(n)
    F = np.column_stack([np.ones(n), Z_feats])
    for k in np.unique(fold_ids):
        tr, te = fold_ids != k, fold_ids == k
        clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000, tol=1e-10)
        clf.fit(Z_feats[tr], D[tr])
        e[te] = clf.predict_proba(Z_feats[te])[:, 1]
        for d, mu in ((1, mu1), (0, mu0)):
            rows = tr & (D == d)
            mu[te] = F[te] @ _ols(F[rows], Y[rows])
    if not np.all((e > 1e-12) & (e < 1 - 1e-12)):
        return {**FAILED, "status": "propensity_range"}
    theta, se = _aipw(D, Y, e, mu1, mu0)
    if not (np.isfinite(theta) and np.isfinite(se)):
        return {**FAILED, "status": "nonfinite"}
    return {"estimate": theta, "se": se, "status": "ok"}


def _gbm_classifier(params, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier

    lr, leaves, iters = params
    return HistGradientBoostingClassifier(
        learning_rate=lr, max_leaf_nodes=leaves, max_iter=iters, min_samples_leaf=20,
        early_stopping=False, random_state=seed,
    )  # fmt: skip


def _gbm_regressor(params, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor

    lr, leaves, iters = params
    return HistGradientBoostingRegressor(
        learning_rate=lr, max_leaf_nodes=leaves, max_iter=iters, min_samples_leaf=20,
        early_stopping=False, random_state=seed,
    )  # fmt: skip


def tune_gbm(D, Z, Y, cv_ids, seed):
    """5-fold CV on pilot data over the registered grid; ties go to the earlier grid point.

    Returns ``(propensity_params, outcome_params)`` chosen by log-loss and MSE.
    """
    from sklearn.metrics import log_loss

    best_p, best_o = (np.inf, None), (np.inf, None)
    DZ = np.column_stack([D, Z])
    for params in GBM_GRID:
        ll, mse = 0.0, 0.0
        for k in np.unique(cv_ids):
            tr, te = cv_ids != k, cv_ids == k
            p = _gbm_classifier(params, seed).fit(Z[tr], D[tr]).predict_proba(Z[te])[:, 1]
            ll += log_loss(D[te], p, labels=[0, 1]) * te.sum()
            mse += np.sum(
                (_gbm_regressor(params, seed).fit(DZ[tr], Y[tr]).predict(DZ[te]) - Y[te]) ** 2
            )
        if ll < best_p[0]:
            best_p = (ll, params)
        if mse < best_o[0]:
            best_o = (mse, params)
    return best_p[1], best_o[1]


def dml_gbm(D, Z, Y, fold_ids, prop_params, out_params, seed):
    n = len(Y)
    e, mu1, mu0 = np.empty(n), np.empty(n), np.empty(n)
    DZ = np.column_stack([D, Z])
    for k in np.unique(fold_ids):
        tr, te = fold_ids != k, fold_ids == k
        e[te] = _gbm_classifier(prop_params, seed).fit(Z[tr], D[tr]).predict_proba(Z[te])[:, 1]
        reg = _gbm_regressor(out_params, seed).fit(DZ[tr], Y[tr])
        mu1[te] = reg.predict(np.column_stack([np.ones(te.sum()), Z[te]]))
        mu0[te] = reg.predict(np.column_stack([np.zeros(te.sum()), Z[te]]))
    if not np.all((e > 0) & (e < 1)):
        return {**FAILED, "status": "propensity_range"}
    theta, se = _aipw(D, Y, e, mu1, mu0)
    if not (np.isfinite(theta) and np.isfinite(se)):
        return {**FAILED, "status": "nonfinite"}
    return {"estimate": theta, "se": se, "status": "ok"}


def eb_dual_newton(Phi_arm, target, n, tol=1e-8, max_iter=200):
    """Weights ``exp(Phi_arm c)`` with ``sum_i exp(Phi_i c) Phi_i = target`` (the EB dual).

    Damped Newton on the convex ``f(c) = sum exp(Phi c) - c' target``; returns the
    weights, or ``None`` if it does not converge. Convergence is the criterion of
    registration §1.3 C on the sample-mean scale of the GRR fit it checks:
    ``max_j |g_j| / n <= tol * max(1, max_j |target_j| / n)``.
    """
    c = np.zeros(Phi_arm.shape[1])

    def f(c_):
        return float(np.sum(np.exp(Phi_arm @ c_)) - c_ @ target)

    for _ in range(max_iter):
        w = np.exp(Phi_arm @ c)
        g = Phi_arm.T @ w - target
        if np.max(np.abs(g)) / n <= tol * max(1.0, np.max(np.abs(target)) / n):
            return w
        H = Phi_arm.T @ (Phi_arm * w[:, None])
        step = -np.linalg.solve(H, g)
        f0, gnorm, t = f(c), float(np.max(np.abs(g))), 1.0
        while t > 1e-12:
            cand = c + t * step
            f1 = f(cand)
            g1 = float(np.max(np.abs(Phi_arm.T @ np.exp(Phi_arm @ cand) - target)))
            rounding = abs(f1 - f0) <= 8 * np.finfo(float).eps * max(1.0, abs(f0))
            if f1 <= f0 + 1e-4 * t * (g @ step) or (rounding and g1 < gnorm):
                break
            t /= 2.0
        c = c + t * step
    return None
