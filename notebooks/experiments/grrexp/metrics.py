"""Metrics of registration §1.5 (P- and B-type experiments).

``R`` is the registered number of replications and ``R_s`` the number whose
status is ``ok``. A failed replication carries ``ok=False``; its estimate and
SE may be NaN. Every function takes the full length-``R`` arrays so that the
unconditional coverage can count failures in the denominator.

Nominal 95% Wald intervals use the registered critical value ``Z_95 = 1.96``,
the same constant as the predicted coverage ``c(n)`` of §1.9.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

Z_95 = 1.96


def _as_arrays(estimate, ok):
    estimate = np.asarray(estimate, dtype=float)
    ok = np.asarray(ok, dtype=bool)
    if estimate.shape != ok.shape or estimate.ndim != 1:
        raise ValueError("estimate and ok must be 1-d arrays of the same length R")
    if not np.all(np.isfinite(estimate[ok])):
        raise ValueError("a replication marked ok has a non-finite estimate")
    return estimate, ok


def _successful(estimate, ok) -> np.ndarray:
    est, ok = _as_arrays(estimate, ok)
    x = est[ok]
    if x.size < 2:
        raise ValueError(f"need at least two successful replications, got {x.size}")
    return x


def clopper_pearson_upper(k: int, n: int, level: float = 0.95) -> float:
    """One-sided Clopper–Pearson upper bound for a binomial proportion."""
    if not 0 <= k <= n or n < 1:
        raise ValueError(f"need 0 <= k <= n and n >= 1, got k={k}, n={n}")
    if k == n:
        return 1.0
    return float(stats.beta.ppf(level, k + 1, n - k))


def failure_rate(ok, status=None) -> dict:
    """``1 - R_s/R``, the counts by status, and the CP upper bound when no failure."""
    ok = np.asarray(ok, dtype=bool)
    r = ok.size
    k = int(r - ok.sum())
    out = {"R": r, "R_s": int(ok.sum()), "failure_rate": k / r}
    if status is not None:
        values, counts = np.unique(np.asarray(status, dtype=str), return_counts=True)
        out["status_counts"] = {str(v): int(c) for v, c in zip(values, counts, strict=True)}
    if k == 0:
        out["failure_rate_cp_upper"] = clopper_pearson_upper(0, r)
    return out


def moment_terms(x) -> tuple[float, float]:
    """Sample variance (ddof=1) and the plug-in fourth central moment."""
    x = np.asarray(x, dtype=float)
    centred = x - x.mean()
    return float(x.var(ddof=1)), float(np.mean(centred**4))


def sd_mcse(x) -> float:
    """Delta-method MCSE of the sample SD: ``{(m4 - s^4)/(4 s^2 R_s)}^{1/2}``."""
    x = np.asarray(x, dtype=float)
    var, m4 = moment_terms(x)
    return float(np.sqrt(max(m4 - var**2, 0.0) / (4.0 * var * x.size)))


def bias(estimate, ok, theta0: float) -> dict:
    x = _successful(estimate, ok)
    return {"bias": float(x.mean() - theta0), "bias_mcse": float(x.std(ddof=1) / np.sqrt(x.size))}


def root_n_sd(estimate, ok, n: int) -> dict:
    x = _successful(estimate, ok)
    return {
        "root_n_sd": float(np.sqrt(n) * x.std(ddof=1)),
        "root_n_sd_mcse": float(np.sqrt(n) * sd_mcse(x)),
    }


def rmse(estimate, ok, theta0: float) -> float:
    x = _successful(estimate, ok)
    return float(np.sqrt(np.mean((x - theta0) ** 2)))


def se_ratio(estimate, se, ok) -> float:
    """Mean SE over MC SD, both over successful replications."""
    est, ok = _as_arrays(estimate, ok)
    se = np.asarray(se, dtype=float)
    if not np.all(np.isfinite(se[ok])):
        raise ValueError("a replication marked ok has a non-finite SE")
    return float(se[ok].mean() / est[ok].std(ddof=1))


def wald_interval(estimate, se) -> tuple[np.ndarray, np.ndarray]:
    estimate = np.asarray(estimate, dtype=float)
    se = np.asarray(se, dtype=float)
    return estimate - Z_95 * se, estimate + Z_95 * se


def coverage(estimate, se, ok, theta0: float) -> dict:
    """Unconditional coverage (failures count as non-covering) and the conditional one."""
    est, ok = _as_arrays(estimate, ok)
    lo, hi = wald_interval(est, se)
    covered = ok & (lo <= theta0) & (theta0 <= hi)
    k = int(covered.sum())
    r_s = int(ok.sum())
    return {
        "covered": k,
        "coverage": k / ok.size,
        "coverage_conditional": (k / r_s) if r_s else float("nan"),
    }


def ci_length(se, ok) -> dict:
    se = np.asarray(se, dtype=float)
    ok = np.asarray(ok, dtype=bool)
    length = 2.0 * Z_95 * se[ok]
    return {"ci_length_mean": float(length.mean()), "ci_length_median": float(np.median(length))}


def max_weight_summary(max_abs_alpha, ok) -> dict:
    """Median and 95% point over replications of ``max_i |alpha_hat(X_i)|``."""
    m = np.asarray(max_abs_alpha, dtype=float)[np.asarray(ok, dtype=bool)]
    return {"max_weight_median": float(np.median(m)), "max_weight_q95": float(np.quantile(m, 0.95))}


def ess(alpha) -> float:
    """``(sum |alpha_i|)^2 / sum alpha_i^2`` for one fitted representer."""
    a = np.asarray(alpha, dtype=float)
    return float(np.abs(a).sum() ** 2 / np.sum(a**2))


def imbalance(alpha, m_phi, phi) -> float:
    """``max_j |Delta_hat(alpha, phi_j)|`` with ``Delta_hat = P_n{alpha phi_j - m(W, phi_j)}``.

    ``phi`` is the ``n x p`` matrix ``phi_j(X_i)`` and ``m_phi`` the ``n x p`` matrix
    ``m(W_i, phi_j)``.
    """
    alpha = np.asarray(alpha, dtype=float)
    delta = np.mean(alpha[:, None] * np.asarray(phi) - np.asarray(m_phi), axis=0)
    return float(np.max(np.abs(delta)))


def summarise(estimate, se, ok, theta0: float, n: int, status=None) -> dict:
    """The registered columns that every P/B main table carries side by side."""
    out = failure_rate(ok, status)
    out.update(bias(estimate, ok, theta0))
    out.update(root_n_sd(estimate, ok, n))
    out["rmse"] = rmse(estimate, ok, theta0)
    out["se_ratio"] = se_ratio(estimate, se, ok)
    out.update(coverage(estimate, se, ok, theta0))
    out.update(ci_length(se, ok))
    return out
