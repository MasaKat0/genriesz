"""Decision rules of registration §1.8 and the predicted coverage of §1.9.

Each statistical hypothesis is an interval null ``H0: theta in [L, U]``. Its
p-value is calibrated on both sides of the interval,
``p = min{1, 2 min(P_L(stat <= observed), P_U(stat >= observed))}``; the
one-sided p-value of the observed side alone is never used. The binomial tests
(coverage, failure rate) are exact; the normal and bootstrap ones are
asymptotic in ``R_s``.

A family (all nulls of one hypothesis ID) is judged by Holm's step-down
procedure at the registered family level (0.01, or 0.05 for E-24).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .metrics import Z_95, fourth_moment_radicand, moment_terms

COVERAGE_TOLERANCE = 0.02
SD_RATIO_BOUNDS = (0.95, 1.05)
BIAS_TOLERANCE_FACTOR = 0.1
FAMILY_LEVEL = 0.01


def predicted_coverage(sigma_se: float, sigma_true: float, b: float, n: int) -> float:
    """``c(n) = Phi(1.96 r - tau) - Phi(-1.96 r - tau)``, ``r = sigma_SE/sigma_true``,
    ``tau = sqrt(n) b / sigma_true``."""
    _finite(sigma_se=sigma_se, sigma_true=sigma_true, b=b)
    if sigma_se <= 0 or sigma_true <= 0 or n < 1:
        raise ValueError("sigma_se, sigma_true and n must be positive")
    r = sigma_se / sigma_true
    tau = np.sqrt(n) * b / sigma_true
    return float(stats.norm.cdf(Z_95 * r - tau) - stats.norm.cdf(-Z_95 * r - tau))


def _finite(**values) -> None:
    bad = [k for k, v in values.items() if not np.isfinite(v)]
    if bad:
        raise ValueError(f"non-finite input: {bad}")


def _count(k, R) -> None:
    for name, v in (("k", k), ("R", R)):
        if not isinstance(v, (int, np.integer)) or isinstance(v, bool):
            raise ValueError(f"{name} must be an int, got {v!r}")
    if R < 1 or not 0 <= k <= R:
        raise ValueError(f"need R >= 1 and 0 <= k <= R, got k={k}, R={R}")


def _two_sided(p_low: float, p_high: float) -> float:
    return float(min(1.0, 2.0 * min(p_low, p_high)))


def coverage_pvalue(covered: int, R: int, c_n: float, tol: float = COVERAGE_TOLERANCE) -> float:
    """Exact test of ``H0: |c - c(n)| <= tol`` from ``k ~ Bin(R, c)``."""
    _count(covered, R)
    _finite(c_n=c_n, tol=tol)
    if not 0.0 <= c_n <= 1.0 or tol < 0.0:
        raise ValueError("c(n) must be in [0, 1] and tol >= 0")
    lo = max(0.0, c_n - tol)
    hi = min(1.0, c_n + tol)
    return _two_sided(stats.binom.cdf(covered, R, lo), stats.binom.sf(covered - 1, R, hi))


def interval_null_pvalue(estimate: float, se: float, lower: float, upper: float) -> float:
    """Normal-approximation test of ``H0: theta in [lower, upper]``."""
    if not (np.isfinite(estimate) and np.isfinite(se) and se > 0):
        raise ValueError("estimate must be finite and se positive")
    _finite(lower=lower, upper=upper)
    if lower > upper:
        raise ValueError("lower must not exceed upper")
    return _two_sided(
        stats.norm.cdf((estimate - lower) / se), stats.norm.sf((estimate - upper) / se)
    )


def bias_pvalue(x, theta0: float, b: float, sigma_true: float, n: int) -> float:
    """``H0: |mu - b| <= 0.1 sigma_true / sqrt(n)``, ``mu = E theta_hat - theta0``.

    ``x`` holds the successful estimates; ``se = SD/sqrt(R_s)``.
    """
    x = np.asarray(x, dtype=float)
    moment_terms(x)  # validates a finite sample of size >= 2
    _finite(theta0=theta0, b=b, sigma_true=sigma_true)
    if sigma_true <= 0 or n < 1:
        raise ValueError("sigma_true and n must be positive")
    delta = BIAS_TOLERANCE_FACTOR * sigma_true / np.sqrt(n)
    se = x.std(ddof=1) / np.sqrt(x.size)
    return interval_null_pvalue(x.mean() - theta0, se, b - delta, b + delta)


def sd_ratio_pvalue(x, sigma_true: float, n: int, bounds=SD_RATIO_BOUNDS) -> float:
    """``H0: |sqrt(n) SD / sigma_true - 1| <= 0.05`` with the fourth-moment delta se."""
    x = np.asarray(x, dtype=float)
    _finite(sigma_true=sigma_true)
    if sigma_true <= 0 or n < 1:
        raise ValueError("sigma_true and n must be positive")
    radicand, var = fourth_moment_radicand(x)
    se_sd = np.sqrt(radicand / (4.0 * var * x.size))
    ratio = np.sqrt(n) * np.sqrt(var) / sigma_true
    return interval_null_pvalue(ratio, np.sqrt(n) * se_sd / sigma_true, *bounds)


def variance_ratio_pvalue(x, n: int, v_star: float, delta: float) -> float:
    """E-19 Cex (i): ``H0: |Q - 1| <= delta``, ``Q = n Var(theta_hat)/(V* + 1/2)``."""
    x = np.asarray(x, dtype=float)
    _finite(v_star=v_star, delta=delta)
    if n < 1 or delta < 0 or v_star + 0.5 <= 0:
        raise ValueError("need n >= 1, delta >= 0 and V* + 1/2 > 0")
    radicand, var = fourth_moment_radicand(x)
    denom = v_star + 0.5
    se = n * np.sqrt(radicand / x.size) / denom
    return interval_null_pvalue(n * var / denom, se, 1.0 - delta, 1.0 + delta)


def failure_rate_pvalue(failures: int, R: int, f0: float) -> float:
    """One-directional exact test of ``H0: f <= f0``: ``P_{f0}(K >= k)``."""
    _count(failures, R)
    _finite(f0=f0)
    if not 0.0 <= f0 <= 1.0:
        raise ValueError("f0 must be in [0, 1]")
    return float(stats.binom.sf(failures - 1, R, f0))


def holm(pvalues, level: float = FAMILY_LEVEL) -> np.ndarray:
    """Holm's step-down rejections, in the order of ``pvalues``."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or p.size == 0 or not np.all((p >= 0) & (p <= 1)):
        raise ValueError("pvalues must be a non-empty 1-d array in [0, 1]")
    order = np.argsort(p, kind="stable")
    reject = np.zeros(p.size, dtype=bool)
    m = p.size
    for rank, idx in enumerate(order):
        if p[idx] <= level / (m - rank):
            reject[idx] = True
        else:
            break
    return reject


@dataclass(frozen=True)
class FamilyVerdict:
    hypothesis: str
    labels: tuple[str, ...]
    pvalues: tuple[float, ...]
    rejected: tuple[str, ...]
    level: float

    @property
    def negative(self) -> bool:
        return bool(self.rejected)

    def sentence(self) -> str:
        if self.negative:
            return f"{self.hypothesis}: negative result; rejected cells: {', '.join(self.rejected)}"
        return f"{self.hypothesis}: no departure from the prediction was detected"


def judge_family(
    hypothesis: str, tests: dict[str, float], level: float = FAMILY_LEVEL
) -> FamilyVerdict:
    """Holm over all nulls of one hypothesis ID; any rejection makes the family negative."""
    labels = tuple(tests)
    pvalues = tuple(float(tests[k]) for k in labels)
    rejected = holm(pvalues, level)
    return FamilyVerdict(
        hypothesis=hypothesis,
        labels=labels,
        pvalues=pvalues,
        rejected=tuple(lab for lab, r in zip(labels, rejected, strict=True) if r),
        level=level,
    )
