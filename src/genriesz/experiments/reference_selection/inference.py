"""Confidence intervals for the selected estimator.

Section 12 of ``notebooks/experiments/REFERENCE_SELECTION_PLAN.md``.

Three of the four intervals have a theorem behind them. ``bias_aware_pooled``
does not: it applies the single-split bounded-normal-mean critical value to a
cross-fitted point estimate and a pooled standard error. It is computed so the
gap against the conservative cross-fitted interval can be measured, and it is
reported as unsupported rather than presented alongside the others.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from .dgp import FloatArray


def bias_aware_critical_value(t: float, coverage: float) -> float:
    """Smallest ``c`` with ``inf_{|u| <= t} P(|Z + u| <= c) >= coverage``.

    The infimum is attained at ``|u| = t``, so the defining equation is
    ``Phi(c - t) - Phi(-c - t) = coverage``.
    """

    if t < 0:
        raise ValueError("t must be nonnegative.")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must lie in (0, 1).")

    def equation(c: float) -> float:
        return stats.norm.cdf(c - t) - stats.norm.cdf(-c - t) - coverage

    upper = t + stats.norm.ppf((1.0 + coverage) / 2.0) + 10.0
    return float(optimize.brentq(equation, 0.0, upper))


@dataclass(frozen=True)
class Interval:
    """A confidence interval.

    Whether a theorem covers the combination of estimator and critical value is
    recorded once, in ``refsel.report.INTERVALS``, so the two cannot drift apart.
    """

    name: str
    low: float
    high: float

    @property
    def length(self) -> float:
        return self.high - self.low

    def covers(self, theta0: float) -> bool:
        return bool(self.low <= theta0 <= self.high)


def wald_interval(
    estimate: float, standard_error: float, *, tau: float, name: str = "wald"
) -> Interval:
    z = float(stats.norm.ppf(1.0 - tau / 2.0))
    half = z * standard_error
    return Interval(name, estimate - half, estimate + half)


def bias_aware_interval(
    estimate: float,
    standard_error: float,
    bias_bound: float,
    *,
    coverage: float,
    name: str = "bias_aware_split",
) -> Interval:
    """Bounded-normal-mean interval of Theorem ``uniform_selected_inference``.

    A non-positive or non-finite standard error degenerates to the point
    interval ``[estimate, estimate]``, matching :func:`wald_interval`. Returning
    an infinite interval instead would score the degenerate case as covering for
    the bias-aware interval and as non-covering for the Wald interval, biasing
    exactly the contrast that the uniform-coverage table reports.
    """

    if standard_error <= 0.0 or not np.isfinite(standard_error):
        return Interval(name, estimate, estimate)
    critical = bias_aware_critical_value(bias_bound / standard_error, coverage)
    half = critical * standard_error
    return Interval(name, estimate - half, estimate + half)


def conservative_crossfit_interval(
    estimate: float,
    weights: FloatArray,
    bias_bounds: FloatArray,
    standard_errors: FloatArray,
    *,
    tau: float,
    delta: float,
    n_folds: int,
    name: str = "conservative_cf",
) -> Interval:
    """Additive cross-fitted interval of equation ``crossfit_bias_aware_half_length``.

    ``sum_k w_k (U_k + z_{1 - (tau - delta) / (2K)} se_k)``. A union bound over
    folds gives the coverage statement, so this is the cross-fitted interval the
    manuscript actually proves.
    """

    z = float(stats.norm.ppf(1.0 - (tau - delta) / (2.0 * n_folds)))
    half = float(
        np.sum(np.asarray(weights) * (np.asarray(bias_bounds) + z * np.asarray(standard_errors)))
    )
    return Interval(name, estimate - half, estimate + half)


def pooled_bias_aware_interval(
    estimate: float,
    standard_error: float,
    total_bias_bound: float,
    *,
    coverage: float,
) -> Interval:
    """Single-split critical value applied to the pooled cross-fitted estimate.

    No theorem in the manuscript covers this combination. Reported separately
    and flagged as unsupported.
    """

    return bias_aware_interval(
        estimate,
        standard_error,
        total_bias_bound,
        coverage=coverage,
        name="bias_aware_pooled",
    )


def monte_carlo_standard_error(probability: FloatArray, replications: FloatArray) -> FloatArray:
    """Bernoulli Monte Carlo standard error of a coverage or frequency estimate."""

    p = np.asarray(probability, dtype=float)
    r = np.asarray(replications, dtype=float)
    return np.sqrt(np.maximum(p * (1.0 - p), 0.0) / np.maximum(r, 1.0))
