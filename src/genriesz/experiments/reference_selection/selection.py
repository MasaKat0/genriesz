"""Diagnostic bounds, the error-probability budget, and the selection rules.

Sections 5 and 11 of ``notebooks/experiments/REFERENCE_SELECTION_PLAN.md``.

The error budget separates two events that the earlier implementation merged.
Theorem ``uniform_selected_inference`` needs only ``|B_a| <= U_a``, and
``U_a = |D_a| + q_a + b_r`` does not involve the variance bound; the variance
bound appears only in the risk statement of Theorem ``nested_oracle``. Charging
both to the same ``delta`` overspent the budget by half.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .candidates import FIXED_BENCHMARKS, FoldLibrary
from .dgp import BoolArray, FloatArray


@dataclass(frozen=True)
class DeltaBudget:
    """Allocation of the miscoverage probability across the diagnostic events."""

    tau: float = 0.05
    delta: float = 0.01
    delta_variance: float = 0.01
    n_folds: int = 5

    def __post_init__(self) -> None:
        if not 0.0 < self.delta < self.tau < 1.0:
            raise ValueError("Require 0 < delta < tau < 1.")
        if not 0.0 < self.delta_variance < 1.0:
            raise ValueError("Require 0 < delta_variance < 1.")
        if self.n_folds < 1:
            raise ValueError("n_folds must be positive.")

    @property
    def fold_delta(self) -> float:
        """Bias-bound failure probability allotted to one fold."""

        return self.delta / self.n_folds

    @property
    def mean_radius_delta(self) -> float:
        """Failure probability of the simultaneous mean radius within a fold."""

        return self.delta / (2.0 * self.n_folds)

    @property
    def reference_delta(self) -> float:
        """Failure probability of the reference allowance within a fold."""

        return self.delta / (2.0 * self.n_folds)

    @property
    def ellipsoid_probability(self) -> float:
        """Coverage of each of the two coefficient ellipsoids forming ``b_r``."""

        return 1.0 - self.delta / (4.0 * self.n_folds)

    @property
    def variance_delta(self) -> float:
        """Failure probability of the variance upper bound within a fold.

        Charged to a separate budget because it does not enter the coverage
        statement.
        """

        return self.delta_variance / self.n_folds

    @property
    def normal_coverage(self) -> float:
        """Conditional coverage required of the evaluation-sample interval."""

        return 1.0 - (self.tau - self.delta)

    def total_bias_spend(self) -> float:
        """Probability charged to the bias-bound event across all folds.

        This is the quantity the earlier implementation got wrong. It also
        charged the variance bound here, which made the total one and a half
        times ``delta``; the variance bound belongs to the risk statement of
        Theorem ``nested_oracle``, not to the coverage statement, so it has its
        own budget and must not appear in this sum.
        """

        return self.n_folds * (self.mean_radius_delta + self.reference_delta)

    def bias_budget_is_exhausted(self) -> bool:
        """Whether the bias allocations spend exactly ``delta`` in total."""

        return bool(np.isclose(self.total_bias_spend(), self.delta))


def gaussian_multiplier_mean_radii(
    values: FloatArray,
    *,
    delta: float,
    draws: int,
    seed: int,
) -> FloatArray:
    """Simultaneous Gaussian-multiplier radii for the column means.

    A column whose values are constant receives radius zero, which matches the
    convention in the design document.
    """

    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    centered = values - values.mean(axis=0)
    sd = centered.std(axis=0, ddof=1)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    rng = np.random.default_rng(seed)
    multipliers = rng.normal(size=(draws, n))
    boot_means = (multipliers @ centered) / np.sqrt(n)
    standardized = np.abs(boot_means / sd_safe)
    critical = float(np.quantile(np.max(standardized, axis=1), 1.0 - delta))
    return critical * sd / np.sqrt(n)


def gaussian_multiplier_variance_upper(
    values: FloatArray,
    *,
    delta: float,
    draws: int,
    seed: int,
) -> FloatArray:
    """Simultaneous upper bounds for the column variances."""

    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    centered = values - values.mean(axis=0)
    squared = centered * centered
    variances = squared.mean(axis=0)
    squared_centered = squared - variances
    sd_squared = squared_centered.std(axis=0, ddof=1)
    sd_safe = np.where(sd_squared > 1e-12, sd_squared, 1.0)
    rng = np.random.default_rng(seed)
    multipliers = rng.normal(size=(draws, n))
    boot = (multipliers @ squared_centered) / np.sqrt(n)
    standardized = np.abs(boot / sd_safe)
    critical = float(np.quantile(np.max(standardized, axis=1), 1.0 - delta))
    return np.maximum(variances + critical * sd_squared / np.sqrt(n), 0.0)


def bias_upper_bound(
    absolute_drift: FloatArray, radius: FloatArray, allowance: float
) -> FloatArray:
    """``U_a = |D_a| + q_a + b_r``."""

    return np.asarray(absolute_drift, dtype=float) + np.asarray(radius, dtype=float) + allowance


def minimum_bias_upper_bound(bounds: dict[str, FloatArray]) -> FloatArray:
    """``U_a^R = min_r U_{a,r}`` of Proposition ``several_references``.

    A candidate that is inadmissible under every reference stays ``nan`` rather
    than raising the all-``nan`` slice warning.

    The proposition assumes ``|B_r| <= b_r`` for *every* reference in the set, so
    the minimum is only as trustworthy as the weakest member. Admitting one
    invalid reference degrades the combined bound, because the invalid reference
    tends to supply the smaller value and therefore wins the minimum. Section
    18.1 of the plan measures how badly.
    """

    if not bounds:
        raise ValueError("At least one reference bound is required.")
    stacked = np.vstack([np.asarray(value, dtype=float) for value in bounds.values()])
    out = np.full(stacked.shape[1], np.nan, dtype=float)
    usable = np.any(np.isfinite(stacked), axis=0)
    if np.any(usable):
        out[usable] = np.nanmin(stacked[:, usable], axis=0)
    return out


#: Every selection rule evaluated on the shared candidate library.
RULES: tuple[str, ...] = (
    "proposed",
    "proposed_min",
    "bregman_cv",
    "lsif_cv",
    "abs_drift",
    "score_var",
    *FIXED_BENCHMARKS,
    "oracle",
)

#: Rules whose selection depends on the reference estimator and its allowance.
REFERENCE_DEPENDENT_RULES: frozenset[str] = frozenset(
    {"proposed", "proposed_min", "abs_drift"}
)


@dataclass
class SelectionInputs:
    """Everything a selection rule may read for one fold."""

    admissible: BoolArray
    variance_upper: FloatArray
    bregman: FloatArray
    lsif: FloatArray
    audit_risk: FloatArray
    n_evaluation: int
    fixed_index: dict[str, int]
    absolute_drift: FloatArray | None = None
    bias_bound: FloatArray | None = None


def _masked_argmin(values: FloatArray, admissible: BoolArray) -> int | None:
    masked = np.where(admissible, np.asarray(values, dtype=float), np.nan)
    if not np.any(np.isfinite(masked)):
        return None
    return int(np.nanargmin(masked))


def apply_rule(rule: str, inputs: SelectionInputs) -> int | None:
    """Return the candidate index chosen by ``rule``, or ``None`` if it has none.

    Returning ``None`` is a recorded outcome, not an error: a fixed benchmark
    whose specification failed to fit produces no estimate on that fold, and the
    replication counts it in the denominator.
    """

    if rule in {"proposed", "proposed_min"}:
        if inputs.bias_bound is None:
            raise ValueError("The proposed rule requires a bias bound.")
        criterion = inputs.bias_bound**2 + inputs.variance_upper / inputs.n_evaluation
        return _masked_argmin(criterion, inputs.admissible)
    if rule == "abs_drift":
        if inputs.absolute_drift is None:
            raise ValueError("The absolute-drift rule requires held-out drift estimates.")
        return _masked_argmin(inputs.absolute_drift, inputs.admissible)
    if rule == "bregman_cv":
        return _masked_argmin(inputs.bregman, inputs.admissible)
    if rule == "lsif_cv":
        return _masked_argmin(inputs.lsif, inputs.admissible)
    if rule == "score_var":
        return _masked_argmin(inputs.variance_upper, inputs.admissible)
    if rule == "oracle":
        return _masked_argmin(inputs.audit_risk, inputs.admissible)
    if rule in inputs.fixed_index:
        index = inputs.fixed_index[rule]
        return index if inputs.admissible[index] else None
    raise ValueError(f"Unknown selection rule: {rule}")


def ranked_count(rule: str, inputs: SelectionInputs) -> int:
    """How many candidates a rule could actually rank on this fold.

    The held-out Bregman and squared criteria return ``nan`` when a fitted
    coefficient overflows, so ``bregman_cv`` and ``lsif_cv`` silently optimize
    over a smaller set than ``proposed`` or ``oracle``. Recording the count keeps
    the horse race honest about that difference.
    """

    criterion = _rule_criterion(rule, inputs)
    if criterion is None:
        return int(np.sum(inputs.admissible))
    return int(np.sum(inputs.admissible & np.isfinite(criterion)))


def _rule_criterion(rule: str, inputs: SelectionInputs) -> FloatArray | None:
    if rule in {"proposed", "proposed_min"} and inputs.bias_bound is not None:
        return inputs.bias_bound**2 + inputs.variance_upper / inputs.n_evaluation
    if rule == "abs_drift":
        return inputs.absolute_drift
    if rule == "bregman_cv":
        return inputs.bregman
    if rule == "lsif_cv":
        return inputs.lsif
    if rule == "score_var":
        return inputs.variance_upper
    if rule == "oracle":
        return inputs.audit_risk
    return None


def fixed_benchmark_indices(library: FoldLibrary) -> dict[str, int]:
    """Map each fixed-specification benchmark to its position in the library."""

    lookup = {spec.label: j for j, spec in enumerate(library.specs)}
    out: dict[str, int] = {}
    for name, spec in FIXED_BENCHMARKS.items():
        if spec.label not in lookup:
            raise ValueError(f"Fixed benchmark {name} ({spec.label}) is not in the candidate grid.")
        out[name] = lookup[spec.label]
    return out


def effective_sample_ratio(alpha: FloatArray) -> FloatArray:
    """Kish effective sample size of each column, divided by the sample size.

    ``(sum_i |a_i|)^2 / (n sum_i a_i^2)`` equals one for uniform weights and
    ``1/n`` when a single observation carries all of the weight. It is invariant
    to rescaling a column, so screening on it neither depends on the units of the
    target parameter nor interferes with the rescaling invariance of Section 4.

    Each column is divided by its peak absolute value before squaring, so the
    ratio survives weights near the floating-point range in either direction:
    uniform weights of ``1e308`` are ratio one, not ``inf / inf``, and uniform
    tiny weights do not underflow to zero. A column that is identically zero has
    no effective sample and receives zero; a column with a non-finite entry
    receives ``nan``.
    """

    alpha = np.asarray(alpha, dtype=float)
    n = alpha.shape[0]
    absolute = np.abs(alpha)
    finite = np.all(np.isfinite(alpha), axis=0)
    peak = np.max(np.where(np.isfinite(absolute), absolute, 0.0), axis=0)
    scaled = absolute / np.where(peak > 0.0, peak, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        numerator = scaled.sum(axis=0) ** 2
        denominator = n * (scaled * scaled).sum(axis=0)
        ratio = np.where(denominator > 0.0, numerator / denominator, 0.0)
    return np.where(finite, ratio, np.nan)


def candidate_scores(
    library: FoldLibrary,
    X: FloatArray,
    y: FloatArray,
    contrast: FloatArray,
    gamma: FloatArray,
    *,
    min_ess_ratio: float | None = None,
) -> tuple[FloatArray, BoolArray, FloatArray, FloatArray]:
    """Return scores, admissibility flags, maximum weights, and ESS ratios.

    ``min_ess_ratio`` is the pre-specified weight-concentration restriction that
    Section 4 of the manuscript allows. A candidate whose representer spreads its
    weight over less than that fraction of the diagnostic fold is inadmissible.
    ``None`` imposes no restriction, which is the behaviour the plan measured.

    The restriction screens candidates; it does not cap any weight, so the
    estimand is unchanged. Admissibility still requires a converged fit and
    finite scores, and the ratio is reported for every candidate whether or not
    the restriction is imposed.
    """

    if min_ess_ratio is not None and not 0.0 <= min_ess_ratio < 1.0:
        raise ValueError("Require 0 <= min_ess_ratio < 1.")
    alpha = library.alpha_matrix(X)
    residual = np.asarray(y, dtype=float) - np.asarray(gamma, dtype=float)
    scores = np.asarray(contrast, dtype=float)[:, None] + alpha * residual[:, None]
    finite_alpha = np.all(np.isfinite(alpha), axis=0)
    finite_scores = np.all(np.isfinite(scores), axis=0)
    admissible = np.asarray(library.success, dtype=bool) & finite_alpha & finite_scores
    ess_ratio = effective_sample_ratio(alpha)
    if min_ess_ratio is not None:
        admissible &= np.nan_to_num(ess_ratio, nan=0.0) >= min_ess_ratio
    scores[:, ~admissible] = np.nan
    # The maximum weight describes the fit, not the selection, so it is recorded
    # for every finite representer whether or not the candidate is admissible.
    # Tying it to admissibility would drop exactly the heavy candidates that
    # report.failure_table is meant to describe.
    with np.errstate(invalid="ignore"):
        max_weight = np.where(finite_alpha, np.max(np.abs(alpha), axis=0), np.nan)
    return scores, admissible, max_weight, ess_ratio


def theorem_upper_slack(
    bias_bound: FloatArray, audit_bias: FloatArray, radius: FloatArray, allowance: float
) -> NDArray[np.bool_]:
    """Check the upper half of Theorem ``data_dependent_bias``.

    The theorem states ``|B_a| <= U_a <= |B_a| + 2(q_a + b_r)``. Verifying only
    the lower half would not distinguish a valid bound from a vacuous one.
    """

    return np.asarray(bias_bound) <= np.abs(np.asarray(audit_bias)) + 2.0 * (
        np.asarray(radius) + allowance
    ) + 1e-12
