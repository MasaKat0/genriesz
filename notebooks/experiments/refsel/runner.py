"""Configuration, fold orchestration, and batched execution.

Sections 13 and 16 of ``notebooks/experiments/REFERENCE_SELECTION_PLAN.md``.

One repetition fits the ninety-candidate library once per fold. Every selection
rule, every reference, and every allowance scale is then evaluated on that same
library, so widening the comparison costs almost no additional computation.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from hashlib import blake2b
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .audit import (
    audit_from_values,
    audit_library,
    audit_true_reference,
    integration_sample,
    scenario_key,
)
from .candidates import (
    CandidateSpec,
    FoldLibrary,
    OutcomeFit,
    candidate_grid,
    fit_outcome,
)
from .dgp import (
    THETA0,
    Design,
    FloatArray,
    FoldRoles,
    GeneratedData,
    generate_data,
    make_fold_roles,
    stable_seed,
)
from .inference import (
    bias_aware_interval,
    conservative_crossfit_interval,
    pooled_bias_aware_interval,
    wald_interval,
)
from .reference import (
    ReferenceEstimator,
    TruthReference,
    fit_logistic_reference,
    fit_rff_reference,
    reference_check,
)
from .selection import (
    REFERENCE_DEPENDENT_RULES,
    RULES,
    DeltaBudget,
    SelectionInputs,
    apply_rule,
    bias_upper_bound,
    candidate_scores,
    fixed_benchmark_indices,
    gaussian_multiplier_mean_radii,
    gaussian_multiplier_variance_upper,
    minimum_bias_upper_bound,
    ranked_count,
    theorem_upper_slack,
)

Tier = Literal["smoke", "pilot", "publication"]

#: Replication counts per tier and grid. Only this number changes across tiers;
#: the candidate library, selection rules, designs, and seed construction do not.
#:
#: Grid B carries the uniform-coverage claim and gets the most replications
#: (coverage Monte Carlo standard error 0.0049 at the nominal level). Grid A is
#: supporting evidence on overlap, at 1,000 (0.0069). Grid C is the
#: high-dimensional check and is by far the most expensive per replication, so it
#: runs at 500 (0.0097), which is adequate for a secondary design.
#:
#: Measured single-core costs per replication: 7.6 s at n=1000, 12.7 s at
#: n=3000, and 57.5 s in the high-dimensional design. The publication tier is
#: therefore about 94 core-hours.
TIER_REPLICATIONS: dict[str, dict[str, int]] = {
    "smoke": {"A": 2, "B": 2, "C": 2},
    "pilot": {"A": 25, "B": 50, "C": 25},
    "publication": {"A": 1000, "B": 2000, "C": 500},
}

#: Tables written per batch.
TABLES = ("candidate", "selection", "repetition", "bound", "check", "oracle")

#: The single split whose interval is covered by Theorem ``uniform_selected_inference``.
SPLIT_FOLD = 0


@dataclass(frozen=True)
class Numerics:
    """Numerical settings shared by every scenario."""

    n_folds: int = 5
    max_iter: int = 1000
    tolerance: float = 1e-8
    gradient_tolerance: float = 1e-2
    #: Draws for the simultaneous multiplier bootstrap. The mean radius is a
    #: ``1 - delta/(2K) = 0.999`` quantile, so 2,000 draws put the critical value
    #: at roughly the third-largest observation: measured coefficient of
    #: variation 3.5 percent with a 0.7 percent downward bias against 50,000
    #: draws. Since the radius is the dominant term of the bias bound, that noise
    #: is not worth saving.
    multiplier_draws: int = 10000
    low_integration_size: int = 100_000
    high_integration_size: int = 50_000
    rff_features: int = 2000
    rff_lam: float = 1e-3
    allowance_scales: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
    reference_constants: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    base_seed: int = 20260720
    budget: DeltaBudget = field(default_factory=DeltaBudget)
    store_candidates: bool = True

    def __post_init__(self) -> None:
        if self.budget.n_folds != self.n_folds:
            raise ValueError(
                "The error budget is allocated per fold, so DeltaBudget.n_folds "
                f"({self.budget.n_folds}) must equal Numerics.n_folds ({self.n_folds}). "
                "Otherwise the bias event is charged at the wrong rate and the "
                "total silently exceeds delta."
            )

    def integration_size(self, design: Design) -> int:
        return self.low_integration_size if design == "low" else self.high_integration_size


@dataclass(frozen=True)
class Scenario:
    """One cell of the publication grid."""

    grid: str
    design: Design
    sample_size: int
    overlap_scale: float
    target_t: float
    hidden_scale: float

    @property
    def label(self) -> str:
        return (
            f"{self.grid}|{self.design}|n={self.sample_size}"
            f"|s={self.overlap_scale:g}|t={self.target_t:g}"
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """A full run: a set of scenarios at one tier."""

    name: str
    tier: str
    scenarios: tuple[Scenario, ...]
    numerics: Numerics = field(default_factory=Numerics)
    batch_size: int = 20
    max_workers: int | None = None

    def replications(self, scenario: Scenario) -> int:
        return TIER_REPLICATIONS[self.tier][scenario.grid]


def primary_reference(design: Design) -> str:
    """The reference used for inference when a rule does not pick one itself."""

    return "correct" if design == "low" else "rff"


def reference_names(design: Design) -> tuple[str, ...]:
    """Every reference a design defines, regardless of whether its fit succeeds."""

    return ("truth", "correct", "misspecified") if design == "low" else ("truth", "rff")


def expected_procedures(
    design: Design, numerics: Numerics
) -> tuple[tuple[str, str, float], ...]:
    """The full set of ``(rule, reference, scale)`` a scenario should produce.

    Enumerated from the configuration rather than from what a fold happened to
    return. Building the set from the successful folds would drop a procedure
    from the denominator exactly when it failed everywhere, which is the
    reporting convention the plan forbids.
    """

    keys: list[tuple[str, str, float]] = [
        (rule, "none", 1.0) for rule in RULES if rule not in REFERENCE_DEPENDENT_RULES
    ]
    keys.extend(("abs_drift", name, 1.0) for name in reference_names(design))
    keys.extend(("proposed", name, scale) for name, scale in reference_variants(design, numerics))
    if design == "low":
        keys.extend(("proposed_min", "min", scale) for scale in numerics.allowance_scales)
    return tuple(sorted(keys))


def reference_variants(design: Design, numerics: Numerics) -> tuple[tuple[str, float], ...]:
    """Return the ``(reference, scale)`` pairs evaluated in every fold.

    For the random-feature reference the scale plays the role of ``c_r`` in
    ``b_r = c_r / sqrt(n_eval)``, so the sensitivity sweep reuses the same
    mechanism as the allowance-scaling sweep.
    """

    variants: list[tuple[str, float]] = [("truth", 1.0)]
    if design == "low":
        for name in ("correct", "misspecified"):
            variants.extend((name, scale) for scale in numerics.allowance_scales)
    else:
        variants.extend(("rff", constant) for constant in numerics.reference_constants)
    return tuple(variants)


def _build_references(
    X_train: FloatArray,
    y_train: FloatArray,
    *,
    scenario: Scenario,
    numerics: Numerics,
    outcome: OutcomeFit,
    n_evaluation: int,
    seed: int,
) -> dict[str, ReferenceEstimator]:
    references: dict[str, ReferenceEstimator] = {
        "truth": TruthReference(
            design=scenario.design,
            overlap_scale=scenario.overlap_scale,
            hidden_scale=scenario.hidden_scale,
        )
    }
    if scenario.design == "low":
        for name in ("correct", "misspecified"):
            references[name] = fit_logistic_reference(
                X_train,
                y_train,
                name=name,  # type: ignore[arg-type]
                ellipsoid_probability=numerics.budget.ellipsoid_probability,
                max_iter=numerics.max_iter,
                tolerance=numerics.tolerance,
            )
    else:
        references["rff"] = fit_rff_reference(
            X_train,
            outcome=outcome,
            n_features=numerics.rff_features,
            lam=numerics.rff_lam,
            seed=seed,
            reference_constant=1.0,
            n_evaluation=n_evaluation,
        )
    return references


@dataclass
class FoldOutcome:
    """Per-rule results of one fold, keyed by ``(rule, reference, scale)``."""

    selection: dict[tuple[str, str, float], dict[str, object]]
    rows_candidate: list[dict[str, object]]
    rows_bound: list[dict[str, object]]
    rows_check: list[dict[str, object]]


def run_fold(
    *,
    scenario: Scenario,
    numerics: Numerics,
    data: GeneratedData,
    integration: GeneratedData,
    integration_key: tuple,
    roles: FoldRoles,
    specs: tuple[CandidateSpec, ...],
    scenario_seed: int,
    repetition: int,
    fold_index: int,
) -> FoldOutcome:
    """Fit, diagnose, select, and evaluate one rotation of the fold roles."""

    X_train, y_train = data.X[roles.training], data.outcomes()[roles.training]
    X_diag, y_diag = data.X[roles.diagnostic], data.outcomes()[roles.diagnostic]
    X_eval, y_eval = data.X[roles.evaluation], data.outcomes()[roles.evaluation]
    n_evaluation = len(roles.evaluation)
    budget = numerics.budget

    outcome = fit_outcome(
        X_train,
        y_train,
        design=scenario.design,
        seed=stable_seed(scenario_seed, repetition, fold_index, "outcome"),
    )
    library = FoldLibrary(
        X_train,
        specs,
        max_iter=numerics.max_iter,
        tolerance=numerics.tolerance,
        gradient_tolerance=numerics.gradient_tolerance,
    )
    references = _build_references(
        X_train,
        y_train,
        scenario=scenario,
        numerics=numerics,
        outcome=outcome,
        n_evaluation=n_evaluation,
        seed=stable_seed(scenario_seed, repetition, fold_index, "reference"),
    )

    scores_diag, admissible, max_weight = candidate_scores(
        library, X_diag, y_diag, outcome.contrast(X_diag), outcome.predict(X_diag)
    )
    valid = np.where(admissible)[0]
    if valid.size == 0:
        raise RuntimeError(
            "Every candidate failed on a diagnostic fold of scenario "
            f"{scenario.label!r}, repetition {repetition}, fold {fold_index}. "
            "This is a design problem, not an expected numerical status: at least "
            "the unpenalized squared candidate should fit."
        )

    n_candidates = len(library)
    variance_upper = np.full(n_candidates, np.nan)
    variance_upper[valid] = gaussian_multiplier_variance_upper(
        scores_diag[:, valid],
        delta=budget.variance_delta,
        draws=numerics.multiplier_draws,
        seed=stable_seed(scenario_seed, repetition, fold_index, "variance_bootstrap"),
    )

    # A reference whose fit failed carries no allowance guarantee. Exclude it
    # before anything reads it: RFFSquaredReference.alpha raises when its
    # conjugate-gradient solve did not converge, so merely filtering the outputs
    # would still abort the fold on the way there.
    usable = {name for name, reference in references.items() if reference.success}
    references = {name: reference for name, reference in references.items() if name in usable}

    drift: dict[str, FloatArray] = {}
    radius: dict[str, FloatArray] = {}
    for name, reference in references.items():
        differences = scores_diag[:, valid] - reference.score(X_diag, y_diag)[:, None]
        d_full = np.full(n_candidates, np.nan)
        q_full = np.full(n_candidates, np.nan)
        d_full[valid] = differences.mean(axis=0)
        q_full[valid] = gaussian_multiplier_mean_radii(
            differences,
            delta=budget.mean_radius_delta,
            draws=numerics.multiplier_draws,
            seed=stable_seed(scenario_seed, repetition, fold_index, name, "mean_bootstrap"),
        )
        drift[name] = d_full
        radius[name] = q_full

    audit = audit_library(
        library, outcome, integration, key=integration_key, n_evaluation=n_evaluation
    )
    truth_reference_bias = audit_true_reference(
        integration,
        design=scenario.design,
        overlap_scale=scenario.overlap_scale,
        hidden_scale=scenario.hidden_scale,
        n_evaluation=n_evaluation,
    )[0]
    # The realized drift of each reference, so that |B_r| <= b_r can be checked
    # rather than assumed. Section 17 of the plan records that the manuscript
    # claims this quantity is measured.
    reference_drift: dict[str, float] = {"truth": truth_reference_bias}
    for name, reference in references.items():
        if name == "truth":
            continue
        reference_drift[name] = audit_from_values(
            alpha_hat=reference.alpha(integration.X),
            alpha0=integration.alpha0,
            gamma_hat=reference.gamma(integration.X),
            gamma0=integration.gamma0,
            m_hat=reference.contrast(integration.X),
            n_evaluation=n_evaluation,
        )[0]

    bregman = library.heldout_bregman(X_diag)
    lsif = library.heldout_lsif(X_diag)

    scores_eval, admissible_eval, _ = candidate_scores(
        library, X_eval, y_eval, outcome.contrast(X_eval), outcome.predict(X_eval)
    )

    variants = tuple(
        (name, scale)
        for name, scale in reference_variants(scenario.design, numerics)
        if name in usable
    )
    bounds: dict[tuple[str, float], FloatArray] = {
        (name, scale): bias_upper_bound(
            np.abs(drift[name]), radius[name], references[name].allowance(scale)
        )
        for name, scale in variants
    }
    if scenario.design == "low" and {"correct", "misspecified"} <= usable:
        for scale in numerics.allowance_scales:
            bounds[("min", scale)] = minimum_bias_upper_bound(
                {name: bounds[(name, scale)] for name in ("correct", "misspecified")}
            )

    primary = primary_reference(scenario.design)
    inference_bound = bounds.get((primary, 1.0))
    if inference_bound is None:
        # Without the primary reference no rule has a bound to build an interval
        # from. Record the fold as producing nothing rather than silently falling
        # back to a different reference.
        return FoldOutcome(
            selection={},
            rows_candidate=[],
            rows_bound=[],
            rows_check=[],
        )
    selection: dict[tuple[str, str, float], dict[str, object]] = {}

    def record(
        rule: str,
        reference_name: str,
        scale: float,
        index: int | None,
        inputs: SelectionInputs,
    ) -> None:
        """Store one rule's outcome, always with a bound usable for inference.

        A rule that does not consult a reference still receives the primary
        reference's bound, because interval construction is separate from
        selection.
        """

        entry: dict[str, object] = {
            "rule": rule,
            "reference": reference_name,
            "allowance_scale": scale,
            "available": bool(index is not None and admissible_eval[index]),
            "n_ranked": ranked_count(rule, inputs),
            "reference_status": (
                references[reference_name].status if reference_name in references else "none"
            ),
        }
        if entry["available"]:
            assert index is not None
            column = scores_eval[:, index]
            selection_bound = bounds.get((reference_name, scale))
            entry.update(
                {
                    "candidate": library.specs[index].label,
                    "loss": library.specs[index].loss,
                    "dictionary": library.specs[index].dictionary,
                    "penalty_multiplier": library.specs[index].penalty_multiplier,
                    "theta": float(column.mean()),
                    "standard_error": float(column.std(ddof=1) / np.sqrt(column.size)),
                    "bias_bound": float(
                        selection_bound[index]
                        if selection_bound is not None
                        else inference_bound[index]
                    ),
                    "audit_bias": float(audit.bias[index]),
                    "audit_risk": float(audit.risk[index]),
                    "n_evaluation": n_evaluation,
                }
            )
        selection[(rule, reference_name, scale)] = entry

    base_inputs = SelectionInputs(
        admissible=admissible,
        variance_upper=variance_upper,
        bregman=bregman,
        lsif=lsif,
        audit_risk=audit.risk,
        n_evaluation=n_evaluation,
        fixed_index=fixed_benchmark_indices(library),
    )
    for rule in RULES:
        if rule not in REFERENCE_DEPENDENT_RULES:
            record(rule, "none", 1.0, apply_rule(rule, base_inputs), base_inputs)

    for name in sorted(usable):
        inputs = replace(base_inputs, absolute_drift=np.abs(drift[name]))
        record("abs_drift", name, 1.0, apply_rule("abs_drift", inputs), inputs)

    for key, bound in bounds.items():
        name, scale = key
        rule = "proposed_min" if name == "min" else "proposed"
        inputs = replace(base_inputs, bias_bound=bound)
        record(rule, name, scale, apply_rule(rule, inputs), inputs)

    rows_bound: list[dict[str, object]] = []
    for key, bound in bounds.items():
        name, scale = key
        mask = admissible & np.isfinite(audit.bias) & np.isfinite(bound)
        if not np.any(mask):
            continue
        has_radius = name in radius
        allowance = references[name].allowance(scale) if name in references else np.nan
        upper_coverage = (
            float(
                np.mean(
                    theorem_upper_slack(
                        bound[mask], audit.bias[mask], radius[name][mask], allowance
                    )
                )
            )
            if has_radius
            else np.nan
        )
        rows_bound.append(
            {
                "reference": name,
                "allowance_scale": scale,
                "fold": fold_index,
                "n_candidates": int(mask.sum()),
                "lower_coverage": float(np.mean(np.abs(audit.bias[mask]) <= bound[mask])),
                "upper_coverage": upper_coverage,
                "mean_bias_bound": float(np.mean(bound[mask])),
                "mean_absolute_bias": float(np.mean(np.abs(audit.bias[mask]))),
                "mean_radius": float(np.mean(radius[name][mask])) if has_radius else np.nan,
                "allowance": float(allowance),
                "truth_reference_bias": float(truth_reference_bias),
                "reference_status": (
                    references[name].status if name in references else "minimum"
                ),
                "reference_drift": float(reference_drift.get(name, np.nan)),
                # None for the minimum-bound row, which has no single reference
                # and therefore no single allowance to check.
                "allowance_covers_reference": (
                    bool(abs(reference_drift[name]) <= allowance)
                    if name in reference_drift and np.isfinite(allowance)
                    else None
                ),
            }
        )

    rows_check: list[dict[str, object]] = []
    estimated = [name for name in references if name != "truth" and name in usable]
    for i, first in enumerate(estimated):
        for second in estimated[i + 1 :]:
            check = reference_check(
                references[first],
                references[second],
                X_diag,
                y_diag,
                delta=budget.mean_radius_delta,
            )
            rows_check.append(
                {
                    "fold": fold_index,
                    "first": check.first,
                    "second": check.second,
                    "difference": check.difference,
                    "radius": check.radius,
                    "allowance_sum": check.allowance_sum,
                    "checkable": check.checkable,
                    "violated": check.violated,
                }
            )

    rows_candidate: list[dict[str, object]] = []
    if numerics.store_candidates:
        name = primary_reference(scenario.design)
        for j, spec in enumerate(library.specs):
            fit = library.fits[j]
            rows_candidate.append(
                {
                    "fold": fold_index,
                    "candidate": spec.label,
                    "loss": spec.loss,
                    "omega": spec.omega,
                    "dictionary": spec.dictionary,
                    "penalty_multiplier": spec.penalty_multiplier,
                    "fit_success": bool(fit.success),
                    "fit_status": fit.status,
                    "gradient_norm": fit.gradient_norm,
                    "kkt_residual": fit.kkt_residual,
                    "binding_rate": fit.binding_rate,
                    "admissible": bool(admissible[j]),
                    "max_abs_alpha": max_weight[j],
                    "relative_drift": drift[name][j],
                    "diagnostic_radius": radius[name][j],
                    "bias_bound": inference_bound[j],
                    "variance_upper": variance_upper[j],
                    "heldout_bregman": bregman[j],
                    "heldout_lsif": lsif[j],
                    "audit_bias": audit.bias[j],
                    "audit_variance": audit.variance[j],
                    "audit_risk": audit.risk[j],
                }
            )

    return FoldOutcome(
        selection=selection,
        rows_candidate=rows_candidate,
        rows_bound=rows_bound,
        rows_check=rows_check,
    )


def run_repetition(job: tuple[ExperimentConfig, Scenario, int]) -> dict[str, pd.DataFrame]:
    """Run one replication: five folds, every rule, every reference variant."""

    config, scenario, repetition = job
    numerics = config.numerics
    budget = numerics.budget
    scenario_seed = stable_seed(numerics.base_seed, scenario.label)

    data = generate_data(
        n=scenario.sample_size,
        design=scenario.design,
        overlap_scale=scenario.overlap_scale,
        hidden_scale=scenario.hidden_scale,
        seed=stable_seed(scenario_seed, repetition, "sample"),
    )
    size = numerics.integration_size(scenario.design)
    integration_seed = stable_seed(scenario_seed, "integration")
    shared = dict(
        design=scenario.design,
        overlap_scale=scenario.overlap_scale,
        hidden_scale=scenario.hidden_scale,
        size=size,
        seed=integration_seed,
    )
    integration = integration_sample(**shared)  # type: ignore[arg-type]
    key = scenario_key(**shared)  # type: ignore[arg-type]
    roles = make_fold_roles(
        scenario.sample_size,
        numerics.n_folds,
        stable_seed(scenario_seed, repetition, "folds"),
    )
    specs = candidate_grid()

    fold_outcomes = [
        run_fold(
            scenario=scenario,
            numerics=numerics,
            data=data,
            integration=integration,
            integration_key=key,
            roles=fold_roles,
            specs=specs,
            scenario_seed=scenario_seed,
            repetition=repetition,
            fold_index=k,
        )
        for k, fold_roles in enumerate(roles)
    ]

    identifiers: dict[str, object] = {
        "experiment": config.name,
        "tier": config.tier,
        "grid": scenario.grid,
        "design": scenario.design,
        "sample_size": scenario.sample_size,
        "overlap_scale": scenario.overlap_scale,
        "target_t": scenario.target_t,
        "hidden_scale": scenario.hidden_scale,
        "repetition": repetition,
    }
    weights = np.asarray(
        [len(role.evaluation) / scenario.sample_size for role in roles], dtype=float
    )
    # Iterate over the procedures the configuration defines, not over the ones
    # that happened to succeed, so a procedure that failed on every fold still
    # contributes an incomplete row to its own denominator.
    keys = expected_procedures(scenario.design, numerics)

    selection_rows: list[dict[str, object]] = []
    repetition_rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []

    for rule_key in keys:
        rule, reference_name, scale = rule_key
        entries = [outcome.selection.get(rule_key) for outcome in fold_outcomes]
        for k, entry in enumerate(entries):
            if entry is not None:
                selection_rows.append({**identifiers, "fold": k, **entry})

        split_entry = entries[SPLIT_FOLD]
        split_available = split_entry is not None and bool(split_entry["available"])
        base: dict[str, object] = {
            **identifiers,
            "rule": rule,
            "reference": reference_name,
            "allowance_scale": scale,
            "folds_available": sum(
                1 for entry in entries if entry is not None and entry["available"]
            ),
            # The single-split intervals verify Theorem ``uniform_selected_inference``
            # and depend only on fold ``SPLIT_FOLD``. Gating them on the
            # cross-fitted completeness flag would let an unrelated fold's failure
            # count as a coverage failure, which would depress the headline
            # uniform-coverage row for reasons that have nothing to do with the
            # theorem.
            "split_available": split_available,
        }
        if split_available:
            assert split_entry is not None
            split_theta = float(split_entry["theta"])
            split_se = float(split_entry["standard_error"])
            split_bound = float(split_entry["bias_bound"])
            split_intervals = [
                wald_interval(split_theta, split_se, tau=budget.tau, name="wald_split"),
                bias_aware_interval(
                    split_theta, split_se, split_bound, coverage=budget.normal_coverage
                ),
            ]
            base.update(
                {
                    "split_theta": split_theta,
                    "split_se": split_se,
                    "split_bias_bound": split_bound,
                    "split_audit_bias": float(split_entry["audit_bias"]),
                    "split_ranked": split_entry.get("n_ranked"),
                }
            )
            for interval in split_intervals:
                base[f"{interval.name}_low"] = interval.low
                base[f"{interval.name}_high"] = interval.high
                base[f"{interval.name}_covers"] = interval.covers(THETA0)
                base[f"{interval.name}_length"] = interval.length

        if any(entry is None or not entry["available"] for entry in entries):
            repetition_rows.append({**base, "complete": False})
            continue

        thetas = np.asarray([float(entry["theta"]) for entry in entries])  # type: ignore[index]
        ses = np.asarray([float(entry["standard_error"]) for entry in entries])  # type: ignore[index]
        bound_values = np.asarray([float(entry["bias_bound"]) for entry in entries])  # type: ignore[index]
        estimate = float(np.sum(weights * thetas))
        pooled_se = float(np.sqrt(np.sum(weights**2 * ses**2)))
        total_bound = float(np.sum(weights * bound_values))

        intervals = [
            wald_interval(estimate, pooled_se, tau=budget.tau, name="wald_cf"),
            conservative_crossfit_interval(
                estimate,
                weights,
                bound_values,
                ses,
                tau=budget.tau,
                delta=budget.delta,
                n_folds=numerics.n_folds,
            ),
            pooled_bias_aware_interval(
                estimate, pooled_se, total_bound, coverage=budget.normal_coverage
            ),
        ]

        row: dict[str, object] = {
            **base,
            "complete": True,
            "estimate": estimate,
            "bias": estimate - THETA0,
            "squared_error": (estimate - THETA0) ** 2,
            "pooled_se": pooled_se,
            "total_bias_bound": total_bound,
            "mean_audit_bias": float(
                np.mean([float(entry["audit_bias"]) for entry in entries])  # type: ignore[index]
            ),
            "mean_audit_risk": float(
                np.mean([float(entry["audit_risk"]) for entry in entries])  # type: ignore[index]
            ),
        }
        for interval in intervals:
            row[f"{interval.name}_low"] = interval.low
            row[f"{interval.name}_high"] = interval.high
            row[f"{interval.name}_covers"] = interval.covers(THETA0)
            row[f"{interval.name}_length"] = interval.length
        repetition_rows.append(row)

    for k, outcome in enumerate(fold_outcomes):
        oracle = outcome.selection.get(("oracle", "none", 1.0))
        if oracle is None or not oracle["available"]:
            continue
        oracle_risk = float(oracle["audit_risk"])
        for rule_key, entry in outcome.selection.items():
            if not entry["available"]:
                continue
            oracle_rows.append(
                {
                    **identifiers,
                    "fold": k,
                    "rule": rule_key[0],
                    "reference": rule_key[1],
                    "allowance_scale": rule_key[2],
                    "audit_risk": float(entry["audit_risk"]),
                    "oracle_audit_risk": oracle_risk,
                    "oracle_regret": float(entry["audit_risk"]) - oracle_risk,
                }
            )

    def frame(rows: list[dict[str, object]], prefix: bool = True) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([{**identifiers, **row} if prefix else row for row in rows])

    return {
        "candidate": frame(
            [row for outcome in fold_outcomes for row in outcome.rows_candidate]
        ),
        "selection": frame(selection_rows, prefix=False),
        "repetition": frame(repetition_rows, prefix=False),
        "bound": frame([row for outcome in fold_outcomes for row in outcome.rows_bound]),
        "check": frame([row for outcome in fold_outcomes for row in outcome.rows_check]),
        "oracle": frame(oracle_rows, prefix=False),
    }


def expand_jobs(config: ExperimentConfig) -> list[tuple[ExperimentConfig, Scenario, int]]:
    """Enumerate the replication jobs of a configuration."""

    return [
        (config, scenario, repetition)
        for scenario in config.scenarios
        for repetition in range(config.replications(scenario))
    ]


def batch_identities(config: ExperimentConfig) -> list[list[str]]:
    """The job identity of every batch, in order.

    Two configurations can share a configuration record and still place different
    jobs in the same batch, for instance when an earlier grid's replication count
    changes. Comparing the expanded job list catches that; comparing the record
    alone does not.
    """

    jobs = expand_jobs(config)
    size = config.batch_size
    return [
        [f"{scenario.label}#{repetition}" for _, scenario, repetition in jobs[i : i + size]]
        for i in range(0, len(jobs), size)
    ]


def configuration_digest(config: ExperimentConfig) -> str:
    """Stable digest of everything that determines the numbers in a run.

    Reusing a completed batch is only safe when the configuration that produced
    it is the one being asked for now. Skipping on file existence alone would let
    a changed calibration table, multiplier count, or scenario list read stale
    Parquet as if it were the new run's output.
    """

    payload = json.dumps(
        {
            "configuration": configuration_record(config),
            "batches": batch_identities(config),
        },
        sort_keys=True,
        default=str,
    )
    return blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def configuration_record(config: ExperimentConfig) -> dict[str, object]:
    """Serializable description of a configuration, written next to the results."""

    numerics = asdict(config.numerics)
    numerics["budget"] = asdict(config.numerics.budget)
    return {
        "name": config.name,
        "tier": config.tier,
        "batch_size": config.batch_size,
        "scenarios": [asdict(scenario) for scenario in config.scenarios],
        "numerics": numerics,
    }


def run_experiment(config: ExperimentConfig, output_dir: str | Path) -> None:
    """Run a configuration and write one Parquet file per table per batch.

    A batch is skipped when all of its files already exist, so an interrupted
    run resumes without changing the random-number allocation.

    With ``max_workers`` other than 1 this uses ``ProcessPoolExecutor``. On
    platforms whose default start method is ``spawn`` (macOS and Windows) each
    worker re-imports the calling module, so a *script* that calls this function
    must guard its entry point::

        if __name__ == "__main__":
            run_experiment(config, output_dir)

    Without the guard the workers re-execute the script body and the pool dies
    with ``BrokenProcessPool``. Calling from a notebook is fine, because the
    module being re-imported is the kernel rather than the cell.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    jobs = expand_jobs(config)
    batches = [jobs[i : i + config.batch_size] for i in range(0, len(jobs), config.batch_size)]
    digest = configuration_digest(config)
    manifest_path = output / "run_manifest.json"
    existing = _batch_files(output)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("digest") != digest:
            raise RuntimeError(
                f"{output} holds results from a different configuration "
                f"(digest {previous.get('digest')} on disk, {digest} requested). "
                "Resuming would mix runs. Write to a new directory, or delete the "
                "existing one if the old results are no longer wanted."
            )
    elif existing:
        raise RuntimeError(
            f"{output} holds {len(existing)} batch files but no run_manifest.json, "
            "so their provenance cannot be checked. Reusing them would report an "
            "unknown configuration's numbers as this run's. Delete them or write "
            "to a new directory."
        )
    _reject_batches_beyond(output, len(batches))
    manifest_path.write_text(
        json.dumps(
            {
                "digest": digest,
                "n_batches": len(batches),
                "batches": batch_identities(config),
                "configuration": configuration_record(config),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    for batch_index, batch in enumerate(batches):
        paths = {table: output / f"{table}_{batch_index:05d}.parquet" for table in TABLES}
        if all(path.exists() for path in paths.values()):
            continue
        if config.max_workers == 1:
            results = [run_repetition(job) for job in batch]
        else:
            with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
                results = list(executor.map(run_repetition, batch))
        for table, path in paths.items():
            collected = [result[table] for result in results if not result[table].empty]
            combined = pd.concat(collected, ignore_index=True) if collected else pd.DataFrame()
            combined.to_parquet(path, index=False)


def _batch_files(output: Path) -> list[Path]:
    return sorted(path for table in TABLES for path in output.glob(f"{table}_*.parquet"))


def _reject_batches_beyond(output: Path, n_batches: int) -> None:
    """Refuse batch files whose index lies outside the current run."""

    stale = sorted(
        path.name
        for path in _batch_files(output)
        if int(path.stem.rsplit("_", 1)[1]) >= n_batches
    )
    if stale:
        raise RuntimeError(
            f"{output} holds {len(stale)} batch files beyond the {n_batches} batches "
            f"of this configuration, starting with {stale[0]}. They would be read as "
            "part of this run."
        )


def load_experiment(output_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load every completed batch of an experiment directory.

    Refuses a directory without a manifest, or one holding batch files outside
    the manifest's range, either of which would mix another run's output into
    these tables.
    """

    output = Path(output_dir)
    manifest_path = output / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{output} has no run_manifest.json, so the provenance of its batch "
            "files cannot be checked. Re-run the experiment into this directory."
        )
    _reject_batches_beyond(
        output, int(json.loads(manifest_path.read_text(encoding="utf-8"))["n_batches"])
    )
    loaded: dict[str, pd.DataFrame] = {}
    for table in TABLES:
        frames = [pd.read_parquet(path) for path in sorted(output.glob(f"{table}_*.parquet"))]
        frames = [frame for frame in frames if not frame.empty]
        loaded[table] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if loaded["repetition"].empty:
        raise FileNotFoundError(f"No completed batches were found in {output}.")
    return loaded
