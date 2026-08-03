"""Tables for the manuscript.

Section 14 of ``notebooks/experiments/REFERENCE_SELECTION_PLAN.md``.

Two reporting conventions are enforced here rather than left to the notebook.

First, every coverage and frequency is unconditional: a replication in which a
rule produced no estimate counts in the denominator as a non-covering
replication. Reporting coverage conditional on success would flatter rules that
fail often, and the fixed BKL benchmark fails on every fold of the ATE designs.

Second, a claim of uniform validity is summarized by the *minimum* over the
data-generating family, not the average. ``uniform_coverage_table`` therefore
reports a ``min`` row alongside the sweep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .inference import monte_carlo_standard_error

#: Intervals produced for every rule, with the availability flag that governs
#: them and whether a theorem in the manuscript covers the combination of
#: estimator and critical value.
#:
#: The single-split intervals are governed by ``split_available`` rather than by
#: the cross-fitted ``complete`` flag. They depend only on the split fold, so a
#: failure in an unrelated fold must not be scored as a coverage failure.
INTERVALS: dict[str, tuple[str, bool]] = {
    "wald_split": ("split_available", True),
    "bias_aware_split": ("split_available", True),
    "wald_cf": ("complete", True),
    "conservative_cf": ("complete", True),
    "bias_aware_pooled": ("complete", False),
}

SCENARIO_KEYS = ("grid", "design", "sample_size", "overlap_scale", "target_t")

#: The keys that identify one selection rule. A rule evaluated against different
#: references is a different procedure and must not be averaged with the others.
RULE_KEYS = ("rule", "reference", "allowance_scale")


def _coverage_frame(repetitions: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Aggregate unconditional coverage and length for every interval."""

    frame = repetitions.copy()
    flags = {
        name: frame[name].fillna(False).astype(bool)
        if name in frame
        else pd.Series(False, index=frame.index)
        for name in ("complete", "split_available")
    }
    for interval, (flag, _) in INTERVALS.items():
        column = f"{interval}_covers"
        if column not in frame:
            continue
        available = flags[flag]
        frame[f"_{interval}_cov"] = available & frame[column].fillna(False).astype(bool)
        frame[f"_{interval}_ok"] = available
        length = frame.get(f"{interval}_length")
        if length is not None:
            frame[f"_{interval}_len"] = np.where(available, length, np.nan)
    grouped = frame.groupby(keys, dropna=False)
    out = grouped.agg(
        replications=("repetition", "nunique"),
        availability=("complete", "mean"),
        split_availability=("split_available", "mean"),
        bias=("bias", "mean"),
        rmse=("squared_error", lambda x: float(np.sqrt(np.nanmean(x)))),
    )
    for interval in INTERVALS:
        if f"_{interval}_cov" not in frame:
            continue
        out[f"{interval}_coverage"] = grouped[f"_{interval}_cov"].mean()
        out[f"{interval}_length"] = grouped[f"_{interval}_len"].mean()
    out = out.reset_index()
    for interval in INTERVALS:
        column = f"{interval}_coverage"
        if column in out:
            out[f"{column}_mcse"] = monte_carlo_standard_error(
                out[column].to_numpy(), out["replications"].to_numpy()
            )
    return out


def selection_rule_table(
    tables: dict[str, pd.DataFrame],
    *,
    reference: str | None = "correct",
    allowance_scale: float = 1.0,
) -> pd.DataFrame:
    """Experiment E1b: how every selection rule performs on the same library.

    The grouping keeps ``reference`` and ``allowance_scale`` separate from
    ``rule``. Pooling them would average the infeasible ``truth`` reference, the
    honest ``correct`` one, and the deliberately broken ``misspecified`` one into
    a single "proposed" row, and would leave the Monte Carlo standard error
    computed against a replication count that no longer matches the number of
    rows aggregated.

    ``reference`` selects which reference the reference-dependent rules are shown
    at; pass ``None`` to keep them all as separate rows.
    """

    repetitions = tables["repetition"]
    mask = np.isclose(repetitions["allowance_scale"], allowance_scale) | repetitions[
        "reference"
    ].eq("none")
    if reference is not None:
        mask &= repetitions["reference"].isin([reference, "none"])
    frame = repetitions.loc[mask]
    if frame.empty:
        return pd.DataFrame()
    keys = [*RULE_KEYS, *SCENARIO_KEYS]
    table = _coverage_frame(frame, keys)

    oracle = tables.get("oracle")
    if oracle is not None and not oracle.empty:
        regret = (
            oracle.groupby(keys, dropna=False)["oracle_regret"]
            .agg(["mean", "median"])
            .rename(columns={"mean": "mean_oracle_regret", "median": "median_oracle_regret"})
            .reset_index()
        )
        table = table.merge(regret, on=keys, how="left")

    selection = tables.get("selection")
    if selection is not None and not selection.empty and "n_ranked" in selection:
        ranked = (
            selection.groupby(keys, dropna=False)["n_ranked"]
            .mean()
            .rename("mean_candidates_ranked")
            .reset_index()
        )
        table = table.merge(ranked, on=keys, how="left")
    return table.sort_values([*SCENARIO_KEYS, "rule", "reference"])


def uniform_coverage_table(
    tables: dict[str, pd.DataFrame],
    *,
    rule: str = "proposed",
    reference: str = "correct",
    allowance_scale: float = 1.0,
    grid: str = "B",
) -> pd.DataFrame:
    """Experiment E4: coverage across the calibrated bias sweep, plus its minimum.

    The final rows, labelled ``min``, are the smallest coverage over the sweep
    for each sample size. A uniform-coverage claim stands or falls on those rows.
    """

    repetitions = tables["repetition"]
    frame = repetitions.loc[
        repetitions["rule"].eq(rule)
        & repetitions["reference"].eq(reference)
        & np.isclose(repetitions["allowance_scale"], allowance_scale)
        & repetitions["grid"].eq(grid)
    ]
    if frame.empty:
        return pd.DataFrame()
    return _coverage_frame(frame, ["sample_size", "target_t"]).sort_values(
        ["sample_size", "target_t"]
    )


def worst_case_coverage_table(
    tables: dict[str, pd.DataFrame],
    *,
    rule: str = "proposed",
    reference: str = "correct",
    allowance_scale: float = 1.0,
    grid: str = "B",
) -> pd.DataFrame:
    """Experiment E4 headline: the least favourable point of the bias family.

    One row per ``(sample_size, interval)`` giving the smallest coverage over the
    sweep, the ``t`` at which it occurs, and that same row's Monte Carlo standard
    error and interval length. Taking a column-wise minimum instead would pair a
    coverage from one scenario with a length from another, and would leave the
    headline row without a standard error.
    """

    sweep = uniform_coverage_table(
        tables,
        rule=rule,
        reference=reference,
        allowance_scale=allowance_scale,
        grid=grid,
    )
    if sweep.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for sample_size, frame in sweep.groupby("sample_size", dropna=False):
        for interval, (_, supported) in INTERVALS.items():
            column = f"{interval}_coverage"
            if column not in frame or frame[column].isna().all():
                continue
            worst = frame.loc[frame[column].idxmin()]
            rows.append(
                {
                    "sample_size": sample_size,
                    "interval": interval,
                    "theorem_supported": supported,
                    "min_coverage": float(worst[column]),
                    "min_coverage_mcse": float(worst.get(f"{column}_mcse", np.nan)),
                    "attained_at_t": worst["target_t"],
                    "length_at_that_t": float(worst.get(f"{interval}_length", np.nan)),
                    "replications": int(worst["replications"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["sample_size", "min_coverage"])


def bias_bound_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Experiment E2: validity and tightness of the candidate bias bound.

    ``lower_coverage`` is the share of candidates satisfying ``|B_a| <= U_a``.
    ``upper_coverage`` is the share satisfying the theorem's upper comparison
    ``U_a <= |B_a| + 2 (q_a + b_r)``; checking only the lower half cannot
    distinguish a valid bound from a vacuous one.
    """

    bounds = tables["bound"]
    if bounds.empty:
        return pd.DataFrame()
    grouped = bounds.groupby(
        ["reference", "allowance_scale", *SCENARIO_KEYS], dropna=False
    ).agg(
        folds=("fold", "size"),
        lower_coverage=("lower_coverage", "mean"),
        upper_coverage=("upper_coverage", "mean"),
        mean_bias_bound=("mean_bias_bound", "mean"),
        median_absolute_bias=("mean_absolute_bias", "median"),
        mean_radius=("mean_radius", "mean"),
        mean_allowance=("allowance", "mean"),
        truth_reference_bias=("truth_reference_bias", "mean"),
    )
    grouped["radius_share"] = grouped["mean_radius"] / grouped["mean_bias_bound"]
    grouped["allowance_share"] = grouped["mean_allowance"] / grouped["mean_bias_bound"]
    return grouped.reset_index()


def oracle_regret_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Experiment E3: realized regret against the theorem's predicted remainder.

    Regret is only defined on folds where the rule produced an estimate, so it is
    unavoidably conditional. ``folds_used`` and ``fold_share`` state that
    denominator explicitly, measured against the folds on which the oracle itself
    was available: a rule that fails often would otherwise show a flattering mean
    computed over its easy folds. The median is reported next to the mean because
    the distribution has a heavy right tail.
    """

    oracle = tables.get("oracle")
    if oracle is None or oracle.empty:
        return pd.DataFrame()
    keys = ["rule", "reference", "allowance_scale", *SCENARIO_KEYS]
    table = (
        oracle.groupby(keys, dropna=False)
        .agg(
            folds_used=("fold", "size"),
            mean_oracle_regret=("oracle_regret", "mean"),
            median_oracle_regret=("oracle_regret", "median"),
            p90_oracle_regret=("oracle_regret", lambda x: float(np.quantile(x, 0.9))),
            max_oracle_regret=("oracle_regret", "max"),
            mean_risk=("audit_risk", "mean"),
            mean_oracle_risk=("oracle_audit_risk", "mean"),
        )
        .assign(risk_ratio=lambda f: f["mean_risk"] / f["mean_oracle_risk"])
        .reset_index()
    )
    totals = (
        oracle.loc[oracle["rule"].eq("oracle")]
        .groupby(list(SCENARIO_KEYS), dropna=False)
        .size()
        .rename("folds_with_oracle")
        .reset_index()
    )
    table = table.merge(totals, on=list(SCENARIO_KEYS), how="left")
    table["fold_share"] = table["folds_used"] / table["folds_with_oracle"]
    return table


def reference_robustness_table(
    tables: dict[str, pd.DataFrame], *, rule: str = "proposed"
) -> pd.DataFrame:
    """Experiment E5: how coverage responds to the reference and its allowance.

    Scaling the honest allowance by ``rho`` quantifies how much of the coverage
    guarantee is carried by ``b_r`` rather than by the diagnostic comparison.
    """

    repetitions = tables["repetition"]
    frame = repetitions.loc[repetitions["rule"].eq(rule)]
    if frame.empty:
        return pd.DataFrame()
    return _coverage_frame(frame, ["reference", "allowance_scale", *SCENARIO_KEYS])


def reference_check_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Experiment E5: violation rate of the pairwise reference check.

    ``violation_rate`` is the fold-level rate among *decidable* folds. A fold
    whose difference, radius, or allowance was non-finite is neither a pass nor a
    violation; counting it as a pass would let a blown-up reference score be
    reported as a clean check. ``undecidable_rate`` reports how often that
    happened, over all folds.

    ``violation_mcse`` is clustered by replication. The folds of one replication
    share a sample and are therefore not independent Bernoulli trials, so the
    standard error comes from the between-replication spread of the per-
    replication rates rather than from a fold-level Bernoulli formula, which
    would understate it by roughly ``sqrt(K)`` when folds agree.
    """

    checks = tables.get("check")
    if checks is None or checks.empty:
        return pd.DataFrame()
    checks = checks.copy()
    if "checkable" not in checks:
        checks["checkable"] = True
    checks["checkable"] = checks["checkable"].eq(True)
    checks["_violated"] = checks["violated"].eq(True)

    keys = ["first", "second", *SCENARIO_KEYS]
    per_replication = (
        checks.assign(_decidable=checks["checkable"].astype(float))
        .assign(_fired=(checks["_violated"] & checks["checkable"]).astype(float))
        .groupby([*keys, "repetition"], dropna=False)[["_fired", "_decidable"]]
        .sum()
        .reset_index()
    )

    # Pooled fold-level rate with a replication-clustered standard error.
    #
    # The point estimate is the ratio of totals, not the average of the
    # per-replication rates: those differ whenever the number of decidable folds
    # varies across replications, and only the ratio of totals is the fold-level
    # rate the column claims to report.
    #
    # The standard error is the linearized cluster-robust one for a ratio, with
    # replications as clusters. Expanding sum((v - p d)^2) into
    # sum(v^2) - 2 p sum(v d) + p^2 sum(d^2) keeps the whole computation inside
    # one groupby aggregation, which also avoids GroupBy.apply and its
    # ``include_groups`` argument -- that keyword only exists from pandas 2.2,
    # while this project supports pandas 2.0.
    per_replication = per_replication.assign(
        _vv=lambda f: f["_fired"] ** 2,
        _vd=lambda f: f["_fired"] * f["_decidable"],
        _dd=lambda f: f["_decidable"] ** 2,
        _has=lambda f: (f["_decidable"] > 0).astype(float),
    )
    sums = per_replication.groupby(keys, dropna=False).agg(
        fired_total=("_fired", "sum"),
        decidable_folds=("_decidable", "sum"),
        sum_vv=("_vv", "sum"),
        sum_vd=("_vd", "sum"),
        sum_dd=("_dd", "sum"),
        clusters=("repetition", "nunique"),
        decidable_replications=("_has", "sum"),
    )
    total = sums["decidable_folds"].to_numpy(dtype=float)
    clusters = sums["clusters"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(total > 0, sums["fired_total"].to_numpy(dtype=float) / total, np.nan)
        residual_sq = (
            sums["sum_vv"].to_numpy(dtype=float)
            - 2.0 * rate * sums["sum_vd"].to_numpy(dtype=float)
            + rate**2 * sums["sum_dd"].to_numpy(dtype=float)
        )
        variance = np.where(
            (total > 0) & (clusters > 1),
            clusters / np.maximum(clusters - 1.0, 1.0) * residual_sq / total**2,
            np.nan,
        )
    grouped = pd.DataFrame(
        {
            "violation_rate": rate,
            "violation_mcse": np.sqrt(np.maximum(variance, 0.0)),
            "decidable_folds": total,
            "decidable_replications": sums["decidable_replications"].to_numpy(dtype=float),
        },
        index=sums.index,
    )
    detail = checks.groupby(keys, dropna=False).agg(
        folds=("fold", "size"),
        undecidable_rate=("checkable", lambda x: 1.0 - float(np.mean(x))),
        mean_absolute_difference=("difference", lambda x: float(np.nanmean(np.abs(x)))),
        mean_radius=("radius", "mean"),
        mean_allowance_sum=("allowance_sum", "mean"),
    )
    return detail.join(grouped, how="left").reset_index()


def interval_length_table(
    tables: dict[str, pd.DataFrame],
    *,
    rule: str = "proposed",
    reference: str = "correct",
    allowance_scale: float = 1.0,
) -> pd.DataFrame:
    """Experiment E6: the price of bias awareness when the bias is negligible."""

    repetitions = tables["repetition"]
    frame = repetitions.loc[
        repetitions["rule"].eq(rule)
        & repetitions["reference"].eq(reference)
        & np.isclose(repetitions["allowance_scale"], allowance_scale)
        & repetitions["complete"].fillna(False).astype(bool)
    ]
    if frame.empty:
        return pd.DataFrame()
    table = frame.groupby([*SCENARIO_KEYS], dropna=False).agg(
        replications=("repetition", "nunique"),
        wald_split_length=("wald_split_length", "mean"),
        bias_aware_split_length=("bias_aware_split_length", "mean"),
        conservative_cf_length=("conservative_cf_length", "mean"),
        mean_bias_bound=("split_bias_bound", "mean"),
        mean_split_se=("split_se", "mean"),
    )
    table["bound_to_se"] = table["mean_bias_bound"] / table["mean_split_se"]
    table["bias_aware_over_wald"] = (
        table["bias_aware_split_length"] / table["wald_split_length"]
    )
    return table.reset_index()


def failure_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Numerical failure by loss and dictionary, with the reason recorded."""

    candidates = tables.get("candidate")
    if candidates is None or candidates.empty:
        return pd.DataFrame()
    aggregations = dict(
        fits=("fit_success", "size"),
        failure_rate=("fit_success", lambda x: 1.0 - float(np.mean(x))),
        inadmissible_rate=("admissible", lambda x: 1.0 - float(np.mean(x))),
        median_max_abs_alpha=("max_abs_alpha", "median"),
        median_ess_ratio=("ess_ratio", "median"),
    )
    # Results written before manifest schema 3 have no ess_ratio column; they
    # remain readable, so the report degrades to the columns that exist rather
    # than raising on a directory the loader accepted.
    if "ess_ratio" not in candidates.columns:
        del aggregations["median_ess_ratio"]
    grouped = candidates.groupby(
        ["loss", "dictionary", "design", "sample_size", "overlap_scale"], dropna=False
    ).agg(**aggregations)
    status = (
        candidates.assign(count=1)
        .pivot_table(
            index=["loss", "dictionary", "design", "sample_size", "overlap_scale"],
            columns="fit_status",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .add_prefix("status_")
    )
    return grouped.join(status, how="left").reset_index()


def selection_frequency_table(
    tables: dict[str, pd.DataFrame], *, rule: str = "proposed"
) -> pd.DataFrame:
    """Which specifications a rule actually picks, with Monte Carlo error."""

    selection = tables.get("selection")
    if selection is None or selection.empty:
        return pd.DataFrame()
    frame = selection.loc[selection["rule"].eq(rule) & selection["available"]]
    if frame.empty:
        return pd.DataFrame()
    totals = frame.groupby([*SCENARIO_KEYS], dropna=False).size().rename("folds")
    counts = (
        frame.groupby([*SCENARIO_KEYS, "loss", "dictionary"], dropna=False)
        .size()
        .rename("selected")
        .reset_index()
        .merge(totals.reset_index(), on=list(SCENARIO_KEYS), how="left")
    )
    counts["frequency"] = counts["selected"] / counts["folds"]
    counts["frequency_mcse"] = monte_carlo_standard_error(
        counts["frequency"].to_numpy(), counts["folds"].to_numpy()
    )
    return counts.sort_values([*SCENARIO_KEYS, "frequency"], ascending=False)
