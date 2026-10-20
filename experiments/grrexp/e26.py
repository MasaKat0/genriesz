"""Learned histogram regression and Gaussian Wald inference for ATT."""

from __future__ import annotations

import base64
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

from genriesz.att_histogram import (
    att_score_record,
    fit_att_odds,
    fit_cell_ols,
    predict_cell_ols,
)

from . import e26_population as population
from . import study
from .e25 import quantile_interval
from .seeds import CONFIRMATORY_ENTROPY, Seeds, fold_ids

EXP, R, K, THETA0 = 26, 1000, 2, 1.0
B_VALUES = (0.60, 5 / 7, 0.85)
J_VALUES = (3, 4, 5, 6)
CELLS = [[b, j, 2 * j**6] for b in B_VALUES for j in J_VALUES]
ARM_LABELS = ("UKL0", "SQ")
BOXES = {b: ((-1.0, 1.0), (b - 0.2, (1 + b) / 2 - 0.01)) for b in B_VALUES}
Z_975 = stats.norm.ppf(0.975)
SUMMARY_METRICS = (
    [f"{prefix}_{name}" for prefix in ("root_n", "stable", "max_scale")
     for name in ("error", "oracle", "linear_drift", "population_drift",
                  "taylor_remainder", "evaluation_remainder")]
    + [f"fold{k}_{name}" for k in range(K)
       for name in ("regression_l2", "representer_l2", "full_norm_product")]
    + [f"fold{k}_{label}_{key}" for k in range(K) for label in ("intercept", "slope")
       for key in ("coefficient_error", "scaled_coefficient_error",
                   "root_l_coefficient_error", "normalized_directional_error")]
    + ["variance_ratio", "variance_minus_oracle", "linear_drift", "population_drift",
       "taylor_remainder", "evaluation_remainder", "decomposition_identity_error",
       "max_abs_alpha", "ess", "score_concentration"]
)


def outcome_mean(t):
    t = np.asarray(t, float)
    return 10 * np.sum(t + t**2 / 4, axis=1)


def draw(rng, n, b0):
    t = rng.uniform(size=(n, 4))
    if np.any(t <= 0) or np.any(t >= 1):
        raise ValueError("the DGP requires coordinates strictly between zero and one")
    d = (rng.uniform(size=n) < 1 / (1 + t[:, 0] ** b0)).astype(float)
    epsilon = rng.standard_normal(n)
    y = d + outcome_mean(t) + epsilon
    return t, d, y, epsilon


def moment_facts():
    return [population.constants(b0, j) for b0, j, _ in CELLS]


def oracle_record(t, d, epsilon, b0, j):
    facts = population.constants(b0, j)
    empirical_pi = float(d.mean())
    if empirical_pi == 0:
        return {"oracle_status": "no_treated", "oracle_estimate": 0.0,
                "oracle_variance": 0.0, "oracle_se": 0.0,
                "oracle_known_variance_se": np.nan, "oracle_mean": np.nan,
                "oracle_variance_ratio": np.nan, "share_scale": np.nan,
                "true_variance": facts["variance"]}
    alpha0 = (d - (1 - d) * t[:, 0] ** (-b0)) / facts["pi"]
    score = alpha0 * epsilon
    oracle = float(score.mean())
    share_scale = facts["pi"] / empirical_pi
    influence = share_scale * (score - oracle * d / empirical_pi)
    variance = float(np.mean(influence**2))
    return {
        "oracle_status": "ok", "oracle_mean": oracle, "share_scale": share_scale,
        "oracle_estimate": float(THETA0 + share_scale * oracle),
        "oracle_variance": variance, "oracle_se": float(np.sqrt(variance / len(d))),
        "oracle_variance_ratio": variance / facts["variance"],
        "true_variance": facts["variance"],
        "oracle_known_variance_se": float(np.sqrt(facts["variance"] / len(d))),
    }


def decomposition(t, d, y, epsilon, folds, outcome_fits, betas, estimate, b0, j):
    """Evaluate the separate population and evaluation-fold score differences."""
    facts = population.constants(b0, j)
    pi, n = facts["pi"], len(d)
    alpha0 = (d - (1 - d) * t[:, 0] ** (-b0)) / pi
    oracle = float(np.mean(alpha0 * epsilon))
    share_scale = pi / float(d.mean())
    linear, exact, empirical, details = [], [], [], []
    f = outcome_mean(t)
    for k in range(K):
        te = folds == k
        ols, beta = outcome_fits[k], np.asarray(betas[k], float)
        delta_beta = beta - np.array([0.0, b0])
        diagnostics = population.regression_diagnostics(
            ols["coefficients"], ols["tau"], b0, j
        )
        projection = np.asarray(diagnostics["directional_error"])
        mu = population.population_drift(diagnostics["mean_cell_coefficient"], beta, b0, j)
        first_order = float(-delta_beta @ projection)
        g = predict_cell_ols(ols, t[te], d[te], j)
        delta = g - (d[te] + f[te])
        odds = np.exp(beta[0] - beta[1] * np.log(t[te, 0]))
        alpha = (d[te] - (1 - d[te]) * odds) / pi
        score_difference = (
            d[te] * (ols["tau"] - THETA0) / pi
            + (alpha - alpha0[te]) * epsilon[te] - alpha * delta
        )
        empirical.append(float(score_difference.mean() - mu))
        linear.append(first_order)
        exact.append(mu)
        alpha_error = population.representer_l2(beta, b0, pi, float(d.mean()))
        details.append({
            **diagnostics, "coefficient_error": delta_beta.tolist(),
            "scaled_coefficient_error": (delta_beta / facts["rho"]).tolist(),
            "root_l_coefficient_error": (delta_beta * np.sqrt(facts["L"])).tolist(),
            "normalized_directional_error": (projection / facts["directional_scale"]).tolist(),
            "linear_drift": first_order, "population_drift": mu,
            "representer_l2": alpha_error,
            "full_norm_product": float(np.sqrt(n) * alpha_error * diagnostics["regression_l2"]),
        })
    drift_linear, drift_population = float(np.mean(linear)), float(np.mean(exact))
    taylor = drift_population - drift_linear
    evaluation = float(np.mean(empirical))
    total = share_scale * (oracle + drift_linear + taylor + evaluation)
    result = {
        "linear_drift": drift_linear, "population_drift": drift_population,
        "taylor_remainder": taylor, "evaluation_remainder": evaluation,
        "oracle_mean": oracle, "share_scale": share_scale,
        "decomposition_total": total,
        "decomposition_identity_error": float(estimate - THETA0 - total),
        "fold_diagnostics": details,
    }
    root_scale, stable_scale = 1 / np.sqrt(n), facts["stable_scale"]
    for name, value in (
        ("error", estimate - THETA0), ("oracle", oracle),
        ("linear_drift", drift_linear), ("population_drift", drift_population),
        ("taylor_remainder", taylor), ("evaluation_remainder", evaluation),
    ):
        for prefix, scale in (("root_n", root_scale), ("stable", stable_scale),
                              ("max_scale", max(root_scale, stable_scale))):
            result[f"{prefix}_{name}"] = float(value / scale)
    for k, detail in enumerate(details):
        for name in ("regression_l2", "representer_l2", "full_norm_product"):
            result[f"fold{k}_{name}"] = detail[name]
        for component, label in enumerate(("intercept", "slope")):
            for key in ("coefficient_error", "scaled_coefficient_error",
                        "root_l_coefficient_error", "normalized_directional_error"):
                result[f"fold{k}_{label}_{key}"] = detail[key][component]
    scalars = [v for v in result.values() if isinstance(v, (float, int))]
    if not np.isfinite(scalars).all():
        raise ArithmeticError("nonfinite population or score decomposition diagnostic")
    return result


def _ols_record(fit):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in fit.items()}


def _fold_record(folds, cell, rep, entropy):
    values = np.asarray(folds, dtype=np.uint8)
    packed = np.packbits(values, bitorder="little")
    return {
        "encoding": "base64-packbits-little", "n": len(values),
        "counts": np.bincount(values, minlength=K).tolist(),
        "packed_base64": base64.b64encode(packed.tobytes()).decode("ascii"),
        "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "entropy": int(entropy), "seed_key": [EXP, int(cell), int(rep), 1],
    }


def replicate(task):
    cell, rep, entropy = task
    b0, j, n = CELLS[cell]
    seeds = Seeds(EXP, entropy)
    t, d, y, epsilon = draw(seeds.data(cell, rep), n, b0)
    folds = fold_ids(n, K, seeds.folds(cell, rep))
    outcome_fits = []
    g, g1, g0 = (np.empty(n) for _ in range(3))
    for k in range(K):
        tr, te = folds != k, folds == k
        fit = fit_cell_ols(t[tr], d[tr], y[tr], j)
        outcome_fits.append(fit)
        g[te] = predict_cell_ols(fit, t[te], d[te], j)
        g1[te] = predict_cell_ols(fit, t[te], np.ones(te.sum()), j)
        g0[te] = predict_cell_ols(fit, t[te], np.zeros(te.sum()), j)
    oracle = oracle_record(t, d, epsilon, b0, j)
    arms = {}
    for arm in ARM_LABELS:

        def run(arm=arm):
            fits, betas, odds = [], [], np.empty(n)
            for k in range(K):
                tr, te = folds != k, folds == k
                beta, info = fit_att_odds(t[tr, 0], d[tr], arm, BOXES[b0])
                fits.append({"fold": k, **info, "coefficient": beta.tolist()})
                betas.append(beta)
                if info["status"] == "ok":
                    odds[te] = np.exp(beta[0] - beta[1] * np.log(t[te, 0]))
            details = {"fits": fits, "outcome_fits": [_ols_record(f) for f in outcome_fits],
                       **oracle}
            failed_fits = [fit for fit in fits if fit["status"] != "ok"]
            if failed_fits:
                return study.failed(failed_fits[0]["status"], **details)
            record = {**att_score_record(d, y, odds, g, g1, g0), **details}
            if record["status"] != "ok":
                return record
            diagnostics = decomposition(
                t, d, y, epsilon, folds, outcome_fits, betas, record["estimate"], b0, j
            )
            record.update(diagnostics)
            record["variance_ratio"] = record["variance"] / oracle["true_variance"]
            record["variance_minus_oracle"] = record["variance"] - oracle["oracle_variance"]
            record["boundary_any"] = any(fit["boundary"] for fit in fits)
            record["boundary_both"] = all(fit["boundary"] for fit in fits)
            return record

        arms[arm] = study.recorded(run)
    return {"arms": arms, "folds": _fold_record(folds, cell, rep, entropy)}


def tasks_for(entropy=CONFIRMATORY_ENTROPY):
    return study.tasks_for(sys.modules[__name__], entropy)


def _binomial_interval(count, size):
    interval = stats.binomtest(int(count), int(size)).proportion_ci(method="exact")
    return float(interval.low), float(interval.high)


def _quantiles(row, name, values):
    values = np.asarray(values, float)
    row[name + "_count"] = len(values)
    row[name + "_nonfinite"] = int((~np.isfinite(values)).sum())
    values = values[np.isfinite(values)]
    for q in (0.05, 0.5, 0.95):
        key = f"{name}_q{int(100 * q):02}"
        row[key] = float(np.quantile(values, q)) if len(values) else np.nan
        row[key + "_mc_lo"], row[key + "_mc_hi"] = quantile_interval(values, q)


def summarise(raw):
    expected = {(c, arm) for c in range(len(CELLS)) for arm in ARM_LABELS}
    actual = set(zip(raw.cell, raw.arm, strict=True))
    if actual != expected:
        raise ValueError("E-26 aggregation requires all twelve conditions and both procedures")
    rows = []
    for cell in range(len(CELLS)):
        b0, j, n = CELLS[cell]
        for arm in ARM_LABELS:
            frame = raw[(raw.cell == cell) & raw.arm.eq(arm)]
            if len(frame) != R or set(frame.rep) != set(range(R)) or not frame.n.eq(n).all():
                raise ValueError("E-26 aggregation requires exactly replications 0 through 999")
            ok = frame.status.eq("ok") & np.isfinite(frame.estimate) & np.isfinite(frame.se)
            successful = frame.loc[ok]
            errors = successful.estimate.to_numpy() - THETA0
            covered = int(np.sum(np.abs(errors) <= Z_975 * successful.se.to_numpy()))
            returned = int(ok.sum())
            row = {
                "cell": cell, "b0": b0, "J": j, "L": j**6, "n": n, "arm": arm,
                "R": R, "returned": returned, "coverage": covered / R,
                "coverage_conditional": covered / returned if returned else np.nan,
                "failure_rate": 1 - returned / R,
                "status_counts": json.dumps(frame.status.value_counts().to_dict()),
                "bias_conditional": float(np.mean(errors)) if returned else np.nan,
                "rmse_conditional": float(np.sqrt(np.mean(errors**2))) if returned else np.nan,
                "length_conditional": float(2 * Z_975 * successful.se.mean())
                if returned else np.nan,
            }
            row["coverage_mc_lo"], row["coverage_mc_hi"] = _binomial_interval(covered, R)
            row["failure_mc_lo"], row["failure_mc_hi"] = _binomial_interval(R - returned, R)
            if returned:
                row["coverage_conditional_mc_lo"], row["coverage_conditional_mc_hi"] = (
                    _binomial_interval(covered, returned)
                )
            else:
                row["coverage_conditional_mc_lo"] = row["coverage_conditional_mc_hi"] = np.nan
            for label, se_column in (("oracle", "oracle_se"),
                                     ("oracle_known_variance", "oracle_known_variance_se")):
                valid = frame.oracle_status.eq("ok")
                oracle_covered = int(np.sum(
                    np.abs(frame.loc[valid, "oracle_estimate"] - THETA0)
                    <= Z_975 * frame.loc[valid, se_column]
                ))
                row[label + "_coverage"] = oracle_covered / R
                row[label + "_coverage_mc_lo"], row[label + "_coverage_mc_hi"] = (
                    _binomial_interval(oracle_covered, R)
                )
            for name in ("oracle_variance_ratio", "oracle_mean", "share_scale"):
                if name in frame:
                    _quantiles(row, name, frame.loc[frame.oracle_status.eq("ok"), name])
            for name in SUMMARY_METRICS:
                _quantiles(row, name, successful[name] if name in successful else [])
            parsed = [json.loads(value) if isinstance(value, str) else value
                      for value in frame.fits]
            fitted = [fit for fits in parsed for fit in fits]
            row["coefficient_fits"] = len(fitted)
            row["coefficient_status_counts"] = json.dumps(
                pd.Series([fit["status"] for fit in fitted]).value_counts().to_dict()
            )
            boundary = [any(fit["boundary"] for fit in fits) for fits in parsed]
            both = [all(fit["boundary"] for fit in fits) for fits in parsed]
            row["boundary_frequency"] = float(np.mean(boundary))
            row["boundary_both_frequency"] = float(np.mean(both))
            row["boundary_frequency_conditional"] = (
                float(np.mean(np.array(boundary)[ok])) if returned else np.nan
            )
            for fold in range(K):
                for coordinate, label in enumerate(("intercept", "slope")):
                    for side, bound in (("lower", BOXES[b0][coordinate][0]),
                                        ("upper", BOXES[b0][coordinate][1])):
                        row[f"fold{fold}_{label}_{side}_frequency"] = float(np.mean([
                            fits[fold]["coefficient"][coordinate] == bound for fits in parsed
                        ]))
                for name in ("gap", "stationarity_scaled", "objective"):
                    _quantiles(row, f"fold{fold}_{name}",
                               [fits[fold].get(name, np.nan) for fits in parsed])
            rows.append(row)
    return pd.DataFrame(rows)


def tables(summary):
    columns = ["b0", "n", "arm", "coverage", "oracle_coverage", "variance_ratio_q50"]
    diagnostics = ["b0", "n", "arm", "boundary_frequency_conditional",
                   "boundary_both_frequency", "oracle_variance_ratio_q50",
                   "fold0_full_norm_product_q50", "fold1_full_norm_product_q50"]
    normalization = ["b0", "n", "arm", "root_n_error_q05", "root_n_error_q50",
                     "root_n_error_q95", "stable_error_q05", "stable_error_q50",
                     "stable_error_q95"]
    headers = {
        "b0": "$b_0$", "n": "$N$", "arm": "Loss", "coverage": "Coverage",
        "oracle_coverage": "Oracle", "variance_ratio_q50": "Variance ratio",
        "boundary_frequency_conditional": "Boundary", "boundary_both_frequency": "Both folds",
        "oracle_variance_ratio_q50": "Oracle variance ratio",
        "fold0_full_norm_product_q50": "Norm product 1",
        "fold1_full_norm_product_q50": "Norm product 2",
        "root_n_error_q05": "$q^{G}_{.05}$", "root_n_error_q50": "$q^{G}_{.50}$",
        "root_n_error_q95": "$q^{G}_{.95}$", "stable_error_q05": "$q^{S}_{.05}$",
        "stable_error_q50": "$q^{S}_{.50}$", "stable_error_q95": "$q^{S}_{.95}$",
    }
    result = {}
    for name, cols in (("tab_E26", columns), ("tab_E26_diagnostics", diagnostics),
                       ("tab_E26_normalization", normalization)):
        frame = summary[cols].copy()
        frame["arm"] = frame.arm.replace({"UKL0": "UKL"})
        result[name] = frame.rename(columns=headers).to_latex(
            index=False, escape=False, float_format="%.4g"
        )
    return result


def figures(summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main, axes = plt.subplots(2, 3, figsize=(11, 6), sharex="col")
    detail, detail_axes = plt.subplots(3, 3, figsize=(11, 8), sharex="col")
    for column, b0 in enumerate(B_VALUES):
        for arm in ARM_LABELS:
            frame = summary[(summary.b0 == b0) & summary.arm.eq(arm)].sort_values("n")
            label = "UKL" if arm == "UKL0" else arm
            axes[0, column].plot(frame.n, frame.coverage, "o-", label=label)
            axes[0, column].fill_between(
                frame.n, frame.coverage_mc_lo, frame.coverage_mc_hi, alpha=0.12
            )
            axes[1, column].plot(frame.n, frame.variance_ratio_q50, "o-", label=label)
            axes[1, column].fill_between(
                frame.n, frame.variance_ratio_q05, frame.variance_ratio_q95, alpha=0.12
            )
            for row, field in enumerate(("root_n_error", "stable_error")):
                detail_axes[row, column].plot(frame.n, frame[field + "_q50"], "o-", label=label)
                detail_axes[row, column].fill_between(
                    frame.n, frame[field + "_q05"], frame[field + "_q95"], alpha=0.12
                )
        oracle = summary[(summary.b0 == b0) & summary.arm.eq(ARM_LABELS[0])].sort_values("n")
        axes[0, column].plot(oracle.n, oracle.oracle_coverage, "k--", label="Oracle")
        axes[1, column].plot(oracle.n, oracle.oracle_variance_ratio_q50, "k--", label="Oracle")
        sq = summary[(summary.b0 == b0) & summary.arm.eq("SQ")].sort_values("n")
        for field, label in (("linear_drift", "Linear drift"),
                             ("taylor_remainder", "Taylor remainder"),
                             ("evaluation_remainder", "Evaluation remainder")):
            detail_axes[2, column].plot(
                sq.n, sq[f"max_scale_{field}_q50"], "o-", label=label
            )
        b_label = r"\frac{5}{7}" if b0 == 5 / 7 else f"{b0:.2f}"
        axes[0, column].set(title=rf"$b_0={b_label}$", ylim=(0, 1.02), ylabel="Coverage")
        axes[0, column].axhline(0.95, color="gray", linewidth=0.8)
        axes[1, column].set(ylabel="Variance ratio", xlabel="$N$", xscale="log",
                            xlim=(2 * J_VALUES[0]**6, 2 * J_VALUES[-1]**6))
        axes[1, column].axhline(1, color="gray", linewidth=0.8)
        for row, label in enumerate((r"$\sqrt{N}(\widehat\theta-\tau)$",
                                     r"$(\widehat\theta-\tau)/(\rho_Lt_L)$", "SQ components")):
            detail_axes[row, column].set(
                ylabel=label, xscale="log", xlim=(2 * J_VALUES[0]**6, 2 * J_VALUES[-1]**6)
            )
            detail_axes[row, column].axhline(0, color="gray", linewidth=0.8)
        detail_axes[0, column].set_title(rf"$b_0={b_label}$")
        detail_axes[2, column].set_xlabel("$N$")
    axes[0, 0].legend(fontsize=8)
    detail_axes[0, 0].legend(fontsize=8)
    detail_axes[2, 0].legend(fontsize=7)
    main.tight_layout()
    detail.tight_layout()
    return {"fig_E26": main, "fig_E26_diagnostics": detail}


def figure(summary):
    import matplotlib.pyplot as plt

    output = figures(summary)
    plt.close(output["fig_E26_diagnostics"])
    return output["fig_E26"]


def additional_figures(summary):
    import matplotlib.pyplot as plt

    output = figures(summary)
    plt.close(output["fig_E26"])
    return {"fig_E26_diagnostics": output["fig_E26_diagnostics"]}
