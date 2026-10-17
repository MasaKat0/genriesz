"""E-25: the ATT rare-control example and score moment conditions."""

from __future__ import annotations

import inspect
import itertools
import json

import numpy as np
import pandas as pd
from scipy import integrate, optimize, special, stats

from . import study
from .seeds import CONFIRMATORY_ENTROPY, Seeds, fold_ids

EXP = 25
B_VALUES = (0.2, 0.6)
N_VALUES = (500, 2000, 8000)
CELLS = [list(c) for c in itertools.product(B_VALUES, N_VALUES)]
R, K, THETA0 = 500, 2, 1.0
ARM_LABELS = ("UKL0", "SQ", "Logistic")
BOXES = {0.2: ((-1.0, 1.0), (0.05, 0.35)), 0.6: ((-1.0, 1.0), (0.4, 0.75))}


def draw(rng, n, b0):
    U = rng.uniform(size=n)
    if np.any(U <= 0):
        raise ValueError("U must be strictly positive")
    D = (rng.uniform(size=n) < special.expit(-b0 * np.log(U))).astype(float)
    Y = D + U + U**2 + rng.standard_normal(n)
    return U, D, Y


def objective(coef, F, D, arm):
    t = F @ coef
    if arm == "Logistic":
        return (float(np.mean(np.logaddexp(0, t) - D * t)), F.T @ (special.expit(t) - D) / len(D))
    r = np.exp(t)
    if arm == "UKL0":
        v, score = (1 - D) * r - D * t, (1 - D) * r - D
    elif arm == "SQ":
        v, score = (1 - D) * r**2 - 2 * D * r, 2 * ((1 - D) * r**2 - D * r)
    else:
        raise ValueError(arm)
    return float(v.mean()), F.T @ score / len(D)


def fit(U, D, b0, arm, *, options=None):
    F = np.column_stack([np.ones(len(U)), -np.log(U)])
    bounds = None if arm == "Logistic" else BOXES[b0]
    start = np.array([0.0, np.mean(BOXES[b0][1])])
    result = optimize.minimize(
        lambda c: objective(c, F, D, arm), start, jac=True, bounds=bounds, options=options
    )
    grad = np.asarray(result.jac)
    projected = grad.copy()
    boundary = False
    if bounds is not None:
        for j, (lo, hi) in enumerate(bounds):
            # Bound activity is a diagnostic, never a rejection rule.
            boundary |= bool(result.x[j] <= lo or result.x[j] >= hi)
            if result.x[j] <= lo:
                projected[j] = min(0.0, grad[j])
            if result.x[j] >= hi:
                projected[j] = max(0.0, grad[j])
    _, initial_gradient = objective(start, F, D, arm)
    finite = np.isfinite(result.x).all() and np.isfinite(result.fun) and np.isfinite(grad).all()
    info = {
        "status": "ok"
        if result.success and finite
        else ("nonfinite" if not finite else "optimizer_failure"),
        "library_success": bool(result.success),
        "library_status": int(result.status),
        "message": str(result.message),
        "coefficient": result.x.tolist(),
        "objective": float(result.fun),
        "iterations": int(result.nit),
        "boundary": boundary,
        "stationarity_scaled": float(
            np.max(np.abs(projected)) / max(1.0, np.max(np.abs(initial_gradient)))
        ),
    }
    return result.x, info


def treated_share(b0):
    return integrate.quad(lambda u: 1 / (1 + u**b0), 0.0, 1.0)[0]


def moment_facts(b0):
    return {
        "theta0": THETA0,
        "b0": b0,
        "representer_L2_finite": b0 < 1.0,
        "compatible_score_variance_finite": b0 < 1.0,
        "SQ_score_variance_finite": 3 * b0 < 1.0,
        "SQ_stable_index": (1 + b0) / (2 * b0) if 3 * b0 > 1 else 2.0,
        "pi": treated_share(b0),
    }


def representer_l2(coef, b0):
    # Only the control branch differs when the common population pi is used.
    a, b = coef
    pi = treated_share(b0)
    # Nonfinite L2 for an unrestricted logistic estimate is a result, not a fit failure.
    if 2 * b - b0 >= 1:
        return np.inf
    if a == 0 and b == b0:
        return 0.0

    rate = 1 + b0 - 2 * max(b, b0)
    magnitude = max(abs(np.expm1(a)), -np.expm1(-abs(b - b0) / rate))

    def integrand(v):
        t = v / rate
        difference = (
            np.expm1(a) - np.expm1(-(b - b0) * t)
            if b >= b0
            else np.expm1(a + (b - b0) * t)
        )
        return np.exp(-v) * (difference / magnitude) ** 2 / (1 + np.exp(-b0 * t))

    # u = exp(-v / rate) removes the endpoint singularity without subtracting moments.
    # Resolve the transition near zero even when the integrability boundary is close.
    epsrel = inspect.signature(integrate.quad).parameters["epsrel"].default
    transition_rate = min(b0, abs(b - b0)) if b != b0 else b0
    split = min(1.0, -np.log(epsrel) * rate / transition_rate)
    value = 0.0
    for lo, hi in ((0.0, split), (split, np.inf)):
        result = integrate.quad(integrand, lo, hi, full_output=True)
        if len(result) != 3:
            raise RuntimeError(f"representer L2 quadrature failed: {result[3]}")
        value += result[0]
    return float(magnitude * np.sqrt(value / rate) / pi)


def att_record(D, Y, odds, g, g1, g0):
    pi = D.mean()
    if not 0 < pi < 1:
        return study.failed("degenerate_functional")
    alpha = (D - (1 - D) * odds) / pi
    uncentered = D * (g1 - g0) / pi + alpha * (Y - g)
    theta = uncentered.mean()
    influence = uncentered - D * theta / pi
    if not np.isfinite(influence).all():
        return study.failed("nonfinite")
    return {
        "estimate": float(theta),
        "se": float(influence.std() / np.sqrt(len(D))),
        "status": "ok",
        "max_abs_alpha": float(np.max(np.abs(alpha))),
        "score_concentration": float(np.max(influence**2) / np.sum(influence**2)),
        "ess": float(np.sum(np.abs(alpha)) ** 2 / np.sum(alpha**2)),
    }


def replicate(task):
    cell, rep, entropy = task
    b0, n = CELLS[cell]
    seeds = Seeds(EXP, entropy)
    U, D, Y = draw(seeds.data(cell, rep), n, b0)
    folds = fold_ids(n, K, seeds.folds(cell, rep))
    F = np.column_stack([np.ones(n), U, U**2])
    # Each treatment arm has its own feasible quadratic OLS on outer training rows.
    g, g1, g0 = (np.empty(n) for _ in range(3))
    for k in range(K):
        tr, te = folds != k, folds == k
        mu = []
        for d in (0, 1):
            rows = tr & (D == d)
            if not rows.any():
                raise ValueError("empty outcome training arm")
            coef = np.linalg.lstsq(F[rows], Y[rows], rcond=None)[0]
            mu.append(F[te] @ coef)
        g0[te], g1[te] = mu
        g[te] = np.where(D[te] == 1, mu[1], mu[0])
    arms = {}
    for arm in ARM_LABELS:

        def run(arm=arm):
            coef, full = fit(U, D, b0, arm)
            details = {
                "full_fit": full,
                "slope_error": float(coef[1] - b0) if full["status"] == "ok" else np.nan,
                "representer_l2": representer_l2(coef, b0) if full["status"] == "ok" else np.nan,
            }
            fits, odds = [], np.empty(n)
            for k in range(K):
                tr, te = folds != k, folds == k
                beta, info = fit(U[tr], D[tr], b0, arm)
                fits.append({"fold": k, **info})
                if info["status"] != "ok":
                    return study.failed(info["status"], fits=fits, **details)
                odds[te] = np.exp(beta[0] - beta[1] * np.log(U[te]))
            return {**att_record(D, Y, odds, g, g1, g0), "fits": fits, **details}

        arms[arm] = study.recorded(run)
    return {"arms": arms, "folds": folds.tolist()}


def tasks_for(entropy=CONFIRMATORY_ENTROPY):
    import sys

    return study.tasks_for(sys.modules[__name__], entropy)


def quantile_interval(values, q, level=0.95):
    x = np.sort(np.asarray(values, float))
    if not len(x):
        return np.nan, np.nan
    alpha = (1 - level) / 2
    lo = int(stats.binom.ppf(alpha, len(x), q))
    hi = int(stats.binom.ppf(1 - alpha, len(x), q)) + 1
    return (float(x[lo - 1]) if lo >= 1 else -np.inf, float(x[hi - 1]) if hi <= len(x) else np.inf)


def summarise(raw):
    summary = study.summarise(raw, THETA0)
    for idx, row in summary.iterrows():
        frame = raw[(raw.cell == row.cell) & (raw.arm == row.arm)]
        b0, n = CELLS[int(row.cell)]
        errors = frame.slope_error.dropna().to_numpy()
        summary.loc[idx, "b0"] = b0
        summary.loc[idx, "coefficient_returned"] = len(errors)
        facts = moment_facts(b0)
        power = 1 - 1 / facts["SQ_stable_index"]
        reference_scale = (
            np.sqrt(n) if facts["SQ_score_variance_finite"] else n**power / np.log(n)
        )
        for name, values in (
            ("slope_error", errors),
            ("root_n_slope_error", errors * np.sqrt(n)),
            ("SQ_reference_scaled_error", errors * reference_scale),
        ):
            for q in (0.05, 0.5, 0.95):
                key = f"{name}_q{int(q * 100):02}"
                summary.loc[idx, key] = np.quantile(values, q) if len(values) else np.nan
                lo, hi = quantile_interval(values, q)
                summary.loc[idx, key + "_mc_lo"], summary.loc[idx, key + "_mc_hi"] = lo, hi
        summary.loc[idx, "median_abs_slope_error"] = (
            np.median(np.abs(errors)) if len(errors) else np.nan
        )
        lo, hi = quantile_interval(np.abs(errors), 0.5)
        summary.loc[idx, "median_abs_slope_error_mc_lo"] = lo
        summary.loc[idx, "median_abs_slope_error_mc_hi"] = hi
        summary.loc[idx, "coefficient_failure_rate"] = 1 - len(errors) / len(frame)
        summary.loc[idx, "root_n_median_abs_slope_error"] = summary.loc[
            idx, "median_abs_slope_error"
        ] * np.sqrt(n)
        summary.loc[idx, "SQ_reference_median_abs_slope_error"] = (
            summary.loc[idx, "median_abs_slope_error"] * reference_scale
        )
        fits = [json.loads(v) for v in frame.full_fit]
        summary.loc[idx, "optimizer_status_counts"] = json.dumps(
            pd.Series([f["status"] for f in fits]).value_counts().to_dict()
        )
        good = [f for f in fits if f["status"] == "ok"]
        summary.loc[idx, "boundary_frequency_conditional"] = (
            np.mean([f["boundary"] for f in good]) if good else np.nan
        )
        summary.loc[idx, "stationarity_scaled_median"] = np.median(
            [f["stationarity_scaled"] for f in fits]
        )
    return summary


def figure(summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for j, b0 in enumerate(B_VALUES):
        for arm in ARM_LABELS:
            s = summary[(summary.b0 == b0) & (summary.arm == arm)].sort_values("n")
            axes[0, j].plot(s.n, s.median_abs_slope_error, "o-", label=arm)
            axes[1, j].plot(s.n, s.root_n_slope_error_q50, "o-", label=arm)
            axes[1, j].fill_between(
                s.n, s.root_n_slope_error_q05, s.root_n_slope_error_q95, alpha=0.15
            )
        positive = summary.loc[summary.b0 == b0, "median_abs_slope_error"].gt(0).any()
        axes[0, j].set(
            xscale="log",
            yscale="log" if positive else "linear",
            xlim=(N_VALUES[0], N_VALUES[-1]),
            title=f"b0 = {b0}",
            ylabel="Median absolute slope error",
        )
        axes[1, j].set(
            xscale="log",
            xlim=(N_VALUES[0], N_VALUES[-1]),
            xlabel="n",
            ylabel="Root-n scaled slope error (central 90%)",
        )
        axes[0, j].legend()
    fig.tight_layout()
    return fig


def tables(summary):
    columns = [
        "b0",
        "n",
        "arm",
        "median_abs_slope_error",
        "slope_error_q05",
        "slope_error_q95",
        "representer_l2_median",
    ]
    diagnostics = [
        "b0",
        "n",
        "arm",
        "coefficient_returned",
        "boundary_frequency_conditional",
        "stationarity_scaled_median",
    ]
    normalization = [
        "b0",
        "n",
        "arm",
        "root_n_median_abs_slope_error",
        "SQ_reference_median_abs_slope_error",
    ]
    return {
        **study.tables(summary, EXP),
        "tab_E25_coefficients": summary[columns].to_latex(
            index=False, escape=True, float_format="%.4f"
        ),
        "tab_E25_diagnostics": summary[diagnostics].to_latex(
            index=False, escape=True, float_format="%.4g"
        ),
        "tab_E25_normalization": summary[normalization].to_latex(
            index=False, escape=True, float_format="%.4f"
        ),
    }
