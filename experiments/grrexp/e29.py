"""E-29: Nonparametric density ratios and validation-selected increasing series."""

from __future__ import annotations

import itertools
import math
from functools import cache

import numpy as np
import pandas as pd
from scipy import integrate, stats

from genriesz.partition_regression import (
    fit_ratio,
    fit_regression,
    local_basis,
    predict,
    predict_ratio,
    regression_moments,
    select_penalty,
)

from .e27 import interior_uniform
from .seeds import CONFIRMATORY_ENTROPY

EXP, R, K = 29, 2000, 2
DGPS = ("Log", "Osc028", "Osc035", "Osc042")
B_VALUES = dict(Log=0.0, Osc028=0.28, Osc035=0.35, Osc042=0.42)
N_VALUES = (8192, 32768, 131072)
CELLS = tuple(itertools.product(DGPS, N_VALUES))
LOSSES, REGRESSIONS = ("SQ", "UKL"), ("OLS", "CV")
METHODS = (("OLS", "SQ"), ("OLS", "UKL"), ("CV", "SQ"), ("CV", "UKL"), ("Oracle", "Oracle"))
PENALTIES = np.r_[0.0, 10.0 ** np.arange(-6, 1)]
Z_975 = float(stats.norm.ppf(0.975))


def outcome(u):
    return 10 * u + np.sin(4 * np.pi * u)


@cache
def normalizer(dgp):
    if dgp not in DGPS:
        raise ValueError("unknown E-29 density ratio")
    if dgp == "Log":
        return 6 / 25
    b = B_VALUES[dgp]
    return 1 / ((2 / 3) * sum(1 / a - 0.25 / (a**2 + 1) for a in (1 - b, 2 - b)))


def true_ratio(u, dgp):
    u = np.asarray(u, dtype=float)
    if not np.isfinite(u).all() or np.any((u <= 0) | (u > 1)):
        raise ValueError("oracle density ratios require covariates in (0,1]")
    if dgp == "Log":
        return normalizer(dgp) * (1 - np.log(u)) ** 2
    return normalizer(dgp) * u ** (-B_VALUES[dgp]) * (1 + 0.25 * np.sin(np.log(u)))


def source_cdf(u):
    return (2 * u + u**2) / 3


def target_cdf(u, dgp):
    u = np.asarray(u, dtype=float)
    if not np.isfinite(u).all() or np.any((u < 0) | (u > 1)):
        raise ValueError("CDF evaluation requires values in [0,1]")
    positive = u > 0
    x = u[positive]
    values = np.zeros_like(u)
    if dgp == "Log":
        z = 1 - np.log(x)
        values[positive] = (
            normalizer(dgp)
            * (2 / 3)
            * sum(x**a * (z**2 / a + 2 * z / a**2 + 2 / a**3) for a in (1, 2))
        )
    else:
        b = B_VALUES[dgp]
        values[positive] = (
            normalizer(dgp)
            * (2 / 3)
            * sum(x**a / a + 0.25 * np.imag(x ** (a + 1j) / (a + 1j)) for a in (1 - b, 2 - b))
        )
    return values


def target_t_density(t, dgp):
    b, k = B_VALUES[dgp], normalizer(dgp)
    factor = (1 + t) ** 2 if dgp == "Log" else 1 - 0.25 * np.sin(t)
    return k * (2 / 3) * (np.exp(-(1 - b) * t) + np.exp(-(2 - b) * t)) * factor


def _checked_quad(function):
    value, error = integrate.quad(function, 0, np.inf, epsabs=2e-11, epsrel=2e-12, limit=200)
    if not np.isfinite([value, error]).all() or error > 1e-9 * (1 + abs(value)):
        raise ArithmeticError("the population quadrature accuracy check failed")
    return value


@cache
def population(dgp, n):
    if (dgp, n) not in CELLS:
        raise ValueError("condition is outside the complete E-29 design")
    size = n // 4
    h, j = int(np.ceil(2 * size ** (1 / 9))), max(1, int(round(size**0.75 / 4)))
    k, b = normalizer(dgp), B_VALUES[dgp]
    theta = _checked_quad(lambda t: target_t_density(t, dgp) * outcome(np.exp(-t)))
    second = _checked_quad(lambda t: target_t_density(t, dgp) * outcome(np.exp(-t)) ** 2)
    if dgp == "Log":
        vs = (
            k**2
            * (2 / 3)
            * sum(
                math.comb(4, q) * math.factorial(q) / a ** (q + 1) for a in (1, 2) for q in range(5)
            )
        )
    else:
        vs = (
            k**2
            * (2 / 3)
            * sum(
                1 / a - 0.5 / (a**2 + 1) + (0.25**2 / 2) * (1 / a - a / (a**2 + 4))
                for a in (1 - 2 * b, 2 - 2 * b)
            )
        )
    vt = second - theta**2
    if min(vs, vt) <= 0 or abs(float(target_cdf(np.array([1.0]), dgp)[0]) - 1) > 1e-13:
        raise ArithmeticError("nondegenerate normalized population constants are required")
    edges = np.linspace(0, 1, j + 1)
    p, q = np.diff(source_cdf(edges)), np.diff(target_cdf(edges, dgp))
    if min(p.min(), q.min()) <= 0 or max(abs(p.sum() - 1), abs(q.sum() - 1)) > 1e-13:
        raise ArithmeticError("the population cell probabilities failed")
    moments = population_regression(h)
    return dict(
        dgp=dgp,
        n=n,
        L=size,
        H=h,
        d=4 * h,
        J=j,
        k=k,
        theta0=theta,
        source_variance=vs,
        target_variance=vt,
        V=2 * (vs + vt),
        source_probability=p,
        target_probability=q,
        regression_moments=moments,
    )


@cache
def population_regression(h):
    """Deterministic source moments, checked at two Gaussian quadrature orders."""
    records = []
    for order in (16, 32):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        u = ((np.arange(h)[:, None] + (nodes + 1) / 2) / h).ravel()
        _, phi = local_basis(u, h)
        phi = phi.reshape(h, order, 4)
        w = weights[None, :] / (2 * h) * (2 / 3) * (1 + u.reshape(h, order))
        f = outcome(u).reshape(h, order)
        records.append(
            dict(
                gram=np.einsum("ho,hoj,hok->hjk", w, phi, phi),
                xy=np.einsum("ho,hoj,ho->hj", w, phi, f),
                yy=float(np.sum(w * f**2)),
            )
        )
    if max(np.max(np.abs(records[0][key] - records[1][key])) for key in records[0]) > 2e-11:
        raise ArithmeticError("the deterministic regression moments failed their accuracy check")
    return records[1]


def generator(cell, rep, stream, fold):
    if not (
        cell in range(len(CELLS)) and rep in range(R) and stream in range(4) and fold in range(K)
    ):
        raise ValueError("invalid E-29 random stream")
    return np.random.default_rng(
        np.random.SeedSequence(CONFIRMATORY_ENTROPY, spawn_key=(EXP, cell, rep, stream, fold))
    )


def sample_source(rng, size):
    component = interior_uniform(rng, size) < (1 / 3)
    uniform = interior_uniform(rng, size)
    return np.where(component, np.sqrt(uniform), uniform)


def sample_target(rng, size, dgp):
    if dgp == "Log":
        component = rng.choice(6, size=size, p=np.array([4, 8, 8, 2, 2, 1]) / 25)
        t = rng.gamma(np.array([1, 2, 3, 1, 2, 3])[component])
        t /= np.array([1, 1, 1, 2, 2, 2])[component]
    else:
        b = B_VALUES[dgp]
        rates = np.array([1 - b, 2 - b])
        probability = (1 / rates) / np.sum(1 / rates)
        accepted, remaining = [], size
        while remaining:
            component = interior_uniform(rng, remaining) < probability[0]
            proposal = -np.log(interior_uniform(rng, remaining)) / np.where(
                component, rates[0], rates[1]
            )
            keep = interior_uniform(rng, remaining) < (1 - 0.25 * np.sin(proposal)) / 1.25
            accepted.append(proposal[keep])
            remaining -= int(keep.sum())
        t = np.concatenate(accepted)
    u = np.exp(-t)
    if not np.isfinite(u).all() or not np.all((u > 0) & (u < 1)):
        raise ArithmeticError("target observations must lie strictly inside the domain")
    return u


def generate_fold(cell, rep, fold):
    dgp, n = CELLS[cell]
    size = n // 4
    u = sample_source(generator(cell, rep, 0, fold), size)
    epsilon = generator(cell, rep, 1, fold).standard_normal(size)
    tu = sample_target(generator(cell, rep, 2, fold), size, dgp)
    return dict(u=u, y=outcome(u) + epsilon, epsilon=epsilon, tu=tu)


def fit_nuisances(data, h, j):
    """Fit the increasing classes from observations, without a density-ratio formula."""
    size = len(data["u"])
    ratio, ratio_checks = fit_ratio(data["u"], data["tu"], j)
    inner = [
        regression_moments(data["u"][s], data["y"][s], h)
        for s in (slice(0, size // 2), slice(size // 2, size))
    ]
    selected, scores = select_penalty(inner, PENALTIES)
    full = regression_moments(data["u"], data["y"], h)
    regressions = {}
    for name, penalty in (("OLS", 0.0), ("CV", selected)):
        coef, checks = fit_regression(full, penalty)
        matrix = full["gram"] / size + penalty * np.diag([0, 1, 1, 1])
        rhs = full["xy"] / size
        residual = float(np.max(np.abs(np.einsum("hjk,hk->hj", matrix, coef) - rhs)))
        residual /= 1 + float(np.max(np.abs(rhs)))
        if (
            not np.isfinite([*checks.values(), residual]).all()
            or max(checks["normal_equation_scaled"], residual) > 1e-12
        ):
            raise ArithmeticError("the fitted regression equations failed")
        checks["normal_equation_scaled"] = residual
        regressions[name] = coef, float(penalty), checks
    return ratio, regressions, {**ratio_checks, "cv_minimum": float(scores.min())}


def population_diagnostics(ratio, coefficient, facts):
    ratio2 = float(
        facts["source_probability"] @ ratio**2
        - 2 * facts["target_probability"] @ ratio
        + facts["source_variance"]
    )
    moments = facts["regression_moments"]
    regression2 = float(
        np.einsum("hj,hjk,hk->", coefficient, moments["gram"], coefficient)
        - 2 * np.sum(coefficient * moments["xy"])
        + moments["yy"]
    )
    if min(ratio2, regression2) < 0:
        raise ArithmeticError("an exact population squared error is negative")
    return dict(ratio_l2=np.sqrt(ratio2), regression_l2=np.sqrt(regression2))


def _result(source_scores, target_scores, facts):
    theta = float(sum(x.mean() for x in source_scores + target_scores) / K)
    vs = float(sum(np.mean(x**2) for x in source_scores) / K)
    vt = float(sum(np.mean((x - theta) ** 2) for x in target_scores) / K)
    se = float(np.sqrt(2 * (vs + vt) / facts["n"]))
    error = theta - facts["theta0"]
    return dict(
        theta=theta,
        se=se,
        error=error,
        theta0=facts["theta0"],
        source_variance=vs,
        target_variance=vt,
        source_variance_ratio=vs / facts["source_variance"],
        target_variance_ratio=vt / facts["target_variance"],
        covered=float(abs(error) <= Z_975 * se),
        covered_known_variance=float(abs(error) <= Z_975 * np.sqrt(facts["V"] / facts["n"])),
    )


def replicate(task):
    cell, rep = task
    dgp, n = CELLS[cell]
    facts = population(dgp, n)
    data = [generate_fold(cell, rep, fold) for fold in range(K)]
    fits = [fit_nuisances(d, facts["H"], facts["J"]) for d in data]
    oracle_s = [true_ratio(d["u"], dgp) * d["epsilon"] for d in data]
    oracle_t = [outcome(d["tu"]) for d in data]
    oracle = _result(oracle_s, oracle_t, facts)
    base = dict(
        cell=cell,
        rep=rep,
        dgp=dgp,
        n=n,
        L=facts["L"],
        H=facts["H"],
        dimension=facts["d"],
        ratio_cells=facts["J"],
    )
    rows = []
    for regression in REGRESSIONS:
        source_scores, target_scores = [], []
        row = {**base, "regression": regression}
        for evaluation in range(K):
            ratio, regressions, ratio_checks = fits[1 - evaluation]
            coef, penalty, regression_checks = regressions[regression]
            d = data[evaluation]
            source_scores.append(predict_ratio(d["u"], ratio) * (d["y"] - predict(d["u"], coef)))
            target_scores.append(predict(d["tu"], coef))
            diagnostics = population_diagnostics(ratio, coef, facts)
            for key, value in {
                **diagnostics,
                **ratio_checks,
                **regression_checks,
                "lambda": penalty,
            }.items():
                row[f"fold{evaluation}_{key}"] = float(value)
            row[f"fold{evaluation}_norm_product"] = (
                np.sqrt(n) * diagnostics["ratio_l2"] * diagnostics["regression_l2"]
            )
        result = _result(source_scores, target_scores, facts)
        row.update(
            result,
            oracle=oracle["theta"],
            root_n_oracle_difference=np.sqrt(n) * (result["theta"] - oracle["theta"]),
            loss_identity_error=0.0,
        )
        for loss in LOSSES:
            record = {**row, "loss": loss}
            check_result(record)
            rows.append(record)
    oracle_record = {
        **base,
        **oracle,
        "regression": "Oracle",
        "loss": "Oracle",
        "oracle": oracle["theta"],
        "root_n_oracle_difference": 0.0,
        "loss_identity_error": 0.0,
    }
    check_result(oracle_record)
    rows.append(oracle_record)
    return rows


def check_result(row):
    numeric = [value for value in row.values() if isinstance(value, (int, float, np.number))]
    if not np.isfinite(numeric).all() or row["se"] <= 0 or row["loss_identity_error"] != 0:
        raise ArithmeticError(
            "a scientific result is nonfinite or violates the shared-loss identity"
        )
    expected_se = np.sqrt(2 * (row["source_variance"] + row["target_variance"]) / row["n"])
    if (
        abs(row["error"] - (row["theta"] - row["theta0"])) > 1e-12
        or not np.isclose(row["se"], expected_se, rtol=1e-13, atol=0)
        or row["covered"] != float(abs(row["error"]) <= Z_975 * row["se"])
    ):
        raise ArithmeticError("the estimate, variance, or coverage identity failed")
    if row["regression"] != "Oracle":
        for fold in range(K):
            residuals = [
                row[f"fold{fold}_{name}"]
                for name in ("SQ_stationarity", "UKL_stationarity", "normal_equation_scaled")
            ]
            if min(residuals) < 0 or max(residuals) > 1e-12:
                raise ArithmeticError("a fitted first-order equation failed")
            if (
                row[f"fold{fold}_ratio_min"] <= 0
                or row[f"fold{fold}_lambda"] not in PENALTIES
                or (row["regression"] == "OLS" and row[f"fold{fold}_lambda"] != 0)
            ):
                raise ArithmeticError("a ratio-domain or regression-selection check failed")


def tasks_for():
    return tuple(itertools.product(range(len(CELLS)), range(R)))


def summarise(raw):
    keys = ["cell", "rep", "regression", "loss"]
    if len(raw) != len(CELLS) * R * len(METHODS) or raw.duplicated(keys).any():
        raise ValueError("complete E-29 observations are required")
    if raw[["cell", "rep", "regression", "loss", "dgp"]].isna().any().any():
        raise ValueError("missing scientific key")
    expected = set(itertools.product(range(len(CELLS)), METHODS))
    actual = {
        (c, (r, loss))
        for c, r, loss in raw[["cell", "regression", "loss"]].itertuples(index=False, name=None)
    }
    if actual != expected:
        raise ValueError("all twelve conditions and five method records are required")
    numeric = [
        "theta",
        "se",
        "error",
        "theta0",
        "covered",
        "covered_known_variance",
        "source_variance",
        "target_variance",
        "root_n_oracle_difference",
    ]
    if not np.isfinite(raw[numeric]).all().all() or (raw.se <= 0).any():
        raise ValueError("nonfinite scientific record")
    learned = raw[raw.regression != "Oracle"]
    if not np.isfinite(learned.select_dtypes(include=np.number)).all().all():
        raise ValueError("nonfinite fitted nuisance record")
    if not (raw.loss_identity_error == 0).all():
        raise ValueError("the shared-loss identity failed")
    results = []
    for (cell, regression, loss), group in raw.groupby(["cell", "regression", "loss"], sort=True):
        dgp, n = CELLS[cell]
        facts = population(dgp, n)
        if (
            len(group) != R
            or set(group.rep) != set(range(R))
            or not (
                (group.n == n).all()
                and (group.dgp == dgp).all()
                and (group.dimension == facts["d"]).all()
                and (group.ratio_cells == facts["J"]).all()
            )
        ):
            raise ValueError("normal-design replications and metadata must be complete")
        row = dict(
            cell=cell,
            dgp=dgp,
            n=n,
            regression=regression,
            loss=loss,
            replications=R,
            dimension=facts["d"],
            ratio_cells=facts["J"],
            oracle_variance=facts["V"],
        )
        for field in ("covered", "covered_known_variance"):
            if not group[field].isin([0.0, 1.0]).all():
                raise ValueError("coverage records must be binary")
            count = int(group[field].sum())
            interval = stats.binomtest(count, R).proportion_ci(method="exact")
            row[field], row[field + "_lo"], row[field + "_hi"] = (
                count / R,
                interval.low,
                interval.high,
            )
        row["bias"] = float(group.error.mean())
        row["rmse"] = float(np.sqrt(np.mean(group.error**2)))
        row["length"] = float(2 * Z_975 * group.se.mean())
        for field in ("source_variance_ratio", "target_variance_ratio", "root_n_oracle_difference"):
            for label, probability in (("q05", 0.05), ("median", 0.5), ("q95", 0.95)):
                row[field + "_" + label] = float(np.quantile(group[field], probability))
        row["oracle_difference_abs_q95"] = float(
            np.quantile(np.abs(group.root_n_oracle_difference), 0.95)
        )
        if regression != "Oracle":
            fields = (
                "lambda",
                "ratio_l2",
                "regression_l2",
                "norm_product",
                "ratio_max",
                "gram_min_eigenvalue",
                "source_empty",
                "target_empty",
            )
            for field in fields:
                values = group[[f"fold{f}_{field}" for f in range(K)]].to_numpy().ravel()
                if not np.isfinite(values).all():
                    raise ValueError("nonfinite fitted nuisance diagnostics")
                for label, probability in (("q05", 0.05), ("median", 0.5), ("q95", 0.95)):
                    row[field + "_" + label] = float(np.quantile(values, probability))
            for field in ("SQ_stationarity", "UKL_stationarity", "normal_equation_scaled"):
                row[field + "_max"] = float(
                    group[[f"fold{f}_{field}" for f in range(K)]].max().max()
                )
                if not 0 <= row[field + "_max"] <= 1e-12:
                    raise ValueError("a fitted first-order residual is invalid")
        results.append(row)
    paired = raw[raw.regression != "Oracle"].set_index(keys).sort_index()
    left, right = paired.xs("SQ", level="loss"), paired.xs("UKL", level="loss")
    if not left.equals(right):
        raise ValueError("SQ and UKL must have exactly identical scientific records")
    return pd.DataFrame(results)


def tables(summary):
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Ratio & $N$ & $d_L$ & OLS & CV & Oracle \\",
        r"\midrule",
    ]
    for dgp, n in CELLS:
        values = []
        for regression, loss in (("OLS", "SQ"), ("CV", "SQ"), ("Oracle", "Oracle")):
            group = summary[
                (summary.dgp == dgp)
                & (summary.n == n)
                & (summary.regression == regression)
                & (summary.loss == loss)
            ]
            if len(group) != 1:
                raise ValueError("each table cell requires one scientific summary")
            values.append(f"{group.covered.iloc[0]:.4f}")
        label = "Log" if dgp == "Log" else f"Osc., $b={B_VALUES[dgp]:.2f}$"
        lines.append(f"{label} & {n} & {population(dgp, n)['d']} & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return {"tab_E29_coverage": "\n".join(lines) + "\n"}


def figures(summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.8), sharex=True, sharey=True)
    for dgp, ax in zip(DGPS, axes.ravel(), strict=True):
        for regression, loss, color in (
            ("OLS", "SQ", "tab:green"),
            ("CV", "SQ", "tab:blue"),
            ("Oracle", "Oracle", "black"),
        ):
            group = summary[
                (summary.dgp == dgp) & (summary.regression == regression) & (summary.loss == loss)
            ].sort_values("n")
            ax.errorbar(
                group.n,
                group.covered,
                yerr=np.vstack(
                    (group.covered - group.covered_lo, group.covered_hi - group.covered)
                ),
                label=regression,
                color=color,
                marker="o",
                capsize=2,
            )
        ax.axhline(0.95, color="black", linewidth=0.7, linestyle=":")
        ax.set_xscale("log", base=2)
        ax.set_xlim(min(N_VALUES) / 1.15, max(N_VALUES) * 1.15)
        ax.set_xticks(N_VALUES, [r"$2^{13}$", r"$2^{15}$", r"$2^{17}$"])
        ax.set_ylim(0, 1.01)
        ax.set_title(
            "Logarithmic ratio" if dgp == "Log" else f"Oscillating ratio, b = {B_VALUES[dgp]:.2f}"
        )
    for ax in axes[-1]:
        ax.set_xlabel("N")
    for ax in axes[:, 0]:
        ax.set_ylabel("Coverage")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    return {"fig_E29_coverage": fig}
