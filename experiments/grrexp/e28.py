"""E-28: Estimated singular exponents and prediction-selected regression."""

from __future__ import annotations

import itertools
from functools import cache

import numpy as np
import pandas as pd
from scipy import stats

from genriesz.density_ratio_pilot import fit_exponent, fit_outcome, select_penalty
from genriesz.shape_regression import basis, fit_shape

from .e27 import BETA0, SHAPE_POLYNOMIAL, cell_moments, interior_uniform, polynomial_moments
from .seeds import CONFIRMATORY_ENTROPY

EXP, R, K = 28, 2000, 2
B_VALUES = (0.28, 0.35, 0.42)
N_VALUES = (8192, 32768, 131072)
CELLS = tuple(itertools.product(B_VALUES, N_VALUES))
EXPONENTS = ("Estimated", "Known")
REGRESSIONS = ("OLS", "CV", "Ridge")
LOSSES = ("SQ", "UKL")
UPPER = 0.49
PENALTIES = np.r_[0.0, 10.0 ** np.arange(-6, 1)]
Z_975 = float(stats.norm.ppf(0.975))


def source_moment(power, z_moments):
    """Exact moments under source density 2(1+u)/3."""
    first, second = polynomial_moments(power, z_moments)
    first_u, second_u = polynomial_moments(power + 1, z_moments)
    return (2 / 3) * (first + first_u), (2 / 3) * (second + second_u)


def negative_moment(power, derivative=0):
    if power >= 1 or derivative not in (0, 1, 2):
        raise ValueError("integrable exponent and derivative zero, one or two required")
    factor = (1, 1, 2)[derivative]
    return (2 / 3) * factor * ((1 - power) ** (-derivative - 1) + (2 - power) ** (-derivative - 1))


@cache
def population(b, n):
    if (b, n) not in CELLS:
        raise ValueError("condition is outside the complete E-28 design")
    train, m = n // 4, n // 8
    h = int(round(m**0.25))
    z_uniform = 1 / (np.arange(5) + 1)
    z_shape = sum(SHAPE_POLYNOMIAL[j] / (np.arange(5) + j + 1) for j in range(3))
    squared_shape = np.convolve(SHAPE_POLYNOMIAL, SHAPE_POLYNOMIAL)
    shape2 = sum(squared_shape[j] / (j + 1) for j in range(5))
    k = 1 / negative_moment(b)
    direction, target_gram = source_moment(-b, z_shape)
    target_mean, target_gram = k * direction, k * target_gram
    _, gram = source_moment(0, z_uniform)
    theta = float(target_mean @ BETA0)
    vs = k**2 * negative_moment(2 * b) * shape2
    vt = float(BETA0 @ target_gram @ BETA0 - theta**2)
    mu_x = negative_moment(b, 1) * k
    information = negative_moment(b, 2) * k - mu_x**2
    omega = (
        k**2
        * (
            negative_moment(2 * b, 2)
            - 2 * mu_x * negative_moment(2 * b, 1)
            + mu_x**2 * negative_moment(2 * b)
        )
        + information
    )
    penalty_direction = -float(direction @ np.linalg.solve(gram, np.r_[0, BETA0[1:]]))
    if min(vs, vt, information, omega) <= 0 or abs(penalty_direction) < 1e-8:
        raise ArithmeticError("nondegenerate population constants are required")
    return dict(
        b=b,
        n=n,
        L=train,
        M=m,
        H=h,
        k=k,
        theta0=theta,
        source_variance=vs,
        target_variance=vt,
        V=2 * (vs + vt),
        gram=gram,
        target_gram=target_gram,
        target_mean=target_mean,
        direction=direction,
        shape2=shape2,
        exponent_variance=omega / information**2,
        penalty_direction=penalty_direction,
        rho=m ** (2 * b - 1),
    )


def generator(cell, rep, stream, fold):
    if not (0 <= cell < len(CELLS) and 0 <= rep < R and 0 <= stream < 6 and fold in range(K)):
        raise ValueError("invalid E-28 random stream")
    return np.random.default_rng(
        np.random.SeedSequence(CONFIRMATORY_ENTROPY, spawn_key=(EXP, cell, rep, stream, fold))
    )


def generate_fold(cell, rep, fold):
    """Draw every observation from the nonuniform source and its tilted target."""
    b, n = CELLS[cell]
    size, k = n // 4, population(b, n)["k"]
    streams = [generator(cell, rep, j, fold) for j in range(6)]
    v = interior_uniform(streams[0], size)
    u = 3 * v / (np.sqrt(1 + 3 * v) + 1)
    z = interior_uniform(streams[1], size)
    epsilon = streams[2].standard_normal(size)
    phi = basis(u, z)
    y = phi @ BETA0 + epsilon
    target_component = interior_uniform(streams[3], size) < (2 * k / 3) / (1 - b)
    target_v = interior_uniform(streams[4], size)
    tu = target_v ** (1 / np.where(target_component, 1 - b, 2 - b))
    w = interior_uniform(streams[5], size)
    root = np.cbrt(1 + 7 * w)
    tz = 7 * w / (root**2 + root + 1)
    target_phi = basis(tu, tz)
    if not all(np.all((x > 0) & (x < 1)) for x in (u, z, tu, tz)):
        raise ArithmeticError("all observations must lie strictly inside the domain")
    return dict(u=u, z=z, phi=phi, y=y, epsilon=epsilon, tu=tu, tz=tz, tphi=target_phi)


def regression_moments(phi, y):
    return dict(gram=phi.T @ phi, xy=phi.T @ y, yy=float(y @ y), size=len(y))


def fit_nuisances(data, h):
    """Learn all unknown functions and coefficients without DGP parameters."""
    size = len(data["y"])
    m = size // 2
    source_counts = np.bincount((data["z"][:m] * h).astype(int), minlength=h)
    target_counts = np.bincount((data["tz"][:m] * h).astype(int), minlength=h)
    shape, shape_info = fit_shape(source_counts, target_counts)
    if shape.shape != (h,) or not np.isfinite(shape).all() or np.any(shape <= 0):
        raise ArithmeticError("the shape pilot must be finite and positive in every cell")
    q_mean = float(np.mean((target_counts + 1) / (source_counts + 1)))
    gradient = (source_counts + 1) - (target_counts + 1) / (shape * q_mean)
    shape_residual = float(np.max(np.abs(gradient)) / (1 + np.max(source_counts + 1)))
    integral_error = float(abs(shape.mean() - 1))
    if (
        not np.isfinite(list(shape_info.values())).all()
        or max(shape_residual, integral_error) > 1e-12
        or not 0 <= shape_info["pilot_stationarity_scaled"] <= 1e-12
        or not 0 <= shape_info["shape_integral_error"] <= 1e-12
        or shape_info["shape_min"] <= 0
    ):
        raise ArithmeticError("a shape stationarity or normalization check failed")
    shape_info["pilot_stationarity_scaled"] = shape_residual
    shape_info["shape_integral_error"] = integral_error
    exponent, exponent_info = fit_exponent(data["u"][:m], data["tu"][:m], upper=UPPER)
    if (
        not np.isfinite([exponent, *exponent_info.values()]).all()
        or not 0 < exponent < UPPER
        or not 0 <= exponent_info["exponent_stationarity_scaled"] <= 1e-12
        or exponent_info["exponent_hessian"] <= 0
    ):
        raise ArithmeticError("an exponent stationarity or interior check failed")
    inner = [
        regression_moments(data["phi"][s], data["y"][s]) for s in (slice(0, m), slice(m, size))
    ]
    selected, cv_scores = select_penalty(inner, PENALTIES)
    full = regression_moments(data["phi"], data["y"])
    regressions = {}
    for name, penalty in (("OLS", 0.0), ("CV", selected), ("Ridge", size ** (-0.2))):
        coef, info = fit_outcome(full["gram"], full["xy"], size, penalty)
        matrix = full["gram"] / size + np.diag([0, *([penalty] * 5)])
        vector = full["xy"] / size
        residual = float(np.max(np.abs(matrix @ coef - vector)) / (1 + np.max(np.abs(vector))))
        if (
            not np.isfinite([*coef, residual, *info.values()]).all()
            or not 0 <= info["normal_equation_scaled"] <= 1e-12
            or residual > 1e-12
            or (penalty > 0 and info["ridge_min_eigenvalue"] <= 0)
        ):
            raise ArithmeticError(
                "an outcome-regression fitted equation or positivity check failed"
            )
        info["normal_equation_scaled"] = residual
        regressions[name] = (coef, float(penalty), info)
    checks = {**shape_info, **exponent_info}
    checks["cv_minimum"] = float(np.min(cv_scores))
    checks["exponent"] = float(exponent)
    return shape, exponent, regressions, checks


def scale_fit(data, shape, exponent):
    m, h = len(data["u"]) // 2, len(shape)
    sw = data["u"][m:] ** (-exponent) * shape[(data["z"][m:] * h).astype(int)]
    tw = data["tu"][m:] ** (-exponent) * shape[(data["tz"][m:] * h).astype(int)]
    s, s2, t = float(sw.mean()), float(sw @ sw / m), float(tw.mean())
    scales = {"SQ": t / s2, "UKL": 1 / s}
    if not np.isfinite([s, s2, t, *scales.values()]).all() or min(scales.values()) <= 0:
        raise ArithmeticError("both scalar minimizers must be finite positive interior solutions")
    checks = dict(
        source_first=s,
        source_second=s2,
        target_first=t,
        SQ_stationarity=abs(scales["SQ"] * s2 - t) / (1 + abs(t)),
        UKL_stationarity=abs(scales["UKL"] * s - 1),
    )
    if max(checks["SQ_stationarity"], checks["UKL_stationarity"]) > 1e-12:
        raise ArithmeticError("a scalar GRR stationarity check failed")
    return scales, checks


def population_diagnostics(shape, exponent, scale, coefficient, facts):
    h, b, k = len(shape), facts["b"], facts["k"]
    integrals = cell_moments(h)
    z_shape = shape @ integrals
    shape_cross = float(np.sum(shape * (integrals[:, :3] @ SHAPE_POLYNOMIAL)))
    fitted_moment, _ = source_moment(-exponent, z_shape)
    error = coefficient - BETA0
    drift = -float((scale * fitted_moment - k * facts["direction"]) @ error)
    ratio2 = (
        scale**2 * negative_moment(2 * exponent) * float(shape @ shape / h)
        - 2 * scale * k * negative_moment(exponent + b) * shape_cross
        + k**2 * negative_moment(2 * b) * facts["shape2"]
    )
    if ratio2 < 0:
        raise ArithmeticError("negative exact squared ratio norm")
    regression2 = float(error @ facts["gram"] @ error)
    return dict(
        population_drift=drift,
        ratio_l2=float(np.sqrt(ratio2)),
        regression_l2=float(np.sqrt(regression2)),
        directional_error=float(facts["direction"] @ error),
        coefficient_error=scale - k,
        coefficient_stable=(scale - k) / facts["rho"],
        coefficient_norm=float(np.linalg.norm(error)),
    )


def replicate(task):
    cell, rep = task
    b, n = CELLS[cell]
    facts = population(b, n)
    data = [generate_fold(cell, rep, fold) for fold in range(K)]
    fits = [fit_nuisances(d, facts["H"]) for d in data]
    oracle_s = [
        facts["k"] * d["u"] ** (-b) * (3 * (1 + d["z"]) ** 2 / 7) * d["epsilon"] for d in data
    ]
    oracle_t = [d["tphi"] @ BETA0 for d in data]
    oracle = float(sum(x.mean() for x in oracle_s + oracle_t) / 2)
    rows = []
    for exponent_mode in EXPONENTS:
        scales = [
            scale_fit(d, fit[0], fit[1] if exponent_mode == "Estimated" else b)
            for d, fit in zip(data, fits, strict=True)
        ]
        for regression, loss in itertools.product(REGRESSIONS, LOSSES):
            source_scores, target_scores, population_drifts = [], [], []
            row = dict(
                cell=cell,
                rep=rep,
                b=b,
                n=n,
                exponent_mode=exponent_mode,
                regression=regression,
                loss=loss,
            )
            for evaluation in range(K):
                training = 1 - evaluation
                d, fit = data[evaluation], fits[training]
                shape, estimated_b, regressions, checks = fit
                coef, penalty, coef_checks = regressions[regression]
                exponent = estimated_b if exponent_mode == "Estimated" else b
                scale = scales[training][0][loss]
                ratio = scale * d["u"] ** (-exponent) * shape[(d["z"] * facts["H"]).astype(int)]
                score_s = ratio * (d["y"] - d["phi"] @ coef)
                score_t = d["tphi"] @ coef
                source_scores.append(score_s)
                target_scores.append(score_t)
                diagnostics = population_diagnostics(shape, exponent, scale, coef, facts)
                population_drifts.append(diagnostics["population_drift"])
                for key, value in diagnostics.items():
                    row[f"fold{evaluation}_{key}"] = value
                row[f"fold{evaluation}_lambda"] = penalty
                row[f"fold{evaluation}_exponent"] = estimated_b
                row[f"fold{evaluation}_root_M_exponent_error"] = np.sqrt(facts["M"]) * (
                    estimated_b - b
                )
                row[f"fold{evaluation}_exponent_standardized"] = (
                    np.sqrt(facts["M"]) * (estimated_b - b) / np.sqrt(facts["exponent_variance"])
                )
                row[f"fold{evaluation}_scale"] = scale
                row[f"fold{evaluation}_shape_stationarity"] = checks["pilot_stationarity_scaled"]
                row[f"fold{evaluation}_shape_integral_error"] = checks["shape_integral_error"]
                row[f"fold{evaluation}_exponent_stationarity"] = checks[
                    "exponent_stationarity_scaled"
                ]
                row[f"fold{evaluation}_regression_stationarity"] = coef_checks[
                    "normal_equation_scaled"
                ]
                row[f"fold{evaluation}_scale_stationarity"] = scales[training][1][
                    loss + "_stationarity"
                ]
                row[f"fold{evaluation}_evaluation_remainder"] = float(
                    (score_s - oracle_s[evaluation]).mean()
                    + (score_t - oracle_t[evaluation]).mean()
                    - diagnostics["population_drift"]
                )
                row[f"fold{evaluation}_norm_product"] = (
                    np.sqrt(n) * diagnostics["ratio_l2"] * diagnostics["regression_l2"]
                )
                for j, c in enumerate(coef):
                    row[f"fold{evaluation}_beta{j}"] = float(c)
            theta = float(sum(x.mean() for x in source_scores + target_scores) / 2)
            vs = float(sum(x @ x / len(x) for x in source_scores) / 2)
            vt = float(sum(np.mean((x - theta) ** 2) for x in target_scores) / 2)
            se = float(np.sqrt(2 * (vs + vt) / n))
            drift = float(sum(population_drifts) / 2)
            evaluation_remainder = float(
                sum(row[f"fold{j}_evaluation_remainder"] for j in range(K)) / 2
            )
            row.update(
                theta=theta,
                se=se,
                source_variance=vs,
                target_variance=vt,
                source_variance_ratio=vs / facts["source_variance"],
                target_variance_ratio=vt / facts["target_variance"],
                oracle=oracle,
                theta0=facts["theta0"],
                error=theta - facts["theta0"],
                covered=float(abs(theta - facts["theta0"]) <= Z_975 * se),
                covered_known_variance=float(
                    abs(theta - facts["theta0"]) <= Z_975 * np.sqrt(facts["V"] / n)
                ),
                oracle_covered=float(
                    abs(oracle - facts["theta0"]) <= Z_975 * np.sqrt(facts["V"] / n)
                ),
                population_drift=drift,
                root_n_drift=np.sqrt(n) * drift,
                evaluation_remainder=evaluation_remainder,
                decomposition_error=theta - oracle - drift - evaluation_remainder,
                exponent_variance=facts["exponent_variance"],
                penalty_direction=facts["penalty_direction"],
            )
            rows.append(row)
    known = {
        (r["regression"], r["loss"]): r["theta"] for r in rows if r["exponent_mode"] == "Known"
    }
    for row in rows:
        row["root_n_exponent_effect"] = np.sqrt(n) * (
            row["theta"] - known[row["regression"], row["loss"]]
        )
        check_result(row)
    return rows


def check_result(row):
    numeric = [v for v in row.values() if isinstance(v, (int, float, np.number))]
    if not np.isfinite(numeric).all() or row["se"] <= 0:
        raise ArithmeticError("nonfinite result or nonpositive standard error")
    residuals = [
        row[f"fold{fold}_{name}"]
        for fold in range(K)
        for name in (
            "shape_stationarity",
            "shape_integral_error",
            "exponent_stationarity",
            "regression_stationarity",
            "scale_stationarity",
        )
    ]
    if min(residuals) < 0 or max(residuals) > 1e-12:
        raise ArithmeticError("a fitted first-order equation or normalization check failed")
    identity = row["theta"] - row["oracle"] - row["population_drift"] - row["evaluation_remainder"]
    if max(abs(identity), abs(row["decomposition_error"])) > 1e-10 * (1 + abs(row["theta"])):
        raise ArithmeticError("the finite-sample score decomposition failed")


def tasks_for():
    return tuple(itertools.product(range(len(CELLS)), range(R)))


def summarise(raw):
    keys = ["cell", "rep", "exponent_mode", "regression", "loss"]
    if len(raw) != len(CELLS) * R * 12 or raw.duplicated(keys).any():
        raise ValueError("complete E-28 observations are required")
    if not np.isfinite(raw.select_dtypes(include=np.number)).all().all():
        raise ValueError("nonfinite scientific record")
    expected = set(itertools.product(range(len(CELLS)), EXPONENTS, REGRESSIONS, LOSSES))
    actual = set(
        raw[["cell", "exponent_mode", "regression", "loss"]].itertuples(index=False, name=None)
    )
    if actual != expected:
        raise ValueError("the fixed conditions and twelve methods must all be present")
    result = []
    for (cell, mode, regression, loss), group in raw.groupby(
        ["cell", "exponent_mode", "regression", "loss"], sort=True
    ):
        if len(group) != R or set(group.rep) != set(range(R)):
            raise ValueError("missing normal-design replications")
        b, n = CELLS[cell]
        if not (np.all(group.b == b) and np.all(group.n == n)):
            raise ValueError("condition metadata do not match the fixed design")
        row = dict(
            cell=cell,
            b=b,
            n=n,
            exponent_mode=mode,
            regression=regression,
            loss=loss,
            replications=R,
        )
        for name in ("covered", "covered_known_variance", "oracle_covered"):
            count = int(group[name].sum())
            interval = stats.binomtest(count, R).proportion_ci(method="exact")
            row[name] = count / R
            row[name + "_lo"], row[name + "_hi"] = interval.low, interval.high
        row["bias"] = float(group.error.mean())
        row["rmse"] = float(np.sqrt(np.mean(group.error**2)))
        for name in [
            "se",
            "source_variance_ratio",
            "target_variance_ratio",
            "root_n_drift",
            "root_n_exponent_effect",
            "fold0_lambda",
            "fold1_lambda",
            "fold0_exponent",
            "fold1_exponent",
            "fold0_exponent_standardized",
            "fold1_exponent_standardized",
            "fold0_norm_product",
            "fold1_norm_product",
        ]:
            for label, probability in (("q05", 0.05), ("median", 0.5), ("q95", 0.95)):
                row[name + "_" + label] = float(np.quantile(group[name], probability))
        row["exponent_effect_abs_q95"] = float(
            np.quantile(np.abs(group.root_n_exponent_effect), 0.95)
        )
        for name in (
            "decomposition_error",
            "fold0_shape_stationarity",
            "fold1_shape_stationarity",
            "fold0_shape_integral_error",
            "fold1_shape_integral_error",
            "fold0_exponent_stationarity",
            "fold1_exponent_stationarity",
            "fold0_scale_stationarity",
            "fold1_scale_stationarity",
            "fold0_regression_stationarity",
            "fold1_regression_stationarity",
        ):
            row[name + "_max"] = float(np.max(np.abs(group[name])))
        result.append(row)
    return pd.DataFrame(result)


def tables(summary):
    outputs = {}
    for mode in EXPONENTS:
        lines = [
            r"\begin{tabular}{rr" + "r" * 6 + "}",
            r"\toprule",
            r" & & \multicolumn{3}{c}{SQ} & \multicolumn{3}{c}{UKL} \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
            r"$b$ & $N$ & OLS & CV & Ridge & OLS & CV & Ridge \\",
            r"\midrule",
        ]
        for b, n in CELLS:
            values = []
            for loss, regression in itertools.product(LOSSES, REGRESSIONS):
                g = summary[
                    (summary.b == b)
                    & (summary.n == n)
                    & (summary.exponent_mode == mode)
                    & (summary.loss == loss)
                    & (summary.regression == regression)
                ]
                if len(g) != 1:
                    raise ValueError("each coverage cell must have exactly one summary")
                values.append(f"{g.covered.iloc[0]:.4f}")
            lines.append(f"{b:.2f} & {n} & " + " & ".join(values) + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        outputs["tab_E28_" + mode.lower()] = "\n".join(lines) + "\n"
    return outputs


def figures(summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5), sharex=True, sharey=True)
    colors = dict(OLS="tab:green", CV="tab:blue", Ridge="tab:orange")
    for row, loss in enumerate(LOSSES):
        for col, b in enumerate(B_VALUES):
            ax = axes[row, col]
            for regression in REGRESSIONS:
                g = summary[
                    (summary.b == b)
                    & (summary.loss == loss)
                    & (summary.regression == regression)
                    & (summary.exponent_mode == "Estimated")
                ].sort_values("n")
                ax.errorbar(
                    g.n,
                    g.covered,
                    yerr=np.vstack((g.covered - g.covered_lo, g.covered_hi - g.covered)),
                    color=colors[regression],
                    label=regression,
                    marker="o",
                    capsize=2,
                )
            ax.axhline(0.95, color="black", linewidth=0.7, linestyle=":")
            ax.set_xscale("log", base=2)
            ax.set_ylim(0, 1.01)
            ax.set_title(f"{loss}, b = {b:.2f}")
            if row == 1:
                ax.set_xlabel("N")
            if col == 0:
                ax.set_ylabel("Coverage")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    return {"fig_E28": fig}
