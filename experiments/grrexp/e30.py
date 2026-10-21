"""E-30: Learned nonlinear GRR, regression tuning and loss selection."""

from __future__ import annotations

import itertools
from functools import cache

import numpy as np
import pandas as pd
from scipy import integrate, special, stats

from genriesz.nonlinear_partition_ratio import (
    DEFAULT_BOUNDS,
    fit_global_ols,
    fit_partition_ratio,
    fit_tensor_histogram_ols,
    predict_global_ols,
    predict_partition_ratio,
    predict_tensor_histogram_ols,
)

from .e27 import interior_uniform
from .seeds import CONFIRMATORY_ENTROPY

EXP, R, K = 30, 2000, 2
B = 0.43
N_VALUES = (32768, 131072, 524288)
DGPS = ("Linear", "Nonlinear")
CELLS = tuple(itertools.product(DGPS, N_VALUES))
LOSSES = ("SQ", "UKL")
REGRESSIONS = ("Global", "Histogram", "Anisotropic")
METHODS = tuple(itertools.product(("Histogram", "Anisotropic", "PredictionCV"), LOSSES)) + (
    ("Selected", "Selected"),
    ("Oracle", "Oracle"),
)
Z_975 = float(stats.norm.ppf(0.975))


def shape(z):
    return 3 * (1 + np.asarray(z)) ** 2 / 7


def mean_function(t, z, dgp):
    if dgp not in DGPS:
        raise ValueError("unknown E-30 mean function")
    u = np.asarray(t)[:, 0]
    first = 10 * u if dgp == "Linear" else -10 * np.expm1(-6 * u)
    return first + 10 * np.asarray(t)[:, 1:].sum(axis=1) + np.asarray(z)


def negative_moment(power, order=0):
    if power >= 1 or order not in (0, 1, 2):
        raise ValueError("integrable power and logarithmic order zero, one or two required")
    return special.factorial(order, exact=True) / (1 - power) ** (order + 1)


def exponential_moment(power, rate):
    """Integral of u**(-power) exp(-rate*u) on the unit interval."""
    if power >= 1 or rate <= 0:
        raise ValueError("require an integrable power and a positive rate")
    a = 1 - np.asarray(power)
    return float(rate) ** (-a) * special.gamma(a) * special.gammainc(a, rate)


def u_mean_moment(dgp, power, squared=False):
    if dgp == "Linear":
        return (100 if squared else 10) / ((3 if squared else 2) - power)
    if dgp != "Nonlinear":
        raise ValueError("unknown mean function")
    if squared:
        return 100 * (
            negative_moment(power)
            - 2 * exponential_moment(power, 6)
            + exponential_moment(power, 12)
        )
    return 10 * (negative_moment(power) - exponential_moment(power, 6))


def cell_integrals(power, n_bins, order=0):
    """Exact integrals of u**(-power) (-log u)**order in equal cells."""
    if power >= 1 or n_bins < 1 or order not in (0, 1):
        raise ValueError("invalid weighted cell integral")
    edges = np.arange(n_bins + 1, dtype=float) / n_bins
    a = 1 - power
    primitive = np.zeros_like(edges)
    positive = edges[1:]
    primitive[1:] = positive**a / a
    if order:
        primitive[1:] *= -np.log(positive) + 1 / a
    return np.diff(primitive)


@cache
def shape_moments(n_cells):
    edges = np.arange(n_cells + 1, dtype=float) / n_cells
    mass = np.diff((1 + edges) ** 3) / 7
    z_mass = (3 / 7) * np.diff(edges**2 / 2 + 2 * edges**3 / 3 + edges**4 / 4)
    shape2 = (9 / 49) * np.diff((1 + edges) ** 5) / 5
    uniform_z = np.diff(edges**2) / 2
    return mass, z_mass, shape2, uniform_z


@cache
def population(dgp, n):
    if (dgp, n) not in CELLS:
        raise ValueError("condition is outside the complete E-30 design")
    m = n // 8
    j = int(np.ceil(np.log(m) / 2))
    h = int(np.ceil(m**0.2))
    hu, hw = int(np.ceil(m**0.4)), int(np.ceil(m**0.1))
    k = 1 - B
    target_u = k * u_mean_moment(dgp, B)
    target_u2 = k * u_mean_moment(dgp, B, True)
    target_z = 17 / 28
    target_z2 = 31 / 70
    theta0 = target_u + 10 + target_z
    vs = k**2 * (279 / 245) / (1 - 2 * B)
    vt = target_u2 - target_u**2 + 200 / 12 + target_z2 - target_z**2
    if min(vs, vt) <= 0:
        raise ArithmeticError("positive source and target variances required")
    return dict(
        dgp=dgp,
        n=n,
        L=n // 4,
        M=m,
        J=j,
        H=h,
        HU=hu,
        HW=hw,
        theta0=float(theta0),
        source_variance=float(vs),
        target_variance=float(vt),
        V=float(2 * (vs + vt)),
        penalties=(0.0, m**-0.5),
    )


def generator(cell, rep, stream, fold):
    if not (0 <= cell < len(CELLS) and 0 <= rep < R and 0 <= stream < 3 and fold in range(K)):
        raise ValueError("invalid E-30 random stream")
    return np.random.default_rng(
        np.random.SeedSequence(
            CONFIRMATORY_ENTROPY,
            spawn_key=(EXP, cell, rep, stream, fold),
        )
    )


def generate_fold(cell, rep, fold):
    dgp, n = CELLS[cell]
    size = n // 4
    x = interior_uniform(generator(cell, rep, 0, fold), (size, 4))
    epsilon = 2 * generator(cell, rep, 1, fold).integers(0, 2, size=size) - 1
    tx = interior_uniform(generator(cell, rep, 2, fold), (size, 4))
    tx[:, 0] **= 1 / (1 - B)
    tx[:, 3] = np.cbrt(1 + 7 * tx[:, 3]) - 1
    t, z = x[:, :3], x[:, 3]
    tt, tz = tx[:, :3], tx[:, 3]
    f = mean_function(t, z, dgp)
    return dict(
        t=t, z=z, y=f + epsilon, epsilon=epsilon, tt=tt, tz=tz, f=f, tf=mean_function(tt, tz, dgp)
    )


def _regression(data, indices, name, facts):
    if name not in REGRESSIONS:
        raise ValueError("unknown E-30 regression")
    args = data["t"][indices], data["z"][indices], data["y"][indices]
    if name == "Global":
        fit = fit_global_ols(*args)
    else:
        bins = (
            (facts["H"],) * 3
            if name == "Histogram"
            else (
                facts["HU"],
                facts["HW"],
                facts["HW"],
            )
        )
        fit = fit_tensor_histogram_ols(*args, bins)
    if fit["status"] != "ok":
        raise ArithmeticError("the registered least-squares fit failed")
    return dict(name=name, fit=fit)


def predict_regression(regression, t, z):
    fit = regression["fit"]
    predictor = (
        predict_global_ols if regression["name"] == "Global" else predict_tensor_histogram_ols
    )
    return predictor(fit, t, z)


def _ratio(data, indices, penalty, facts, loss):
    coefficient, info = fit_partition_ratio(
        data["t"][indices, 0],
        data["z"][indices],
        data["tt"][indices, 0],
        data["tz"][indices],
        facts["J"],
        loss,
        penalty=penalty,
        bounds=DEFAULT_BOUNDS,
    )
    if info["status"] != "ok":
        raise ArithmeticError(f"E-30 {loss} global box minimization failed: {info}")
    return dict(coefficient=coefficient, info=info)


def fit_training(data, facts):
    """Inner CV and full fitting refits precede independent loss validation."""
    m = facts["M"]
    fitting = np.arange(m)
    validation = np.arange(m, 2 * m)
    halves = np.array_split(fitting, 2)
    ratio_risks = np.zeros(len(facts["penalties"]))
    regression_risks = np.zeros(len(REGRESSIONS))
    for inner in range(2):
        train, test = halves[1 - inner], halves[inner]
        for index, penalty in enumerate(facts["penalties"]):
            for loss in LOSSES:
                fit = _ratio(data, train, penalty, facts, loss)
                rs = predict_partition_ratio(
                    fit["coefficient"], data["t"][test, 0], data["z"][test]
                )
                rt = predict_partition_ratio(
                    fit["coefficient"], data["tt"][test, 0], data["tz"][test]
                )
                ratio_risks[index] += (np.mean(rs**2) - 2 * np.mean(rt)) / 4
        for index, name in enumerate(REGRESSIONS):
            reg = _regression(data, train, name, facts)
            residual = predict_regression(reg, data["t"][test], data["z"][test]) - data["y"][test]
            regression_risks[index] += np.mean(residual**2) / 2
    penalty_index, regression_index = int(np.argmin(ratio_risks)), int(np.argmin(regression_risks))
    penalty = facts["penalties"][penalty_index]
    ratios = {loss: _ratio(data, fitting, penalty, facts, loss) for loss in LOSSES}
    regressions = {name: _regression(data, fitting, name, facts) for name in REGRESSIONS}
    cv_name = REGRESSIONS[regression_index]
    candidates = tuple(dict.fromkeys((cv_name, "Anisotropic")))
    source_u, source_z = data["t"][validation, 0], data["z"][validation]
    contrast = predict_partition_ratio(ratios["SQ"]["coefficient"], source_u, source_z)
    contrast -= predict_partition_ratio(ratios["UKL"]["coefficient"], source_u, source_z)
    contrast_estimates, prediction_scores = {}, {}
    for name in candidates:
        residual = predict_regression(regressions[name], data["t"][validation], source_z)
        residual -= data["y"][validation]
        contrast_estimates[name] = float(np.mean(contrast * residual))
        prediction_scores[name] = float(np.mean(residual**2))
    epsilon_n = float(1 / np.log(np.log(facts["n"] + np.exp(np.e))))
    admitted = [
        name
        for name in candidates
        if np.sqrt(facts["n"]) * abs(contrast_estimates[name]) <= epsilon_n
    ]
    selected_name = min(
        admitted or candidates, key=lambda name: (prediction_scores[name], candidates.index(name))
    )
    selected_loss = "SQ" if admitted else "UKL"
    return dict(
        ratios=ratios,
        regressions=regressions,
        cv_name=cv_name,
        selected_name=selected_name,
        selected_loss=selected_loss,
        penalty=float(penalty),
        ratio_risks=ratio_risks,
        regression_risks=regression_risks,
        contrast_estimates=contrast_estimates,
        epsilon_n=epsilon_n,
    )


def _regression_integrals(regression, dgp, power, order=0):
    """Integral over T of u**(-power) V**order times the fitted T component."""
    fit = regression["fit"]
    if regression["name"] == "Global":
        beta = fit["coefficients"]
        value = (beta[0] + beta[2:4].sum() / 2) * negative_moment(power, order)
        value += beta[1] * negative_moment(power - 1, order)
        return float(value)
    coefficient = np.asarray(fit["coefficients"]).reshape(fit["bins"])
    return float(coefficient.mean(axis=(1, 2)) @ cell_integrals(power, fit["bins"][0], order))


def _eta(regression):
    return (
        float(regression["fit"]["coefficients"][-1])
        if regression["name"] == "Global"
        else regression["fit"]["slope"]
    )


@cache
def true_log_direction(dgp):
    def integrand(u):
        first = 10 * u if dgp == "Linear" else -10 * np.expm1(-6 * u)
        return u ** (-B) * (-np.log(u)) * (first + 10)

    value, error = integrate.quad(integrand, 0, 1, epsabs=2e-10, epsrel=2e-12, limit=300)
    if error > 1e-7:
        raise ArithmeticError("population logarithmic integration did not reach its accuracy bound")
    return float(value)


def regression_l2_squared(regression, dgp):
    fit = regression["fit"]
    eta = _eta(regression)
    if regression["name"] == "Global":
        beta = fit["coefficients"]
        eu, eu2 = u_mean_moment(dgp, 0), u_mean_moment(dgp, 0, True)
        gu = u_mean_moment(dgp, -1)
        vector = beta[:4]
        gram = np.full((4, 4), 0.25)
        gram[0] = gram[:, 0] = 0.5
        gram[0, 0] = 1
        np.fill_diagonal(gram[1:, 1:], 1 / 3)
        moment = np.r_[eu + 10, gu + 5, eu / 2 + 35 / 6, eu / 2 + 35 / 6]
        eg, eg2 = float(vector @ np.r_[1.0, 0.5, 0.5, 0.5]), float(vector @ gram @ vector)
        cross = float(vector @ moment)
    else:
        bins = fit["bins"]
        edges = np.arange(bins[0] + 1, dtype=float) / bins[0]
        if dgp == "Linear":
            um = 10 * (edges[:-1] + edges[1:]) / 2
        else:
            um = 10 * (1 - bins[0] * (np.exp(-6 * edges[:-1]) - np.exp(-6 * edges[1:])) / 6)
        w1, w2 = (np.arange(bins[1]) + 0.5) / bins[1], (np.arange(bins[2]) + 0.5) / bins[2]
        truth_cells = um[:, None, None] + 10 * w1[None, :, None] + 10 * w2[None, None, :]
        coefficient = np.asarray(fit["coefficients"]).reshape(bins)
        eg, eg2 = float(coefficient.mean()), float(np.mean(coefficient**2))
        cross = float(np.mean(coefficient * truth_cells))
        eu, eu2 = u_mean_moment(dgp, 0), u_mean_moment(dgp, 0, True)
    truth_second = eu2 + 20 * eu + 350 / 3
    value = eg2 - 2 * cross + truth_second + (eta - 1) * (eg - eu - 10) + (eta - 1) ** 2 / 3
    if value < -1e-9:
        raise ArithmeticError("an exact population regression squared norm is negative")
    return float(max(0.0, value))


def population_diagnostics(ratio, regression, facts):
    coefficient = ratio["coefficient"]
    a, beta = coefficient.T
    scales = np.exp(a)
    mass, _, shape2, uniform_z = shape_moments(len(a))
    p, k = 1 / len(a), 1 - B
    s_j = mass / p
    eta = _eta(regression)
    true_moment = u_mean_moment(facts["dgp"], B) + 10 * negative_moment(B)
    fitted_moment = _regression_integrals(regression, facts["dgp"], B)
    direction0 = fitted_moment - true_moment
    direction1 = _regression_integrals(regression, facts["dgp"], B, 1) - true_log_direction(
        facts["dgp"]
    )
    target_fitted = k * fitted_moment + eta * (17 / 28)
    source_fitted = 0.0
    source_truth = 0.0
    for index, (scale, exponent) in enumerate(zip(scales, beta, strict=True)):
        source_fitted += scale * (
            p * _regression_integrals(regression, facts["dgp"], exponent)
            + eta * uniform_z[index] * negative_moment(exponent)
        )
        source_truth += scale * (
            p * (u_mean_moment(facts["dgp"], exponent) + 10 * negative_moment(exponent))
            + uniform_z[index] * negative_moment(exponent)
        )
    drift = float(target_fitted - facts["theta0"] - source_fitted + source_truth)
    delta = coefficient - np.column_stack((np.log(k * s_j), np.full(len(a), B)))
    d0 = k * s_j * (p * direction0 + (eta - 1) * uniform_z * negative_moment(B))
    d1 = k * s_j * (p * direction1 + (eta - 1) * uniform_z * negative_moment(B, 1))
    linear = -float(np.sum(d0 * delta[:, 0] + d1 * delta[:, 1]))
    shape_approximation = (eta - 1) * float(17 / 28 - np.sum(s_j * uniform_z))
    ratio2 = float(
        np.sum(
            p * scales**2 / (1 - 2 * beta)
            - 2 * k * scales * mass / (1 - beta - B)
            + k**2 * shape2 / (1 - 2 * B)
        )
    )
    if ratio2 < -1e-9:
        raise ArithmeticError("an exact population ratio squared norm is negative")
    regression2 = regression_l2_squared(regression, facts["dgp"])
    return dict(
        drift=drift,
        linear_drift=linear,
        shape_drift=float(shape_approximation),
        nonlinear_remainder=float(drift - linear - shape_approximation),
        direction_intercept=float(direction0),
        direction_slope=float(direction1),
        ratio_l2=float(np.sqrt(max(0.0, ratio2))),
        regression_l2=float(np.sqrt(regression2)),
        norm_product=float(np.sqrt(facts["n"] * max(0.0, ratio2) * regression2)),
        coefficient_max_error=float(np.max(np.linalg.norm(delta, axis=1))),
        aggregate_intercept_error=float(np.sum(k * mass * delta[:, 0])),
        aggregate_slope_error=float(np.sum(k * mass * delta[:, 1])),
    )


def _result(source_scores, target_scores, facts):
    source, target = np.concatenate(source_scores), np.concatenate(target_scores)
    theta = float(source.mean() + target.mean())
    vs = float(np.mean(source**2))
    vt = float(np.mean((target - theta) ** 2))
    se = float(np.sqrt((vs + vt) / len(source)))
    error = theta - facts["theta0"]
    return dict(
        theta=theta,
        theta0=facts["theta0"],
        error=float(error),
        se=se,
        source_average=float(source.mean()),
        target_average=float(target.mean()),
        source_variance=vs,
        target_variance=vt,
        covered=float(abs(error) <= Z_975 * se),
        covered_known_variance=float(abs(error) <= Z_975 * np.sqrt(facts["V"] / facts["n"])),
        interval_length=2 * Z_975 * se,
    )


def replicate(task):
    cell, rep = task
    dgp, n = CELLS[cell]
    facts = population(dgp, n)
    data = [generate_fold(cell, rep, fold) for fold in range(K)]
    fits = [fit_training(d, facts) for d in data]
    oracle_source = [(1 - B) * d["t"][:, 0] ** (-B) * shape(d["z"]) * d["epsilon"] for d in data]
    oracle = _result(oracle_source, [d["tf"] for d in data], facts)
    base = dict(
        cell=cell,
        rep=rep,
        dgp=dgp,
        n=n,
        M=facts["M"],
        ratio_cells=facts["J"],
        dimension=2 * facts["J"],
    )
    rows = []
    for regression, loss in METHODS:
        if regression == "Oracle":
            rows.append(
                {
                    **base,
                    **oracle,
                    "regression": regression,
                    "loss": loss,
                    "oracle": oracle["theta"],
                    "root_n_drift": 0.0,
                    "root_n_oracle_difference": 0.0,
                    "root_n_centered_difference": 0.0,
                }
            )
            continue
        source_scores, target_scores, drifts = [], [], []
        row = {**base, "regression": regression, "loss": loss}
        for evaluation in range(K):
            training, d = fits[1 - evaluation], data[evaluation]
            name = (
                training["selected_name"]
                if regression == "Selected"
                else training["cv_name"]
                if regression == "PredictionCV"
                else regression
            )
            used_loss = training["selected_loss"] if regression == "Selected" else loss
            ratio, reg = training["ratios"][used_loss], training["regressions"][name]
            rs = predict_partition_ratio(ratio["coefficient"], d["t"][:, 0], d["z"])
            source_scores.append(rs * (d["y"] - predict_regression(reg, d["t"], d["z"])))
            target_scores.append(predict_regression(reg, d["tt"], d["tz"]))
            diagnostics = population_diagnostics(ratio, reg, facts)
            drifts.append(diagnostics["drift"])
            for key, value in diagnostics.items():
                row[f"fold{evaluation}_{key}"] = value
            info = ratio["info"]
            extra = dict(
                penalty=training["penalty"],
                boundary_cells=info["boundary_cells"],
                ratio_stationarity=info["stationarity_scaled_max"],
                ratio_gap=info["gap"],
                regression_family=REGRESSIONS.index(name),
                selected_sq=int(used_loss == "SQ"),
                cv_family=REGRESSIONS.index(training["cv_name"]),
                empty_cells=reg["fit"].get("empty_cells", 0),
                validation_drift=training["contrast_estimates"].get(name, np.nan),
            )
            # Validation is computed only for the candidates admitted by the design.
            if not np.isfinite(extra["validation_drift"]):
                v = np.arange(facts["M"], facts["L"])
                tr = data[1 - evaluation]
                difference = predict_partition_ratio(
                    training["ratios"]["SQ"]["coefficient"], tr["t"][v, 0], tr["z"][v]
                )
                difference -= predict_partition_ratio(
                    training["ratios"]["UKL"]["coefficient"], tr["t"][v, 0], tr["z"][v]
                )
                residual = predict_regression(reg, tr["t"][v], tr["z"][v]) - tr["y"][v]
                extra["validation_drift"] = float(np.mean(difference * residual))
            for key, value in extra.items():
                row[f"fold{evaluation}_{key}"] = float(value)
            row[f"fold{evaluation}_ratio_coefficients"] = ratio["coefficient"].ravel().tolist()
            row[f"fold{evaluation}_regression_coefficients"] = np.asarray(
                reg["fit"]["coefficients"]
            ).tolist()
            row[f"fold{evaluation}_regression_bins"] = list(reg["fit"].get("bins", ()))
            row[f"fold{evaluation}_regression_z_slope"] = _eta(reg)
        row.update(_result(source_scores, target_scores, facts))
        row.update(
            oracle=oracle["theta"],
            root_n_drift=float(np.sqrt(n) * np.mean(drifts)),
            root_n_oracle_difference=float(np.sqrt(n) * (row["theta"] - oracle["theta"])),
        )
        row["root_n_centered_difference"] = row["root_n_oracle_difference"] - row["root_n_drift"]
        numeric = [value for value in row.values() if isinstance(value, (int, float, np.number))]
        if not np.isfinite(numeric).all() or row["se"] <= 0:
            raise ArithmeticError("nonfinite E-30 scientific result")
        rows.append(row)
    return rows


def tasks_for():
    return tuple(itertools.product(range(len(CELLS)), range(R)))


def summarise(raw):
    keys = ["cell", "rep", "regression", "loss"]
    if len(raw) != len(CELLS) * R * len(METHODS) or raw.duplicated(keys).any():
        raise ValueError("the complete six-condition E-30 records are required")
    for name in ("cell", "rep"):
        if not all(
            isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
            for value in raw[name]
        ):
            raise ValueError("E-30 condition and replication keys must be integers")
    expected = set(itertools.product(range(len(CELLS)), range(R), METHODS))
    actual = {
        (int(c), int(r), (reg, loss))
        for c, r, reg, loss in raw[keys].itertuples(index=False, name=None)
    }
    if actual != expected:
        raise ValueError("missing or unexpected E-30 scientific keys")
    rows = []
    for (cell, regression, loss), group in raw.groupby(["cell", "regression", "loss"], sort=True):
        coverage = stats.binomtest(int(group["covered"].sum()), R).proportion_ci()
        row = dict(
            cell=int(cell),
            dgp=CELLS[cell][0],
            n=CELLS[cell][1],
            regression=regression,
            loss=loss,
            replications=R,
            coverage=float(group["covered"].mean()),
            coverage_lower=float(coverage.low),
            coverage_upper=float(coverage.high),
            known_variance_coverage=float(group["covered_known_variance"].mean()),
            bias=float(group["error"].mean()),
            rmse=float(np.sqrt(np.mean(group["error"] ** 2))),
            interval_length=float(group["interval_length"].mean()),
        )
        for name in ("root_n_drift", "root_n_oracle_difference", "root_n_centered_difference"):
            for quantile in (0.25, 0.5, 0.75):
                row[f"{name}_q{int(quantile * 100)}"] = float(group[name].quantile(quantile))
        if regression != "Oracle":
            for name in (
                "boundary_cells",
                "selected_sq",
                "cv_family",
                "regression_family",
                "direction_slope",
                "ratio_l2",
                "regression_l2",
                "norm_product",
                "linear_drift",
                "nonlinear_remainder",
                "shape_drift",
                "validation_drift",
            ):
                values = np.concatenate([group[f"fold{k}_{name}"].to_numpy() for k in range(K)])
                row[name + "_mean"] = float(values.mean())
                row[name + "_median"] = float(np.median(values))
            for name in ("direction_slope", "drift", "linear_drift", "nonlinear_remainder"):
                values = np.concatenate(
                    [group[f"fold{k}_{name}"].to_numpy() for k in range(K)]
                )
                row[name + "_absolute_median"] = float(np.median(np.abs(values)))
            cv = np.concatenate([group[f"fold{k}_cv_family"].to_numpy() for k in range(K)])
            row.update(
                {
                    "cv_" + name.lower() + "_frequency": float(np.mean(cv == j))
                    for j, name in enumerate(REGRESSIONS)
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def tables(summary):
    labels = {
        "Histogram": "Isotropic", "Anisotropic": "Anisotropic",
        "PredictionCV": "Prediction CV", "Selected": "Validation selection", "Oracle": "Oracle",
    }
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\hline",
        r"Regression & Loss & Coverage & Bias & RMSE \\",
        r"\hline",
    ]
    for dgp in DGPS:
        lines.append(r"\multicolumn{5}{l}{" + dgp + r" regression, $N=524288$} \\")
        part = summary[(summary.dgp == dgp) & (summary.n == N_VALUES[-1])]
        for reg, loss in METHODS:
            row = part[(part.regression == reg) & (part.loss == loss)].iloc[0]
            lines.append(
                f"{labels[reg]} & {loss} & {row.coverage:.3f} & {row.bias:.4f} & {row.rmse:.4f}"
                + r" \\"
            )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return {"tab_E30_coverage": "\n".join(lines) + "\n"}


def figures(summary):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.1), constrained_layout=True)
    styles = (
        ("Histogram", "SQ", "o", "#b44b42"),
        ("Histogram", "UKL", "s", "#267c9b"),
        ("Anisotropic", "SQ", "^", "#b18b23"),
        ("PredictionCV", "SQ", "D", "#6c5ca5"),
        ("Selected", "Selected", "v", "#2b785a"),
    )
    for column, dgp in enumerate(DGPS):
        for reg, loss, marker, color in styles:
            part = summary[
                (summary.dgp == dgp) & (summary.regression == reg) & (summary.loss == loss)
            ].sort_values("n")
            x = np.arange(len(part))
            axes[0, column].errorbar(
                x,
                part.coverage,
                yerr=np.vstack(
                    (part.coverage - part.coverage_lower, part.coverage_upper - part.coverage)
                ),
                marker=marker,
                color=color,
                capsize=2,
                linewidth=1,
                label=f"{reg}, {loss}",
            )
            axes[1, column].plot(x, part.root_n_drift_q50, marker=marker, color=color, linewidth=1)
        axes[0, column].axhline(0.95, color="black", linestyle="--", linewidth=0.8)
        axes[0, column].set_title(dgp + " outcome regression")
        axes[0, column].set_ylabel("95% Wald coverage")
        axes[0, column].set_ylim(0, 1)
        axes[1, column].axhline(0, color="black", linewidth=0.7)
        axes[1, column].set_ylabel(r"Median $\sqrt{N}$ population drift")
        for row in range(2):
            axes[row, column].set_xticks(range(3), [str(n) for n in N_VALUES])
            axes[row, column].set_xlabel("Total sample size")
            axes[row, column].grid(alpha=0.15)
    axes[0, 0].legend(fontsize=7, loc="lower left")
    return {"fig_E30": figure}
