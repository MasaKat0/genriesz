"""E-27: Unknown density-ratio shape and fitted ridge regression."""

from __future__ import annotations

import itertools
import json
from functools import cache

import numpy as np
import pandas as pd
from scipy import integrate, special, stats

from genriesz.shape_regression import (
    SourceMoments,
    TargetMoments,
    basis,
    fit_ridge,
    fit_scales,
    fit_shape,
)

from .e25 import quantile_interval
from .seeds import CONFIRMATORY_ENTROPY

EXP, R, K = 27, 2000, 2
B_VALUES = (0.28, 0.35, 0.42)
N_VALUES = (2048, 32768, 524288, 8388608)
CELLS = tuple(itertools.product(B_VALUES, N_VALUES))
ARM_LABELS = ("SQ", "UKL")
CHUNK_SIZE = 65536
BETA0 = np.array([0, 10, 1, 1, 0.5, 0.5], dtype=float)
SHAPE_POLYNOMIAL = np.array([3, 6, 3], dtype=float) / 7
Z_975 = float(stats.norm.ppf(0.975))


def _basis_polynomial():
    c = np.zeros((6, 3, 3))
    c[0, 0, 0] = 1
    c[1, :2, 0] = np.sqrt(3) * np.array([-1, 2])
    c[2, 0, :2] = np.sqrt(3) * np.array([-1, 2])
    c[3, :2, :2] = 3 * np.array([[1, -2], [-2, 4]])
    c[4, :, 0] = np.sqrt(5) * np.array([1, -6, 6])
    c[5, 0, :] = np.sqrt(5) * np.array([1, -6, 6])
    return c


BASIS_POLYNOMIAL = _basis_polynomial()


def polynomial_moments(power, z_moments):
    """Exact integrals of phi and phi phi' with the weight u**power."""
    um = 1 / (np.arange(5) + power + 1)
    zm = np.asarray(z_moments, float)
    if power <= -1 or zm.shape != (5,):
        raise ValueError("integrable U power and five Z moments are required")
    moment = np.einsum("iab,a,b->i", BASIS_POLYNOMIAL, um[:3], zm[:3])
    indices = np.add.outer(np.arange(3), np.arange(3))
    gram = np.einsum(
        "iab,jcd,ac,bd->ij", BASIS_POLYNOMIAL, BASIS_POLYNOMIAL, um[indices], zm[indices]
    )
    return moment, gram


@cache
def cell_moments(h):
    edges = np.arange(h + 1, dtype=float) / h
    powers = np.arange(1, 6)
    return (edges[1:, None] ** powers - edges[:-1, None] ** powers) / powers


def true_shape(z):
    """The data-generating marginal density; it is not an input to any learner."""
    return 3 * (1 + np.asarray(z)) ** 2 / 7


def gaussian_stable_coverage(b, shape_intensity, kappa, theta0, a0, variance):
    """Known-V SQ coverage reference using finite-N kappa in the limiting mechanism.

    This is a deterministic Gaussian plus stable reference, rather than exact
    finite-sample coverage of the learned estimator.
    """
    values = np.asarray([b, shape_intensity, kappa, theta0, a0, variance], float)
    if (
        not np.isfinite(values).all()
        or not 0.25 < b < 0.5
        or shape_intensity <= 0
        or kappa < 0
        or theta0 >= 0
        or a0 <= 0
        or variance <= 0
    ):
        raise ValueError(
            "the coverage reference requires 1/4 < b < 1/2 and valid population constants"
        )
    index = 1 / (2 * b)
    h = -kappa * theta0 / (2 * a0 * np.sqrt(variance))
    amplitude = (
        2 * shape_intensity * special.gamma(1 - index) * np.cos(np.pi * index / 2) * h**index
    )
    if amplitude < 0 or not np.isfinite(amplitude):
        raise ArithmeticError("the stable characteristic exponent must have nonnegative amplitude")

    def integrand(t):
        stable_power = amplitude * t**index
        return (
            Z_975
            * np.sinc(Z_975 * t / np.pi)
            * np.exp(-(t**2) / 2 - stable_power)
            * np.cos(stable_power * np.tan(np.pi * index / 2))
        )

    integral, error = integrate.quad(integrand, 0, np.inf, epsabs=1e-11, epsrel=1e-11, limit=300)
    coverage = float(2 * integral / np.pi)
    if error > 1e-9 or not 0 <= coverage <= 0.95 + 1e-12:
        raise ArithmeticError("the Gaussian plus stable coverage quadrature failed")
    return coverage


@cache
def population(b, n):
    train_size, m = n // 4, n // 8
    h = int(round(m**0.25))
    if n % 8 or h**4 != m:
        raise ValueError("E-27 requires integer M and the exact fourth root H")
    z_uniform = 1 / (np.arange(5) + 1)
    z_shape = sum(SHAPE_POLYNOMIAL[j] / (np.arange(5) + j + 1) for j in range(3))
    vs_moment, vs_gram = polynomial_moments(-b, z_shape)
    _, v2_gram = polynomial_moments(-2 * b, z_uniform)
    k = 1 - b
    theta = float(k * vs_moment @ BETA0)
    target_variance = float(k * BETA0 @ vs_gram @ BETA0 - theta**2)
    shape_squared = float(
        sum(
            SHAPE_POLYNOMIAL[i] * SHAPE_POLYNOMIAL[j] / (i + j + 1)
            for i in range(3)
            for j in range(3)
        )
    )
    source_variance = k**2 * shape_squared / (1 - 2 * b)
    squared_shape = np.convolve(SHAPE_POLYNOMIAL, SHAPE_POLYNOMIAL)
    z_shape_squared = sum(squared_shape[j] / (np.arange(5) + j + 1) for j in range(5))
    _, v2_shape2_gram = polynomial_moments(-2 * b, z_shape_squared)
    projected_shape = h * (cell_moments(h)[:, :3] @ SHAPE_POLYNOMIAL)
    shape_projection_error2 = shape_squared - float(projected_shape @ projected_shape / h)
    penalty = train_size ** (-0.2)
    q = penalty / (1 + penalty)
    rho = m ** (2 * b - 1)
    a0 = shape_squared / (1 - 2 * b)
    shape_intensity = (3 / 7) ** (1 / b) * (2 ** (2 / b + 1) - 1) / (2 / b + 1)
    variance = 2 * (source_variance + target_variance)
    kappa = np.sqrt(n) * rho * q
    coverage_reference = gaussian_stable_coverage(b, shape_intensity, kappa, theta, a0, variance)
    return {
        "n": n,
        "L": train_size,
        "M": m,
        "H": h,
        "b": b,
        "k": k,
        "lambda": penalty,
        "q": q,
        "rho": rho,
        "kappa": kappa,
        "gaussian_stable_coverage_reference": coverage_reference,
        "theta0": theta,
        "mu": float(vs_moment @ BETA0),
        "A0": a0,
        "D_s": shape_intensity,
        "stable_index": 1 / (2 * b),
        "source_variance": source_variance,
        "target_variance": target_variance,
        "finite_q_source_variance": float(
            source_variance + q**2 * k**2 * BETA0 @ v2_shape2_gram @ BETA0
        ),
        "finite_q_target_variance": (1 - q) ** 2 * target_variance + q**2 * theta**2,
        "V": variance,
        "vs_moment": vs_moment,
        "target_gram": k * vs_gram,
        "v2_gram": v2_gram,
        "projected_shape": projected_shape,
        "shape_projection_error2": shape_projection_error2,
    }


def generator(cell, rep, stream, fold, entropy=CONFIRMATORY_ENTROPY):
    if not (
        0 <= cell < len(CELLS)
        and 0 <= rep < R
        and stream in range(5)
        and fold in range(K)
        and entropy == CONFIRMATORY_ENTROPY
    ):
        raise ValueError("invalid E-27 seed key")
    return np.random.default_rng(
        np.random.SeedSequence(
            entropy, spawn_key=(EXP, int(cell), int(rep), int(stream), int(fold))
        )
    )


def interior_uniform(rng, size):
    return rng.integers(1, 1 << 53, size=size).astype(float) / float(1 << 53)


def collect_fold(cell, rep, fold, *, chunk_size=CHUNK_SIZE):
    """Process every observation once; changing chunk size preserves each RNG stream."""
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    b, n = CELLS[cell]
    facts = population(b, n)
    h, m, train_size = facts["H"], facts["M"], facts["L"]
    source, target = SourceMoments(h, b), TargetMoments(h, b)
    streams = [generator(cell, rep, stream, fold) for stream in range(5)]
    oracle = np.zeros(3)
    minima, maxima = np.ones(4), np.zeros(4)
    for pilot in (True, False):
        for start in range(0, m, chunk_size):
            size = min(chunk_size, m - start)
            u, z = interior_uniform(streams[0], size), interior_uniform(streams[1], size)
            epsilon = streams[2].standard_normal(size)
            phi = basis(u, z)
            y = phi @ BETA0 + epsilon
            source.add(u, z, y, pilot=pilot)
            oracle_score = facts["k"] * u ** (-b) * true_shape(z) * epsilon
            oracle[:2] += (oracle_score.sum(), oracle_score @ oracle_score)
            if not pilot:
                oracle[2] += np.sum((u ** (-b) * true_shape(z)) ** 2)
            tu = interior_uniform(streams[3], size) ** (1 / facts["k"])
            w = interior_uniform(streams[4], size)
            root = np.cbrt(1 + 7 * w)
            tz = 7 * w / (root**2 + root + 1)
            target.add(tu, tz, pilot=pilot)
            values = (u, z, tu, tz)
            minima = np.minimum(minima, [x.min() for x in values])
            maxima = np.maximum(maxima, [x.max() for x in values])
    if (
        source.n != train_size
        or target.n != train_size
        or source.pilot_n != m
        or source.scale_n != m
        or target.pilot_n != m
        or target.scale_n != m
        or source.cell_counts.sum() != train_size
        or target.cell_counts.sum() != train_size
        or not np.all(minima > 0)
        or not np.all(maxima < 1)
    ):
        raise ArithmeticError("incomplete data processing or non-interior covariates")
    return source, target, oracle, minima, maxima


def nuisance(source, target, penalty):
    """Learn the shape, GRR scales and regression from observations alone."""
    if source.n != target.n or source.scale_n != target.scale_n:
        raise ValueError("source and target training halves must have equal sizes")
    shape, info = fit_shape(source.pilot_counts, target.pilot_counts)
    scales, scale_info = fit_scales(
        shape, source.scale_v, source.scale_v2, target.scale_v, source.scale_n
    )
    coefficient, ridge_info = fit_ridge(source.gram, source.xy, source.n, penalty)
    checks = {**info, **scale_info, **ridge_info}
    residuals = [
        checks[name]
        for name in (
            "pilot_stationarity_scaled",
            "shape_integral_error",
            "SQ_stationarity_scaled",
            "UKL_stationarity_scaled",
            "normal_equation_scaled",
        )
    ]
    if max(residuals) > 1e-12 or checks["ridge_min_eigenvalue"] <= 0:
        raise ArithmeticError("a fitted first-order equation or positivity check failed")
    return shape, scales, coefficient, checks


def fold_diagnostics(shape, scale, coefficient, facts, scale_true_v2_mean):
    """Exact polynomial and cell integrals for the fitted nuisance functions."""
    b, k, q, h = (facts[key] for key in ("b", "k", "q", "H"))
    delta = coefficient - BETA0
    ridge_error = coefficient - (1 - q) * BETA0
    shape_moment, _ = polynomial_moments(-b, shape @ cell_moments(h))
    difference_moment = shape_moment - facts["vs_moment"]
    directional = float(facts["vs_moment"] @ delta)
    shape_directional = float(difference_moment @ delta)
    shape_true_regression = float(difference_moment @ BETA0)
    shape_ridge_error = float(difference_moment @ ridge_error)
    shape_bias = k * q * shape_true_regression
    shape_sampling = -k * shape_ridge_error
    cross_bias = (scale - k) * q * shape_true_regression
    cross_sampling = -(scale - k) * shape_ridge_error
    da_source = -k / facts["A0"] * (scale_true_v2_mean - facts["A0"])
    da_remainder = (scale - k) - da_source
    source_stable_leading_drift = q * facts["mu"] * da_source
    coefficient_expansion_drift = q * facts["mu"] * da_remainder
    leading = (scale - k) * q * facts["mu"]
    ridge_drift = -(scale - k) * float(facts["vs_moment"] @ ridge_error)
    pilot_drift = -k * shape_directional
    cross = -(scale - k) * shape_directional
    drift = -(scale - k) * directional - scale * shape_directional
    projected = facts["projected_shape"]
    ratio_norm2 = (
        np.mean((scale * shape - k * projected) ** 2) + k**2 * facts["shape_projection_error2"]
    ) / (1 - 2 * b)
    shape_norm2 = np.mean((shape - projected) ** 2) + facts["shape_projection_error2"]
    regression_norm = float(np.linalg.norm(delta))
    shape_weighted_norm = float(np.sqrt(shape_norm2 / (1 - 2 * b)))
    cross_norm = abs(scale - k) * shape_weighted_norm
    result = {
        "coefficient": scale,
        "coefficient_error": scale - k,
        "coefficient_stable": (scale - k) / facts["rho"],
        "coefficient_root_M": np.sqrt(facts["M"]) * (scale - k),
        "coefficient_error_l2": abs(scale - k) * np.sqrt(facts["A0"]),
        "shape_term_l2": k * shape_weighted_norm,
        "cross_l2": cross_norm,
        "cross_norm_product": np.sqrt(facts["n"]) * cross_norm * regression_norm,
        "scale_true_v2_mean": float(scale_true_v2_mean),
        "da_source": da_source,
        "da_remainder": da_remainder,
        "da_source_over_rho": da_source / facts["rho"],
        "da_remainder_over_rho": da_remainder / facts["rho"],
        "root_n_da_source": np.sqrt(facts["n"]) * da_source,
        "root_n_da_remainder": np.sqrt(facts["n"]) * da_remainder,
        "population_drift": drift,
        "leading_drift": leading,
        "ridge_drift": ridge_drift,
        "pilot_shape_drift": pilot_drift,
        "coefficient_shape_cross": cross,
        "shape_bias": shape_bias,
        "shape_sampling": shape_sampling,
        "cross_bias": cross_bias,
        "cross_sampling": cross_sampling,
        "shape_decomposition_identity_error": pilot_drift - shape_bias - shape_sampling,
        "cross_decomposition_identity_error": cross - cross_bias - cross_sampling,
        "source_stable_leading_drift": source_stable_leading_drift,
        "coefficient_expansion_drift": coefficient_expansion_drift,
        "coefficient_decomposition_identity_error": (
            leading - source_stable_leading_drift - coefficient_expansion_drift
        ),
        "drift_identity_error": drift - leading - ridge_drift - pilot_drift - cross,
        "directional_error": directional,
        "directional_over_q": directional / q,
        "directional_limit": -facts["mu"],
        "shape_directional": shape_directional,
        "root_L_shape_directional": np.sqrt(facts["L"]) * shape_directional,
        "regression_l2": regression_norm,
        "regression_l2_over_q": regression_norm / q,
        "regression_v_l2": float(np.sqrt(delta @ facts["v2_gram"] @ delta)),
        "regression_target_l2": float(np.sqrt(delta @ facts["target_gram"] @ delta)),
        "ridge_error_l2": float(np.linalg.norm(ridge_error)),
        "representer_l2": float(np.sqrt(ratio_norm2)),
        "shape_l2": float(np.sqrt(shape_norm2)),
        "shape_sampling_l2": float(np.sqrt(np.mean((shape - projected) ** 2))),
        "shape_projection_l2": float(np.sqrt(facts["shape_projection_error2"])),
        "full_norm_product": float(np.sqrt(facts["n"] * ratio_norm2) * regression_norm),
        "full_norm_product_lower_bound": float(
            np.sqrt(facts["n"] * k**2 * facts["shape_projection_error2"] / (1 - 2 * b))
            * regression_norm
        ),
        "SQ_population_scale": float(k * np.mean(projected * shape) / np.mean(shape**2)),
        "scale_denominator_center": float(np.mean(shape**2) / (1 - 2 * b)),
        "scale_numerator_center": float(k * np.mean(projected * shape) / (1 - 2 * b)),
    }
    if not np.isfinite(list(result.values())).all():
        raise ArithmeticError("nonfinite population diagnostic")
    return result


def score_records(source, target, fitted, oracle_sums, facts):
    """Evaluate opposite-fold fits exactly from sufficient statistics."""
    train_size, n = facts["L"], facts["n"]
    theta = facts["theta0"]
    oracle_source_mean = float(sum(x[0] for x in oracle_sums) / (2 * train_size))
    oracle_target_mean = float(sum(t.phi @ BETA0 for t in target) / (2 * train_size))
    oracle_estimate = oracle_source_mean + oracle_target_mean
    oracle_source_variance = float(sum(x[1] for x in oracle_sums) / (2 * train_size))
    oracle_target_variance = float(sum(t.score(BETA0, oracle_estimate)[1] for t in target) / 2)
    oracle_se = np.sqrt((oracle_source_variance + oracle_target_variance) / (2 * train_size))
    records = {}
    for arm in ARM_LABELS:
        source_first, source_second, target_first = [], [], []
        diagnostics = []
        for evaluation in range(K):
            shape, scales, coefficient, _ = fitted[1 - evaluation]
            first, second = source[evaluation].score(coefficient, shape, scales[arm])
            source_first.append(first)
            source_second.append(second)
            target_first.append(float(target[evaluation].phi @ coefficient / train_size))
            scale_true_v2_mean = oracle_sums[1 - evaluation][2] / facts["M"]
            diagnostics.append(
                fold_diagnostics(shape, scales[arm], coefficient, facts, scale_true_v2_mean)
            )
        estimate = float((sum(source_first) + sum(target_first)) / 2)
        target_second = [target[j].score(fitted[1 - j][2], estimate)[1] for j in range(K)]
        source_variance, target_variance = (
            float(np.mean(source_second)),
            float(np.mean(target_second)),
        )
        se = float(np.sqrt((source_variance + target_variance) / (2 * train_size)))
        components = {
            key: float(np.mean([d[key] for d in diagnostics]))
            for key in (
                "population_drift",
                "leading_drift",
                "ridge_drift",
                "pilot_shape_drift",
                "coefficient_shape_cross",
            )
        }
        evaluation_source = float(np.mean(source_first) - oracle_source_mean)
        evaluation_target = float(np.mean(target_first) - oracle_target_mean)
        evaluation_remainder = (
            evaluation_source + evaluation_target - components["population_drift"]
        )
        components["evaluation_remainder"] = evaluation_remainder
        additional_components = {
            key: float(np.mean([d[key] for d in diagnostics]))
            for key in (
                "shape_bias",
                "shape_sampling",
                "cross_bias",
                "cross_sampling",
                "source_stable_leading_drift",
                "coefficient_expansion_drift",
            )
        }
        finite_sample_identity = (
            estimate
            - theta
            - (
                oracle_estimate
                - theta
                + components["leading_drift"]
                + components["ridge_drift"]
                + components["pilot_shape_drift"]
                + components["coefficient_shape_cross"]
                + evaluation_remainder
            )
        )
        if abs(finite_sample_identity) > 1e-10 * (1 + abs(estimate)):
            raise ArithmeticError("the finite-sample score decomposition failed")
        row = {
            "status": "ok",
            "estimate": estimate,
            "se": se,
            "source_variance": source_variance,
            "target_variance": target_variance,
            "variance_ratio": 2 * (source_variance + target_variance) / facts["V"],
            "source_variance_ratio": source_variance / facts["source_variance"],
            "target_variance_ratio": target_variance / facts["target_variance"],
            "root_n_error": np.sqrt(n) * (estimate - theta),
            "studentized_error": (estimate - theta) / se,
            "known_variance_studentized_error": np.sqrt(n / facts["V"]) * (estimate - theta),
            "finite_q_source_variance": facts["finite_q_source_variance"],
            "finite_q_target_variance": facts["finite_q_target_variance"],
            "source_variance_over_finite_q": source_variance / facts["finite_q_source_variance"],
            "target_variance_over_finite_q": target_variance / facts["finite_q_target_variance"],
            "stable_error": (estimate - theta) / (facts["rho"] * facts["q"]),
            "oracle_estimate": oracle_estimate,
            "oracle_se": oracle_se,
            "oracle_known_variance_se": np.sqrt(facts["V"] / n),
            "oracle_source_variance": oracle_source_variance,
            "oracle_target_variance": oracle_target_variance,
            "oracle_variance_ratio": 2
            * (oracle_source_variance + oracle_target_variance)
            / facts["V"],
            "oracle_root_n_error": np.sqrt(n) * (oracle_estimate - theta),
            "finite_sample_identity_error": finite_sample_identity,
            **components,
            **additional_components,
        }
        normalization = n ** (-0.5) + facts["rho"] * facts["q"]
        for name, value in {**components, **additional_components}.items():
            row["root_n_" + name] = np.sqrt(n) * value
            row["stable_" + name] = value / (facts["rho"] * facts["q"])
            row["combined_scale_" + name] = value / normalization
        for evaluation, d in enumerate(diagnostics):
            training = 1 - evaluation
            _, scales, coefficient, info = fitted[training]
            for name, value in {**d, **info}.items():
                row[f"fold{evaluation}_{name}"] = value
            row[f"fold{evaluation}_regression_coefficient"] = json.dumps(coefficient.tolist())
            row[f"fold{evaluation}_shape_coefficient"] = json.dumps(fitted[training][0].tolist())
            row[f"fold{evaluation}_source_pilot_counts"] = json.dumps(
                source[training].pilot_counts.tolist()
            )
            row[f"fold{evaluation}_target_pilot_counts"] = json.dumps(
                target[training].pilot_counts.tolist()
            )
            for name, center in (
                ("scale_source_v2", d["scale_denominator_center"]),
                ("scale_target_v", d["scale_numerator_center"]),
            ):
                row[f"fold{evaluation}_{name}_stable"] = (info[name] - center) / facts["rho"]
            row[f"fold{evaluation}_scale_positive"] = float(scales[arm] > 0)
            row[f"fold{evaluation}_source_observations"] = source[evaluation].n
            row[f"fold{evaluation}_target_observations"] = target[evaluation].n
        if not np.isfinite([v for v in row.values() if isinstance(v, (float, int))]).all():
            raise ArithmeticError("nonfinite E-27 result")
        records[arm] = row
    return records


def replicate(task):
    cell, rep, entropy = task
    if entropy != CONFIRMATORY_ENTROPY:
        raise ValueError("E-27 uses its fixed confirmatory entropy")
    b, n = CELLS[cell]
    facts = population(b, n)
    collected = [collect_fold(cell, rep, fold) for fold in range(K)]
    source, target, oracle_sums = ([item[j] for item in collected] for j in range(3))
    fitted = [nuisance(source[j], target[j], facts["lambda"]) for j in range(K)]
    records = score_records(source, target, fitted, oracle_sums, facts)
    for row in records.values():
        for fold in range(K):
            for coordinate, name in enumerate(("source_U", "source_Z", "target_U", "target_Z")):
                row[f"fold{fold}_{name}_min"] = float(collected[fold][3][coordinate])
                row[f"fold{fold}_{name}_max"] = float(collected[fold][4][coordinate])
    return records


def tasks_for():
    return [(cell, rep, CONFIRMATORY_ENTROPY) for cell in range(len(CELLS)) for rep in range(R)]


def raw_rows(task, result):
    cell, rep, entropy = task
    b, n = CELLS[cell]
    facts = population(b, n)
    constants = {
        key: facts[key]
        for key in (
            "L",
            "M",
            "H",
            "lambda",
            "q",
            "rho",
            "theta0",
            "mu",
            "A0",
            "D_s",
            "stable_index",
            "V",
        )
    }
    return [
        {
            "cell": cell,
            "rep": rep,
            "entropy": entropy,
            "b": b,
            "n": n,
            "arm": arm,
            **constants,
            **result[arm],
        }
        for arm in ARM_LABELS
    ]


def _quantiles(row, name, values):
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        raise ValueError(f"nonfinite values in {name}")
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        key = f"{name}_q{int(100 * q):02}"
        row[key] = float(np.quantile(values, q))
        row[key + "_mc_lo"], row[key + "_mc_hi"] = quantile_interval(values, q)


def summarise(raw):
    expected = set(itertools.product(range(len(CELLS)), ARM_LABELS))
    if set(zip(raw.cell, raw.arm, strict=True)) != expected:
        raise ValueError("E-27 aggregation requires every condition and both losses")
    rows = []
    for cell, (b, n) in enumerate(CELLS):
        facts = population(b, n)
        for arm in ARM_LABELS:
            frame = raw[(raw.cell == cell) & raw.arm.eq(arm)]
            if len(frame) != R or set(frame.rep) != set(range(R)) or not frame.n.eq(n).all():
                raise ValueError("E-27 aggregation requires exactly replications 0 through 1999")
            if not frame.status.eq("ok").all():
                raise ValueError("E-27 has a failed replication; aggregation cannot exclude it")
            numeric = frame.select_dtypes(include=[np.number])
            if not np.isfinite(numeric.to_numpy()).all():
                raise ValueError("E-27 aggregation requires finite numerical results")
            errors = frame.estimate.to_numpy() - facts["theta0"]
            row = {
                "cell": cell,
                "b": b,
                "n": n,
                "L": facts["L"],
                "M": facts["M"],
                "H": facts["H"],
                "arm": arm,
                "R": R,
                "theta0": facts["theta0"],
                "q": facts["q"],
                "rho": facts["rho"],
                "kappa": facts["kappa"],
                "gaussian_stable_coverage_reference": facts["gaussian_stable_coverage_reference"],
                "source_variance_limit": facts["source_variance"],
                "target_variance_limit": facts["target_variance"],
                "bias": float(errors.mean()),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "interval_length": float(2 * Z_975 * frame.se.mean()),
            }
            for label, estimate, standard_error in (
                ("coverage", "estimate", "se"),
                ("known_variance_coverage", "estimate", "oracle_known_variance_se"),
                ("oracle_coverage", "oracle_estimate", "oracle_se"),
                ("oracle_known_variance_coverage", "oracle_estimate", "oracle_known_variance_se"),
            ):
                count = int(
                    (
                        np.abs(frame[estimate] - facts["theta0"]) <= Z_975 * frame[standard_error]
                    ).sum()
                )
                interval = stats.binomtest(count, R).proportion_ci(method="exact")
                row[label], row[label + "_mc_lo"], row[label + "_mc_hi"] = (
                    count / R,
                    float(interval.low),
                    float(interval.high),
                )
            excluded = {
                "cell",
                "rep",
                "entropy",
                "b",
                "n",
                "L",
                "M",
                "H",
                "lambda",
                "q",
                "rho",
                "theta0",
                "mu",
                "A0",
                "D_s",
                "stable_index",
                "V",
            }
            for name in numeric.columns:
                if name not in excluded:
                    _quantiles(row, name, frame[name])
            rows.append(row)
    return pd.DataFrame(rows)


def tables(summary):
    groups = {
        "tab_E27": [
            "b",
            "n",
            "arm",
            "coverage",
            "coverage_mc_lo",
            "coverage_mc_hi",
            "known_variance_coverage",
            "oracle_coverage",
            "interval_length",
            "variance_ratio_q50",
        ],
        "tab_E27_variance": [
            "b",
            "n",
            "arm",
            "source_variance_ratio_q50",
            "target_variance_ratio_q50",
            "oracle_variance_ratio_q50",
        ],
        "tab_E27_normalization": [
            "b",
            "n",
            "arm",
            "root_n_error_q05",
            "root_n_error_q50",
            "root_n_error_q95",
            "stable_error_q05",
            "stable_error_q50",
            "stable_error_q95",
        ],
        "tab_E27_diagnostics": [
            "b",
            "n",
            "arm",
            "fold0_full_norm_product_q50",
            "fold1_full_norm_product_q50",
            "fold0_directional_over_q_q50",
            "fold1_directional_over_q_q50",
            "finite_sample_identity_error_q50",
        ],
        "tab_E27_decomposition": [
            "b",
            "n",
            "arm",
            "population_drift_q50",
            "leading_drift_q50",
            "ridge_drift_q50",
            "pilot_shape_drift_q50",
            "coefficient_shape_cross_q50",
            "evaluation_remainder_q50",
        ],
    }
    return {
        name: summary[columns].to_latex(index=False, float_format="%.5g")
        for name, columns in groups.items()
    }


def figures(summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main, axes = plt.subplots(2, 3, figsize=(11, 6), sharex="col")
    norms, norm_axes = plt.subplots(3, 3, figsize=(11, 8), sharex="col")
    decomposition, drift_axes = plt.subplots(2, 3, figsize=(11, 6), sharex="col")
    for col, b in enumerate(B_VALUES):
        for arm in ARM_LABELS:
            frame = summary[summary.b.eq(b) & summary.arm.eq(arm)].sort_values("n")
            for axis, field in ((axes[0, col], "coverage"), (axes[1, col], "variance_ratio_q50")):
                axis.plot(frame.n, frame[field], "o-", label=arm)
            axes[0, col].fill_between(
                frame.n, frame.coverage_mc_lo, frame.coverage_mc_hi, alpha=0.12
            )
            for row, field in enumerate(
                ("root_n_error", "stable_error", "fold0_full_norm_product")
            ):
                norm_axes[row, col].plot(frame.n, frame[field + "_q50"], "o-", label=arm)
                norm_axes[row, col].fill_between(
                    frame.n, frame[field + "_q05"], frame[field + "_q95"], alpha=0.12
                )
        sq = summary[summary.b.eq(b) & summary.arm.eq("SQ")].sort_values("n")
        axes[0, col].plot(sq.n, sq.oracle_coverage, "k--", label="Oracle")
        axes[0, col].plot(
            sq.n,
            sq.gaussian_stable_coverage_reference,
            "k-.",
            label="SQ Gaussian + stable reference",
        )
        for arm in ARM_LABELS:
            frame = summary[summary.b.eq(b) & summary.arm.eq(arm)].sort_values("n")
            axes[0, col].plot(
                frame.n, frame.known_variance_coverage, ":", label=arm + " with known V"
            )
        for field, label in (
            ("population_drift", "Population drift"),
            ("leading_drift", "Leading term"),
            ("ridge_drift", "Ridge estimation term"),
            ("pilot_shape_drift", "Shape term"),
            ("coefficient_shape_cross", "Scale and shape interaction"),
            ("evaluation_remainder", "Evaluation remainder"),
        ):
            for row, prefix in enumerate(("root_n_", "stable_")):
                drift_axes[row, col].plot(sq.n, sq[prefix + field + "_q50"], "o-", label=label)
        axes[0, col].set(title=f"$b={b}$", ylabel="Coverage", ylim=(0, 1.02))
        axes[0, col].axhline(0.95, color="gray", linewidth=0.8)
        axes[1, col].set(ylabel="Variance ratio")
        axes[1, col].axhline(1, color="gray", linewidth=0.8)
        for row, label in enumerate(
            (
                r"$\sqrt{N}(\widehat\theta-\theta_0)$",
                r"$(\widehat\theta-\theta_0)/(\rho_Lq_L)$",
                "Full norm product",
            )
        ):
            norm_axes[row, col].set(ylabel=label)
        for row, label in enumerate(("Root N drift", "Stable normalized drift")):
            drift_axes[row, col].set(ylabel=label)
        for matrix in (axes, norm_axes, drift_axes):
            for axis in matrix[:, col]:
                axis.set_xscale("log")
                axis.set_xlabel("$N$")
        norm_axes[0, col].set_title(f"$b={b}$")
        drift_axes[0, col].set_title(f"$b={b}$")
    axes[0, 0].legend(fontsize=8)
    norm_axes[0, 0].legend(fontsize=8)
    drift_axes[0, 0].legend(fontsize=6)
    for fig in (main, norms, decomposition):
        fig.tight_layout()
    return {"fig_E27": main, "fig_E27_norms": norms, "fig_E27_decomposition": decomposition}
