"""Marginal UKL exponent estimation and outcome-regression selection."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

from .shape_regression import fit_ridge


def fit_exponent(source_u, target_u, upper=0.49):
    """Fit the profiled marginal UKL criterion with its logarithmic barrier."""
    source, target = np.asarray(source_u, float), np.asarray(target_u, float)
    if (
        source.ndim != 1
        or source.shape != target.shape
        or not len(source)
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
        or np.any((source <= 0) | (source >= 1))
        or np.any((target <= 0) | (target >= 1))
        or not np.isscalar(upper)
        or not np.isfinite(upper)
        or upper <= 0
    ):
        raise ValueError("equal sizes, interior covariates and a positive bound are required")
    x, target_mean = -np.log(source), float(np.mean(-np.log(target)))
    size = len(source)

    def tilted(exponent):
        log_sum = logsumexp(exponent * x)
        weights = np.exp(exponent * x - log_sum)
        mean = float(weights @ x)
        variance = float(weights @ (x - mean) ** 2)
        return float(log_sum - np.log(size)), mean, variance

    def gradient(exponent):
        if exponent == 0:
            return -np.inf
        if exponent == upper:
            return np.inf
        _, mean, _ = tilted(exponent)
        return mean - target_mean - 1 / (size * exponent) + 1 / (size * (upper - exponent))

    exponent = brentq(
        gradient,
        0,
        upper,
        xtol=np.finfo(float).tiny,
        rtol=4 * np.finfo(float).eps,
    )
    if not 0 < exponent < upper:
        raise ArithmeticError("the marginal UKL root is not strictly interior")
    log_mean, mean, variance = tilted(exponent)
    left, right = 1 / (size * exponent), 1 / (size * (upper - exponent))
    derivative = mean - target_mean - left + right
    criterion = (
        log_mean - exponent * target_mean - (np.log(exponent) + np.log(upper - exponent)) / size
    )
    hessian = variance + 1 / (size * exponent**2) + 1 / (size * (upper - exponent) ** 2)
    diagnostics = {
        "exponent_criterion": float(criterion),
        "exponent_gradient": float(derivative),
        "exponent_stationarity_scaled": float(
            abs(derivative) / (1 + abs(mean) + abs(target_mean) + left + right)
        ),
        "exponent_hessian": float(hessian),
        "exponent_upper": float(upper),
    }
    if not np.isfinite(list(diagnostics.values())).all() or hessian <= 0:
        raise ArithmeticError("the marginal UKL objective or derivative is not finite")
    if diagnostics["exponent_stationarity_scaled"] > 1e-12:
        raise ArithmeticError("the marginal UKL stationarity check failed")
    return float(exponent), diagnostics


def fit_outcome(gram, xy, size, penalty):
    """Fit six coefficients by ridge or minimum-norm OLS from sample sums."""
    gram, xy = np.asarray(gram, float), np.asarray(xy, float)
    if (
        not np.isscalar(size)
        or not np.isfinite(size)
        or size <= 0
        or size != int(size)
        or not np.isscalar(penalty)
        or not np.isfinite(penalty)
        or penalty < 0
        or gram.shape != (6, 6)
        or xy.shape != (6,)
        or not np.isfinite(gram).all()
        or not np.isfinite(xy).all()
        or not np.allclose(gram, gram.T, rtol=1e-13, atol=0)
    ):
        raise ValueError(
            "finite symmetric moments, integer size and nonnegative penalty are required"
        )
    matrix = gram / size + np.diag([0, *([penalty] * 5)])
    vector = xy / size
    if penalty > 0:
        coefficient, diagnostics = fit_ridge(gram, xy, size, penalty)
    else:
        coefficient, _, rank, _ = np.linalg.lstsq(matrix, vector, rcond=None)
        diagnostics = {
            "ols_rank": int(rank),
            "ols_min_eigenvalue": float(np.linalg.eigvalsh(matrix).min()),
        }
    residual = float(np.max(np.abs(matrix @ coefficient - vector)) / (1 + np.max(np.abs(vector))))
    reported = diagnostics["normal_equation_scaled"] if penalty > 0 else residual
    if (
        not np.isfinite(coefficient).all()
        or not np.isfinite([residual, *diagnostics.values()]).all()
        or not 0 <= reported <= 1e-12
        or residual > 1e-12
        or (penalty > 0 and diagnostics["ridge_min_eigenvalue"] <= 0)
    ):
        raise ArithmeticError("an outcome-regression fitted equation or positivity check failed")
    diagnostics["normal_equation_scaled"] = residual
    return coefficient, diagnostics


def select_penalty(inner_moments, penalties):
    """Select a ridge penalty by pooled two-fold held-out squared error.

    Each moment dictionary contains the unnormalized sums ``gram``, ``xy``
    and ``yy``, together with its observation count ``size``. The two folds
    exchange training and validation roles. Exact ties choose the smallest
    candidate penalty.
    """
    penalties = np.asarray(penalties, float)
    if (
        penalties.ndim != 1
        or len(penalties) < 2
        or not np.isfinite(penalties).all()
        or penalties[0] != 0
        or penalties[-1] != 1
        or np.any(np.diff(penalties) <= 0)
        or len(inner_moments) != 2
    ):
        raise ValueError(
            "two inner folds and strictly increasing finite penalties from zero to one are required"
        )
    for moments in inner_moments:
        if not np.isscalar(moments["yy"]) or not np.isfinite(moments["yy"]) or moments["yy"] < 0:
            raise ValueError("the sum of squared outcomes must be finite and nonnegative")
    scores = []
    for penalty in penalties:
        squared_error, size = 0.0, 0
        for fold, validation in enumerate(inner_moments):
            training = inner_moments[1 - fold]
            coefficient, _ = fit_outcome(
                training["gram"], training["xy"], training["size"], penalty
            )
            squared_error += float(
                validation["yy"]
                - 2 * coefficient @ np.asarray(validation["xy"], float)
                + coefficient @ np.asarray(validation["gram"], float) @ coefficient
            )
            size += validation["size"]
        scores.append(squared_error / size)
    scores = np.asarray(scores)
    if not np.isfinite(scores).all():
        raise ArithmeticError("the held-out prediction scores are not finite")
    return float(penalties[np.argmin(scores)]), scores
