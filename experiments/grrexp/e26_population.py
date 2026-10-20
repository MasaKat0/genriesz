"""Population integrals for the four-dimensional ATT histogram example."""

from __future__ import annotations

from functools import cache, lru_cache

import numpy as np
from scipy import integrate, optimize, special

QUAD_TOL = 2e-10


def _quad(function, lo, hi):
    result = integrate.quad(
        function, lo, hi, epsabs=QUAD_TOL, epsrel=QUAD_TOL, limit=250, full_output=True
    )
    if len(result) != 3 or not np.isfinite(result[:2]).all():
        raise ArithmeticError(f"population quadrature failed: {result[3:]}")
    value, error = result[:2]
    if error > QUAD_TOL * max(1.0, abs(value)):
        raise ArithmeticError("population quadrature did not attain its tolerance")
    return float(value)


def primitive(b0, power, upper):
    if power <= -1:
        raise ValueError("the population integral requires power > -1")
    x = np.asarray(upper, float)
    shape = (power + 1) / b0
    value = x ** (power + 1) / (power + 1) * special.hyp2f1(
        1.0, shape, 1 + shape, -(x**b0)
    )
    if not np.isfinite(value).all():
        raise ArithmeticError("nonfinite hypergeometric population integral")
    return value


@lru_cache(maxsize=256)
def bin_moments(b0, j, power=0.0, log_power=0):
    edges = np.arange(j + 1, dtype=float) / j
    if log_power == 0:
        values = np.diff(primitive(b0, power, edges))
    else:
        values = np.array([
            _quad(
                lambda x: np.exp(-(power + 1) * x) * x**log_power
                / (1 + np.exp(-b0 * x)),
                -np.log(hi), -np.log(lo) if lo > 0 else np.inf,
            )
            for lo, hi in zip(edges[:-1], edges[1:], strict=True)
        ])
    if np.any(values <= 0) or not np.isfinite(values).all():
        raise ArithmeticError("invalid positive population bin integral")
    values.setflags(write=False)
    return values


@cache
def grid_moments(j):
    lo, hi = np.arange(j) / j, np.arange(1, j + 1) / j

    def mean_power(power):
        return j * (hi ** (power + 1) - lo ** (power + 1)) / (power + 1)

    mean = mean_power(1) + mean_power(2) / 4
    variance = mean_power(2) + mean_power(3) / 2 + mean_power(4) / 16 - mean**2
    z_mean = (
        mean[:, None, None] + mean[None, :, None] + mean[None, None, :]
    ).ravel()
    return mean, variance, z_mean


@cache
def constants(b0, j):
    length = j**6
    pi = float(primitive(b0, 0.0, 1.0))
    jacobian_moments = [float(bin_moments(b0, j, -b0, m).sum()) for m in range(3)]
    jacobian = 2 / pi**2 * np.array([
        [jacobian_moments[0], jacobian_moments[1]],
        [jacobian_moments[1], jacobian_moments[2]],
    ])
    u_tail = optimize.brentq(
        lambda u: length * float(primitive(b0, b0, u)) - 1,
        0.0, 1.0, xtol=1e-15, rtol=1e-14,
    )
    a_length = 2 / pi**2 * u_tail ** (-2 * b0) * (-np.log(u_tail))
    rho = a_length / length
    directional_scale = np.log(j) / j**2
    return {
        "b0": float(b0), "J": int(j), "L": int(length), "N": int(2 * length),
        "theta0": 1.0, "sigma_squared": 1.0, "pi": pi,
        "variance": 1 / (pi**2 * (1 - b0)),
        "sq_jacobian": jacobian.tolist(),
        "sq_jacobian_inverse": np.linalg.inv(jacobian).tolist(),
        "stable_index": (1 + b0) / (2 * b0), "tail_u": float(u_tail),
        "a_L": float(a_length), "rho": float(rho),
        "directional_scale": float(directional_scale),
        "stable_scale": float(rho * directional_scale),
        "root_n_drift_scale": float(np.sqrt(2 * length) * rho * directional_scale),
        "directional_limit": [0.0, -10 / (12 * pi)],
        "b_norm": 0.5, "b_crit": 5 / 7, "gradient_squared_integral": 1900 / 3,
    }


def regression_diagnostics(coefficients, tau_hat, b0, j):
    cells = np.asarray(coefficients, float).reshape(j, j**3)
    if not np.isfinite(cells).all() or not np.isfinite(tau_hat):
        raise ArithmeticError("nonfinite outcome regression coefficients")
    _, within_variance, z_mean = grid_moments(j)
    average = cells.mean(axis=1)
    centered = cells - 10 * z_mean
    centered_variance = np.mean((centered - centered.mean(axis=1)[:, None]) ** 2, axis=1)
    w = np.column_stack((average - 17.5, np.full(j, -10.0), np.full(j, -2.5)))
    width = 1 / j
    lo, hi = np.arange(j) / j, np.arange(1, j + 1) / j
    uniform = np.array([(hi ** (m + 1) - lo ** (m + 1)) / (m + 1) for m in range(5)])
    square_integral = sum(
        np.polynomial.polynomial.polymul(row, row) @ uniform[:, idx]
        for idx, row in enumerate(w)
    )
    e_w = sum(w[:, m] @ bin_moments(b0, j, float(m)) for m in range(3))
    delta_tau = tau_hat - 1.0
    pi = constants(b0, j)["pi"]
    squared = (
        width * centered_variance.sum() + 300 * within_variance.mean()
        + square_integral + 2 * delta_tau * e_w + delta_tau**2 * pi
    )
    if squared < 0 or not np.isfinite(squared):
        raise ArithmeticError("invalid regression squared L2 error")
    projection = np.array([
        -sum(w[:, m] @ bin_moments(b0, j, float(m), log_power) for m in range(3)) / pi
        for log_power in range(2)
    ])
    return {
        "regression_l2": float(np.sqrt(squared)),
        "directional_error": projection.tolist(),
        "mean_cell_coefficient": average.tolist(),
    }


def population_drift(mean_cell_coefficient, beta, b0, j):
    average = np.asarray(mean_cell_coefficient, float)
    a, b = np.asarray(beta, float)

    def weighted(power):
        return (
            (average - 17.5) @ bin_moments(b0, j, float(power))
            - 10 * bin_moments(b0, j, float(power + 1)).sum()
            - 2.5 * bin_moments(b0, j, float(power + 2)).sum()
        )

    return float((np.exp(a) * weighted(b0 - b) - weighted(0.0)) / constants(b0, j)["pi"])


def representer_l2(beta, b0, pi, empirical_pi):
    if empirical_pi <= 0:
        raise ValueError("representer L2 requires positive empirical treatment share")
    a, b = np.asarray(beta, float)
    rate = 1 + b0 - 2 * max(b, b0)
    if rate <= 0:
        raise ArithmeticError("representer squared error is not integrable")
    log_share = np.log(pi / empirical_pi)
    shift, difference = a + log_share, b - b0
    magnitude = max(abs(np.expm1(shift)), -np.expm1(-abs(difference) / rate))
    treated = np.expm1(log_share) ** 2 / pi
    if magnitude == 0:
        return float(np.sqrt(treated))

    def integrand(v):
        x = v / rate
        contrast = (
            np.expm1(shift) - np.expm1(-difference * x)
            if difference >= 0 else np.expm1(shift + difference * x)
        ) / magnitude
        return np.exp(-v) * contrast**2 / (1 + np.exp(-b0 * x))

    transition_rate = min(b0, abs(difference)) if difference != 0 else b0
    split = min(1.0, -np.log(QUAD_TOL) * rate / transition_rate)
    integral = _quad(integrand, 0.0, split) + _quad(integrand, split, np.inf)
    squared = treated + magnitude**2 * integral / (rate * pi**2)
    if not np.isfinite(squared) or squared < 0:
        raise ArithmeticError("invalid representer squared L2 error")
    return float(np.sqrt(squared))
