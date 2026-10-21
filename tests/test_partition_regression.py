"""Independent dense-design checks for local regression and count-smoothed GRR."""

import numpy as np
import pytest
from scipy.optimize import minimize

import genriesz.partition_regression as partition


def dense_basis(u, h):
    result = np.zeros((len(u), 4 * h))
    for j in range(h):
        inside = (u >= j / h) & ((u < (j + 1) / h) | ((j == h - 1) & (u == 1)))
        coordinate = 2 * (h * u[inside] - j) - 1
        values = np.polynomial.legendre.legvander(coordinate, 3)
        result[inside, 4 * j : 4 * j + 4] = values * np.sqrt(h * np.array([1, 3, 5, 7]))
    return result


def test_basis_is_lebesgue_orthonormal_with_unknown_source_gram():
    h = 5
    nodes, weights = np.polynomial.legendre.leggauss(8)
    u = ((np.arange(h)[:, None] + (nodes + 1) / 2) / h).ravel()
    w = np.tile(weights / (2 * h), h)
    phi = dense_basis(u, h)
    np.testing.assert_allclose(phi.T @ (w[:, None] * phi), np.eye(4 * h), atol=5e-15)
    gram = phi.T @ ((w * 2 * (1 + u) / 3)[:, None] * phi)
    assert not np.allclose(gram, np.eye(4 * h))
    assert 2 / 3 < np.linalg.eigvalsh(gram).min() < np.linalg.eigvalsh(gram).max() < 4 / 3
    np.testing.assert_array_equal(partition.local_basis(np.array([0, 0.2, 1]), h)[0], [0, 1, 4])


@pytest.mark.parametrize("penalty", [0.0, 1e-6, 0.17, 1.0])
def test_block_fit_and_prediction_equal_direct_dense_criterion(penalty):
    rng = np.random.default_rng(13497)
    u = rng.beta(1.3, 1, 347)
    y = 3 * u + np.cos(3 * u) + rng.normal(size=len(u))
    h = 4
    phi = dense_basis(u, h)
    moment = partition.regression_moments(u, y, h)
    coef, checks = partition.fit_regression(moment, penalty)
    d = np.diag(np.tile([0, 1, 1, 1], h))
    matrix = phi.T @ phi / len(y) + penalty * d
    expected = np.linalg.lstsq(matrix, phi.T @ y / len(y), rcond=None)[0]
    np.testing.assert_allclose(coef.ravel(), expected, atol=3e-14)
    np.testing.assert_allclose(partition.predict(u, coef), phi @ expected, atol=8e-14)
    assert checks["normal_rank"] == 4 * h
    assert checks["normal_equation_scaled"] < 1e-14
    if penalty == 0:
        np.testing.assert_allclose(coef.ravel(), np.linalg.lstsq(phi, y, rcond=None)[0], atol=3e-14)


@pytest.mark.parametrize("penalty", [0, 0.2])
def test_rank_deficient_samples_and_empty_cells_use_the_minimum_norm_solution(penalty):
    u = np.r_[np.full(11, 0.1), np.full(7, 0.8)]
    y = np.arange(len(u), dtype=float)
    h = 4
    phi = dense_basis(u, h)
    coef, checks = partition.fit_regression(partition.regression_moments(u, y, h), penalty)
    d = np.diag(np.tile([0, 1, 1, 1], h))
    expected = np.linalg.lstsq(phi.T @ phi / len(y) + penalty * d, phi.T @ y / len(y), rcond=None)[
        0
    ]
    np.testing.assert_allclose(coef.ravel(), expected, atol=5e-13)
    np.testing.assert_array_equal(coef[1:3], np.zeros((2, 4)))
    assert checks["normal_equation_scaled"] < 1e-13
    if penalty == 0:
        np.testing.assert_allclose(coef.ravel(), np.linalg.lstsq(phi, y, rcond=None)[0], atol=2e-14)


def test_cv_matches_actual_held_out_predictions_and_full_refit():
    rng = np.random.default_rng(71038)
    u = rng.uniform(size=209)
    y = np.sin(8 * u) + rng.normal(size=len(u))
    h, split = 3, 93
    penalties = np.array([0, 1e-6, 0.003, 0.1, 1])
    parts = (slice(0, split), slice(split, len(u)))
    moments = [partition.regression_moments(u[s], y[s], h) for s in parts]
    selected, scores = partition.select_penalty(moments, penalties)
    direct = []
    for penalty in penalties:
        error = 0.0
        for validation in range(2):
            train, valid = parts[1 - validation], parts[validation]
            x = dense_basis(u[train], h)
            coef = np.linalg.solve(
                x.T @ x / len(x) + penalty * np.diag(np.tile([0, 1, 1, 1], h)),
                x.T @ y[train] / len(x),
            )
            error += np.sum((y[valid] - dense_basis(u[valid], h) @ coef) ** 2)
        direct.append(error / len(u))
    np.testing.assert_allclose(scores, direct, atol=3e-14)
    assert selected == penalties[np.argmin(direct)]
    refit, _ = partition.fit_regression(partition.regression_moments(u, y, h), selected)
    x = dense_basis(u, h)
    expected = np.linalg.solve(
        x.T @ x / len(x) + selected * np.diag(np.tile([0, 1, 1, 1], h)), x.T @ y / len(x)
    )
    np.testing.assert_allclose(refit.ravel(), expected, atol=3e-14)


def test_cv_exact_ties_select_the_smallest_candidate():
    u = np.linspace(0.01, 0.99, 42)
    moments = partition.regression_moments(u, np.zeros(len(u)), 4)
    selected, scores = partition.select_penalty([moments, moments], [0, 0.03, 1])
    assert selected == 0
    np.testing.assert_array_equal(scores, [0, 0, 0])


def test_both_ratio_objectives_have_the_count_smoothed_positive_minimizer():
    source = np.array([0.1, 0.2, 0.4, 0.8, 0.9])
    target = np.array([0.1, 0.1, 0.15, 0.16, 0.7, 0.7, 0.9])
    h = 8
    ratio, checks = partition.fit_ratio(source, target, h)
    s = (np.histogram(source, bins=np.linspace(0, 1, h + 1))[0] + 1) / len(source)
    t = (np.histogram(target, bins=np.linspace(0, 1, h + 1))[0] + 1) / len(target)
    np.testing.assert_array_equal(ratio, t / s)
    for loss in ("SQ", "UKL"):
        objective = (
            (lambda a: np.sum(s * a**2 - 2 * t * a))
            if loss == "SQ"
            else (lambda a: np.sum(s * a - t * np.log(a)))
        )
        solution = minimize(objective, ratio * 1.1, method="BFGS", tol=1e-9)
        np.testing.assert_allclose(solution.x, ratio, atol=3e-7)
        assert objective(ratio) < objective(ratio * 1.01)
        assert checks[loss + "_stationarity"] < 1e-14
    assert checks["source_empty"] > 0 and checks["target_empty"] > 0
    assert np.min(ratio) > 0


@pytest.mark.parametrize(
    "u,h", [([], 2), ([np.nan], 2), ([-0.1], 2), ([1.1], 2), ([0.5], 0), ([0.5], 2.1)]
)
def test_basis_rejects_invalid_inputs(u, h):
    with pytest.raises(ValueError):
        partition.local_basis(u, h)


@pytest.mark.parametrize("penalty", [-1, np.nan, np.inf])
def test_fit_rejects_invalid_penalties(penalty):
    moments = partition.regression_moments(np.linspace(0, 1, 20), np.zeros(20), 2)
    with pytest.raises(ValueError):
        partition.fit_regression(moments, penalty)


@pytest.mark.parametrize("penalties", [[0], [0.1, 1], [0, 0.1, 0.1], [0, np.nan], [0, -1]])
def test_cv_rejects_invalid_candidate_sets(penalties):
    moments = partition.regression_moments(np.linspace(0, 1, 20), np.zeros(20), 2)
    with pytest.raises(ValueError):
        partition.select_penalty([moments, moments], penalties)


def test_fitted_equation_guard_checks_the_actual_coefficients(monkeypatch):
    moments = partition.regression_moments(np.linspace(0, 1, 20), np.ones(20), 2)
    monkeypatch.setattr(
        partition.np.linalg, "lstsq", lambda *args, **kwargs: (np.zeros(4), [], 4, [])
    )
    with pytest.raises(ArithmeticError, match="normal equations"):
        partition.fit_regression(moments, 0)


def test_cv_rejects_negative_squared_errors_without_clipping_or_a_fallback():
    moments = partition.regression_moments(np.linspace(0, 1, 20), np.ones(20), 2)
    inconsistent = {**moments, "yy": 0.0}
    with pytest.raises(ArithmeticError, match="squared validation error"):
        partition.select_penalty([inconsistent, inconsistent], [0, 0.1, 1])
