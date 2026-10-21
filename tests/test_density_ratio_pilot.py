"""Sample-array checks for exponent fitting and regression selection."""

import inspect

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

import genriesz.density_ratio_pilot as pilot
from genriesz.density_ratio_pilot import fit_exponent, fit_outcome, select_penalty
from genriesz.shape_regression import basis


def profile(source, target, exponent, upper):
    return (
        logsumexp(-exponent * np.log(source))
        - np.log(len(source))
        + exponent * np.log(target).mean()
        - (np.log(exponent) + np.log(upper - exponent)) / len(source)
    )


def test_exponent_root_matches_direct_minimization_and_derivatives():
    rng = np.random.default_rng(842)
    source = rng.beta(1.3, 1, 3000)
    target = rng.beta(0.9, 1, 3000)
    upper = 0.49
    exponent, info = fit_exponent(source, target, upper)
    direct = minimize_scalar(
        lambda d: profile(source, target, d, upper),
        bounds=(0, upper),
        method="bounded",
        options={"xatol": 1e-14},
    )
    assert direct.success
    assert exponent == pytest.approx(direct.x, abs=2e-8)
    assert 0 < exponent < upper
    assert info["exponent_criterion"] == pytest.approx(direct.fun, abs=2e-14)
    assert info["exponent_hessian"] > 0
    assert info["exponent_stationarity_scaled"] < 2e-14
    step = 1e-5
    below = profile(source, target, exponent - step, upper)
    center = profile(source, target, exponent, upper)
    above = profile(source, target, exponent + step, upper)
    assert (above - below) / (2 * step) == pytest.approx(info["exponent_gradient"], abs=3e-9)
    assert (above - 2 * center + below) / step**2 == pytest.approx(
        info["exponent_hessian"], rel=2e-5
    )
    assert tuple(inspect.signature(fit_exponent).parameters) == (
        "source_u",
        "target_u",
        "upper",
    )


@pytest.mark.parametrize("target_value", [0.001, 0.5, 0.999])
def test_barrier_has_an_interior_solution_even_for_constant_samples(target_value):
    source, target = np.full(23, 0.5), np.full(23, target_value)
    exponent, info = fit_exponent(source, target)
    assert 0 < exponent < 0.49
    assert np.isfinite(list(info.values())).all()
    assert info["exponent_stationarity_scaled"] < 2e-14
    assert profile(source, target, exponent, 0.49) < profile(source, target, exponent / 2, 0.49)


@pytest.mark.parametrize(
    "source,target,upper",
    [
        ([], [], 0.49),
        ([0.5], [0.5, 0.5], 0.49),
        ([[0.5]], [[0.5]], 0.49),
        ([0], [0.5], 0.49),
        ([0.5], [1], 0.49),
        ([np.nan], [0.5], 0.49),
        ([0.5], [np.inf], 0.49),
        ([0.5], [0.5], 0),
        ([0.5], [0.5], np.inf),
    ],
)
def test_exponent_rejects_invalid_inputs(source, target, upper):
    with pytest.raises(ValueError):
        fit_exponent(source, target, upper)


def test_exponent_rejects_a_finite_interior_nonstationary_root(monkeypatch):
    monkeypatch.setattr(pilot, "brentq", lambda *args, **kwargs: 0.1)
    with pytest.raises(ArithmeticError, match="stationarity check failed"):
        fit_exponent(np.full(23, 0.5), np.full(23, 0.5))


def sample(size, seed):
    rng = np.random.default_rng(seed)
    u, z = rng.uniform(size=(2, size))
    phi = basis(u, z)
    y = phi @ np.array([0.4, 0.9, -0.5, 0.2, 0.1, 0.7]) + rng.normal(size=size)
    moments = {"gram": phi.T @ phi, "xy": phi.T @ y, "yy": y @ y, "size": size}
    return phi, y, moments


@pytest.mark.parametrize("penalty", [0, 0.07, 1])
def test_outcome_fit_matches_direct_normal_equations(penalty):
    phi, y, moments = sample(79, 129)
    coefficient, info = fit_outcome(moments["gram"], moments["xy"], moments["size"], penalty)
    matrix = phi.T @ phi / len(y) + np.diag([0, *([penalty] * 5)])
    expected = np.linalg.solve(matrix, phi.T @ y / len(y))
    np.testing.assert_allclose(coefficient, expected, atol=5e-15)
    assert info["normal_equation_scaled"] < 1e-14
    if penalty == 0:
        np.testing.assert_allclose(coefficient, np.linalg.lstsq(phi, y, rcond=None)[0], atol=5e-15)


def test_rank_deficient_ols_is_the_minimum_norm_solution():
    phi = basis(np.full(11, 0.3), np.full(11, 0.7))
    y = np.linspace(2, 5, 11)
    coefficient, info = fit_outcome(phi.T @ phi, phi.T @ y, len(y), 0)
    expected = np.linalg.lstsq(phi, y, rcond=None)[0]
    np.testing.assert_allclose(coefficient, expected, atol=4e-15)
    assert info["ols_rank"] == 1
    assert info["normal_equation_scaled"] < 1e-14
    ridge, _ = fit_outcome(phi.T @ phi, phi.T @ y, len(y), 0.4)
    np.testing.assert_allclose(ridge, [y.mean(), 0, 0, 0, 0, 0], atol=5e-14)


@pytest.mark.parametrize("penalty", [0, 0.07])
def test_outcome_rejects_finite_coefficients_that_fail_the_equations(monkeypatch, penalty):
    _, _, moments = sample(79, 129)
    if penalty == 0:
        monkeypatch.setattr(
            pilot.np.linalg, "lstsq", lambda *args, **kwargs: (np.zeros(6), [], 6, [])
        )
    else:
        monkeypatch.setattr(
            pilot,
            "fit_ridge",
            lambda *args: (
                np.zeros(6),
                {"normal_equation_scaled": 0.0, "ridge_min_eigenvalue": 1.0},
            ),
        )
    with pytest.raises(ArithmeticError, match="fitted equation"):
        fit_outcome(moments["gram"], moments["xy"], moments["size"], penalty)


@pytest.mark.parametrize(
    "key,value", [("normal_equation_scaled", 2e-12), ("ridge_min_eigenvalue", 0.0)]
)
def test_outcome_rejects_finite_invalid_ridge_diagnostics(monkeypatch, key, value):
    _, _, moments = sample(79, 129)
    coefficient, info = fit_outcome(moments["gram"], moments["xy"], moments["size"], 0.07)
    monkeypatch.setattr(pilot, "fit_ridge", lambda *args: (coefficient, {**info, key: value}))
    with pytest.raises(ArithmeticError, match="fitted equation or positivity"):
        fit_outcome(moments["gram"], moments["xy"], moments["size"], 0.07)


@pytest.mark.parametrize(
    "gram,xy,size,penalty",
    [
        (np.eye(5), np.ones(6), 10, 0),
        (np.eye(6), np.ones(5), 10, 0),
        (np.eye(6), np.ones(6), 0, 0),
        (np.eye(6), np.ones(6), 1.5, 0),
        (np.eye(6), np.ones(6), 10, -0.1),
        (np.eye(6), np.ones(6), 10, np.nan),
        (np.eye(6), np.full(6, np.inf), 10, 0),
        (np.triu(np.ones((6, 6))), np.ones(6), 10, 0),
    ],
)
def test_outcome_rejects_invalid_inputs(gram, xy, size, penalty):
    with pytest.raises(ValueError):
        fit_outcome(gram, xy, size, penalty)


def test_penalty_scores_equal_direct_held_out_prediction_error():
    data = [sample(31, 109), sample(46, 425)]
    penalties = np.array([0, 0.001, 0.03, 0.2, 1])
    selected, scores = select_penalty([row[2] for row in data], penalties)
    direct_scores = []
    for penalty in penalties:
        error = 0.0
        for fold, (validation_phi, validation_y, _) in enumerate(data):
            training_phi, training_y, _ = data[1 - fold]
            matrix = training_phi.T @ training_phi / len(training_y) + np.diag(
                [0, *([penalty] * 5)]
            )
            coefficient = np.linalg.solve(matrix, training_phi.T @ training_y / len(training_y))
            error += np.sum((validation_y - validation_phi @ coefficient) ** 2)
        direct_scores.append(error / sum(len(row[1]) for row in data))
    np.testing.assert_allclose(scores, direct_scores, rtol=3e-15, atol=3e-15)
    assert selected == penalties[np.argmin(direct_scores)]


def test_penalty_ties_choose_zero_without_changing_any_candidate():
    _, _, moments = sample(20, 928)
    moments["xy"] = np.zeros(6)
    moments["yy"] = 0.0
    selected, scores = select_penalty([moments, moments], [0, 0.01, 1])
    assert selected == 0
    np.testing.assert_array_equal(scores, [0, 0, 0])


@pytest.mark.parametrize(
    "penalties", [[0], [0, 0.1], [0.1, 1], [0, 1, 0.1], [0, 0, 1], [0, np.nan, 1]]
)
def test_penalty_selection_rejects_invalid_candidates(penalties):
    _, _, moments = sample(20, 928)
    with pytest.raises(ValueError):
        select_penalty([moments, moments], penalties)


def test_penalty_selection_rejects_invalid_fold_count_or_outcome_sum():
    _, _, moments = sample(20, 928)
    with pytest.raises(ValueError):
        select_penalty([moments], [0, 1])
    for bad in [-1, np.inf, np.nan]:
        with pytest.raises(ValueError):
            select_penalty([moments, {**moments, "yy": bad}], [0, 1])
