from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest
from scipy import optimize
from scipy.optimize._numdiff import approx_derivative

from genriesz import att_histogram as att


@pytest.mark.parametrize("D", [
    [0, 1, 0, 1, 0, 1], [0, 0, 0, 1, 1, 1], [0] * 6, [1] * 6,
])
def test_cell_ols_matches_dense_minimum_norm(D):
    T = np.array([[.1, .1], [.2, .1], [.1, .6], [.2, .7], [.7, .7], [.8, .9]])
    Y = np.array([2.0, -1.0, 4.0, 3.0, 8.0, 5.0])
    fit = att.fit_cell_ols(T, D, Y, 2)
    ids = att.cell_ids(T, 2)
    design = np.column_stack([D, np.eye(4)[ids]])
    expected, _, rank, _ = np.linalg.lstsq(design, Y, rcond=None)
    np.testing.assert_allclose([fit["tau"], *fit["coefficients"]], expected, atol=2e-14)
    np.testing.assert_allclose(att.predict_cell_ols(fit, T, D, 2), design @ expected, atol=2e-14)
    assert fit["rank"] == rank and fit["empty_cells"] == 1
    assert fit["coefficients"][2] == 0


def test_cell_ols_pure_mixed_cells_and_integer_indexing():
    T = np.array([[.1, .1], [.1, .2], [.1, .7], [.7, .1], [.7, .7]])
    D = np.array([0, 0, 1, 1, 0])
    Y = np.array([1, 3, 8, 12, 5.0])
    ids = att.cell_ids(T, 2)
    np.testing.assert_array_equal(ids, [0, 0, 1, 2, 3])
    fit = att.fit_cell_ols(T, D, Y, 2)
    expected = np.linalg.lstsq(np.column_stack([D, np.eye(4)[ids]]), Y, rcond=None)[0]
    np.testing.assert_allclose([fit["tau"], *fit["coefficients"]], expected, atol=1e-14)
    assert fit["rank"] == 4
    with pytest.raises(ValueError, match="coordinates"):
        att.cell_ids([[1.0, .5]], 2)


def _specimen():
    V = np.array([0.0, .1, .3, .7, 1.0, 2.0, 4.0, 7.0])
    return np.exp(-V), np.array([0, 1, 0, 1, 1, 0, 1, 0.0])


@pytest.mark.parametrize("arm", ["SQ", "UKL0"])
def test_objective_gradient_and_profile(arm):
    U, D = _specimen()
    coefficient = np.array([.2, .43])
    actual, gradient = att.odds_objective(coefficient, U, D, arm)
    t = coefficient[0] - coefficient[1] * np.log(U)
    values = ((1 - D) * np.exp(2 * t) - 2 * D * np.exp(t)) if arm == "SQ" else (
        (1 - D) * np.exp(t) - D * t
    )
    assert actual == pytest.approx(values.mean(), rel=1e-14)
    numerical = approx_derivative(
        lambda c: att.odds_objective(c, U, D, arm)[0], coefficient
    ).ravel()
    np.testing.assert_allclose(gradient, numerical, rtol=2e-8)
    profile = att._OddsProfile(U, D, arm, ((-1, 1), (.2, .8)))
    for b in (.2, .4, .8):
        value, derivative, a = profile.at(b)
        direct, grad = att.odds_objective([a, b], U, D, arm)
        assert value == pytest.approx(direct, abs=2e-13)
        assert derivative == pytest.approx(grad[1], abs=2e-12)


def test_sq_analytic_interval_bound_against_high_precision_objective():
    U, D = _specimen()
    profile = att._OddsProfile(U, D, "SQ", ((-1, 1), (.2, .8)))
    with mp.workdps(60):
        for lo, hi in ((.2, .8), (.3, .45), (.7, .71)):
            bound = att._sq_bound(profile, lo, hi)
            for fraction in (0, .125, .5, .875, 1):
                b = lo + fraction * (hi - lo)
                for a in (-1, -.35, .2, 1):
                    value = mp.fsum(
                        (1 - int(d)) * mp.exp(2 * (mp.mpf(a) + mp.mpf(b) * mp.mpf(float(v))))
                        - 2 * int(d) * mp.exp(mp.mpf(a) + mp.mpf(b) * mp.mpf(float(v)))
                        for d, v in zip(D, profile.V, strict=True)
                    ) / len(D)
                    assert bound <= float(value) + 2e-12 * (1 + abs(float(value)))


def test_sq_global_profile_with_two_stationary_points():
    V = np.array([0.0] * 5 + [4.0] + [1.0] * 400 + [6.0])
    D = np.array([1.0] * 6 + [0.0] * 401)
    U = np.exp(-V)
    bounds = ((-10.0, 10.0), (0.0, .9))
    profile = att._OddsProfile(U, D, "SQ", bounds)
    grid = np.linspace(0, .9, 201)
    critical = [0.0, .9]
    for lo, hi in zip(grid[:-1], grid[1:], strict=True):
        if profile.at(lo)[1] * profile.at(hi)[1] < 0:
            critical.append(optimize.brentq(lambda b: profile.at(float(b))[1], lo, hi))
    assert len(critical) == 4
    expected = min(profile.at(b)[0] for b in critical)
    coefficient, info = att.fit_att_odds(U, D, "SQ", bounds)
    assert info["status"] == "ok", info
    assert info["objective"] == pytest.approx(expected, abs=1e-12)
    assert info["gap"] <= att.OBJECTIVE_TOL * (1 + abs(expected))
    assert info["gap"] >= -info["bound_roundoff_tolerance"]
    assert info["stationarity_scaled"] <= att.GRADIENT_TOL
    assert bounds[1][0] <= coefficient[1] <= bounds[1][1]


@pytest.mark.parametrize("arm", ["SQ", "UKL0"])
def test_fitting_global_objective_and_projected_gradient(arm):
    U, D = _specimen()
    bounds = ((-1.0, 1.0), (.2, .8))
    coefficient, info = att.fit_att_odds(U, D, arm, bounds)
    assert info["status"] == "ok", info
    independent = optimize.shgo(
        lambda c: att.odds_objective(c, U, D, arm)[0], bounds, n=128, iters=2,
        minimizer_kwargs={"method": "SLSQP", "options": {"ftol": 1e-13}},
    )
    assert independent.success
    assert info["objective"] == pytest.approx(independent.fun, rel=1e-9, abs=1e-11)
    _, gradient = att.odds_objective(coefficient, U, D, arm)
    projected, active = att._projected(gradient, coefficient, np.array(bounds))
    np.testing.assert_allclose(info["projected_gradient"], projected, atol=0)
    assert info["active_bounds"] == active


@pytest.mark.parametrize("arm", ["SQ", "UKL0"])
def test_exact_interior_odds_fit(arm):
    V = np.repeat([0., 1., 2.], [8, 12, 20])
    D = np.array([1.] * 4 + [0.] * 4 + [1.] * 8 + [0.] * 4 + [1.] * 16 + [0.] * 4)
    coefficient, info = att.fit_att_odds(np.exp(-V), D, arm, ((-1, 1), (.2, .8)))
    assert info["status"] == "ok", info
    np.testing.assert_allclose(coefficient, [0, np.log(2)], atol=2e-8)
    assert not info["boundary"]


@pytest.mark.parametrize("arm", ["SQ", "UKL0"])
def test_convex_ukl_and_exact_constant_v_tie_rule(arm):
    U = np.exp(-np.full(6, .5))
    D = np.array([1, 1, 0, 0, 0, 0])
    bounds = ((-1.0, 1.0), (.2, .8))
    coefficient, info = att.fit_att_odds(U, D, arm, bounds)
    assert info["status"] == "ok", info
    assert coefficient[1] == .2
    assert coefficient[0] == pytest.approx(np.log(.5) - .1)
    if arm == "SQ":
        assert info["objective"] == pytest.approx(-1 / 6)
    else:
        profile = att._OddsProfile(*_specimen(), arm, bounds)
        derivative = [profile.at(float(b))[1] for b in np.linspace(.2, .8, 30)]
        assert np.min(np.diff(derivative)) >= 0


@pytest.mark.parametrize("arm", ["SQ", "UKL0"])
@pytest.mark.parametrize("all_treated", [False, True])
def test_single_treatment_sample_exact_endpoint(arm, all_treated):
    U = np.exp(-np.array([.2, .5, 1.0]))
    D = np.full(3, float(all_treated))
    coefficient, info = att.fit_att_odds(U, D, arm, ((-1, 1), (.2, .8)))
    np.testing.assert_array_equal(coefficient, [1, .8] if all_treated else [-1, .2])
    assert info["status"] == "ok" and info["boundary"]
    coefficient, info = att.fit_att_odds(np.ones(3), D, arm, ((-1, 1), (.2, .8)))
    assert coefficient[1] == .2 and info["status"] == "ok"


def test_solver_reports_unmet_gap_and_negative_bound(monkeypatch):
    U, D = _specimen()
    monkeypatch.setattr(att, "MAX_NODES", 1)
    multi_v = np.array([0.0] * 5 + [4.0] + [1.0] * 400 + [6.0])
    multi_d = np.array([1.0] * 6 + [0.0] * 401)
    _, info = att.fit_att_odds(np.exp(-multi_v), multi_d, "SQ", ((-10, 10), (0, .9)))
    assert info["status"] == "global_gap_not_met"
    profile = att._OddsProfile(U, D, "SQ", ((-1, 1), (.2, .8)))
    coef = np.array([0, .4])
    value, _ = profile.objective(coef)
    invalid = att._fit_info(profile, coef, value + 1e-3, 1)
    assert invalid["status"] == "numerical_bound_failure" and invalid["negative_gap"]
    _, overflow = att.fit_att_odds([1e-300, .5], [0, 1], "SQ", ((-1, 1), (.2, .8)))
    assert overflow["status"] == "nonfinite"


def test_att_recentered_score_and_empirical_share():
    D = np.array([1., 0, 1, 0, 0])
    Y = np.array([4., 1, 7, -2, 3])
    odds = np.array([1., 2, 3, 4, 5])
    g0 = np.array([.5, -.5, 1, 0, 2])
    g1 = g0 + np.array([1., 1.5, 2, .5, .25])
    g = D * g1 + (1 - D) * g0
    result = att.att_score_record(D, Y, odds, g, g1, g0)
    pi = D.mean()
    alpha = (D - (1 - D) * odds) / pi
    uncentered = D * (g1 - g0) / pi + alpha * (Y - g)
    influence = uncentered - D * uncentered.mean() / pi
    assert influence.mean() == pytest.approx(0, abs=1e-14)
    assert result["estimate"] == pytest.approx(uncentered.mean())
    assert result["variance"] == pytest.approx(np.mean(influence**2))
    assert result["se"] == pytest.approx(np.sqrt(np.mean(influence**2) / len(D)))
    empty = att.att_score_record(np.zeros(5), Y, odds, g, g1, g0)
    assert empty == {"estimate": 0, "variance": 0, "se": 0, "status": "no_treated"}
    nonfinite = att.att_score_record([1., 0], [0., 1e200], [1., 1], [0., 0], [0., 0], [0., 0])
    assert nonfinite["status"] == "nonfinite" and np.isnan(nonfinite["se"])
