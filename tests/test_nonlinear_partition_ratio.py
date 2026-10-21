from __future__ import annotations

import numpy as np
import pytest
from scipy import optimize
from scipy.optimize._numdiff import approx_derivative

from genriesz import nonlinear_partition_ratio as nr


def _observations():
    sv = np.array([0., .1, .3, .7, 1., 2., 4., 7.])
    tv = np.array([.2, .4, .5, .6, 1., 1.5, 2., 3.])
    sz = np.array([.1, .2, .3, .4, .6, .7, .8, .9])
    tz = np.array([.1, .2, .6, .7, .8, .9, .8, .9])
    return np.exp(-sv), sz, np.exp(-tv), tz


@pytest.mark.parametrize("loss", ["SQ", "UKL"])
@pytest.mark.parametrize("penalty", [0., .3])
def test_complete_objective_normalization_and_gradient(loss, penalty):
    su, sz, tu, tz = _observations()
    beta = np.array([[-.4, .2], [-.1, .4]])
    sr = nr.predict_partition_ratio(beta, su, sz)
    ti = beta[(2 * tz).astype(int), 0] - beta[(2 * tz).astype(int), 1] * np.log(tu)
    expected = (np.mean(sr**2) - 2 * np.mean(np.exp(ti))) if loss == "SQ" else (
        np.mean(sr) - np.mean(ti)
    )
    expected += penalty * np.sum(beta[:, 1]**2) / 4
    value, gradient = nr.partition_ratio_objective(beta, su, sz, tu, tz, loss, penalty)
    assert value == pytest.approx(expected, rel=2e-15)
    numerical = approx_derivative(
        lambda b: nr.partition_ratio_objective(b.reshape(2, 2), su, sz, tu, tz, loss, penalty)[0],
        beta.ravel(),
    ).ravel()
    np.testing.assert_allclose(gradient.ravel(), numerical, rtol=3e-8, atol=1e-8)


@pytest.mark.parametrize("loss", ["SQ", "UKL"])
def test_profile_exact_intercept_and_penalty_derivative(loss):
    su, _, tu, _ = _observations()
    p = nr._CellProfile(su[:3], tu[:5], loss, 4., .7)
    for b in (.05, .2, .49):
        denominator = np.exp((2 if loss == "SQ" else 1) * b * p.vc).sum() / 4
        numerator = np.exp(b * p.vt).sum() / 4 if loss == "SQ" else 5 / 4
        expected = np.log(np.clip(numerator / denominator, p.clo, p.chi))
        value, derivative, a = p.at(b)
        assert a == pytest.approx(expected)
        actual, grad = p.objective([a, b])
        assert value == pytest.approx(actual, abs=2e-13)
        assert derivative == pytest.approx(grad[1], abs=2e-13)
        assert derivative == pytest.approx(
            approx_derivative(lambda x: p.at(float(x[0]))[0], [b]).item(), rel=3e-8,
        )


@pytest.mark.parametrize("loss", ["SQ", "UKL"])
def test_exact_nonlinear_cell_fit(loss):
    su = np.exp(-np.repeat([0., 2., 4.], [4, 4, 4]))
    tu = np.exp(-np.repeat([0., 2., 4.], [1, 2, 4]))
    beta, info = nr._fit_cell(su, tu, loss, 12, 0, nr.DEFAULT_BOUNDS)
    assert info["status"] == "ok", info
    np.testing.assert_allclose(beta, [np.log(.25), np.log(2) / 2], atol=2e-8)
    assert not info["boundary"]


@pytest.mark.parametrize("penalty", [0., .03, 4.])
@pytest.mark.parametrize("empty", ["none", "source", "target", "both"])
def test_sq_lower_bounds_against_dense_profile(penalty, empty):
    su, _, tu, _ = _observations()
    p = nr._CellProfile([] if empty in ("source", "both") else su,
                        [] if empty in ("target", "both") else tu, "SQ", 5, penalty)
    for lo, hi in ((.05, .49), (.13, .24), (.4, .401)):
        bound = nr._sq_bound(p, lo, hi)
        values = [p.at(float(b))[0] for b in np.linspace(lo, hi, 151)]
        assert bound <= min(values) + 3e-12 * (1 + max(abs(x) for x in values))


@pytest.mark.parametrize("loss", ["SQ", "UKL"])
@pytest.mark.parametrize("penalty", [0., .3, 40.])
def test_global_fit_matches_dense_profile_and_stationarity(loss, penalty):
    su, sz, tu, tz = _observations()
    beta, info = nr.fit_partition_ratio(su, sz, tu, tz, 2, loss, penalty)
    assert info["status"] == "ok", info
    for j, cell in enumerate(info["cells"]):
        p = nr._CellProfile(su[(2 * sz).astype(int) == j], tu[(2 * tz).astype(int) == j],
                            loss, 4, penalty)
        grid_value = min(p.at(float(b))[0] for b in np.linspace(.05, .49, 1001))
        assert cell["objective"] <= grid_value + 2e-11
        assert cell["gap"] <= nr.OBJECTIVE_TOL * (1 + abs(cell["objective"]))
        assert cell["gap"] >= -cell["bound_roundoff_tolerance"]
        assert cell["stationarity_scaled"] <= nr.GRADIENT_TOL
    value, _ = nr.partition_ratio_objective(beta, su, sz, tu, tz, loss, penalty)
    assert value == pytest.approx(info["objective"], abs=3e-13)
    repeated, repeated_info = nr.fit_partition_ratio(su, sz, tu, tz, 2, loss, penalty)
    np.testing.assert_array_equal(beta, repeated)
    assert info == repeated_info


def test_sq_global_fit_handles_multiple_stationary_points():
    tv = np.array([0.] * 5 + [8.])
    sv = np.array([2.] * 400 + [12.])
    p = nr._CellProfile(np.exp(-sv), np.exp(-tv), "SQ", 407, 0, ((-10., 10.), (0., .49)))
    grid = np.linspace(0, .49, 301)
    critical = [0., .49]
    for lo, hi in zip(grid[:-1], grid[1:], strict=True):
        if p.at(lo)[1] * p.at(hi)[1] < 0:
            critical.append(optimize.brentq(lambda b: p.at(b)[1], lo, hi))
    assert len(critical) == 4
    beta, info = nr._solve_profile(p)
    assert info["status"] == "ok", info
    assert info["objective"] == pytest.approx(min(p.at(b)[0] for b in critical), abs=1e-12)
    assert 0 <= beta[1] <= .49


@pytest.mark.parametrize("loss", ["SQ", "UKL"])
@pytest.mark.parametrize("penalty", [0., 2.])
def test_empty_cells_solve_the_same_problem(loss, penalty):
    su, sz = np.array([1., .8, .6, .4]), np.array([.1, .1, .3, .3])
    tu, tz = np.array([1., .7, .5, .3]), np.array([.1, .1, .6, .6])
    beta, info = nr.fit_partition_ratio(su, sz, tu, tz, 4, loss, penalty)
    assert info["status"] == "ok", info
    assert info["cells"][1]["target_count"] == 0
    assert info["cells"][2]["source_count"] == 0
    assert info["cells"][3]["source_count"] == info["cells"][3]["target_count"] == 0
    np.testing.assert_array_equal(beta[1], [-3, .05])
    np.testing.assert_array_equal(beta[3], [-3, .05])
    assert beta[2, 0] == 2
    if loss == "UKL":
        expected = np.clip(-np.log(tu[2:]).sum() / penalty, .05, .49) if penalty else .49
        assert beta[2, 1] == pytest.approx(expected)


@pytest.mark.parametrize("loss", ["SQ", "UKL"])
def test_constant_u_and_active_bounds(loss):
    beta, info = nr.fit_partition_ratio(np.ones(6), [.1] * 4 + [.7] * 2,
                                         np.ones(6), [.1] * 2 + [.7] * 4, 2, loss, .4)
    assert info["status"] == "ok", info
    np.testing.assert_allclose(beta[:, 0], np.log([.5, 2]), atol=1e-14)
    np.testing.assert_array_equal(beta[:, 1], [.05, .05])


def test_losses_do_not_collapse_to_a_common_fit():
    data = _observations()
    sq, si = nr.fit_partition_ratio(*data, 2, "SQ")
    uk, ui = nr.fit_partition_ratio(*data, 2, "UKL")
    assert si["status"] == ui["status"] == "ok"
    assert np.linalg.norm(sq - uk) > .01
    assert sq[0, 1] != sq[1, 1] or uk[0, 1] != uk[1, 1]
    values = nr.predict_partition_ratio(sq, [.2, .4, .2, .4], [.1, .1, .8, .8])
    np.testing.assert_allclose(values[0] / values[1], 2**sq[0, 1])
    np.testing.assert_allclose(values[2] / values[3], 2**sq[1, 1])


def test_failures_are_reported(monkeypatch):
    tv, sv = np.array([0.] * 5 + [8.]), np.array([2.] * 400 + [12.])
    p = nr._CellProfile(np.exp(-sv), np.exp(-tv), "SQ", 407, 0, ((-10., 10.), (0., .49)))
    monkeypatch.setattr(nr, "MAX_NODES", 1)
    _, info = nr._solve_profile(p)
    assert info["status"] == "global_gap_not_met"
    _, overflow = nr._fit_cell([np.nextafter(0., 1.)], [1.], "SQ", 1, 0, nr.DEFAULT_BOUNDS)
    assert overflow["status"] == "nonfinite"


@pytest.mark.parametrize("z", [
    [.1, .4, .3, .6, .8, .9], [.1, .1, .3, .3, .8, .8], [2.] * 6, [0.] * 6,
])
def test_continuous_histogram_matches_dense_minimum_norm(z):
    T = np.array([[.1, .1], [.2, .1], [.1, .5], [.2, .5], [.7, .8], [.8, .9]])
    Y = np.array([2., -1., 4., 3., 8., 5.])
    fit = nr.fit_tensor_histogram_ols(T, z, Y, (2, 3))
    ids = nr.tensor_cell_ids(T, (2, 3))
    design = np.column_stack((z, np.eye(6)[ids]))
    expected, _, rank, _ = np.linalg.lstsq(design, Y, rcond=None)
    np.testing.assert_allclose([fit["slope"], *fit["coefficients"]], expected, atol=4e-14)
    np.testing.assert_allclose(nr.predict_tensor_histogram_ols(fit, T, z), design @ expected,
                               atol=4e-14)
    assert fit["rank"] == rank and fit["empty_cells"] == 3
    assert np.all(fit["coefficients"][fit["counts"] == 0] == 0)


def test_scalar_grid_and_one_observation_minimum_norm():
    fit = nr.fit_tensor_histogram_ols([[.1, .2]], [3.], [8.], 2)
    assert fit["slope"] == pytest.approx(2.4)
    np.testing.assert_allclose(fit["coefficients"], [.8, 0, 0, 0])
    np.testing.assert_array_equal(nr.tensor_cell_ids([[.1, .2], [.9, .8]], 2), [0, 3])


@pytest.mark.parametrize("constant", [False, True])
def test_global_ols_is_numpy_minimum_norm(constant):
    T = np.ones((4, 2)) if constant else np.array([[.1, .2], [.3, .6], [.5, .4], [.7, .8]])
    z, y = np.array([.2] * 4), np.array([2., 4., -2., 1.])
    fit = nr.fit_global_ols(T, z, y)
    design = np.column_stack((np.ones(4), T, z))
    expected, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    np.testing.assert_array_equal(fit["coefficients"], expected)
    np.testing.assert_allclose(nr.predict_global_ols(fit, T, z), design @ expected)
    assert fit["rank"] == rank


@pytest.mark.parametrize("kwargs", [
    {"n_cells": 0}, {"n_cells": 1.2}, {"loss": "bad"}, {"penalty": -.1},
    {"source_u": [0.] * 8}, {"source_u": [np.nan] * 8}, {"source_z": [1.] * 8},
    {"target_u": [.2]}, {"target_z": [.2]}, {"bounds": ((-3, 2), (.05, .5))},
])
def test_invalid_ratio_domains(kwargs):
    su, sz, tu, tz = _observations()
    inputs = dict(source_u=su, source_z=sz, target_u=tu, target_z=tz, n_cells=2, loss="SQ")
    inputs.update(kwargs)
    with pytest.raises(ValueError):
        nr.fit_partition_ratio(**inputs)


@pytest.mark.parametrize("T,bins", [([[1., .2]], 2), ([[.2, .3]], (2,)),
                                    ([[.2, .3]], (2, 1.5)), ([[.2, .3]], 0)])
def test_invalid_histogram_domains(T, bins):
    with pytest.raises(ValueError):
        nr.fit_tensor_histogram_ols(T, [.2], [1.], bins)


def test_source_empty_penalty_can_have_an_interior_global_minimum():
    p = nr._CellProfile([], np.exp([-1.]), "SQ", 1, 100.)
    beta, info = nr._solve_profile(p)
    root = optimize.brentq(lambda b: 100 * b - 2 * np.exp(2 + b), .05, .49)
    assert info["status"] == "ok", info
    np.testing.assert_allclose(beta, [2., root], atol=2e-13)
    u = nr._CellProfile([], np.exp([-1.]), "UKL", 1, 4.)
    beta, info = nr._solve_profile(u)
    assert info["status"] == "ok", info
    np.testing.assert_allclose(beta, [2., .25], atol=2e-14)


def test_source_size_tightens_tolerances_without_changing_normalization():
    p = nr._CellProfile([1.], [1.], "UKL", 1, source_size=10**8)
    assert p.n == 1 and p.source_size == 10**8
    value, _ = p.objective([0., .05])
    info = nr._fit_info(p, np.array([0., .05]), value - 5e-13, 1)
    assert info["status"] == "global_gap_not_met"
    assert info["objective_tolerance"] == pytest.approx(1e-16)
    p = nr._CellProfile([1.], [1.], "UKL", 1, source_size=10**10)
    value, _ = p.objective([1e-9, .05])
    info = nr._fit_info(p, np.array([1e-9, .05]), value, 1)
    assert info["status"] == "stationarity_not_met"
    assert info["gradient_tolerance"] == pytest.approx(1e-10)
    with pytest.raises(ValueError, match="source_size"):
        nr._CellProfile([1.], [1.], "UKL", 1, source_size=0)


def test_tiny_positive_penalty_does_not_overflow_empty_source_bound():
    p = nr._CellProfile([], np.exp([-1.]), "SQ", 1, 1e-300)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        beta, info = nr._solve_profile(p)
    assert info["status"] == "ok", info
    np.testing.assert_array_equal(beta, [2., .49])


def test_sq_profile_retains_stationary_root_when_objectives_round_equal():
    b0 = .270000004
    penalty = float(2 * np.exp(2 + b0) / b0)
    profile = nr._CellProfile([], np.exp([-1.]), "SQ", 1., penalty, source_size=4096)
    midpoint = (.05 + .49) / 2
    root = nr._root_candidate(profile, .05, .49)
    assert root is not None
    assert profile.at(midpoint)[0] == profile.at(root)[0]
    midpoint_coefficient = np.array([profile.at(midpoint)[2], midpoint])
    bad = nr._fit_info(profile, midpoint_coefficient, profile.at(midpoint)[0], 0)
    assert bad["status"] == "stationarity_not_met"
    assert bad["stationarity_scaled"] > bad["gradient_tolerance"]
    coefficient, info = nr._solve_profile(profile)
    assert info["status"] == "ok", info
    np.testing.assert_allclose(coefficient, [2., b0], rtol=0, atol=2e-14)
    assert info["objective_tolerance"] == 1e-12
    assert info["gradient_tolerance"] == 1e-8
    assert info["gap"] <= info["objective_tolerance"] * (1 + abs(info["objective"]))
    assert info["gap"] >= -info["bound_roundoff_tolerance"]
    assert info["stationarity_scaled"] <= info["gradient_tolerance"]
    assert info["gradient"][0] < 0
    assert coefficient[0] == 2.
