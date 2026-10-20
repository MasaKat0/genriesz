"""Independent population integrals and finite-sample ATT score identities."""

import base64
import hashlib
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, special

from genriesz.att_histogram import att_score_record, fit_cell_ols, predict_cell_ols

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from grrexp import e26
from grrexp import e26_population as pop


def test_full_design_and_seed_tasks():
    assert e26.R == 1000 and e26.K == 2
    assert len(e26.CELLS) == 12 and e26.J_VALUES == (3, 4, 5, 6)
    assert e26.CELLS == [[b, j, 2 * j**6] for b in e26.B_VALUES for j in e26.J_VALUES]
    assert len(e26.tasks_for()) == 12000
    assert e26.tasks_for()[0] == (0, 0, 20261007)
    assert e26.tasks_for()[-1] == (11, 999, 20261007)
    for b in e26.B_VALUES:
        assert 2 * e26.BOXES[b][1][1] - b == pytest.approx(0.98)


@pytest.mark.parametrize("n", (7, 8, 9, 18))
def test_fold_storage_recovers_every_original_membership(n):
    entropy, cell, rep = 20261007, 3, 19
    folds = e26.fold_ids(n, e26.K, e26.Seeds(e26.EXP, entropy).folds(cell, rep))
    record = e26._fold_record(folds, cell, rep, entropy)
    packed = np.frombuffer(base64.b64decode(record["packed_base64"], validate=True), dtype=np.uint8)
    recovered = np.unpackbits(packed, bitorder="little", count=record["n"])
    np.testing.assert_array_equal(recovered, folds)
    assert record["encoding"] == "base64-packbits-little"
    assert record["n"] == n
    assert record["counts"] == np.bincount(folds, minlength=2).tolist()
    assert record["entropy"] == entropy and record["seed_key"] == [26, cell, rep, 1]
    assert record["sha256"] == hashlib.sha256(folds.astype(np.uint8).tobytes()).hexdigest()
    assert len(packed) == (n + 7) // 8


@pytest.mark.parametrize("b0", e26.B_VALUES)
def test_population_constants_and_tail_equation(b0):
    j = 3
    facts = pop.constants(b0, j)
    pi = integrate.quad(lambda u: 1 / (1 + u**b0), 0, 1, epsabs=1e-12)[0]
    assert facts["pi"] == pytest.approx(pi, rel=1e-11)
    assert facts["variance"] == pytest.approx(1 / (pi**2 * (1 - b0)))
    u = facts["tail_u"]
    mass = integrate.quad(lambda x: x**b0 / (1 + x**b0), 0, u, epsabs=1e-14)[0]
    assert facts["L"] * mass == pytest.approx(1, rel=1e-10)
    assert facts["a_L"] == pytest.approx(2 / pi**2 * u ** (-2 * b0) * (-np.log(u)))
    assert np.linalg.eigvalsh(facts["sq_jacobian"]).min() > 0
    assert facts["gradient_squared_integral"] == pytest.approx(1900 / 3)
    mean, variance, zmean = pop.grid_moments(j)
    assert mean.mean() == pytest.approx(7 / 12)
    assert (variance + mean**2).mean() - mean.mean()**2 == pytest.approx(47 / 360)
    assert zmean.mean() == pytest.approx(7 / 4)


@pytest.mark.parametrize("b0,power", [(0.6, -0.98), (5 / 7, -5 / 7), (0.85, 0.2)])
def test_hypergeometric_bins_against_transformed_quadrature(b0, power):
    j = 3
    actual = pop.bin_moments(b0, j, power)
    edges = np.arange(j + 1) / j
    expected = [integrate.quad(
        lambda x: np.exp(-(power + 1) * x) / (1 + np.exp(-b0 * x)),
        -np.log(hi), -np.log(lo) if lo else np.inf, epsabs=1e-12,
    )[0] for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    np.testing.assert_allclose(actual, expected, rtol=2e-11)


def test_population_diagnostics_against_full_cell_integrals():
    b0, j, tau_hat = 0.6, 2, 1.13
    beta = np.array([0.03, 0.62])
    coefficients = 10 + 0.37 * np.arange(j**4)
    pi = pop.constants(b0, j)["pi"]
    nodes, weights = special.roots_legendre(3)
    q_expected, mu_expected, norm_expected = np.zeros(2), 0.0, 0.0
    for cell in product(range(j), repeat=4):
        index = np.ravel_multi_index(cell, (j,) * 4)
        z_nodes = [((nodes + 1) / 2 + axis) / j for axis in cell[1:]]
        lo, hi = cell[0] / j, (cell[0] + 1) / j
        for z_index in product(range(3), repeat=3):
            z = np.array([z_nodes[axis][z_index[axis]] for axis in range(3)])
            mass = np.prod([weights[index] / (2 * j) for index in z_index])

            def error(u, index=index, z=z):
                return coefficients[index] - 10 * (u + u**2 / 4 + (z + z**2 / 4).sum())

            def e(u):
                return 1 / (1 + u**b0)

            for m in range(2):
                q_expected[m] += mass * integrate.quad(
                    lambda u, m=m: -e(u) * (-np.log(u))**m * error(u) / pi,
                    lo, hi, epsabs=1e-10,
                )[0]
            mu_expected += mass * integrate.quad(
                lambda u: (1 - e(u)) * (np.exp(beta[0]) * u**(-beta[1]) - u**(-b0))
                * error(u) / pi, lo, hi, epsabs=1e-10,
            )[0]
            norm_expected += mass * integrate.quad(
                lambda u: e(u) * (tau_hat - 1 + error(u))**2 + (1 - e(u)) * error(u)**2,
                lo, hi, epsabs=1e-10,
            )[0]
    actual = pop.regression_diagnostics(coefficients, tau_hat, b0, j)
    np.testing.assert_allclose(actual["directional_error"], q_expected, rtol=1e-10)
    assert actual["regression_l2"]**2 == pytest.approx(norm_expected, rel=2e-11)
    assert pop.population_drift(actual["mean_cell_coefficient"], beta, b0, j) == pytest.approx(
        mu_expected, rel=2e-10
    )


@pytest.mark.parametrize("b0", e26.B_VALUES)
def test_nonnegative_representer_norm_near_truth_and_share_change(b0):
    pi = pop.constants(b0, 3)["pi"]
    assert pop.representer_l2([0, b0], b0, pi, pi) == 0
    f_minus_b = float(pop.primitive(b0, -b0, 1))
    expected = np.expm1(1e-10) * np.sqrt(f_minus_b) / pi
    assert pop.representer_l2([1e-10, b0], b0, pi, pi) == pytest.approx(expected, rel=1e-9)
    empirical_pi = pi * 0.93
    expected = abs(pi / empirical_pi - 1) / (pi * np.sqrt(1 - b0))
    assert pop.representer_l2([0, b0], b0, pi, empirical_pi) == pytest.approx(expected, rel=1e-10)
    a, b = -0.2, e26.BOXES[b0][1][1]
    squared = (
        np.exp(2 * a) * float(pop.primitive(b0, b0 - 2 * b, 1))
        - 2 * np.exp(a) * float(pop.primitive(b0, -b, 1)) + f_minus_b
    ) / pi**2
    assert pop.representer_l2([a, b], b0, pi, pi)**2 == pytest.approx(squared, rel=1e-10)


def deterministic_data(n):
    indices = np.arange(n)
    t = np.column_stack([((indices * step + offset) % (n + 1) + 0.5) / (n + 2)
                         for step, offset in ((1, 0), (3, 1), (5, 3), (7, 5))])
    d = (indices % 3 != 0).astype(float)
    epsilon = np.sin(indices) / 5
    return t, d, d + e26.outcome_mean(t) + epsilon, epsilon


def test_empirical_share_decomposition_from_separate_score_differences():
    t, d, y, epsilon = deterministic_data(32)
    folds, j, b0 = np.arange(len(d)) % 2, 2, 0.6
    fits = [fit_cell_ols(t[folds != k], d[folds != k], y[folds != k], j) for k in range(2)]
    betas = [np.array([0.02, 0.63]), np.array([-0.03, 0.58])]
    g, g1, g0, odds = (np.empty(len(d)) for _ in range(4))
    for k in range(2):
        te = folds == k
        g[te] = predict_cell_ols(fits[k], t[te], d[te], j)
        g1[te] = predict_cell_ols(fits[k], t[te], np.ones(te.sum()), j)
        g0[te] = predict_cell_ols(fits[k], t[te], np.zeros(te.sum()), j)
        odds[te] = np.exp(betas[k][0] - betas[k][1] * np.log(t[te, 0]))
    record = att_score_record(d, y, odds, g, g1, g0)
    result = e26.decomposition(t, d, y, epsilon, folds, fits, betas,
                               record["estimate"], b0, j)
    assert abs(result["decomposition_identity_error"]) < 2e-13
    assert result["taylor_remainder"] == pytest.approx(
        result["population_drift"] - result["linear_drift"]
    )
    f = e26.outcome_mean(t)
    true_nuisance = att_score_record(d, y, t[:, 0]**(-b0), d + f, 1 + f, f)
    oracle = e26.oracle_record(t, d, epsilon, b0, j)
    assert oracle["oracle_estimate"] == pytest.approx(true_nuisance["estimate"])
    assert oracle["oracle_variance"] == pytest.approx(true_nuisance["variance"])


def test_replicate_shares_two_outcome_fits_and_attempts_all_four_odds_fits(monkeypatch):
    n = e26.CELLS[0][-1]
    data = deterministic_data(n)
    monkeypatch.setattr(e26, "draw", lambda rng, size, b: data)
    outcome_calls, odds_calls = [], []

    def outcome(t, d, y, j):
        outcome_calls.append((t.copy(), d.copy(), y.copy()))
        return fit_cell_ols(t, d, y, j)

    def odds(u, d, arm, bounds):
        odds_calls.append((u.copy(), d.copy(), arm, bounds))
        return np.full(2, np.nan), {"status": "nonfinite", "boundary": False}

    monkeypatch.setattr(e26, "fit_cell_ols", outcome)
    monkeypatch.setattr(e26, "fit_att_odds", odds)
    result = e26.replicate((0, 0, 20261007))
    assert len(outcome_calls) == 2 and len(odds_calls) == 4
    for k in range(2):
        np.testing.assert_array_equal(odds_calls[k][0], odds_calls[k + 2][0])
        np.testing.assert_array_equal(odds_calls[k][1], odds_calls[k + 2][1])
    assert all(record["status"] == "nonfinite" for record in result["arms"].values())


def synthetic_raw():
    frames = []
    for cell, (b0, _, n) in enumerate(e26.CELLS):
        fit = {"status": "ok", "boundary": False, "coefficient": [0, b0],
               "objective": -1, "gap": 0, "stationarity_scaled": 0}
        for arm in e26.ARM_LABELS:
            status = np.full(e26.R, "ok", dtype=object)
            if cell == 0 and arm == "SQ":
                status[0] = "global_gap_not_met"
            if cell == 1 and arm == "SQ":
                status[:] = "nonfinite"
            frames.append(pd.DataFrame({
                "cell": cell, "rep": np.arange(e26.R), "n": n, "arm": arm, "status": status,
                "estimate": 1.0, "se": 0.1, "oracle_status": "ok", "oracle_estimate": 1.0,
                "oracle_se": 0.1, "oracle_known_variance_se": 0.1, "oracle_variance_ratio": 1.0,
                "oracle_mean": 0.0, "share_scale": 1.0, "variance_ratio": 1.0,
                "fits": json.dumps([fit, fit]),
            }))
    return pd.concat(frames, ignore_index=True)


def test_summary_denominators_and_outputs():
    raw = synthetic_raw()
    result = e26.summarise(raw)
    one_failure = result[(result.cell == 0) & result.arm.eq("SQ")].iloc[0]
    assert one_failure.R == 1000 and one_failure.returned == 999
    assert one_failure.coverage == 0.999 and one_failure.coverage_conditional == 1
    assert one_failure.oracle_coverage == 1 and one_failure.coefficient_fits == 2000
    no_success = result[(result.cell == 1) & result.arm.eq("SQ")].iloc[0]
    assert no_success.coverage == 0 and no_success.failure_rate == 1
    assert np.isnan(no_success.variance_ratio_q50)
    assert len(e26.tables(result)) == 3
    assert "0.999" in e26.tables(result)["tab_E26"]
    with pytest.raises(ValueError, match="exactly replications"):
        e26.summarise(raw.iloc[1:])
    import matplotlib.pyplot as plt

    main = e26.figure(result)
    extra = e26.additional_figures(result)
    assert len(main.axes) == 6 and len(extra["fig_E26_diagnostics"].axes) == 9
    plt.close(main)
    plt.close(extra["fig_E26_diagnostics"])
