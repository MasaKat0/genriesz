"""Deterministic population and complete-record aggregation checks for E-30."""

from __future__ import annotations

import ast
import inspect
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from grrexp import e30

FOLD_FIELDS = (
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
)


def _first(u, dgp):
    return 10 * u if dgp == "Linear" else 10 * (1 - np.exp(-6 * u))


def _global(beta=None):
    return dict(
        name="Global",
        fit=dict(
            coefficients=np.array(
                [0.4, 2.0, 3.0, -1.0, 1.2] if beta is None else beta,
                dtype=float,
            )
        ),
    )


def _histogram():
    return dict(
        name="Histogram",
        fit=dict(
            coefficients=np.arange(12, dtype=float) / 4 + 0.5,
            slope=0.7,
            bins=(2, 2, 3),
        ),
    )


def _quadrature_l2(regression, dgp):
    """Integrate every polynomial cell exactly and its smooth U component independently."""
    fit = regression["fit"]
    bins = (1, 1, 1) if regression["name"] == "Global" else fit["bins"]
    u_nodes, u_weights = np.polynomial.legendre.leggauss(20)
    nodes, weights = np.polynomial.legendre.leggauss(4)
    total = 0.0
    for cell in itertools.product(*(range(b) for b in bins)):
        coordinates, probabilities = [], []
        for axis, (c, b) in enumerate(zip(cell, bins, strict=True)):
            x, w = (u_nodes, u_weights) if axis == 0 else (nodes, weights)
            coordinates.append((c + (x + 1) / 2) / b)
            probabilities.append(w / (2 * b))
        coordinates.append((nodes + 1) / 2)
        probabilities.append(weights / 2)
        mesh = np.meshgrid(*coordinates, indexing="ij")
        weight_mesh = np.meshgrid(*probabilities, indexing="ij")
        t = np.column_stack([a.ravel() for a in mesh[:3]])
        z = mesh[3].ravel()
        truth = _first(t[:, 0], dgp) + 10 * (t[:, 1] + t[:, 2]) + z
        if regression["name"] == "Global":
            beta = fit["coefficients"]
            predicted = beta[0] + t @ beta[1:4] + beta[4] * z
        else:
            coefficient = np.asarray(fit["coefficients"]).reshape(bins)[cell]
            predicted = coefficient + fit["slope"] * z
        total += np.sum((predicted - truth) ** 2 * np.prod(weight_mesh, axis=0).ravel())
    return float(total)


def _diagnostics_integral(coefficient, regression, dgp):
    """Integrate in v=-log(u); no power-moment helper from E-30 is used."""
    j = len(coefficient)
    k, b = 1 - e30.B, e30.B
    fit = regression["fit"]
    eta = fit["coefficients"][-1] if regression["name"] == "Global" else fit["slope"]
    if regression["name"] == "Global":
        beta = fit["coefficients"]

        def fitted_t(v):
            return beta[0] + beta[1] * np.exp(-v) + (beta[2] + beta[3]) / 2

        intervals = [(0.0, np.inf)]
    else:
        average = np.asarray(fit["coefficients"]).reshape(fit["bins"]).mean(axis=(1, 2))
        h = fit["bins"][0]

        def fitted_t(v):
            index = min(int(np.exp(-v) * h), h - 1)
            return average[index]

        intervals = [(-np.log((r + 1) / h), -np.log(r / h) if r else np.inf) for r in range(h)]

    def qv(function):
        return sum(
            integrate.quad(function, lo, hi, epsabs=2e-10, epsrel=2e-11, limit=250)[0]
            for lo, hi in intervals
        )

    def delta_t(v):
        return fitted_t(v) - _first(np.exp(-v), dgp) - 10

    direction0 = qv(lambda v: np.exp(-(1 - b) * v) * delta_t(v))
    direction1 = qv(lambda v: v * np.exp(-(1 - b) * v) * delta_t(v))
    result = dict(
        drift=0.0,
        linear=0.0,
        shape=0.0,
        remainder=0.0,
        ratio2=0.0,
        direction0=direction0,
        direction1=direction1,
    )
    x, w = np.polynomial.legendre.leggauss(6)
    for r, (a, exponent) in enumerate(coefficient):
        lo, hi = r / j, (r + 1) / j
        zs, zw = lo + (x + 1) * (hi - lo) / 2, w * (hi - lo) / 2
        s_j = integrate.quad(lambda z: 3 * (1 + z) ** 2 / 7, lo, hi)[0] * j
        da, db = a - np.log(k * s_j), exponent - b
        for z, weight in zip(zs, zw, strict=True):
            sz = 3 * (1 + z) ** 2 / 7

            def delta(v, z=z):
                return delta_t(v) + (eta - 1) * z

            result["drift"] += weight * qv(
                lambda v, sz=sz, a=a, exponent=exponent, delta=delta: (
                    (k * sz * np.exp(-(1 - b) * v) - np.exp(a - (1 - exponent) * v)) * delta(v)
                )
            )
            result["linear"] += weight * qv(
                lambda v, s_j=s_j, da=da, db=db, delta=delta: (
                    -k * s_j * np.exp(-(1 - b) * v) * (da + db * v) * delta(v)
                )
            )
            result["shape"] += weight * qv(
                lambda v, sz=sz, s_j=s_j, delta=delta: (
                    k * (sz - s_j) * np.exp(-(1 - b) * v) * delta(v)
                )
            )
            result["remainder"] += weight * qv(
                lambda v, s_j=s_j, a=a, exponent=exponent, da=da, db=db, delta=delta: (
                    (
                        k * s_j * np.exp(-(1 - b) * v) * (1 + da + db * v)
                        - np.exp(a - (1 - exponent) * v)
                    )
                    * delta(v)
                )
            )
            result["ratio2"] += weight * qv(
                lambda v, a=a, exponent=exponent, sz=sz: (
                    np.exp(2 * a - (1 - 2 * exponent) * v)
                    - 2 * k * sz * np.exp(a - (1 - exponent - b) * v)
                    + k**2 * sz**2 * np.exp(-(1 - 2 * b) * v)
                )
            )
    return result


def test_registered_design_and_no_truth_input_to_ratio_learner():
    assert e30.R == 2000 and e30.K == 2
    assert e30.CELLS == tuple(itertools.product(("Linear", "Nonlinear"), (32768, 131072, 524288)))
    assert len(e30.METHODS) == 8 and len(e30.tasks_for()) == 12000
    assert e30.tasks_for()[0] == (0, 0) and e30.tasks_for()[-1] == (5, 1999)
    for dgp, n in e30.CELLS:
        facts = e30.population(dgp, n)
        assert facts["L"] == n // 4 and facts["M"] == n // 8
        assert facts["J"] == np.ceil(np.log(facts["M"]) / 2)
        assert facts["H"] == np.ceil(facts["M"] ** 0.2)
        assert facts["HU"] == np.ceil(facts["M"] ** 0.4)
        assert facts["HW"] == np.ceil(facts["M"] ** 0.1)
        assert facts["penalties"] == (0.0, facts["M"] ** -0.5)
    tree = ast.parse(inspect.getsource(e30._ratio))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not names.intersection({"B", "shape", "mean_function", "population"})
    facts_keys = {
        node.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "facts"
        and isinstance(node.slice, ast.Constant)
    }
    assert facts_keys == {"J"}
    with pytest.raises(ValueError, match="complete E-30 design"):
        e30.population("Linear", 1000)


@pytest.mark.parametrize("dgp", e30.DGPS)
def test_population_constants_against_independent_integrals(dgp):
    facts = e30.population(dgp, e30.N_VALUES[0])
    b, k = e30.B, 1 - e30.B
    zm = integrate.quad(lambda z: z * 3 * (1 + z) ** 2 / 7, 0, 1)[0]
    z2 = integrate.quad(lambda z: z**2 * 3 * (1 + z) ** 2 / 7, 0, 1)[0]
    um = integrate.quad(
        lambda v: k * np.exp(-(1 - b) * v) * _first(np.exp(-v), dgp), 0, np.inf, epsabs=1e-11
    )[0]
    u2 = integrate.quad(
        lambda v: k * np.exp(-(1 - b) * v) * _first(np.exp(-v), dgp) ** 2, 0, np.inf, epsabs=1e-10
    )[0]
    shape2 = integrate.quad(lambda z: (3 * (1 + z) ** 2 / 7) ** 2, 0, 1)[0]
    vs = integrate.quad(lambda v: k**2 * np.exp(-(1 - 2 * b) * v), 0, np.inf)[0] * shape2
    vt = u2 - um**2 + 200 / 12 + z2 - zm**2
    assert facts["theta0"] == pytest.approx(um + 10 + zm, abs=2e-12)
    assert facts["source_variance"] == pytest.approx(vs, rel=2e-13)
    assert facts["target_variance"] == pytest.approx(vt, abs=3e-11)
    assert facts["V"] == pytest.approx(2 * (vs + vt), abs=6e-11)


def test_shape_integrals_and_inverse_cdf_are_exact_without_sampling():
    for j in (1, 3, 7):
        mass, zmass, shape2, uniform_z = e30.shape_moments(j)
        for r in range(j):
            lo, hi = r / j, (r + 1) / j
            for actual, fun in (
                (mass[r], lambda z: 3 * (1 + z) ** 2 / 7),
                (zmass[r], lambda z: z * 3 * (1 + z) ** 2 / 7),
                (shape2[r], lambda z: (3 * (1 + z) ** 2 / 7) ** 2),
                (uniform_z[r], lambda z: z),
            ):
                assert actual == pytest.approx(integrate.quad(fun, lo, hi)[0], abs=4e-16)
        assert mass.sum() == pytest.approx(1.0)
    probabilities = np.array([0.0, 1e-9, 0.01, 0.2, 0.6, 0.95, 1.0])
    transformed_z = np.cbrt(1 + 7 * probabilities) - 1
    np.testing.assert_allclose(((1 + transformed_z) ** 3 - 1) / 7, probabilities, atol=3e-16)
    transformed_u = probabilities ** (1 / (1 - e30.B))
    np.testing.assert_allclose(transformed_u ** (1 - e30.B), probabilities, atol=2e-16)


@pytest.mark.parametrize("power", [-1.0, 0.0, 0.43, 0.86])
def test_power_exponential_and_logarithmic_cell_integrals(power):
    for order in (0, 1, 2):
        expected = integrate.quad(
            lambda v, order=order: np.exp(-(1 - power) * v) * v**order, 0, np.inf, epsabs=1e-10
        )[0]
        assert e30.negative_moment(power, order) == pytest.approx(expected, rel=2e-13)
    for rate in (1, 6, 12):
        expected = integrate.quad(
            lambda v, rate=rate: np.exp(-(1 - power) * v - rate * np.exp(-v)),
            0,
            np.inf,
            epsabs=1e-11,
        )[0]
        assert e30.exponential_moment(power, rate) == pytest.approx(expected, rel=3e-13)
    for order in (0, 1):
        values = e30.cell_integrals(power, 5, order)
        for r, actual in enumerate(values):
            lo, hi = -np.log((r + 1) / 5), -np.log(r / 5) if r else np.inf
            expected = integrate.quad(
                lambda v, order=order: np.exp(-(1 - power) * v) * v**order, lo, hi, epsabs=1e-10
            )[0]
            assert actual == pytest.approx(expected, rel=3e-13, abs=2e-15)


@pytest.mark.parametrize("dgp", e30.DGPS)
def test_mean_moments_and_log_direction(dgp):
    for power in (-1.0, 0.0, 0.43):
        for squared in (False, True):
            expected = integrate.quad(
                lambda v, power=power, squared=squared: (
                    np.exp(-(1 - power) * v) * _first(np.exp(-v), dgp) ** (2 if squared else 1)
                ),
                0,
                np.inf,
                epsabs=1e-10,
            )[0]
            assert e30.u_mean_moment(dgp, power, squared) == pytest.approx(expected, abs=3e-11)
    expected = integrate.quad(
        lambda v: np.exp(-(1 - e30.B) * v) * v * (_first(np.exp(-v), dgp) + 10),
        0,
        np.inf,
        epsabs=1e-10,
    )[0]
    assert e30.true_log_direction(dgp) == pytest.approx(expected, abs=2e-10)


@pytest.mark.parametrize("dgp", e30.DGPS)
@pytest.mark.parametrize("regression", [_global(), _histogram()])
def test_regression_l2_formula_matches_deterministic_tensor_integration(regression, dgp):
    assert e30.regression_l2_squared(regression, dgp) == pytest.approx(
        _quadrature_l2(regression, dgp),
        rel=2e-13,
        abs=3e-12,
    )


def test_correct_linear_coefficients_have_zero_error_and_known_eta_error():
    correct = _global([0.0, 10.0, 10.0, 10.0, 1.0])
    assert e30.regression_l2_squared(correct, "Linear") == pytest.approx(0.0, abs=3e-13)
    wrong_eta = _global([0.0, 10.0, 10.0, 10.0, 2.0])
    assert e30.regression_l2_squared(wrong_eta, "Linear") == pytest.approx(1 / 3, abs=3e-13)
    t = np.array([[0.1, 0.2, 0.3], [0.7, 0.4, 0.9]])
    z = np.array([0.25, 0.75])
    np.testing.assert_allclose(e30.mean_function(t, z, "Linear"), 10 * t.sum(axis=1) + z)


@pytest.mark.parametrize("dgp", e30.DGPS)
@pytest.mark.parametrize("regression", [_global(), _histogram()])
def test_population_drift_ratio_norm_and_remainders_match_direct_integrals(regression, dgp):
    coefficient = np.array([[-0.8, 0.31], [-0.1, 0.46]])
    facts = e30.population(dgp, e30.N_VALUES[0])
    actual = e30.population_diagnostics(dict(coefficient=coefficient), regression, facts)
    expected = _diagnostics_integral(coefficient, regression, dgp)
    for key, independent in (
        ("drift", "drift"),
        ("linear_drift", "linear"),
        ("shape_drift", "shape"),
        ("nonlinear_remainder", "remainder"),
        ("direction_intercept", "direction0"),
        ("direction_slope", "direction1"),
    ):
        assert actual[key] == pytest.approx(expected[independent], rel=2e-11, abs=3e-10)
    assert actual["ratio_l2"] ** 2 == pytest.approx(expected["ratio2"], rel=2e-12)
    assert actual["regression_l2"] ** 2 == pytest.approx(_quadrature_l2(regression, dgp), rel=2e-13)
    assert actual["norm_product"] == pytest.approx(
        np.sqrt(facts["n"] * expected["ratio2"] * _quadrature_l2(regression, dgp)),
        rel=2e-12,
    )
    mass = np.array(
        [integrate.quad(lambda z: 3 * (1 + z) ** 2 / 7, r / 2, (r + 1) / 2)[0] for r in range(2)]
    )
    delta = coefficient - np.column_stack((np.log((1 - e30.B) * 2 * mass), np.full(2, e30.B)))
    assert actual["coefficient_max_error"] == pytest.approx(np.linalg.norm(delta, axis=1).max())
    assert actual["aggregate_intercept_error"] == pytest.approx(
        np.sum((1 - e30.B) * mass * delta[:, 0]),
    )
    assert actual["aggregate_slope_error"] == pytest.approx(
        np.sum((1 - e30.B) * mass * delta[:, 1]),
    )


def test_shape_approximation_and_exact_regression_zero_drift_cases():
    mass = e30.shape_moments(3)[0]
    coefficient = np.column_stack((np.log((1 - e30.B) * 3 * mass), np.full(3, e30.B)))
    facts = e30.population("Linear", e30.N_VALUES[0])
    actual = e30.population_diagnostics(
        dict(coefficient=coefficient), _global([0.0, 10.0, 10.0, 10.0, 2.0]), facts
    )
    assert actual["linear_drift"] == pytest.approx(0.0, abs=2e-13)
    assert actual["drift"] == pytest.approx(actual["shape_drift"], abs=3e-13)
    assert actual["nonlinear_remainder"] == pytest.approx(0.0, abs=3e-13)
    actual = e30.population_diagnostics(
        dict(coefficient=coefficient), _global([0.0, 10.0, 10.0, 10.0, 1.0]), facts
    )
    for key in ("drift", "linear_drift", "shape_drift", "nonlinear_remainder"):
        assert actual[key] == pytest.approx(0.0, abs=3e-12)
    roundoff_limit = np.sqrt(facts["n"] * actual["ratio_l2"] ** 2 * 3e-13)
    assert actual["norm_product"] <= roundoff_limit


def test_result_uses_actual_two_sample_means_and_variance_components():
    source = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    target = [np.array([10.0, 12.0]), np.array([14.0, 16.0])]
    facts = dict(theta0=14.0, V=30.0, n=8)
    actual = e30._result(source, target, facts)
    theta = 15.5
    vs, vt = 7.5, np.mean((np.concatenate(target) - theta) ** 2)
    se = np.sqrt((vs + vt) / 4)
    assert actual["theta"] == theta and actual["error"] == 1.5
    assert actual["source_variance"] == vs and actual["target_variance"] == vt
    assert actual["se"] == pytest.approx(se)
    assert actual["interval_length"] == pytest.approx(2 * stats.norm.ppf(0.975) * se)
    assert actual["covered"] == float(1.5 <= stats.norm.ppf(0.975) * se)
    assert actual["covered_known_variance"] == float(1.5 <= stats.norm.ppf(0.975) * np.sqrt(30 / 8))
    uncovered = e30._result(source, target, dict(facts, theta0=-100.0))
    assert uncovered["covered"] == uncovered["covered_known_variance"] == 0


@pytest.fixture(scope="module")
def complete_deterministic_records():
    count, methods = len(e30.CELLS), len(e30.METHODS)
    cell = np.repeat(np.arange(count), e30.R * methods)
    rep = np.tile(np.repeat(np.arange(e30.R), methods), count)
    method = np.tile(np.arange(methods), count * e30.R)
    labels = np.asarray(e30.METHODS)
    error = (rep - (e30.R - 1) / 2) / e30.R + cell / 10 + method / 100
    frame = pd.DataFrame(
        dict(
            cell=cell,
            rep=rep,
            regression=labels[method, 0],
            loss=labels[method, 1],
            covered=((rep + cell + method) % 10 < 9).astype(float),
            covered_known_variance=((rep + method) % 20 < 19).astype(float),
            error=error,
            interval_length=1 + cell / 10 + method / 100 + rep / (100 * e30.R),
            root_n_drift=2 * error,
            root_n_oracle_difference=3 * error,
            root_n_centered_difference=error,
        )
    )
    for fold in range(2):
        for index, name in enumerate(FOLD_FIELDS):
            frame[f"fold{fold}_{name}"] = (
                (rep + fold) % 3
                if name in ("cv_family", "regression_family")
                else (rep + fold) % 2
                if name == "selected_sq"
                else index + fold + rep / e30.R
            )
    for fold in range(2):
        frame[f"fold{fold}_drift"] = (rep - e30.R / 2) / e30.R + fold / 5
    return frame


def test_complete_aggregation_and_table_numbers(complete_deterministic_records):
    raw = complete_deterministic_records
    summary = e30.summarise(raw)
    assert len(summary) == len(e30.CELLS) * len(e30.METHODS)
    assert summary["replications"].eq(2000).all()
    for row in summary.itertuples():
        group = raw[
            (raw.cell == row.cell) & (raw.regression == row.regression) & (raw.loss == row.loss)
        ]
        assert row.coverage == pytest.approx(0.9)
        interval = stats.binomtest(1800, 2000).proportion_ci()
        assert row.coverage_lower == pytest.approx(interval.low)
        assert row.coverage_upper == pytest.approx(interval.high)
        assert row.known_variance_coverage == pytest.approx(0.95)
        assert row.bias == pytest.approx(group.error.mean())
        assert row.rmse == pytest.approx(np.sqrt(group.error.to_numpy() @ group.error / 2000))
        assert row.interval_length == pytest.approx(group.interval_length.mean())
        for field in ("root_n_drift", "root_n_oracle_difference", "root_n_centered_difference"):
            for probability in (0.25, 0.5, 0.75):
                assert getattr(row, f"{field}_q{int(probability * 100)}") == pytest.approx(
                    np.quantile(group[field], probability),
                )
        if row.regression != "Oracle":
            for field in FOLD_FIELDS:
                values = np.r_[group[f"fold0_{field}"], group[f"fold1_{field}"]]
                assert getattr(row, field + "_mean") == pytest.approx(values.mean())
                assert getattr(row, field + "_median") == pytest.approx(np.median(values))
            cv = np.r_[group.fold0_cv_family, group.fold1_cv_family]
            for index, family in enumerate(e30.REGRESSIONS):
                assert getattr(row, "cv_" + family.lower() + "_frequency") == pytest.approx(
                    np.mean(cv == index),
                )
            for field in ("direction_slope", "drift", "linear_drift", "nonlinear_remainder"):
                values = np.r_[group[f"fold0_{field}"], group[f"fold1_{field}"]]
                assert getattr(row, field + "_absolute_median") == pytest.approx(
                    np.median(np.abs(values)),
                )
    table = e30.tables(summary)["tab_E30_coverage"]
    numeric_rows = [line for line in table.splitlines() if line.count(" & ") == 4][1:]
    assert len(numeric_rows) == 2 * len(e30.METHODS)
    labels = {
        "Histogram": "Isotropic",
        "Anisotropic": "Anisotropic",
        "PredictionCV": "Prediction CV",
        "Selected": "Validation selection",
        "Oracle": "Oracle",
    }
    for dgp, lines in zip(e30.DGPS, (numeric_rows[:8], numeric_rows[8:]), strict=True):
        for (regression, loss), line in zip(e30.METHODS, lines, strict=True):
            row = summary[
                (summary.dgp == dgp)
                & (summary.n == e30.N_VALUES[-1])
                & (summary.regression == regression)
                & (summary.loss == loss)
            ].iloc[0]
            assert line == (
                f"{labels[regression]} & {loss} & {row.coverage:.3f} & {row.bias:.4f} "
                f"& {row.rmse:.4f}" + r" \\"
            )


def test_aggregation_rejects_missing_duplicate_and_unregistered_keys(
    complete_deterministic_records,
):
    raw = complete_deterministic_records
    with pytest.raises(ValueError, match="complete six-condition"):
        e30.summarise(raw.iloc[:-1])
    duplicate = pd.concat((raw.iloc[:-1], raw.iloc[[0]]), ignore_index=True)
    with pytest.raises(ValueError, match="complete six-condition"):
        e30.summarise(duplicate)
    unexpected = raw.copy()
    unexpected.loc[0, "rep"] = e30.R
    with pytest.raises(ValueError, match="missing or unexpected"):
        e30.summarise(unexpected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rep", 0.5),
        ("rep", 0.0),
        ("rep", False),
        ("rep", np.bool_(False)),
        ("cell", 0.5),
        ("cell", False),
    ],
)
def test_aggregation_rejects_noninteger_keys(complete_deterministic_records, field, value):
    raw = complete_deterministic_records.copy()
    raw[field] = raw[field].astype(object)
    raw.loc[0, field] = value
    with pytest.raises(ValueError, match="condition and replication keys must be integers"):
        e30.summarise(raw)


def test_pandas_integer_keys_are_accepted(complete_deterministic_records):
    raw = complete_deterministic_records.copy()
    raw["cell"] = raw["cell"].astype(np.int32)
    raw["rep"] = raw["rep"].astype(np.int32)
    summary = e30.summarise(raw)
    assert len(summary) == len(e30.CELLS) * len(e30.METHODS)
