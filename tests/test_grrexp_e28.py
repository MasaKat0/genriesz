"""Independent population integrals and complete-sample E-28 checks."""

import inspect
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import integrate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from grrexp import e28


def phi(u, z):
    u, z = np.broadcast_arrays(u, z)
    return np.stack(
        (
            np.ones_like(u),
            np.sqrt(3) * (2 * u - 1),
            np.sqrt(3) * (2 * z - 1),
            3 * (2 * u - 1) * (2 * z - 1),
            np.sqrt(5) * (6 * u**2 - 6 * u + 1),
            np.sqrt(5) * (6 * z**2 - 6 * z + 1),
        ),
        axis=-1,
    )


def negative_moment(power, derivative=0):
    return integrate.quad(
        lambda x: (2 / 3) * (1 + np.exp(-x)) * np.exp(-(1 - power) * x) * x**derivative,
        0,
        np.inf,
        epsabs=1e-11,
        epsrel=3e-12,
    )[0]


def test_complete_registered_design_and_learner_inputs():
    assert e28.R == 2000 and e28.K == 2
    assert e28.CELLS == tuple(itertools.product((0.28, 0.35, 0.42), (8192, 32768, 131072)))
    assert e28.EXPONENTS == ("Estimated", "Known")
    assert e28.REGRESSIONS == ("OLS", "CV", "Ridge")
    assert e28.LOSSES == ("SQ", "UKL")
    assert e28.tasks_for() == tuple(itertools.product(range(9), range(2000)))
    assert tuple(inspect.signature(e28.fit_nuisances).parameters) == ("data", "h")
    nuisance_source = inspect.getsource(e28.fit_nuisances)
    for name in ("BETA0", "population", "source_moment", "negative_moment", 'data["b"]'):
        assert name not in nuisance_source
    np.testing.assert_array_equal(e28.PENALTIES, np.r_[0, 10.0 ** np.arange(-6, 1)])
    for b, n in e28.CELLS:
        facts = e28.population(b, n)
        assert n == 4 * facts["L"] == 8 * facts["M"]
        assert facts["H"] == round(facts["M"] ** 0.25)
    with pytest.raises(ValueError, match="complete E-28 design"):
        e28.population(0.35, 1024)


@pytest.mark.parametrize("b", e28.B_VALUES)
def test_population_constants_match_independent_quadrature(b):
    facts = e28.population(b, 8192)
    nodes, weights = np.polynomial.legendre.leggauss(5)
    points, weights = (nodes + 1) / 2, weights / 2
    uu, zz = np.meshgrid(points, points, indexing="ij")
    values = phi(uu.ravel(), zz.ravel())
    product_weights = (weights[:, None] * weights[None, :] * 2 * (1 + uu) / 3).ravel()
    gram = values.T @ (product_weights[:, None] * values)
    np.testing.assert_allclose(facts["gram"], gram, atol=3e-14)
    assert not np.allclose(gram, np.eye(6))
    k = 1 / negative_moment(b)
    assert facts["k"] == pytest.approx(k, rel=5e-14)

    def target_integrand(x):
        u = x ** (1 / (1 - b))
        basis = phi(np.full_like(points, u), points)
        z_weights = weights * 3 * (1 + points) ** 2 / 7
        first = z_weights @ basis
        second = basis.T @ (z_weights[:, None] * basis)
        return (k * 2 * (1 + u) / (3 * (1 - b))) * np.r_[first, second.ravel()]

    target = integrate.quad_vec(target_integrand, 0, 1, epsabs=1e-11, epsrel=3e-12)[0]
    target_mean, target_gram = target[:6], target[6:].reshape(6, 6)
    np.testing.assert_allclose(facts["target_mean"], target_mean, atol=4e-13)
    np.testing.assert_allclose(facts["target_gram"], target_gram, atol=4e-13)
    theta = float(target_mean @ e28.BETA0)
    target_variance = float(e28.BETA0 @ target_gram @ e28.BETA0 - theta**2)
    shape2 = integrate.quad(lambda z: (3 * (1 + z) ** 2 / 7) ** 2, 0, 1)[0]
    source_variance = k**2 * negative_moment(2 * b) * shape2
    assert facts["theta0"] == pytest.approx(theta, abs=4e-12)
    assert facts["source_variance"] == pytest.approx(source_variance, rel=5e-14)
    assert facts["target_variance"] == pytest.approx(target_variance, abs=2e-10)
    assert facts["V"] == pytest.approx(2 * (source_variance + target_variance), abs=4e-10)
    mu = k * negative_moment(b, 1)
    information = k * negative_moment(b, 2) - mu**2
    omega = (
        k**2
        * (
            negative_moment(2 * b, 2)
            - 2 * mu * negative_moment(2 * b, 1)
            + mu**2 * negative_moment(2 * b)
        )
        + information
    )
    assert facts["exponent_variance"] == pytest.approx(omega / information**2, rel=3e-12)

    direction = target_mean / k
    penalty_direction = -direction @ np.linalg.solve(gram, np.r_[0, e28.BETA0[1:]])
    assert abs(penalty_direction) > 1
    assert facts["penalty_direction"] == pytest.approx(penalty_direction, abs=4e-12)
    penalty = 1e-6
    population_ridge = np.linalg.solve(
        gram + penalty * np.diag([0, 1, 1, 1, 1, 1]), gram @ e28.BETA0
    )
    assert direction @ (population_ridge - e28.BETA0) / penalty == pytest.approx(
        penalty_direction, rel=3e-6
    )


def test_source_cdf_target_mixture_and_random_streams_match_direct_transforms():
    cell, rep, fold = 6, 11, 1
    b, n = e28.CELLS[cell]
    size = n // 4
    data = e28.generate_fold(cell, rep, fold)
    again = e28.generate_fold(cell, rep, fold)
    for key in data:
        np.testing.assert_array_equal(data[key], again[key])
    streams = [e28.generator(cell, rep, stream, fold) for stream in range(6)]
    uniform = e28.interior_uniform(streams[0], size)
    np.testing.assert_allclose((2 * data["u"] + data["u"] ** 2) / 3, uniform, atol=3e-16)
    np.testing.assert_array_equal(data["z"], e28.interior_uniform(streams[1], size))
    epsilon = streams[2].standard_normal(size)
    np.testing.assert_array_equal(data["epsilon"], epsilon)
    np.testing.assert_allclose(data["y"], phi(data["u"], data["z"]) @ e28.BETA0 + epsilon)
    k = 1 / negative_moment(b)
    component = e28.interior_uniform(streams[3], size) < (2 * k / 3) / (1 - b)
    target_uniform = e28.interior_uniform(streams[4], size)
    expected_target = target_uniform ** (1 / np.where(component, 1 - b, 2 - b))
    np.testing.assert_array_equal(data["tu"], expected_target)
    z_uniform = e28.interior_uniform(streams[5], size)
    np.testing.assert_allclose(((1 + data["tz"]) ** 3 - 1) / 7, z_uniform, atol=4e-16)
    for key in ("u", "z", "tu", "tz"):
        assert np.all((data[key] > 0) & (data[key] < 1))
    baseline = e28.generator(cell, rep, 0, fold).random(20)
    for args in ((cell, rep, 0, 0), (cell, rep + 1, 0, fold), (cell, rep, 1, fold)):
        assert not np.array_equal(baseline, e28.generator(*args).random(20))
    for bad in ((9, 0, 0, 0), (0, 2000, 0, 0), (0, 0, 6, 0), (0, 0, 0, 2)):
        with pytest.raises(ValueError, match="random stream"):
            e28.generator(*bad)


@pytest.fixture(scope="module")
def normal_records():
    return {cell: e28.replicate((cell, 7)) for cell in (0, 3, 6)}


def shape_from_training(data, h):
    m = len(data["y"]) // 2
    source = np.bincount((data["z"][:m] * h).astype(int), minlength=h)
    target = np.bincount((data["tz"][:m] * h).astype(int), minlength=h)
    counts = (target + 1) / (source + 1)
    return counts / counts.mean()


def test_normal_replicates_use_twelve_finite_paired_interior_procedures(normal_records):
    expected = set(itertools.product(e28.EXPONENTS, e28.REGRESSIONS, e28.LOSSES))
    for cell, records in normal_records.items():
        b, n = e28.CELLS[cell]
        facts = e28.population(b, n)
        assert n == 8192 and len(records) == 12
        assert {(r["exponent_mode"], r["regression"], r["loss"]) for r in records} == expected
        frame = pd.DataFrame(records)
        assert np.isfinite(frame.select_dtypes(include=np.number)).all().all()
        assert frame.se.gt(0).all()
        assert frame.loc[frame.exponent_mode.eq("Known"), "root_n_exponent_effect"].eq(0).all()
        known = {(r["regression"], r["loss"]): r for r in records if r["exponent_mode"] == "Known"}
        for row in records:
            paired = known[row["regression"], row["loss"]]
            assert row["root_n_exponent_effect"] == pytest.approx(
                np.sqrt(n) * (row["theta"] - paired["theta"])
            )
            assert abs(row["decomposition_error"]) < 5e-14
            assert row["se"] == pytest.approx(
                np.sqrt(2 * (row["source_variance"] + row["target_variance"]) / n)
            )
            for fold in range(2):
                assert 0 < row[f"fold{fold}_exponent"] < e28.UPPER
                assert row[f"fold{fold}_scale"] > 0
                assert row[f"fold{fold}_shape_stationarity"] < 2e-13
                assert row[f"fold{fold}_shape_integral_error"] < 2e-13
                for residual in ("exponent", "regression", "scale"):
                    assert row[f"fold{fold}_{residual}_stationarity"] < 2e-13
                assert row[f"fold{fold}_exponent"] == paired[f"fold{fold}_exponent"]
                assert row[f"fold{fold}_lambda"] == paired[f"fold{fold}_lambda"]
                if row["regression"] == "OLS":
                    assert row[f"fold{fold}_lambda"] == 0
                elif row["regression"] == "CV":
                    assert row[f"fold{fold}_lambda"] in e28.PENALTIES
                else:
                    assert row[f"fold{fold}_lambda"] == facts["L"] ** (-0.2)


def test_crossfit_scores_variances_and_cv_equal_direct_sample_calculations(normal_records):
    cell, rep = 0, 7
    b, n = e28.CELLS[cell]
    facts = e28.population(b, n)
    data = [e28.generate_fold(cell, rep, fold) for fold in range(2)]
    shape = [shape_from_training(d, facts["H"]) for d in data]
    for row in normal_records[cell]:
        source_scores, target_scores = [], []
        for evaluation in range(2):
            d, training = data[evaluation], data[1 - evaluation]
            coefficient = np.array([row[f"fold{evaluation}_beta{j}"] for j in range(6)])
            penalty = row[f"fold{evaluation}_lambda"]
            matrix = training["phi"].T @ training["phi"] / facts["L"] + np.diag(
                [0, *([penalty] * 5)]
            )
            np.testing.assert_allclose(
                matrix @ coefficient,
                training["phi"].T @ training["y"] / facts["L"],
                atol=3e-14,
            )
            exponent = (
                row[f"fold{evaluation}_exponent"] if row["exponent_mode"] == "Estimated" else b
            )
            ratio = (
                row[f"fold{evaluation}_scale"]
                * d["u"] ** (-exponent)
                * shape[1 - evaluation][(d["z"] * facts["H"]).astype(int)]
            )
            source_scores.append(ratio * (d["y"] - d["phi"] @ coefficient))
            target_scores.append(d["tphi"] @ coefficient)
        source, target = np.concatenate(source_scores), np.concatenate(target_scores)
        theta = source.mean() + target.mean()
        assert row["theta"] == pytest.approx(theta, abs=2e-14)
        assert row["source_variance"] == pytest.approx(np.mean(source**2), rel=5e-14)
        assert row["target_variance"] == pytest.approx(np.mean((target - theta) ** 2), rel=5e-14)

    for training_fold, d in enumerate(data):
        m = facts["M"]
        errors = []
        for penalty in e28.PENALTIES:
            error = 0.0
            for train, validation in ((slice(0, m), slice(m, None)), (slice(m, None), slice(0, m))):
                x, y = d["phi"][train], d["y"][train]
                coefficient = np.linalg.solve(
                    x.T @ x / m + np.diag([0, *([penalty] * 5)]), x.T @ y / m
                )
                error += np.sum((d["y"][validation] - d["phi"][validation] @ coefficient) ** 2)
            errors.append(error / facts["L"])
        selected = e28.PENALTIES[np.argmin(errors)]
        for row in normal_records[cell]:
            if row["regression"] == "CV":
                assert row[f"fold{1 - training_fold}_lambda"] == selected


def test_saved_normal_rows_roundtrip_and_incomplete_aggregation_is_rejected(
    normal_records, tmp_path
):
    raw = pd.DataFrame(normal_records[0])
    destination = tmp_path / "raw.parquet"
    raw.to_parquet(destination, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(destination), raw)
    for incomplete in (raw, raw.iloc[1:], pd.concat([raw, raw.iloc[:1]], ignore_index=True)):
        with pytest.raises(ValueError, match="complete E-28 observations"):
            e28.summarise(incomplete)


@pytest.fixture(scope="module")
def normal_training():
    return e28.generate_fold(0, 7, 0), e28.population(0.28, 8192)["H"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("pilot_stationarity_scaled", 2e-12),
        ("shape_integral_error", 2e-12),
        ("shape_min", 0.0),
    ],
)
def test_nuisance_rejects_finite_invalid_shape_diagnostics(
    monkeypatch, normal_training, key, value
):
    original = e28.fit_shape

    def fit(*args):
        shape, info = original(*args)
        return shape, {**info, key: value}

    monkeypatch.setattr(e28, "fit_shape", fit)
    with pytest.raises(ArithmeticError, match="shape stationarity or normalization"):
        e28.fit_nuisances(*normal_training)


@pytest.mark.parametrize("normalized", [False, True])
def test_nuisance_checks_actual_shape_when_reported_residuals_are_small(
    monkeypatch, normal_training, normalized
):
    original = e28.fit_shape

    def fit(*args):
        shape, info = original(*args)
        if normalized:
            shape[0] += 0.1
            shape[1] -= 0.1
        else:
            shape *= 1.001
        return shape, info

    monkeypatch.setattr(e28, "fit_shape", fit)
    with pytest.raises(ArithmeticError, match="shape stationarity or normalization"):
        e28.fit_nuisances(*normal_training)


@pytest.mark.parametrize(
    "key,value", [("exponent_stationarity_scaled", 2e-12), ("exponent_hessian", 0.0)]
)
def test_nuisance_rejects_finite_invalid_exponent_diagnostics(
    monkeypatch, normal_training, key, value
):
    original = e28.fit_exponent

    def fit(*args, **kwargs):
        exponent, info = original(*args, **kwargs)
        return exponent, {**info, key: value}

    monkeypatch.setattr(e28, "fit_exponent", fit)
    with pytest.raises(ArithmeticError, match="exponent stationarity or interior"):
        e28.fit_nuisances(*normal_training)


@pytest.mark.parametrize("failure", ["coefficient", "reported", "positivity"])
def test_nuisance_checks_actual_outcome_equations_and_reported_diagnostics(
    monkeypatch, normal_training, failure
):
    original = e28.fit_outcome

    def fit(gram, xy, size, penalty):
        coefficient, info = original(gram, xy, size, penalty)
        if failure == "coefficient":
            coefficient[1] += 0.1
        elif failure == "reported":
            info["normal_equation_scaled"] = 2e-12
        elif penalty > 0:
            info["ridge_min_eigenvalue"] = 0.0
        return coefficient, info

    monkeypatch.setattr(e28, "fit_outcome", fit)
    with pytest.raises(ArithmeticError, match="fitted equation or positivity"):
        e28.fit_nuisances(*normal_training)


@pytest.mark.parametrize(
    "field",
    [
        f"fold{fold}_{name}"
        for fold in range(2)
        for name in (
            "shape_stationarity",
            "shape_integral_error",
            "exponent_stationarity",
            "regression_stationarity",
            "scale_stationarity",
        )
    ],
)
def test_result_guard_rejects_finite_excess_residuals(normal_records, field):
    row = {**normal_records[0][0], field: 2e-12}
    with pytest.raises(ArithmeticError, match="fitted first-order equation or normalization"):
        e28.check_result(row)


@pytest.mark.parametrize("field", ["decomposition_error", "evaluation_remainder"])
def test_result_guard_checks_reported_and_actual_decomposition(normal_records, field):
    row = normal_records[0][0].copy()
    row[field] += 1e-4
    with pytest.raises(ArithmeticError, match="score decomposition failed"):
        e28.check_result(row)


@pytest.mark.parametrize("loss", ["SQ", "UKL"])
def test_replication_rejects_finite_invalid_scale_diagnostics(monkeypatch, loss):
    original = e28.scale_fit

    def fit(*args):
        scales, info = original(*args)
        return scales, {**info, loss + "_stationarity": 2e-12}

    monkeypatch.setattr(e28, "scale_fit", fit)
    with pytest.raises(ArithmeticError, match="fitted first-order equation or normalization"):
        e28.replicate((0, 7))
