"""Independent integrals, full-data identities and E-27 design checks."""

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import integrate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from grrexp import e27

from genriesz.shape_regression import SourceMoments, TargetMoments, basis


def test_complete_design_and_entry_point_has_no_reduced_run_option():
    assert e27.R == 2000 and e27.K == 2 and len(e27.CELLS) == 12
    assert e27.N_VALUES == (2048, 32768, 524288, 8388608)
    assert e27.B_VALUES == (0.28, 0.35, 0.42)
    assert len(e27.tasks_for()) == 24000
    assert e27.tasks_for()[0] == (0, 0, 20261007)
    assert e27.tasks_for()[-1] == (11, 1999, 20261007)
    for b, n in e27.CELLS:
        facts = e27.population(b, n)
        assert n == 4 * facts["L"] == 8 * facts["M"]
        assert facts["M"] == facts["H"] ** 4
        assert facts["lambda"] == facts["L"] ** (-0.2)
        assert facts["q"] == facts["lambda"] / (1 + facts["lambda"])
        assert facts["rho"] == facts["M"] ** (2 * b - 1)
    spec = importlib.util.spec_from_file_location(
        "run_e27", Path(e27.__file__).parents[1] / "run_e27.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(inspect.signature(module.run).parameters) == {"directory", "n_jobs"}
    with pytest.raises(SystemExit):
        module.main(["--R", "1", "--output", "/tmp/not-created-e27"])


@pytest.mark.parametrize("b", e27.B_VALUES)
def test_exact_population_integrals_against_independent_quadrature(b):
    facts = e27.population(b, e27.N_VALUES[0])
    assert facts["D_s"] == pytest.approx(
        integrate.quad(lambda z: (3 * (1 + z) ** 2 / 7) ** (1 / b), 0, 1, epsabs=1e-12)[0],
        rel=2e-13,
    )

    # The substitution u=x**(1/(1-b)) removes the singular target density.
    def f(u, z):
        x, w = 2 * u - 1, 2 * z - 1
        return (
            10 * np.sqrt(3) * x
            + np.sqrt(3) * w
            + 3 * x * w
            + 0.5 * np.sqrt(5) * (6 * u**2 - 6 * u + 1)
            + 0.5 * np.sqrt(5) * (6 * z**2 - 6 * z + 1)
        )

    first = integrate.dblquad(
        lambda z, x: f(x ** (1 / (1 - b)), z) * 3 * (1 + z) ** 2 / 7,
        0,
        1,
        lambda _: 0,
        lambda _: 1,
        epsabs=1e-12,
        epsrel=1e-12,
    )[0]
    second = integrate.dblquad(
        lambda z, x: f(x ** (1 / (1 - b)), z) ** 2 * 3 * (1 + z) ** 2 / 7,
        0,
        1,
        lambda _: 0,
        lambda _: 1,
        epsabs=1e-11,
        epsrel=1e-12,
    )[0]
    assert facts["theta0"] == pytest.approx(first, abs=3e-11)
    assert facts["target_variance"] == pytest.approx(second - first**2, abs=3e-9)
    assert facts["mu"] == pytest.approx(first / (1 - b), abs=3e-11)
    assert facts["V"] == pytest.approx(2 * (facts["source_variance"] + facts["target_variance"]))
    assert facts["finite_q_target_variance"] == pytest.approx(
        (1 - facts["q"]) ** 2 * facts["target_variance"] + facts["q"] ** 2 * first**2
    )


def test_polynomial_basis_is_orthonormal_with_canonical_order():
    moment, gram = e27.polynomial_moments(0, 1 / (np.arange(5) + 1))
    np.testing.assert_allclose(moment, [1, 0, 0, 0, 0, 0], atol=2e-15)
    np.testing.assert_allclose(gram, np.eye(6), atol=1e-14)
    u, z = np.array([0.13, 0.72]), np.array([0.37, 0.84])
    polynomial = np.einsum(
        "iab,na,nb->ni",
        e27.BASIS_POLYNOMIAL,
        u[:, None] ** np.arange(3),
        z[:, None] ** np.arange(3),
    )
    np.testing.assert_allclose(polynomial, basis(u, z), atol=2e-15)
    np.testing.assert_allclose(basis(u, z)[:, 3], 3 * (2 * u - 1) * (2 * z - 1))


def test_seed_streams_and_chunk_size_preserve_all_observations():
    first = e27.collect_fold(0, 3, 0, chunk_size=73)
    second = e27.collect_fold(0, 3, 0, chunk_size=181)
    for i in range(2):
        a, b = first[i], second[i]
        np.testing.assert_array_equal(a.pilot_counts, b.pilot_counts)
        np.testing.assert_array_equal(a.cell_counts, b.cell_counts)
        assert a.n == b.n == 512 and a.pilot_n == a.scale_n == 256
        np.testing.assert_allclose(a.gram, b.gram, rtol=2e-14, atol=2e-13)
    np.testing.assert_allclose(first[0].xy, second[0].xy, rtol=2e-14, atol=3e-12)
    np.testing.assert_allclose(first[0].v_augmented, second[0].v_augmented, rtol=2e-14, atol=3e-12)
    np.testing.assert_allclose(
        first[0].v2_augmented, second[0].v2_augmented, rtol=2e-14, atol=2e-11
    )
    np.testing.assert_allclose(first[2], second[2], rtol=2e-14)
    np.testing.assert_array_equal(first[3], second[3])
    np.testing.assert_array_equal(first[4], second[4])
    for values in (first[3], second[3]):
        assert (values > 0).all()
    assert (first[4] < 1).all()
    assert not np.array_equal(
        e27.generator(0, 3, 0, 0).random(10), e27.generator(0, 3, 0, 1).random(10)
    )


def direct_fold(cell, rep, fold):
    b, n = e27.CELLS[cell]
    size = n // 4
    rng = [e27.generator(cell, rep, s, fold) for s in range(5)]
    u, z = e27.interior_uniform(rng[0], size), e27.interior_uniform(rng[1], size)
    epsilon = rng[2].standard_normal(size)
    tu = e27.interior_uniform(rng[3], size) ** (1 / (1 - b))
    w = e27.interior_uniform(rng[4], size)
    root = np.cbrt(1 + 7 * w)
    tz = 7 * w / (root**2 + root + 1)
    return u, z, basis(u, z) @ e27.BETA0 + epsilon, epsilon, tu, tz


def test_crossfit_point_variance_and_identity_equal_direct_full_arrays():
    cell, rep = 0, 7
    facts = e27.population(*e27.CELLS[cell])
    arrays = [direct_fold(cell, rep, fold) for fold in range(2)]
    moments = [e27.collect_fold(cell, rep, fold, chunk_size=61) for fold in range(2)]
    source, target, oracle = ([x[i] for x in moments] for i in range(3))
    fitted = [e27.nuisance(source[j], target[j], facts["lambda"]) for j in range(2)]
    records = e27.score_records(source, target, fitted, oracle, facts)
    for arm in e27.ARM_LABELS:
        ss, tt = [], []
        for j, (u, z, y, _, tu, tz) in enumerate(arrays):
            shape, scale, coefficient, _ = fitted[1 - j]
            ratio = scale[arm] * u ** (-facts["b"]) * shape[(z * facts["H"]).astype(int)]
            ss.append(ratio * (y - basis(u, z) @ coefficient))
            tt.append(basis(tu, tz) @ coefficient)
        direct_estimate = np.concatenate(ss).mean() + np.concatenate(tt).mean()
        source_variance = np.mean(np.concatenate(ss) ** 2)
        target_variance = np.mean((np.concatenate(tt) - direct_estimate) ** 2)
        row = records[arm]
        assert row["estimate"] == pytest.approx(direct_estimate, abs=2e-14)
        assert row["source_variance"] == pytest.approx(source_variance, rel=2e-13)
        assert row["target_variance"] == pytest.approx(target_variance, rel=2e-13)
        assert abs(row["finite_sample_identity_error"]) < 1e-14
        assert row["population_drift"] == pytest.approx(
            sum(
                row[x]
                for x in (
                    "leading_drift",
                    "ridge_drift",
                    "pilot_shape_drift",
                    "coefficient_shape_cross",
                )
            )
        )
        assert row["known_variance_studentized_error"] == pytest.approx(
            np.sqrt(facts["n"] / facts["V"]) * (direct_estimate - facts["theta0"])
        )


def test_population_drift_and_norms_against_separate_cell_integrals():
    facts = e27.population(0.35, 2048)
    shape = np.array([0.4, 0.7, 1.2, 1.7])
    coefficient = (1 - facts["q"]) * e27.BETA0 + np.array([0.01, 0.02, -0.03, 0.01, 0.02, -0.01])
    scale = facts["k"] + 0.03
    diag = e27.fold_diagnostics(shape, scale, coefficient, facts, facts["A0"])
    delta = coefficient - e27.BETA0
    drift, ratio_norm2, shape_norm2 = 0.0, 0.0, 0.0
    for j in range(4):
        lo, hi = j / 4, (j + 1) / 4
        ratio_norm2 += integrate.quad(
            lambda z, j=j: (
                (scale * shape[j] - facts["k"] * 3 * (1 + z) ** 2 / 7) ** 2 / (1 - 2 * facts["b"])
            ),
            lo,
            hi,
            epsabs=1e-12,
        )[0]
        shape_norm2 += integrate.quad(
            lambda z, j=j: (shape[j] - 3 * (1 + z) ** 2 / 7) ** 2, lo, hi, epsabs=1e-12
        )[0]
        drift += integrate.dblquad(
            lambda z, x, j=j: (
                -(scale * shape[j] - facts["k"] * 3 * (1 + z) ** 2 / 7)
                * (basis(np.array([x ** (1 / facts["k"])]), np.array([z])) @ delta)[0]
                / facts["k"]
            ),
            0,
            1,
            lambda _, lo=lo: lo,
            lambda _, hi=hi: hi,
            epsabs=1e-11,
        )[0]
    assert diag["population_drift"] == pytest.approx(drift, abs=2e-11)
    assert diag["representer_l2"] ** 2 == pytest.approx(ratio_norm2, rel=2e-13)
    assert diag["shape_l2"] ** 2 == pytest.approx(shape_norm2, rel=2e-13)
    assert diag["full_norm_product"] >= diag["full_norm_product_lower_bound"]
    assert abs(diag["drift_identity_error"]) < 1e-14


def synthetic_raw():
    frames = []
    metrics = (
        "variance_ratio",
        "source_variance_ratio",
        "target_variance_ratio",
        "oracle_variance_ratio",
        "root_n_error",
        "stable_error",
        "fold0_full_norm_product",
        "fold1_full_norm_product",
        "fold0_directional_over_q",
        "fold1_directional_over_q",
        "finite_sample_identity_error",
        "population_drift",
        "leading_drift",
        "ridge_drift",
        "pilot_shape_drift",
        "coefficient_shape_cross",
        "evaluation_remainder",
    )
    for cell, (b, n) in enumerate(e27.CELLS):
        facts = e27.population(b, n)
        for arm in e27.ARM_LABELS:
            data = {
                "cell": cell,
                "rep": np.arange(e27.R),
                "n": n,
                "arm": arm,
                "status": "ok",
                "estimate": facts["theta0"],
                "se": 0.2,
                "oracle_estimate": facts["theta0"],
                "oracle_se": 0.2,
                "oracle_known_variance_se": np.sqrt(facts["V"] / n),
            }
            data.update({field: np.ones(e27.R) for field in metrics})
            for field in (
                "population_drift",
                "leading_drift",
                "ridge_drift",
                "pilot_shape_drift",
                "coefficient_shape_cross",
                "evaluation_remainder",
            ):
                data["root_n_" + field] = np.zeros(e27.R)
                data["stable_" + field] = np.zeros(e27.R)
            frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)


def test_full_aggregation_quantiles_intervals_outputs_and_failure_rejection():
    row = {}
    e27._quantiles(row, "known", np.arange(101, dtype=float))
    assert row["known_q25"] == 25 and row["known_q50"] == 50 and row["known_q75"] == 75
    assert row["known_q50_mc_lo"] <= 50 <= row["known_q50_mc_hi"]
    raw = synthetic_raw()
    summary = e27.summarise(raw)
    assert len(summary) == 24 and summary.R.eq(2000).all()
    assert summary.coverage.eq(1).all() and summary.known_variance_coverage.eq(1).all()
    assert (summary.coverage_mc_lo < 1).all() and summary.coverage_mc_hi.eq(1).all()
    for cell, (b, n) in enumerate(e27.CELLS):
        values = summary.loc[summary.cell.eq(cell), "gaussian_stable_coverage_reference"]
        assert values.eq(e27.population(b, n)["gaussian_stable_coverage_reference"]).all()
    assert len(e27.tables(summary)) == 5
    import matplotlib.pyplot as plt

    figures = e27.figures(summary)
    assert {name: len(fig.axes) for name, fig in figures.items()} == {
        "fig_E27": 6,
        "fig_E27_norms": 9,
        "fig_E27_decomposition": 6,
    }
    assert figures["fig_E27_decomposition"].axes[0].get_legend_handles_labels()[1] == [
        "Population drift",
        "Leading term",
        "Ridge estimation term",
        "Shape term",
        "Scale and shape interaction",
        "Evaluation remainder",
    ]
    for column, b in enumerate(e27.B_VALUES):
        lines = [
            line
            for line in figures["fig_E27"].axes[column].lines
            if line.get_label() == "SQ Gaussian + stable reference"
        ]
        assert len(lines) == 1
        np.testing.assert_array_equal(lines[0].get_xdata(), e27.N_VALUES)
        np.testing.assert_allclose(
            lines[0].get_ydata(),
            [e27.population(b, n)["gaussian_stable_coverage_reference"] for n in e27.N_VALUES],
        )
    for figure in figures.values():
        plt.close(figure)
    with pytest.raises(ValueError, match="exactly replications"):
        e27.summarise(raw.iloc[1:])
    raw.loc[0, "status"] = "failed"
    with pytest.raises(ValueError, match="failed replication"):
        e27.summarise(raw)
    with pytest.raises(ValueError, match="nonfinite"):
        e27._quantiles({}, "bad", [1, np.nan])


def test_no_true_shape_or_outcome_parameter_is_a_learner_input():
    assert set(inspect.signature(e27.nuisance).parameters) == {"source", "target", "penalty"}
    assert "BETA0" not in inspect.getsource(e27.nuisance)
    assert "true_shape" not in inspect.getsource(e27.nuisance)
    assert set(inspect.signature(SourceMoments).parameters) >= {"h", "b"}
    assert set(inspect.signature(TargetMoments).parameters) >= {"h", "b"}


def test_uniform_and_target_inverse_endpoints_stay_strictly_inside():
    class EndpointGenerator:
        def integers(self, low, high, size):
            assert low == 1 and high == 1 << 53 and size == 2
            return np.array([low, high - 1], dtype=np.int64)

    uniform = e27.interior_uniform(EndpointGenerator(), 2)
    root = np.cbrt(1 + 7 * uniform)
    z = 7 * uniform / (root**2 + root + 1)
    assert ((uniform > 0) & (uniform < 1)).all()
    assert ((z > 0) & (z < 1)).all()
    for b in e27.B_VALUES:
        u = uniform ** (1 / (1 - b))
        assert ((u > 0) & (u < 1)).all()


def test_normal_replicate_and_numerical_raw_roundtrip(tmp_path):
    task = e27.tasks_for()[0]
    result = e27.replicate(task)
    rows = e27.raw_rows(task, result)
    raw = pd.DataFrame(rows)
    path = tmp_path / "raw.parquet"
    raw.to_parquet(path, index=False)
    restored = pd.read_parquet(path)
    pd.testing.assert_frame_equal(raw, restored)
    assert set(raw.arm) == {"SQ", "UKL"} and raw.status.eq("ok").all()
    assert np.isfinite(raw.select_dtypes(include=[np.number]).to_numpy()).all()
    for fold in range(2):
        assert raw[f"fold{fold}_source_observations"].eq(512).all()
        assert raw[f"fold{fold}_target_observations"].eq(512).all()
        assert raw[f"fold{fold}_scale_positive"].eq(1).all()
        assert (raw[f"fold{fold}_normal_equation_scaled"] < 1e-12).all()
        assert (raw[f"fold{fold}_pilot_stationarity_scaled"] < 1e-12).all()


@pytest.mark.parametrize(
    "b, expected",
    [
        (0.28, (0.9426801992, 0.9456301151, 0.9475601126, 0.9486960150)),
        (0.35, (0.9179894982, 0.9129900586, 0.9094780318, 0.9071937315)),
        (0.42, (0.8256204455, 0.6769599706, 0.4439491339, 0.2439983916)),
    ],
)
def test_gaussian_stable_reference_matches_independent_values_and_formula(b, expected):
    from scipy import special

    actual = [e27.population(b, n)["gaussian_stable_coverage_reference"] for n in e27.N_VALUES]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=6e-11)
    facts = e27.population(b, e27.N_VALUES[-1])
    index = 1 / (2 * b)
    h = -facts["kappa"] * facts["theta0"] / (2 * facts["A0"] * np.sqrt(facts["V"]))
    gamma = special.gamma(1 - index)
    cosine = np.cos(np.pi * index / 2)
    amplitude = 2 * facts["D_s"] * gamma * cosine * h**index
    assert 1 < index < 2 and facts["theta0"] < 0 and h > 0
    assert gamma < 0 and cosine < 0 and amplitude > 0
    # A finite interval quadrature independently checks the infinite-interval implementation.
    # Beyond t=16, the absolute integrand is bounded by exp(-t^2/2)/t.
    reference = (
        2
        / np.pi
        * integrate.quad(
            lambda t: (
                np.sin(e27.Z_975 * t)
                / t
                * np.exp(-(t**2) / 2 - amplitude * t**index)
                * np.cos(amplitude * t**index * np.tan(np.pi * index / 2))
            ),
            0,
            16,
            epsabs=1e-12,
            epsrel=1e-12,
            limit=300,
        )[0]
    )
    assert actual[-1] == pytest.approx(reference, abs=3e-12)
    gaussian = e27.gaussian_stable_coverage(
        b, facts["D_s"], 0, facts["theta0"], facts["A0"], facts["V"]
    )
    assert gaussian == pytest.approx(0.95, abs=2e-12)
    assert all(0 < value < gaussian for value in actual)


@pytest.mark.parametrize("b", (0.2, 0.25, 0.5, 0.6, np.nan))
def test_gaussian_stable_reference_rejects_exponents_outside_stated_domain(b):
    with pytest.raises(ValueError, match="1/4 < b < 1/2"):
        e27.gaussian_stable_coverage(b, 1, 1, -1, 1, 1)


def _runner_pool(monkeypatch, future_for):
    spec = importlib.util.spec_from_file_location(
        "run_e27", Path(e27.__file__).parents[1] / "run_e27.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    class Pool:
        def __init__(self):
            self.tasks = []
            self.futures = []
            self.exited = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.exited = True

        def submit(self, function, task):
            self.tasks.append(task)
            future = future_for(self, function, task)
            self.futures.append(future)
            return future

    pool = Pool()
    monkeypatch.setattr(runner, "ProcessPoolExecutor", lambda **kwargs: pool)
    return runner, pool


def _completed_future(rows=None, error=None):
    from concurrent.futures import Future

    future = Future()
    if error is None:
        future.set_result(rows)
    else:
        future.set_exception(error)
    return future


def test_runner_preserves_simultaneous_success_without_submitting_more(tmp_path, monkeypatch):
    error = ArithmeticError("worker failed")

    def future_for(pool, function, task):
        if len(pool.tasks) == 1:
            return _completed_future([{"cell": 0, "rep": 0, "arm": "SQ", "value": 7}])
        return _completed_future(error=error)

    runner, pool = _runner_pool(monkeypatch, future_for)
    destination = tmp_path / "results"
    with pytest.raises(ArithmeticError) as caught:
        runner.run(directory=destination)
    assert caught.value is error
    assert len(pool.tasks) == 2 and pool.exited
    shards = list((destination / "raw").glob("*.parquet"))
    assert len(shards) == 1 and pd.read_parquet(shards[0]).value.tolist() == [7]
    assert not (destination / "summary.csv").exists()


def test_runner_drains_success_finishing_after_worker_failure(tmp_path, monkeypatch):
    from concurrent.futures import Future

    error = ArithmeticError("worker failed")
    late = Future()

    class FailingFuture(Future):
        def result(self, timeout=None):
            late.set_result([{"cell": 0, "rep": 1, "arm": "SQ", "value": 11}])
            return super().result(timeout)

    failed = FailingFuture()
    failed.set_exception(error)
    runner, pool = _runner_pool(
        monkeypatch, lambda pool, function, task: failed if len(pool.tasks) == 1 else late
    )
    destination = tmp_path / "results"
    with pytest.raises(ArithmeticError) as caught:
        runner.run(directory=destination)
    assert caught.value is error
    assert len(pool.tasks) == 2 and pool.exited
    raw = pd.read_parquet(destination / "raw" / "part-000001.parquet")
    assert raw.rep.tolist() == [1] and raw.value.tolist() == [11]
    assert not (destination / "summary.csv").exists()


@pytest.mark.parametrize("stage", ("write", "close", "write_and_close"))
def test_shard_failure_closes_writer_and_keeps_previous_shard(tmp_path, monkeypatch, stage):
    import pyarrow.parquet as pq

    runner, _ = _runner_pool(monkeypatch, lambda *args: None)
    rows = [{"cell": 0, "rep": 0, "arm": "SQ", "value": 13}]
    prior = runner._write_shard(tmp_path, 0, rows)
    real_writer = pq.ParquetWriter
    write_error, close_error = OSError("write failed"), OSError("close failed")
    calls = {"write": 0, "close": 0}

    class Writer:
        def __init__(self, *args, **kwargs):
            self.writer = real_writer(*args, **kwargs)

        def write_table(self, table):
            calls["write"] += 1
            self.writer.write_table(table)
            if stage in ("write", "write_and_close"):
                raise write_error

        def close(self):
            calls["close"] += 1
            self.writer.close()
            if stage in ("close", "write_and_close"):
                raise close_error

    monkeypatch.setattr(pq, "ParquetWriter", Writer)
    with pytest.raises(OSError) as caught:
        runner._write_shard(tmp_path, 1, rows)
    assert caught.value is (close_error if stage == "close" else write_error)
    assert calls == {"write": 1, "close": 1}
    assert list(tmp_path.iterdir()) == [prior]
    assert pd.read_parquet(prior).value.tolist() == [13]


def test_runner_does_not_retry_failed_storage_and_drains_pending_success(tmp_path, monkeypatch):
    from concurrent.futures import Future

    import pyarrow.parquet as pq

    retrieved = []

    class RecordedFuture(Future):
        def result(self, timeout=None):
            retrieved.append(self)
            return super().result(timeout)

    late = RecordedFuture()

    def future_for(pool, function, task):
        if len(pool.tasks) == 4:
            return late
        future = RecordedFuture()
        future.set_result([{"value": len(pool.tasks)}] * (128 if len(pool.tasks) < 3 else 256))
        return future

    runner, pool = _runner_pool(monkeypatch, future_for)
    real_writer = pq.ParquetWriter
    error = OSError("storage failed")
    calls = {"writers": 0, "write": 0, "close": 0}

    class Writer:
        def __init__(self, *args, **kwargs):
            calls["writers"] += 1
            self.number = calls["writers"]
            self.writer = real_writer(*args, **kwargs)

        def write_table(self, table):
            calls["write"] += 1
            self.writer.write_table(table)
            if self.number == 2:
                late.set_result([{"value": 4}])
                raise error

        def close(self):
            calls["close"] += 1
            self.writer.close()

    monkeypatch.setattr(pq, "ParquetWriter", Writer)
    destination = tmp_path / "results"
    with pytest.raises(OSError) as caught:
        runner.run(directory=destination)
    assert caught.value is error
    assert len(pool.tasks) == 4 and pool.exited and retrieved == pool.futures
    assert calls == {"writers": 2, "write": 2, "close": 2}
    shards = list((destination / "raw").iterdir())
    assert len(shards) == 1 and shards[0].name == "part-000001.parquet"
    assert pd.read_parquet(shards[0]).value.tolist() == [1] * 128 + [2] * 128
    assert not (destination / "summary.csv").exists()


def test_runner_preserves_worker_exception_when_short_buffer_storage_fails(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    error = ArithmeticError("worker failed first")
    runner, pool = _runner_pool(
        monkeypatch,
        lambda pool, function, task: (
            _completed_future([{"value": 1}])
            if len(pool.tasks) == 1
            else _completed_future(error=error)
        ),
    )
    real_writer = pq.ParquetWriter
    closed = []

    class Writer:
        def __init__(self, *args, **kwargs):
            self.writer = real_writer(*args, **kwargs)

        def write_table(self, table):
            raise OSError("write failed later")

        def close(self):
            self.writer.close()
            closed.append(True)
            raise OSError("close failed later")

    monkeypatch.setattr(pq, "ParquetWriter", Writer)
    destination = tmp_path / "results"
    with pytest.raises(ArithmeticError) as caught:
        runner.run(directory=destination)
    assert caught.value is error
    assert len(pool.tasks) == 2 and pool.exited and closed == [True]
    assert list((destination / "raw").iterdir()) == []
    assert not (destination / "summary.csv").exists()


@pytest.mark.parametrize("duplicate", (False, True))
def test_runner_reads_all_closed_shards_and_aggregates_fixed_design(
    tmp_path, monkeypatch, duplicate
):
    records = synthetic_raw().to_dict("records")
    runner, pool = _runner_pool(
        monkeypatch, lambda pool, function, task: _completed_future(function(task))
    )

    def compute(task):
        cell, rep, _ = task
        rows = [records[(2 * cell + arm) * e27.R + rep] for arm in range(2)]
        if duplicate and cell == rep == 0:
            rows[1] = rows[0].copy()
        return rows

    monkeypatch.setattr(runner, "_compute", compute)
    destination = tmp_path / "results"
    if duplicate:
        with pytest.raises(ValueError, match="duplicate replications and losses"):
            runner.run(directory=destination, n_jobs=3000)
        assert pool.tasks == e27.tasks_for() and pool.exited
        assert not (destination / "summary.csv").exists()
        return
    summary_path = runner.run(directory=destination, n_jobs=3000)
    assert pool.tasks == e27.tasks_for() and pool.exited
    shards = sorted((destination / "raw").glob("*.parquet"))
    assert len(shards) == 8
    restored = pd.concat([pd.read_parquet(path) for path in shards], ignore_index=True)
    sort_keys = ["cell", "rep", "arm"]
    pd.testing.assert_frame_equal(
        restored.sort_values(sort_keys).reset_index(drop=True),
        pd.DataFrame(records).sort_values(sort_keys).reset_index(drop=True),
    )
    assert len(pd.read_csv(summary_path)) == 24
    assert len(list(destination.glob("tab_E27*.tex"))) == 5
    assert len(list(destination.glob("fig_E27*.pdf"))) == 3
