"""Independent population, sampling, and normal-design E-29 checks."""

import inspect
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from grrexp import e29

from genriesz.partition_regression import predict, predict_ratio


def test_complete_design_and_observed_data_only_learner():
    assert e29.R == 2000 and e29.K == 2
    assert e29.CELLS == tuple(itertools.product(e29.DGPS, (8192, 32768, 131072)))
    assert len(e29.tasks_for()) == 24000 and len(e29.METHODS) == 5
    np.testing.assert_array_equal(e29.PENALTIES, [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1])
    assert tuple(inspect.signature(e29.fit_nuisances).parameters) == ("data", "h", "j")
    source = inspect.getsource(e29.fit_nuisances)
    for name in ("population", "true_ratio", "normalizer", "outcome", "dgp", "B_VALUES"):
        assert name not in source
    dimensions = []
    for dgp, n in e29.CELLS:
        facts = e29.population(dgp, n)
        assert facts["L"] == n // 4
        assert facts["H"] == np.ceil(2 * facts["L"] ** (1 / 9))
        assert facts["d"] == 4 * facts["H"]
        assert facts["J"] == round(facts["L"] ** 0.75 / 4)
        if dgp == "Log":
            dimensions.append(facts["d"])
    assert dimensions[0] < dimensions[1] < dimensions[2]
    with pytest.raises(ValueError, match="complete E-29 design"):
        e29.population("Log", 1000)


@pytest.mark.parametrize("dgp", e29.DGPS)
def test_population_constants_and_cell_cdfs_match_independent_quadrature(dgp):
    facts = e29.population(dgp, 8192)
    k, b = facts["k"], e29.B_VALUES[dgp]

    def logratio(t):
        return k * (1 + t) ** 2 if dgp == "Log" else k * np.exp(b * t) * (1 - 0.25 * np.sin(t))

    def density(t):
        return (2 / 3) * (np.exp(-t) + np.exp(-2 * t))

    def ratio2density(t):
        if dgp == "Log":
            return k**2 * (1 + t) ** 4 * density(t)
        return (
            k**2
            * (2 / 3)
            * (np.exp(-(1 - 2 * b) * t) + np.exp(-(2 - 2 * b) * t))
            * (1 - 0.25 * np.sin(t)) ** 2
        )

    def target_density(t):
        if dgp == "Log":
            return k * (1 + t) ** 2 * density(t)
        return k * (2 / 3) * (np.exp(-(1 - b) * t) + np.exp(-(2 - b) * t)) * (1 - 0.25 * np.sin(t))

    # Factoring the exponential avoids overflow in the independent oracle formula.
    source_variance = integrate.quad(
        ratio2density, 0, np.inf, epsabs=1e-11, epsrel=2e-12, limit=200
    )[0]
    theta = integrate.quad(
        lambda t: target_density(t) * (10 * np.exp(-t) + np.sin(4 * np.pi * np.exp(-t))),
        0,
        np.inf,
        epsabs=1e-11,
        epsrel=2e-12,
    )[0]
    target_second = integrate.quad(
        lambda t: target_density(t) * (10 * np.exp(-t) + np.sin(4 * np.pi * np.exp(-t))) ** 2,
        0,
        np.inf,
        epsabs=1e-11,
        epsrel=2e-12,
    )[0]
    assert facts["source_variance"] == pytest.approx(source_variance, rel=2e-13)
    assert facts["theta0"] == pytest.approx(theta, abs=2e-12)
    assert facts["target_variance"] == pytest.approx(target_second - theta**2, abs=3e-12)
    edges = np.array([0, 0.0001, 0.01, 0.2, 0.7, 1])
    cdf = e29.target_cdf(edges, dgp)
    for i in range(1, len(edges)):
        upper, lower = -np.log(edges[i]), -np.log(edges[i - 1]) if edges[i - 1] else np.inf
        expected = integrate.quad(
            lambda t: e29.target_t_density(t, dgp), upper, lower, epsabs=1e-12, epsrel=2e-12
        )[0]
        assert cdf[i] - cdf[i - 1] == pytest.approx(expected, abs=2e-13)
    for u in (0.001, 0.05, 0.4, 0.8):
        step = u * 1e-5
        derivative = (
            e29.target_cdf(np.array([u + step]), dgp)[0]
            - e29.target_cdf(np.array([u - step]), dgp)[0]
        ) / (2 * step)
        assert derivative == pytest.approx(
            float(e29.true_ratio(u, dgp)) * 2 * (1 + u) / 3, rel=1e-8
        )
        assert e29.true_ratio(np.exp(-u), dgp) == pytest.approx(logratio(u))
    assert facts["V"] == pytest.approx(2 * (facts["source_variance"] + facts["target_variance"]))


@pytest.mark.parametrize("dgp", e29.DGPS)
def test_exact_target_sampler_matches_analytic_cdf_without_clipping(dgp):
    rng = np.random.default_rng(10933)
    sample = e29.sample_target(rng, 100000, dgp)
    assert np.all((sample > 0) & (sample < 1))
    statistic = stats.kstest(sample, lambda u: e29.target_cdf(u, dgp)).statistic
    assert statistic < 0.007
    source = e29.sample_source(np.random.default_rng(22134), 100000)
    assert stats.kstest(source, e29.source_cdf).statistic < 0.007


@pytest.fixture(scope="module")
def normal_records():
    return {cell: e29.replicate((cell, 19)) for cell in (0, 3, 6, 9)}


def test_normal_replicates_share_both_loss_minimizers_and_use_growing_regression(normal_records):
    for cell, rows in normal_records.items():
        facts = e29.population(*e29.CELLS[cell])
        assert len(rows) == 5
        assert {(row["regression"], row["loss"]) for row in rows} == set(e29.METHODS)
        for regression in e29.REGRESSIONS:
            sq = next(
                row for row in rows if row["regression"] == regression and row["loss"] == "SQ"
            )
            ukl = next(
                row for row in rows if row["regression"] == regression and row["loss"] == "UKL"
            )
            assert {k: v for k, v in sq.items() if k != "loss"} == {
                k: v for k, v in ukl.items() if k != "loss"
            }
            assert sq["dimension"] == facts["d"] and sq["ratio_cells"] == facts["J"]
            for fold in range(2):
                assert sq[f"fold{fold}_ratio_min"] > 0
                assert sq[f"fold{fold}_lambda"] in e29.PENALTIES
                assert sq[f"fold{fold}_gram_min_eigenvalue"] > 0
                assert sq[f"fold{fold}_ratio_l2"] > 0
                assert sq[f"fold{fold}_regression_l2"] > 0
                for field in ("SQ_stationarity", "UKL_stationarity", "normal_equation_scaled"):
                    assert sq[f"fold{fold}_{field}"] < 1e-13
            if regression == "OLS":
                assert sq["fold0_lambda"] == sq["fold1_lambda"] == 0


def test_crossfit_estimate_standard_error_and_oracle_are_actual_sample_scores(normal_records):
    cell, rep = 9, 19
    dgp, n = e29.CELLS[cell]
    facts = e29.population(dgp, n)
    data = [e29.generate_fold(cell, rep, fold) for fold in range(2)]
    fits = [e29.fit_nuisances(d, facts["H"], facts["J"]) for d in data]
    for row in normal_records[cell]:
        source, target = [], []
        for evaluation in range(2):
            d = data[evaluation]
            if row["regression"] == "Oracle":
                source.append(e29.true_ratio(d["u"], dgp) * d["epsilon"])
                target.append(e29.outcome(d["tu"]))
            else:
                ratio, regressions, _ = fits[1 - evaluation]
                coef = regressions[row["regression"]][0]
                source.append(predict_ratio(d["u"], ratio) * (d["y"] - predict(d["u"], coef)))
                target.append(predict(d["tu"], coef))
        source, target = np.concatenate(source), np.concatenate(target)
        theta = float(source.mean() + target.mean())
        vs, vt = float(np.mean(source**2)), float(np.mean((target - theta) ** 2))
        assert row["theta"] == pytest.approx(theta, abs=5e-14)
        assert row["source_variance"] == pytest.approx(vs, rel=2e-14)
        assert row["target_variance"] == pytest.approx(vt, rel=2e-14)
        assert row["se"] == pytest.approx(np.sqrt(2 * (vs + vt) / n), rel=2e-14)
        assert row["covered"] == (abs(theta - facts["theta0"]) <= stats.norm.ppf(0.975) * row["se"])


def test_exact_l2_diagnostics_equal_independent_deterministic_integration():
    dgp, n = "Osc042", 8192
    facts = e29.population(dgp, n)
    data = e29.generate_fold(9, 19, 0)
    ratio, regressions, _ = e29.fit_nuisances(data, facts["H"], facts["J"])
    coef = regressions["CV"][0]
    diag = e29.population_diagnostics(ratio, coef, facts)
    ratio_error = 0.0
    edges = np.linspace(0, 1, facts["J"] + 1)
    for j in range(facts["J"]):
        # The change to log coordinates handles the first unbounded-ratio cell.
        left, right = edges[j], edges[j + 1]
        if left == 0:
            integral = integrate.quad(
                lambda t, j=j: (
                    (
                        ratio[j] * np.exp(-e29.B_VALUES[dgp] * t)
                        - facts["k"] * (1 - 0.25 * np.sin(t))
                    )
                    ** 2
                    * (2 / 3)
                    * (
                        np.exp(-(1 - 2 * e29.B_VALUES[dgp]) * t)
                        + np.exp(-(2 - 2 * e29.B_VALUES[dgp]) * t)
                    )
                ),
                -np.log(right),
                np.inf,
                epsabs=1e-11,
                epsrel=2e-12,
                limit=200,
            )[0]
        else:
            integral = integrate.quad(
                lambda u, j=j: float((ratio[j] - e29.true_ratio(u, dgp)) ** 2) * 2 * (1 + u) / 3,
                left,
                right,
                epsabs=1e-11,
            )[0]
        ratio_error += integral
    regression_error = 0.0
    for j in range(facts["H"]):
        regression_error += integrate.quad(
            lambda u: (
                float((predict(np.array([u]), coef)[0] - e29.outcome(u)) ** 2) * 2 * (1 + u) / 3
            ),
            j / facts["H"],
            (j + 1) / facts["H"],
            epsabs=1e-12,
        )[0]
    assert diag["ratio_l2"] ** 2 == pytest.approx(ratio_error, abs=2e-12)
    assert diag["regression_l2"] ** 2 == pytest.approx(regression_error, abs=3e-13)


def test_roundtrip_and_incomplete_aggregation_rejection(normal_records, tmp_path):
    raw = pd.DataFrame(normal_records[0])
    file = tmp_path / "records.parquet"
    raw.to_parquet(file, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(file), raw)
    with pytest.raises(ValueError, match="complete E-29"):
        e29.summarise(raw)


@pytest.mark.parametrize(
    "field", ["fold0_SQ_stationarity", "fold1_UKL_stationarity", "fold0_normal_equation_scaled"]
)
def test_actual_first_order_residuals_are_checked(normal_records, field):
    record = {**normal_records[0][0], field: 2e-12}
    with pytest.raises(ArithmeticError, match="first-order equation"):
        e29.check_result(record)


def test_random_streams_are_independent_and_reproducible():
    reference = e29.generator(0, 19, 0, 0).random(50)
    np.testing.assert_array_equal(reference, e29.generator(0, 19, 0, 0).random(50))
    for args in ((1, 19, 0, 0), (0, 20, 0, 0), (0, 19, 1, 0), (0, 19, 0, 1)):
        assert not np.array_equal(reference, e29.generator(*args).random(50))
    with pytest.raises(ValueError):
        e29.generator(12, 0, 0, 0)


@pytest.fixture(scope="module")
def fabricated_records(normal_records):
    """A complete artificial record set tests aggregation without simulated refits."""
    blocks = []
    for cell, (dgp, n) in enumerate(e29.CELLS):
        facts = e29.population(dgp, n)
        for template in normal_records[3 * e29.DGPS.index(dgp)]:
            block = pd.DataFrame([template] * e29.R)
            block["cell"], block["n"] = cell, n
            block["rep"] = np.arange(e29.R)
            block["L"], block["H"] = facts["L"], facts["H"]
            block["dimension"], block["ratio_cells"] = facts["d"], facts["J"]
            blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def test_complete_aggregation_matches_independent_counts_and_error_metrics(fabricated_records):
    summary = e29.summarise(fabricated_records)
    assert len(summary) == 60
    for row in summary.itertuples(index=False):
        actual = fabricated_records[
            (fabricated_records.cell == row.cell)
            & (fabricated_records.regression == row.regression)
            & (fabricated_records.loss == row.loss)
        ]
        assert row.covered == actual.covered.sum() / 2000
        assert row.bias == pytest.approx(actual.error.iloc[0], abs=1e-15)
        assert row.rmse == pytest.approx(abs(actual.error.iloc[0]), abs=1e-15)
        interval = stats.binomtest(int(actual.covered.sum()), 2000).proportion_ci(method="exact")
        assert (row.covered_lo, row.covered_hi) == pytest.approx((interval.low, interval.high))
    tables = e29.tables(summary)
    assert tables["tab_E29_coverage"].count(r" \\") == 13
    figures = e29.figures(summary)
    import matplotlib.pyplot as plt

    for figure in figures.values():
        assert len(figure.axes) == 4
        for axis in figure.axes:
            assert axis.get_xlim()[0] < min(e29.N_VALUES)
            assert axis.get_xlim()[1] > max(e29.N_VALUES)
        plt.close(figure)


@pytest.mark.parametrize("failure", ["duplicate", "missing", "loss", "diagnostic"])
def test_complete_aggregation_rejects_key_and_diagnostic_errors(fabricated_records, failure):
    raw = fabricated_records.copy()
    if failure == "duplicate":
        raw.loc[1, "rep"] = raw.loc[0, "rep"]
    elif failure == "missing":
        raw = raw.iloc[:-1]
    elif failure == "loss":
        raw.loc[(raw.regression == "CV") & (raw.loss == "UKL"), "theta"] += 0.01
    else:
        raw.loc[0, "fold0_normal_rank"] = np.nan
    with pytest.raises(ValueError):
        e29.summarise(raw)


@pytest.mark.parametrize("name", ["experiments/run_e29.py", "experiments/apps/e29.py"])
def test_e29_source_closure_verifies_runner_and_marimo_app(tmp_path, name):
    from run_e28 import _verify_source
    from run_e29 import EXTRA_PATHS
    from test_run_e28 import git

    names = [
        "src/genriesz/__init__.py",
        "experiments/grrexp/__init__.py",
        "experiments/grrexp/e29.py",
        "src/genriesz/partition_regression.py",
        "pyproject.toml",
        "experiments/run_e28.py",
        "experiments/run_e27.py",
        "experiments/pyproject.toml",
        "experiments/uv.lock",
        "experiments/.python-version",
        *EXTRA_PATHS,
    ]
    for relative in names:
        file = tmp_path / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(relative + "\n")
    git(tmp_path, "init", "-q", "--object-format=sha1")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "Pin source inputs")
    commit = git(tmp_path, "rev-parse", "HEAD")
    assert set(_verify_source(tmp_path, commit, EXTRA_PATHS)) == set(names)
    (tmp_path / name).write_text("different scientific input\n")
    with pytest.raises(RuntimeError, match="differs from the expected reviewed commit"):
        _verify_source(tmp_path, commit, EXTRA_PATHS)


@pytest.mark.parametrize("exit_code", [0, 7])
def test_marimo_launch_uses_a_fresh_interpreter_and_propagates_failure(
    tmp_path, monkeypatch, exit_code
):
    import ast
    import json
    import os
    import subprocess
    from types import ModuleType, SimpleNamespace

    app_path = Path(__file__).resolve().parents[1] / "experiments/apps/e29.py"
    app_tree = ast.parse(app_path.read_text())
    launch = next(
        node
        for node in app_tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "subprocess"
            and call.func.attr == "run"
            for call in ast.walk(node)
        )
    )
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module == "run_e29" for node in ast.walk(app_tree)
    )
    launch.decorator_list = []
    launch.name = "launch_full_design"
    namespace = {}
    exec(compile(ast.Module(body=[launch], type_ignores=[]), str(app_path), "exec"), namespace)

    cached_runner = ModuleType("run_e29")

    def stale_run(*args, **kwargs):
        raise AssertionError("the persistent app executed its stale imported runner")

    cached_runner.run = stale_run
    monkeypatch.setitem(sys.modules, "run_e29", cached_runner)
    runner_script = tmp_path / "run_e29.py"
    runner_script.write_text(
        "import json, os, sys\n"
        "print(json.dumps({'pid': os.getpid(), 'argv': sys.argv}))\n"
        f"raise SystemExit({exit_code})\n"
    )
    invocations = []

    def execute(arguments, **kwargs):
        assert arguments[0] == sys.executable
        assert arguments[1] == str(runner_script)
        assert arguments[2:] == [
            "--output",
            str(tmp_path / "results"),
            "--expected-source-commit",
            "a" * 40,
            "--n-jobs",
            "4",
        ]
        assert kwargs == dict(check=True, capture_output=True, text=True)
        result = subprocess.run(arguments, **kwargs)
        invocations.append(json.loads(result.stdout))
        return result

    arguments = dict(
        mo=SimpleNamespace(stop=lambda condition: None, md=lambda text: text),
        n_jobs=SimpleNamespace(value=4),
        output_directory=SimpleNamespace(value=str(tmp_path / "results")),
        python_executable=sys.executable,
        run_button=SimpleNamespace(value=True),
        runner_script=runner_script,
        source_commit=SimpleNamespace(value="a" * 40),
        subprocess=SimpleNamespace(run=execute),
    )
    if exit_code:
        with pytest.raises(subprocess.CalledProcessError) as failure:
            namespace["launch_full_design"](**arguments)
        assert failure.value.returncode == exit_code
    else:
        namespace["launch_full_design"](**arguments)
        assert len(invocations) == 1 and invocations[0]["pid"] != os.getpid()
