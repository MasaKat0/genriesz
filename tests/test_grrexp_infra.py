from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "experiments"))

from grrexp import bootstrap, env, inference, metrics, outputs, parallel  # noqa: E402
from grrexp.seeds import Seeds  # noqa: E402

# ---------------------------------------------------------------- metrics (§1.5)


def test_cp_upper_matches_registration_value() -> None:
    assert metrics.clopper_pearson_upper(0, 200) == pytest.approx(0.0149, abs=5e-5)
    assert metrics.failure_rate(np.ones(200, bool))["failure_rate_cp_upper"] == pytest.approx(
        1 - 0.05 ** (1 / 200)
    )


def test_coverage_counts_failures_as_not_covering() -> None:
    est = np.array([0.0, 0.0, np.nan, 5.0])
    se = np.array([1.0, 1.0, np.nan, 1.0])
    ok = np.array([True, True, False, True])
    out = metrics.coverage(est, se, ok, theta0=0.0)
    assert out["covered"] == 2
    assert out["coverage"] == 0.5
    assert out["coverage_conditional"] == pytest.approx(2 / 3)


def test_ok_with_nonfinite_estimate_is_rejected() -> None:
    with pytest.raises(ValueError):
        metrics.bias(np.array([np.nan, 1.0, 2.0]), np.array([True, True, True]), 0.0)


def test_sd_mcse_formula() -> None:
    x = np.random.default_rng(0).normal(size=500)
    var = x.var(ddof=1)
    m4 = np.mean((x - x.mean()) ** 4)
    expected = np.sqrt((m4 - var**2) / (4 * var * x.size))
    assert metrics.sd_mcse(x) == pytest.approx(expected)
    out = metrics.root_n_sd(x, np.ones(500, bool), n=100)
    assert out["root_n_sd"] == pytest.approx(10 * x.std(ddof=1))
    assert out["root_n_sd_mcse"] == pytest.approx(10 * expected)


def test_ess_and_imbalance() -> None:
    assert metrics.ess(np.array([1.0, -1.0, 1.0, -1.0])) == pytest.approx(4.0)
    alpha = np.array([2.0, 0.0])
    phi = np.array([[1.0], [1.0]])
    m_phi = np.array([[0.5], [0.5]])
    assert metrics.imbalance(alpha, m_phi, phi) == pytest.approx(0.5)


# ---------------------------------------------------------------- inference (§1.8, §1.9)


def test_predicted_coverage_nominal_and_biased() -> None:
    nominal = stats.norm.cdf(1.96) - stats.norm.cdf(-1.96)
    assert inference.predicted_coverage(1.0, 1.0, 0.0, 100) == pytest.approx(nominal)
    assert inference.predicted_coverage(1.0, 1.0, 0.1, 100) < nominal


def test_coverage_pvalue_two_sided_calibration() -> None:
    # Inside the tolerance band the p-value is large; far below it, tiny.
    assert inference.coverage_pvalue(950, 1000, 0.95) == 1.0
    assert inference.coverage_pvalue(880, 1000, 0.95) < 1e-6
    k, R, lo, hi = 925, 1000, 0.93, 0.97
    expected = min(1, 2 * min(stats.binom.cdf(k, R, lo), stats.binom.sf(k - 1, R, hi)))
    assert inference.coverage_pvalue(k, R, 0.95) == pytest.approx(expected)


def test_interval_null_pvalue() -> None:
    assert inference.interval_null_pvalue(0.0, 1.0, -1.0, 1.0) == 1.0
    p = inference.interval_null_pvalue(5.0, 1.0, -1.0, 1.0)
    assert p == pytest.approx(2 * stats.norm.sf(4.0))


def test_sd_ratio_pvalue_uses_ratio_scale() -> None:
    x = np.random.default_rng(1).normal(scale=2.0 / np.sqrt(400), size=4000)
    assert inference.sd_ratio_pvalue(x, sigma_true=2.0, n=400) > 0.5
    assert inference.sd_ratio_pvalue(x, sigma_true=1.0, n=400) < 1e-6


def test_failure_rate_pvalue_is_one_sided_exact() -> None:
    assert inference.failure_rate_pvalue(0, 200, 0.01) == 1.0
    assert inference.failure_rate_pvalue(10, 200, 0.01) == pytest.approx(
        stats.binom.sf(9, 200, 0.01)
    )


def test_holm_step_down() -> None:
    rej = inference.holm([0.001, 0.02, 0.004, 0.5], level=0.05)
    # sorted: 0.001<=0.0125, 0.004<=0.0167, 0.02<=0.025, 0.5>0.05
    assert rej.tolist() == [True, True, True, False]
    rej = inference.holm([0.004, 0.011], level=0.01)
    assert rej.tolist() == [True, False]


def test_family_verdict_sentences() -> None:
    ok = inference.judge_family("H12-Inf", {"a": 0.5, "b": 0.9})
    assert not ok.negative and "no departure" in ok.sentence()
    bad = inference.judge_family("H12-Inf", {"a": 1e-5, "b": 0.9})
    assert bad.negative and bad.rejected == ("a",)


# ---------------------------------------------------------------- bootstrap (§1.9)


def test_bootstrap_is_chunk_invariant_and_registered_b() -> None:
    x = np.arange(30.0)
    a = bootstrap.FamilyBootstrap(Seeds(12), 1).replicate(lambda idx: x[idx].mean(axis=1), 30)
    b = bootstrap.FamilyBootstrap(Seeds(12), 1).replicate(
        lambda idx: x[idx].mean(axis=1), 30, chunk=7
    )
    assert a.shape == (2000,)
    assert np.array_equal(a, b)
    c = bootstrap.FamilyBootstrap(Seeds(12), 2).replicate(lambda idx: x[idx].mean(axis=1), 30)
    assert c.shape == (10000,) and not np.array_equal(a, c[:2000])


def test_bootstrap_stream_is_registered_key() -> None:
    fb = bootstrap.FamilyBootstrap(Seeds(13), 2)
    first = fb.replicate(lambda idx: idx[:, 0].astype(float), 5, chunk=10000)
    ref = np.random.default_rng(np.random.SeedSequence(20261007, spawn_key=(13, 9999, 2, 4)))
    assert np.array_equal(first, ref.integers(0, 5, size=(10000, 5))[:, 0])


def test_successive_calls_are_independent_draws() -> None:
    fb = bootstrap.FamilyBootstrap(Seeds(14), 4)
    stat = lambda idx: idx.mean(axis=1)  # noqa: E731
    assert not np.array_equal(fb.replicate(stat, 20), fb.replicate(stat, 20))


def test_counts_mode_and_family7_statistic() -> None:
    rng = np.random.default_rng(3)
    ea, eb = rng.normal(size=50), 2 * rng.normal(size=50)
    stat = bootstrap.log_rmse_ratio_counts(ea, eb)
    unit = stat(np.ones((1, 50)))[0]
    assert unit == pytest.approx(np.log(np.sqrt(np.mean(ea**2)) / np.sqrt(np.mean(eb**2))))
    fb = bootstrap.FamilyBootstrap(Seeds(24), 7)
    boot = fb.replicate(stat, 50, mode="counts", chunk=50000)
    assert boot.shape == (200000,)
    p = bootstrap.centred_pvalue(boot, unit)
    assert p == pytest.approx((1 + np.sum(np.abs(boot - unit) >= abs(unit))) / 200001)


# ---------------------------------------------------------------- parallel (§1.2)


def _draw(task):
    cell, rep = task
    return Seeds(12).data(cell, rep).normal(size=4)


def test_parallel_runs_are_bit_identical(monkeypatch) -> None:
    for k in parallel.THREAD_VARIABLES:
        monkeypatch.setenv(k, "1")
    tasks = [(c, r) for c in range(2) for r in range(5)]
    serial = parallel.run_tasks(_draw, tasks, n_jobs=1)
    para = parallel.run_tasks(_draw, tasks, n_jobs=2)
    parallel.assert_bit_identical(serial, para)


def test_serial_run_requires_single_thread_env(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    with pytest.raises(RuntimeError):
        parallel.run_tasks(_draw, [(0, 0)], n_jobs=1)


# ---------------------------------------------------------------- outputs (§1.7)


def test_recorder_writes_manifest_and_refuses_rerun(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(env, "RESULTS_DIR", tmp_path)
    rec = outputs.RunRecorder(12, "stage0", R=None, n=None)
    rec.write_stage0("var", {"sigma2": 7.2906549})
    rec.finalize(warnings=0, failures=0)
    rec = outputs.RunRecorder(12, "stage1", R=3, n=2000)
    rec.write_raw(pd.DataFrame({"rep": [0, 1, 2], "estimate": [1.0, np.nan, 2.0]}))
    rec.write_macros({"EXIIcoverage": "0.948"})
    with pytest.raises(ValueError):
        rec.write_macros({"E12cov": "1"})
    path = rec.finalize(warnings=1, failures=1)
    manifest = json.loads(path.read_text())
    assert manifest["entropy"] == 20261007 and manifest["R"] == 3
    assert set(manifest["outputs_sha256"]) == {"raw.parquet", "macros_E-12.tex"}
    assert manifest["outputs_sha256"]["raw.parquet"] == env.sha256_file(
        tmp_path / "E-12" / "raw.parquet"
    )
    with pytest.raises(FileExistsError):
        outputs.RunRecorder(12, "stage1", R=3, n=2000)
    moved = outputs.archive_run(12, "bug fix in the outcome model")
    assert (moved / "raw.parquet").is_file() and (moved / "REASON.txt").is_file()
    assert (tmp_path / "E-12" / "stage0_var.json").is_file()
    outputs.RunRecorder(12, "stage1", R=3, n=2000)


def test_pilot_keeps_no_estimates(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(env, "RESULTS_DIR", tmp_path)
    rec = outputs.RunRecorder(13, "stage0.5", R=10, n=1000)
    with pytest.raises(ValueError):
        rec.write_raw(pd.DataFrame({"x": [1.0]}))


def test_data_check_rejects_missing_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(env, "DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        env.verify_data(["lalonde/lalonde.csv"])
    (tmp_path / "lalonde").mkdir()
    (tmp_path / "lalonde" / "lalonde.csv").write_text("x")
    with pytest.raises(ValueError):
        env.verify_data(["lalonde/lalonde.csv"])


def test_submodule_genriesz_is_the_imported_source() -> None:
    mod = env.use_submodule_genriesz()
    assert Path(mod.__file__).resolve().parents[1] == env.GENRIESZ_SRC
