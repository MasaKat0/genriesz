from __future__ import annotations

import json
import subprocess
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


@pytest.mark.parametrize("bad_se", [np.inf, -1.0, np.nan])
def test_invalid_se_of_a_success_stops(bad_se) -> None:
    est = np.array([0.0, 1.0, 2.0])
    se = np.array([1.0, bad_se, 1.0])
    ok = np.array([True, True, True])
    with pytest.raises(ValueError):
        metrics.coverage(est, se, ok, 0.0)
    with pytest.raises(ValueError):
        metrics.summarise(est, se, ok, 0.0, n=100)


def test_shapes_and_ok_type_are_checked() -> None:
    with pytest.raises(ValueError):
        metrics.coverage(np.zeros(3), np.ones(2), np.ones(3, bool), 0.0)
    with pytest.raises(ValueError):
        metrics.failure_rate(np.ones(3))  # not boolean
    with pytest.raises(ValueError):
        metrics.failure_rate(np.ones(3, bool), status=["ok", "ok"])


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


def test_negative_radicand_is_reported_not_truncated() -> None:
    # [-1, 1]: s^2 = 2 (ddof=1), m4 = 1, so m4 - s^4 = -3.
    with pytest.raises(ValueError, match="negative fourth-moment radicand"):
        metrics.sd_mcse(np.array([-1.0, 1.0]))
    with pytest.raises(ValueError, match="variance is zero"):
        metrics.sd_mcse(np.array([1.0, 1.0, 1.0]))


def test_ess_and_imbalance() -> None:
    assert metrics.ess(np.array([1.0, -1.0, 1.0, -1.0])) == pytest.approx(4.0)
    alpha = np.array([2.0, 0.0])
    phi = np.array([[1.0], [1.0]])
    m_phi = np.array([[0.5], [0.5]])
    assert metrics.imbalance(alpha, m_phi, phi) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        metrics.imbalance(alpha, m_phi[:1], phi)


# ---------------------------------------------------------------- inference (§1.8, §1.9)


def test_predicted_coverage_nominal_and_biased() -> None:
    nominal = stats.norm.cdf(1.96) - stats.norm.cdf(-1.96)
    assert inference.predicted_coverage(1.0, 1.0, 0.0, 100) == pytest.approx(nominal)
    assert inference.predicted_coverage(1.0, 1.0, 0.1, 100) < nominal
    with pytest.raises(ValueError):
        inference.predicted_coverage(1.0, 1.0, np.nan, 100)


def test_coverage_pvalue_two_sided_calibration() -> None:
    assert inference.coverage_pvalue(950, 1000, 0.95) == 1.0
    assert inference.coverage_pvalue(880, 1000, 0.95) < 1e-6
    k, R, lo, hi = 925, 1000, 0.93, 0.97
    expected = min(1, 2 * min(stats.binom.cdf(k, R, lo), stats.binom.sf(k - 1, R, hi)))
    assert inference.coverage_pvalue(k, R, 0.95) == pytest.approx(expected)


@pytest.mark.parametrize(
    "args", [(95, 100, np.nan), (95, 100, 1.5), (95.0, 100, 0.95), (101, 100, 0.95)]
)
def test_coverage_pvalue_rejects_invalid_inputs(args) -> None:
    with pytest.raises(ValueError):
        inference.coverage_pvalue(*args)


def test_interval_null_pvalue() -> None:
    assert inference.interval_null_pvalue(0.0, 1.0, -1.0, 1.0) == 1.0
    p = inference.interval_null_pvalue(5.0, 1.0, -1.0, 1.0)
    assert p == pytest.approx(2 * stats.norm.sf(4.0))
    with pytest.raises(ValueError):
        inference.interval_null_pvalue(0.0, 1.0, np.nan, 1.0)


def test_sd_ratio_pvalue_uses_ratio_scale() -> None:
    x = np.random.default_rng(1).normal(scale=2.0 / np.sqrt(400), size=4000)
    assert inference.sd_ratio_pvalue(x, sigma_true=2.0, n=400) > 0.5
    assert inference.sd_ratio_pvalue(x, sigma_true=1.0, n=400) < 1e-6


def test_failure_rate_pvalue_is_one_sided_exact() -> None:
    assert inference.failure_rate_pvalue(0, 200, 0.01) == 1.0
    expected = stats.binom.sf(9, 200, 0.01)
    assert inference.failure_rate_pvalue(10, 200, 0.01) == pytest.approx(expected)
    with pytest.raises(ValueError):
        inference.failure_rate_pvalue(1, 200, np.nan)


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

    def stat(idx):
        return idx.mean(axis=1)

    assert not np.array_equal(fb.replicate(stat, 20), fb.replicate(stat, 20))


def test_count_statistic_equals_index_statistic_with_same_counts() -> None:
    rng = np.random.default_rng(3)
    ea, eb = rng.normal(size=40), 2 * rng.normal(size=40)
    stat = bootstrap.log_rmse_ratio_counts(ea, eb)
    idx = rng.integers(0, 40, size=(50, 40))
    counts = np.stack([np.bincount(row, minlength=40) for row in idx])
    direct = np.log(np.sqrt(np.mean(ea[idx] ** 2, axis=1)) / np.sqrt(np.mean(eb[idx] ** 2, axis=1)))
    np.testing.assert_allclose(stat(counts), direct, rtol=1e-12)


def test_family7_counts_and_centred_pvalue() -> None:
    rng = np.random.default_rng(3)
    ea, eb = rng.normal(size=50), 2 * rng.normal(size=50)
    stat = bootstrap.log_rmse_ratio_counts(ea, eb)
    t_hat = stat(np.ones((1, 50)))[0]
    boot = bootstrap.FamilyBootstrap(Seeds(24), 7).replicate(stat, 50, mode="counts", chunk=50000)
    assert boot.shape == (200000,)
    p = bootstrap.centred_pvalue(boot, t_hat)
    assert p == pytest.approx((1 + np.sum(np.abs(boot - t_hat) >= abs(t_hat))) / 200001)
    with pytest.raises(ValueError):
        bootstrap.centred_pvalue(boot, np.nan)
    with pytest.raises(ValueError):
        bootstrap.centred_pvalue(boot[:10], t_hat)
    with pytest.raises(ValueError):
        bootstrap.percentile_interval(boot[:2000], family_id=2)


# ---------------------------------------------------------------- parallel (§1.2)


def _draw(task):
    cell, rep = task
    return Seeds(12).data(cell, rep).normal(size=4) @ np.ones((4, 4))


def test_parallel_runs_are_bit_identical(monkeypatch) -> None:
    for k in parallel.THREAD_VARIABLES:
        monkeypatch.setenv(k, "1")
    tasks = [(c, r) for c in range(2) for r in range(5)]
    digest = parallel.verify_parallel_identity(_draw, tasks, n_jobs=2)
    assert len(digest) == 64


def test_bit_identity_is_not_vacuous() -> None:
    with pytest.raises(AssertionError):
        parallel.assert_bit_identical([], [], expected=1)
    with pytest.raises(AssertionError):
        parallel.assert_bit_identical([1], [1, 2], expected=1)
    with pytest.raises(ValueError):
        parallel.run_tasks(_draw, [], n_jobs=1)


def test_serial_run_requires_single_thread_env(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    with pytest.raises(RuntimeError):
        parallel._call(_draw, (0, 0))


# ---------------------------------------------------------------- outputs (§1.7)

FAKE_SOURCE = {
    "sha": "a" * 40,
    "notebook": "notebooks/experiments/14_E16_exact_examples.ipynb",
    "notebook_code_cells_sha256": "b" * 64,
    "loaded_modules": {"src/genriesz/__init__.py": "c" * 40},
}
FAKE_ENV = {
    "platform": ["Darwin", "arm64"],
    "python": "3.13.6",
    "lock_sha256": "d" * 64,
    "lock_pins_verified": 109,
}


@pytest.fixture
def fake_run(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(env, "check_environment", lambda: dict(FAKE_ENV))
    state = {"source": dict(FAKE_SOURCE)}
    monkeypatch.setattr(env, "source_snapshot", lambda nb: dict(state["source"]))
    parent = {"sha": "e" * 40, "gitlink": "a" * 40, "dirty_tracked": False}
    monkeypatch.setattr(env, "parent_state", lambda: dict(parent))
    monkeypatch.setattr(env, "pip_freeze", lambda: ("numpy==2.3.5\n", "f" * 64))
    return tmp_path, state


def _finish_e16(rec):
    rec.write_macros({"EXVIbias": "-1/15"})
    rec.write_table("tab_E16", "x")
    return rec.finalize(warnings=0, failures=0)


def test_deterministic_run_writes_valid_manifest(fake_run) -> None:
    tmp, _ = fake_run
    rec = outputs.RunRecorder(16, "stage0", notebook="nb", R=None, n=None)
    path = _finish_e16(rec)
    manifest = json.loads(path.read_text())
    outputs.validate_manifest(manifest, 16, "stage0")
    assert path == tmp / "E-16" / "stage0" / "manifest.json"
    assert not (tmp / "E-16" / "stage0" / "RUNNING.json").exists()
    with pytest.raises(RuntimeError):
        rec.write_table("tab_E16b", "y")  # closed after finalize
    with pytest.raises(RuntimeError):
        rec.finalize(warnings=0, failures=0)


def test_stage_is_reserved_even_when_interrupted(fake_run) -> None:
    outputs.RunRecorder(16, "stage0", notebook="nb", R=None, n=None)  # never finalized
    with pytest.raises(FileExistsError):
        outputs.RunRecorder(16, "stage0", notebook="nb", R=None, n=None)
    moved = outputs.archive_run(16, "stage0", "interrupted by a kernel restart")
    assert (moved / "RUNNING.json").is_file()
    assert (moved / "REASON.txt").read_text().startswith("interrupted")
    _finish_e16(outputs.RunRecorder(16, "stage0", notebook="nb", R=None, n=None))


def test_registered_outputs_and_duplicates_are_enforced(fake_run) -> None:
    rec = outputs.RunRecorder(16, "stage0", notebook="nb", R=None, n=None)
    rec.write_table("tab_E16", "x")
    with pytest.raises(FileExistsError):
        rec.write_table("tab_E16", "x")
    with pytest.raises(ValueError, match="registered outputs not written"):
        rec.finalize(warnings=0, failures=0)
    with pytest.raises(ValueError):
        rec.write_macros({"E16x": "1"})


def test_source_change_during_run_stops_finalize(fake_run) -> None:
    _, state = fake_run
    rec = outputs.RunRecorder(16, "stage0", notebook="nb", R=None, n=None)
    state["source"]["sha"] = "9" * 40
    with pytest.raises(RuntimeError, match="source changed"):
        _finish_e16(rec)


def test_stage1_needs_prior_stages_and_pilot_keeps_no_estimates(fake_run) -> None:
    with pytest.raises(RuntimeError, match="needs a finished"):
        outputs.RunRecorder(13, "stage1", notebook="nb", R=10, n=1000)
    pilot = outputs.RunRecorder(13, "stage0.5", notebook="nb", R=10, n=1000)
    with pytest.raises(ValueError):
        pilot.write_raw(pd.DataFrame({"x": [1.0]}))
    with pytest.raises(ValueError):
        pilot.write_table("tab_E13", "x")
    with pytest.raises(ValueError):
        pilot.write_pilot("estimates.csv", pd.DataFrame({"x": [1.0]}))


def test_manifest_validation_rejects_incomplete(fake_run) -> None:
    path = _finish_e16(outputs.RunRecorder(16, "stage0", notebook="nb", R=None, n=None))
    good = json.loads(path.read_text())
    for key in ("environment", "data_sha256", "blas"):
        bad = dict(good)
        del bad[key]
        with pytest.raises(ValueError):
            outputs.validate_manifest(bad, 16, "stage0")
    bad = dict(good, outputs_sha256={"macros_E-16.tex": "0" * 64})
    with pytest.raises(ValueError, match="registered outputs missing"):
        outputs.validate_manifest(bad, 16, "stage0")
    with pytest.raises(ValueError, match="data checks"):
        outputs.validate_manifest(dict(good, experiment="E-21"), 21, "stage0")


def test_data_check_rejects_missing_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(env, "DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        env.verify_data(["lalonde/lalonde.csv"])
    (tmp_path / "lalonde").mkdir()
    (tmp_path / "lalonde" / "lalonde.csv").write_text("x")
    with pytest.raises(ValueError):
        env.verify_data(["lalonde/lalonde.csv"])


# ---------------------------------------------------------------- source and environment


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def fake_checkout(tmp_path, monkeypatch):
    root = tmp_path / "gz"
    (root / "src" / "genriesz").mkdir(parents=True)
    (root / "src" / "genriesz" / "fakemod_grrexp.py").write_text("VALUE = 1\n")
    nbdir = root / "notebooks" / "experiments"
    nbdir.mkdir(parents=True)
    nb = {
        "cells": [{"cell_type": "code", "source": ["x = 1\n"], "outputs": []}],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (nbdir / "run.ipynb").write_text(json.dumps(nb))
    (root / ".gitignore").write_text("__pycache__/\n")  # as in the genriesz repository
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "c")
    monkeypatch.setattr(env, "GENRIESZ_ROOT", root.resolve())
    monkeypatch.syspath_prepend(str(root / "src" / "genriesz"))
    import fakemod_grrexp  # noqa: F401

    yield root, nbdir / "run.ipynb", nb
    sys.modules.pop("fakemod_grrexp", None)


def test_source_snapshot_accepts_clean_checkout_and_outputs(fake_checkout) -> None:
    root, nb_path, nb = fake_checkout
    (root / "notebooks/experiments/results/E-16").mkdir(parents=True)
    (root / "notebooks/experiments/results/E-16/x.json").write_text("{}")
    nb["cells"][0]["outputs"] = [{"output_type": "stream", "name": "stdout", "text": "1"}]
    nb_path.write_text(json.dumps(nb))  # outputs only
    snap = env.source_snapshot(nb_path)
    assert snap["loaded_modules"] == {
        "src/genriesz/fakemod_grrexp.py": env.git_blob_sha1(b"VALUE = 1\n")
    }


def test_source_snapshot_rejects_changes(fake_checkout) -> None:
    root, nb_path, nb = fake_checkout
    (root / "notebooks/experiments/helper.py").write_text("y = 2\n")
    with pytest.raises(RuntimeError, match="not clean"):
        env.source_snapshot(nb_path)
    (root / "notebooks/experiments/helper.py").unlink()
    (root / "src/genriesz/fakemod_grrexp.py").write_text("VALUE = 2\n")
    with pytest.raises(RuntimeError):
        env.source_snapshot(nb_path)
    _git(root, "checkout", "--", "src/genriesz/fakemod_grrexp.py")
    nb["cells"][0]["source"] = ["x = 2\n"]
    nb_path.write_text(json.dumps(nb))
    with pytest.raises(RuntimeError, match="code cells"):
        env.source_snapshot(nb_path)


def test_environment_matches_lock_and_lock_covers_registration() -> None:
    record = env.check_environment()
    pins = env.read_lock()
    assert record["lock_pins_verified"] == len(pins)
    assert {"clarabel", "torch", "cvxpy", "numpy"} <= set(pins)


def test_submodule_genriesz_is_the_imported_source() -> None:
    mod = env.use_submodule_genriesz()
    assert Path(mod.__file__).resolve().parents[1] == env.GENRIESZ_SRC
