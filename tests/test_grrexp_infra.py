from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from grrexp import bootstrap, env, inference, metrics, outputs, parallel, runner  # noqa: E402
from grrexp.seeds import Seeds  # noqa: E402


def _worker_record():
    return {
        "pid": 1,
        "worker_id": 0,
        "scope": "replication process initialization",
        "threads": {
            "variables": {name: "1" for name in parallel.THREAD_VARIABLES},
            "observed_pools": [],
            "torch": {"num_threads": 1, "deterministic": True},
        },
    }


def _manifest(exp):
    import importlib

    module = importlib.import_module(f"grrexp.e{exp}")
    git = {
        "kind": "git_checkout", "sha": "a" * 40,
        "dirty": True, "status_porcelain": " M source.py\n", "gitlink": "b" * 40,
    }
    freeze = "numpy==2.3.5\n"
    return {
        "experiment": f"E-{exp}", "stage": "run", "source_sha256": {"code": "digest"},
        "packages": {}, "entropy": 20261007, "R": module.R, "cells": module.CELLS,
        "outputs_sha256": {"raw.csv": "digest"}, "failures": {},
        "started_utc": "now", "finished_utc": "later", "environment": {},
        "genriesz": dict(git), "genriesz_at_finish": dict(git),
        "parent_repository": dict(git), "parent_repository_at_finish": dict(git),
        "pip_freeze": freeze, "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
        "pip_freeze_command": ["uv", "--no-cache", "pip", "freeze"],
        "blas": "BLAS and LAPACK", "cpu": "test CPU", "wall_seconds": 1.0,
        "threads": {"scope": "replication process initialization", "workers": [_worker_record()]},
        "warning_counts": {},
        "warning_count_scope": "warning records retained in replication results",
    }


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


def test_one_stage_manifest_and_source_provenance():
    assert outputs.STAGE_DIRS == {"run": "run"}
    assert set(outputs.APP_FILES) == {12, 19, 22, 24, 25}
    manifest = _manifest(25)
    outputs.validate_manifest(manifest, 25)
    with pytest.raises(ValueError):
        outputs.validate_manifest(manifest, 24)
    assert "experiments/grrexp/e25.py" in outputs.source_versions(25)


def test_summary_handles_no_estimates_and_two_replications():
    raw = pd.DataFrame(
        {
            "cell": [0, 0],
            "n": [1000, 1000],
            "arm": ["a", "a"],
            "estimate": [1.0, np.nan],
            "se": [0.1, np.nan],
            "status": ["ok", "optimizer_failure"],
        }
    )
    from grrexp import study

    row = study.summarise(raw, 1.0).iloc[0]
    assert row.coverage == 0.5 and row.failure_rate == 0.5 and row.bias_conditional == 0
    assert row.coverage_mcse == pytest.approx(np.sqrt(0.25 / 2))


def test_execute_keeps_failure_records_and_generates_displays(tmp_path, monkeypatch):
    from grrexp import e19, study

    def fake(task):
        return {
            "folds": [0, 1],
            "arms": {
                "SQ|ARW_cf": study.failed(
                    "optimizer_failure", fits=[{"message": "library termination"}]
                )
            },
        }

    monkeypatch.setattr(e19, "replicate", fake)
    def run_fake_tasks(fn, tasks, n_jobs, *, worker_records):
        assert len(tasks) == e19.R * len(e19.CELLS)
        worker_records.append(_worker_record())
        return [fn(task) for task in tasks]

    monkeypatch.setattr(parallel, "run_tasks", run_fake_tasks)
    destination = tmp_path / "run"
    path = outputs.execute(19, destination, n_jobs=1)
    manifest = json.loads(path.read_text())
    assert manifest["R"] == e19.R and manifest["cells"] == e19.CELLS
    assert manifest["failures"] == {
        "('SQ|ARW_cf', 'optimizer_failure')": e19.R * len(e19.CELLS)
    }
    assert manifest["uv_lock_sha256"] == env.sha256_file(env.LOCK_FILE)
    assert manifest["genriesz"]["kind"] == "git_checkout"
    assert manifest["genriesz"]["dirty"] == bool(manifest["genriesz"]["status_porcelain"])
    assert manifest["parent_repository"]["gitlink"]
    assert manifest["pip_freeze"] and manifest["cpu"] and manifest["blas"]
    assert manifest["wall_seconds"] > 0 and manifest["threads"]["workers"]
    assert manifest["warning_counts"] == {}
    assert (destination / "e19.py").read_bytes() == (
        env.EXPERIMENTS_DIR / outputs.APP_FILES[19]
    ).read_bytes()
    assert "library termination" in (destination / "replications.json").read_text()
    assert (destination / "tab_E19.tex").is_file() and (destination / "fig_E19.pdf").is_file()
    for name, digest in manifest["outputs_sha256"].items():
        assert env.sha256_file(destination / name) == digest
    with pytest.raises(FileExistsError):
        outputs.execute(19, destination)


def test_runner_executes_only_matching_marimo_app_and_pins_threads(tmp_path, monkeypatch):
    seen = {}

    def execute(cmd, **kwargs):
        seen.update(cmd=cmd, **kwargs)
        directory = tmp_path / "run"
        directory.mkdir()
        manifest = _manifest(25)
        (directory / "manifest.json").write_text(json.dumps(manifest))
        return subprocess.CompletedProcess(cmd, 0, "output", "diagnostic")

    monkeypatch.setattr(subprocess, "run", execute)
    manifest_path = runner.run(25, directory=tmp_path / "run", n_jobs=2)
    assert seen["cmd"] == [sys.executable, str(env.EXPERIMENTS_DIR / "apps/e25.py")]
    assert all(seen["env"][v] == "1" for v in parallel.THREAD_VARIABLES)
    request = json.loads(seen["env"][outputs.RUN_ENV])
    assert request == {"experiment": 25, "directory": str(tmp_path / "run"), "n_jobs": 2}
    manifest = json.loads(manifest_path.read_text())
    for name in ("stdout.txt", "stderr.txt"):
        assert env.sha256_file(tmp_path / "run" / name) == manifest["outputs_sha256"][name]
    with pytest.raises(ValueError):
        runner.run(25, "apps/e19.py")
    with pytest.raises(FileExistsError):
        runner.run(25, directory=tmp_path / "run")


def test_runner_records_failed_process_and_missing_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, "out", "worker failed"),
    )
    with pytest.raises(runner.RunError, match="execution failed"):
        runner.run(24, directory=tmp_path / "run")
    failure = (tmp_path / "run" / "FAILED.txt").read_text()
    assert "worker failed" in failure and "exit code: 1" in failure
    assert "did not produce manifest.json" in failure


def test_marimo_request_only_executes_full_design(tmp_path, monkeypatch):
    calls = []
    monkeypatch.delenv(outputs.RUN_ENV, raising=False)
    monkeypatch.setattr(outputs, "execute", lambda *a, **kw: calls.append((a, kw)))
    assert outputs.execute_app(25) is None
    assert not calls
    request = {"experiment": 25, "directory": str(tmp_path / "run"), "n_jobs": 2}
    monkeypatch.setenv(outputs.RUN_ENV, json.dumps(request))
    outputs.execute_app(25)
    assert calls == [((25, str(tmp_path / "run")), {"n_jobs": 2})]
    monkeypatch.setenv(outputs.RUN_ENV, json.dumps({**request, "reps": 2}))
    with pytest.raises(ValueError, match="full paper design"):
        outputs.execute_app(25)
    monkeypatch.setenv(outputs.RUN_ENV, json.dumps(request))
    with pytest.raises(ValueError, match="does not match"):
        outputs.execute_app(19)


@pytest.mark.parametrize("exp", [12, 19, 22, 24, 25])
def test_marimo_apps_open_without_executing_experiments(exp, monkeypatch):
    import importlib.util

    monkeypatch.delenv(outputs.RUN_ENV, raising=False)
    monkeypatch.setattr(
        outputs, "execute", lambda *a, **kw: pytest.fail("opening app executed an experiment")
    )
    path = env.EXPERIMENTS_DIR / outputs.APP_FILES[exp]
    spec = importlib.util.spec_from_file_location(f"paper_app_e{exp}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, definitions = module.app.run()
    assert definitions["run_button"].value is False
    assert definitions["n_jobs"].value == 2


def test_headless_marimo_app_runs_requested_design(tmp_path, monkeypatch):
    import importlib.util

    calls = []
    request = {"experiment": 25, "directory": str(tmp_path / "run"), "n_jobs": 2}
    monkeypatch.setenv(outputs.RUN_ENV, json.dumps(request))
    monkeypatch.setattr(
        outputs,
        "execute",
        lambda *a, **kw: calls.append((a, kw)) or tmp_path / "manifest.json",
    )
    path = env.EXPERIMENTS_DIR / outputs.APP_FILES[25]
    spec = importlib.util.spec_from_file_location("paper_app_headless_e25", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.app.run()
    assert calls == [((25, str(tmp_path / "run")), {"n_jobs": 2})]


def test_run_registered_cli(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(
        runner, "run", lambda *a, **kw: called.append((a, kw)) or tmp_path / "manifest.json"
    )
    assert runner.main(["E-25", "--n-jobs", "2"]) == 0
    assert called[0][0] == (25,) and called[0][1]["n_jobs"] == 2


@pytest.mark.parametrize("flag", ["--reps", "--cells"])
def test_cli_has_no_reduced_experiment_mode(flag):
    with pytest.raises(SystemExit):
        runner.main(["E-25", flag, "2"])


def test_fresh_process_initializes_full_design_and_checkout_source(tmp_path):
    script = """
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from grrexp import env, outputs, parallel

def inspect_full_design(fn, tasks, n_jobs, *, worker_records):
    import genriesz
    from grrexp import e19
    assert Path(genriesz.__file__).resolve().parents[1] == env.GENRIESZ_SRC
    assert len(tasks) == len(e19.CELLS) * e19.R
    assert n_jobs == 2
    raise RuntimeError("full design initialized before computation")

parallel.run_tasks = inspect_full_design
directory = Path(sys.argv[2])
try:
    outputs.execute(19, directory, n_jobs=2)
except RuntimeError as error:
    assert str(error) == "full design initialized before computation", str(error)
else:
    raise AssertionError("the computation boundary was not reached")

record = json.loads((directory / "RUNNING.json").read_text())
assert record["R"] == 500 and record["cells"] == [[1000], [4000]]
assert record["environment"]["lock_sha256"] == env.sha256_file(env.LOCK_FILE)
assert record["environment"]["lock_pins_verified"] > 0
assert record["genriesz"]["sha"] and record["parent_repository"]["sha"]
assert record["genriesz"]["app_sha256"] == env.sha256_file(env.EXPERIMENTS_DIR / "apps/e19.py")
assert record["pip_freeze"] and record["blas"] and record["cpu"]
assert record["wall_seconds"] > 0
assert "full design initialized" in (directory / "FAILED.txt").read_text()
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(env.EXPERIMENTS_DIR), str(tmp_path / "run")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr


def _worker_pid(task):
    return os.getpid(), task * task


def test_actual_replication_process_thread_records(monkeypatch):
    monkeypatch.delenv(parallel.MAX_WORKERS_FILE_ENV, raising=False)
    records = []
    values = parallel.run_tasks(_worker_pid, list(range(8)), 2, worker_records=records)
    assert [value for _, value in values] == [i * i for i in range(8)]
    assert {pid for pid, _ in values} <= {record["pid"] for record in records}
    assert all(record["pid"] != os.getpid() for record in records)
    for record in records:
        assert record["scope"] == "replication process initialization"
        assert record["threads"]["torch"] == {"num_threads": 1, "deterministic": True}
        assert all(pool["num_threads"] == 1 for pool in record["threads"]["observed_pools"])


def test_manifest_requires_provenance_and_full_design():
    for field in ("genriesz", "parent_repository", "pip_freeze", "blas", "cpu", "wall_seconds",
                  "threads", "warning_counts"):
        manifest = _manifest(25)
        del manifest[field]
        with pytest.raises(ValueError, match="missing manifest field"):
            outputs.validate_manifest(manifest, 25)
    manifest = _manifest(25)
    manifest["cells"] = manifest["cells"][:1]
    with pytest.raises(ValueError, match="full experiment design"):
        outputs.validate_manifest(manifest, 25)
    manifest = _manifest(25)
    manifest["genriesz"]["dirty"] = False
    with pytest.raises(ValueError, match="dirty state"):
        outputs.validate_manifest(manifest, 25)
    manifest = _manifest(25)
    manifest["threads"]["workers"] = []
    with pytest.raises(ValueError, match="no observed"):
        outputs.validate_manifest(manifest, 25)
    for name, key in (("genriesz", "sha"), ("parent_repository", "sha"),
                      ("parent_repository", "gitlink")):
        manifest = _manifest(25)
        manifest[name + "_at_finish"][key] = "c" * 40
        with pytest.raises(ValueError, match="changed during the run"):
            outputs.validate_manifest(manifest, 25)


def test_warning_records_are_counted_in_nested_fits():
    result = {"arms": [{"warnings": {"ConvergenceWarning: stopped": 2}, "fits": [
        {"warnings": {"RuntimeWarning: overflow": 1}},
    ]}]}
    assert outputs.warning_counts(result) == {
        "ConvergenceWarning: stopped": 2, "RuntimeWarning: overflow": 1,
    }


def test_source_snapshot_preserves_origin_without_reading_ancestor_git(tmp_path, monkeypatch):
    root = tmp_path / "genriesz"
    experiments = root / "experiments"
    files = ("experiments/apps/e25.py", "experiments/run_registered.py",
             "experiments/pyproject.toml", "experiments/uv.lock", "pyproject.toml")
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    metadata = {
        "files": {name: env.sha256_file(root / name) for name in files},
        "source_genriesz": {"sha": "a" * 40, "dirty": True},
        "source_parent_repository": {"sha": "b" * 40, "gitlink": "c" * 40},
    }
    (tmp_path / "SOURCE_SHA256.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(env, "GENRIESZ_ROOT", root)
    monkeypatch.setattr(env, "GENRIESZ_SRC", root / "src")
    monkeypatch.setattr(env, "EXPERIMENTS_DIR", experiments)
    monkeypatch.setattr(env, "_git", lambda *a, **kw: pytest.fail("ancestor git must not be read"))
    state = env.checkout_state(root / files[0])
    assert state["kind"] == "source_snapshot" and state["sha"] is None
    assert state["source_git_state"] == metadata["source_genriesz"]
    assert env.parent_state()["source_git_state"] == metadata["source_parent_repository"]
    (root / files[0]).write_text("changed source")
    with pytest.raises(ValueError, match="differs from its manifest"):
        env.checkout_state(root / files[0])


def test_source_directory_without_snapshot_cannot_borrow_parent_commit(tmp_path, monkeypatch):
    root = tmp_path / "genriesz"
    app = root / "experiments/apps/e25.py"
    app.parent.mkdir(parents=True)
    app.write_text("source")
    monkeypatch.setattr(env, "GENRIESZ_ROOT", root)
    monkeypatch.setattr(env, "_git", lambda *a, **kw: pytest.fail("ancestor git must not be read"))
    with pytest.raises(FileNotFoundError, match="SOURCE_SHA256"):
        env.checkout_state(app)


def test_installed_freeze_uses_current_uv_python_and_propagates_failure(monkeypatch):
    def fail(command, **kwargs):
        assert command == ["uv", "--no-cache", "pip", "freeze", "--python", sys.executable]
        assert kwargs["check"] is True
        raise subprocess.CalledProcessError(1, command, stderr="freeze failed")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        env.pip_freeze()
