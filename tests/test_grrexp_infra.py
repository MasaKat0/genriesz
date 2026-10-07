from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "experiments"))

from grrexp import bootstrap, env, inference, metrics, outputs, parallel, runner  # noqa: E402
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


# ---------------------------------------------------------------- manifest (§1.7)

H40, H64 = "a" * 40, "b" * 64
THREADS = {
    "variables": {k: "1" for k in sorted(parallel.THREAD_VARIABLES)},
    "blas_backend": "accelerate",
    "observed_pools": [{"internal_api": "openmp", "num_threads": 1}],
    "torch": {"num_threads": 1, "deterministic": True},
}


def make_manifest(exp: int, stage: str) -> dict:
    R, n = outputs.registered_conditions(exp, stage)
    if n is not None:
        n = {p: (v if v is not None else [1000]) for p, v in n.items()}
    cells = None if stage == "stage0" else [[0], [1]]
    names = [o.replace("*", "main") for o in outputs.required_outputs(exp, stage)]
    freeze = "numpy==2.3.5\n"
    m = {
        "experiment": f"E-{exp}",
        "stage": stage,
        "parent_repository": {"sha": H40, "gitlink": H40, "dirty_tracked": False},
        "genriesz": {
            "sha": H40,
            "notebook": "notebooks/experiments/x.ipynb",
            "notebook_blob": H40,
            "notebook_code_cells_sha256": H64,
            "computation": {
                "src/genriesz": H40,
                "notebooks/experiments/grrexp": H40,
                "notebooks/experiments/x.ipynb": H40,
                "notebooks/experiments/requirements-lock.txt": H40,
            },
            "loaded_modules": {"src/genriesz/__init__.py": H40},
        },
        "environment": {
            "platform": ["Darwin", "arm64"],
            "python": "3.13.6",
            "lock_sha256": H64,
            "lock_pins_verified": 109,
            "executable": "/usr/bin/python3",
        },
        "threads": copy.deepcopy(THREADS),
        "pip_freeze": freeze,
        "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
        "blas": "accelerate",
        "cpu": "Apple M",
        "data_sha256": {k: env.DATA_SHA256[k] for k in env.REGISTERED_DATA.get(exp, ())},
        "entropy": outputs.STAGE_ENTROPY[stage],
        "R": R,
        "n": n,
        "cells": cells,
        "prerequisites": {
            p: {"manifest_sha256": H64, "commit": H40}
            for p in outputs.prerequisite_stages(exp, stage)
        },
        "started_utc": "2026-10-07T00:00:00+00:00",
        "finished_utc": "2026-10-07T00:01:00+00:00",
        "wall_seconds": 60.0,
        "warnings": 0,
        "failures": 0,
        "outputs_sha256": {name: H64 for name in names},
    }
    if stage == "stage0.5":
        m["parallel_identity"] = {
            "cells": cells, "reps": 10, "n_jobs": 12, "tasks": 20, "digest": H64,
        }  # fmt: skip
    return m


@pytest.mark.parametrize(
    "exp,stage", [(16, "stage0"), (12, "stage0"), (12, "stage0.5"), (12, "stage1"), (22, "stage1")]
)
def test_complete_manifests_validate(exp, stage) -> None:
    outputs.validate_manifest(make_manifest(exp, stage), exp, stage)


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.update(R={"main": 1}),
        lambda m: m.update(n={"main": [-1]}),
        lambda m: m.update(started_utc=""),
        lambda m: m.update(finished_utc="2026-10-06T00:00:00+00:00"),
        lambda m: m.update(wall_seconds=float("inf")),
        lambda m: m.update(cpu=""),
        lambda m: m.update(pip_freeze="other\n"),
        lambda m: m["genriesz"].update(notebook_code_cells_sha256=""),
        lambda m: m["genriesz"].update(loaded_modules={"src/genriesz/__init__.py": "zz"}),
        lambda m: m["threads"]["variables"].update(OMP_NUM_THREADS="4"),
        lambda m: m["threads"].update(torch={"num_threads": 10, "deterministic": False}),
        lambda m: m.update(prerequisites={}),
        lambda m: m.update(cells=[]),
        lambda m: m["outputs_sha256"].pop("tables/tab_E12_ord.tex"),
        lambda m: m.pop("threads"),
        lambda m: m["genriesz"].pop("computation"),
        lambda m: m["environment"].pop("executable"),
    ],
)
def test_incomplete_or_inconsistent_stage1_manifest_is_rejected(change) -> None:
    m = make_manifest(12, "stage1")
    change(m)
    with pytest.raises(ValueError):
        outputs.validate_manifest(m, 12, "stage1")


def test_pilot_identity_must_cover_every_cell() -> None:
    m = make_manifest(12, "stage0.5")
    m["parallel_identity"]["tasks"] = 10
    with pytest.raises(ValueError, match="incomplete"):
        outputs.validate_manifest(m, 12, "stage0.5")
    m = make_manifest(12, "stage0.5")
    m["parallel_identity"]["n_jobs"] = 2
    with pytest.raises(ValueError):
        outputs.validate_manifest(m, 12, "stage0.5")


def test_e22_needs_its_three_tables() -> None:
    m = make_manifest(22, "stage1")
    m["outputs_sha256"].pop("tables/tab_E22_smd.tex")
    with pytest.raises(ValueError, match="registered outputs missing"):
        outputs.validate_manifest(m, 22, "stage1")


def test_data_check_rejects_missing_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(env, "DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        env.verify_data(["lalonde/lalonde.csv"])
    (tmp_path / "lalonde").mkdir()
    (tmp_path / "lalonde" / "lalonde.csv").write_text("x")
    with pytest.raises(ValueError):
        env.verify_data(["lalonde/lalonde.csv"])


# ---------------------------------------------------------------- kernel-side recorder


@pytest.fixture
def reserved(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "RESULTS_DIR", tmp_path)
    for k in parallel.THREAD_VARIABLES:
        monkeypatch.setenv(k, "1")
    d = outputs.stage_dir(16, "stage0")
    d.mkdir(parents=True)
    (d / "RUNNING.json").write_text(json.dumps({"token": "t"}))
    run = {"exp": 16, "stage": "stage0", "token": "t", "executable": sys.executable}
    monkeypatch.setenv(outputs.RUN_ENV, json.dumps(run))
    return d


def test_recorder_refuses_without_the_runner(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(outputs.RUN_ENV, raising=False)
    with pytest.raises(RuntimeError, match="through grrexp.runner"):
        outputs.RunRecorder(16, "stage0")


def test_recorder_checks_token_and_reservation(reserved, monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="reserved"):
        outputs.RunRecorder(17, "stage0")
    run = {"exp": 16, "stage": "stage0", "token": "x", "executable": sys.executable}
    monkeypatch.setenv(outputs.RUN_ENV, json.dumps(run))
    with pytest.raises(RuntimeError, match="does not belong"):
        outputs.RunRecorder(16, "stage0")
    run = {"exp": 16, "stage": "stage0", "token": "t", "executable": "/other/python"}
    monkeypatch.setenv(outputs.RUN_ENV, json.dumps(run))
    with pytest.raises(RuntimeError, match="kernel runs"):
        outputs.RunRecorder(16, "stage0")


@pytest.mark.parametrize("name", ["../../victim", "a/b", ".", "..", "x.y", ""])
def test_output_names_cannot_escape(reserved, name) -> None:
    rec = outputs.RunRecorder(16, "stage0")
    with pytest.raises(ValueError):
        rec.write_table(name, "x")


def test_recorder_writes_kernel_record_once(reserved) -> None:
    rec = outputs.RunRecorder(16, "stage0")
    rec.write_table("tab_E16", "x")
    with pytest.raises(FileExistsError):
        rec.write_table("tab_E16", "x")
    with pytest.raises(ValueError, match="registered outputs not written"):
        rec.finalize(R=None, n=None, warnings=0, failures=0)
    rec.write_macros({"EXVIbias": "-1/15"})
    rec.finalize(R=None, n=None, warnings=0, failures=0)
    record = json.loads((reserved / "kernel.json").read_text())
    assert record["outputs"] == ["macros_E-16.tex", "tables/tab_E16.tex"]
    assert record["threads"]["torch"] == {"num_threads": 1, "deterministic": True}
    assert record["environment"]["executable"] == sys.executable
    with pytest.raises(RuntimeError):
        rec.write_table("tab_E16b", "y")
    with pytest.raises(FileExistsError):
        outputs.RunRecorder(16, "stage0")


def test_archive_keeps_interrupted_runs(reserved) -> None:
    moved = outputs.archive_run(16, "stage0", "kernel died")
    assert (moved / "RUNNING.json").is_file()
    assert (moved / "REASON.txt").read_text().startswith("interrupted")


# ---------------------------------------------------------------- checkout and runner


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


E16_NOTEBOOK = {
    "cells": [
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "from grrexp import env, outputs\n",
                "env.use_submodule_genriesz()\n",
                "rec = outputs.RunRecorder(16, 'stage0')\n",
                "rec.write_macros({'EXVIbias': '-1/15'})\n",
                "rec.write_table('tab_E16', 'x')\n",
                "rec.finalize(R=None, n=None, warnings=0, failures=0)\n",
            ],
        }
    ],
    "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    root = parent / "gz"
    (root / "src" / "genriesz").mkdir(parents=True)
    (root / "src" / "genriesz" / "__init__.py").write_text("VALUE = 1\n")
    exp_dir = root / "notebooks" / "experiments"
    exp_dir.mkdir(parents=True)
    shutil.copytree(
        env.EXPERIMENTS_DIR / "grrexp", exp_dir / "grrexp",
        ignore=shutil.ignore_patterns("__pycache__"),
    )  # fmt: skip
    (exp_dir / "e16.ipynb").write_text(json.dumps(E16_NOTEBOOK))
    shutil.copy(env.LOCK_FILE, exp_dir / "requirements-lock.txt")
    (root / ".gitignore").write_text("__pycache__/\n")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "c")
    _git(parent, "init", "-q")
    _git(parent, "add", "gz")
    _git(parent, "commit", "-qm", "gitlink")
    monkeypatch.setattr(env, "GENRIESZ_ROOT", root.resolve())
    monkeypatch.setattr(env, "GENRIESZ_SRC", (root / "src").resolve())
    monkeypatch.setattr(env, "EXPERIMENTS_DIR", exp_dir.resolve())
    monkeypatch.setattr(env, "RESULTS_DIR", (exp_dir / "results").resolve())
    return root, exp_dir


def test_checkout_state_requires_a_clean_committed_checkout(fake_repo) -> None:
    root, exp_dir = fake_repo
    nb = exp_dir / "e16.ipynb"
    (exp_dir / "results" / "E-16").mkdir(parents=True)
    (exp_dir / "results" / "E-16" / "x.json").write_text("{}")
    state = env.checkout_state(nb)
    assert state["notebook"] == "notebooks/experiments/e16.ipynb"
    (exp_dir / "helper.py").write_text("y = 2\n")
    with pytest.raises(RuntimeError, match="not clean"):
        env.checkout_state(nb)
    (exp_dir / "helper.py").unlink()
    nb.write_text(nb.read_text() + " ")
    with pytest.raises(RuntimeError, match="not clean"):
        env.checkout_state(nb)


def test_module_check_against_commit() -> None:
    blobs = {"src/genriesz/a.py": H40}
    env.check_modules_at_head({"src/genriesz/a.py": H40}, blobs)
    with pytest.raises(RuntimeError, match="differ"):
        env.check_modules_at_head({"src/genriesz/a.py": "c" * 40}, blobs)
    with pytest.raises(RuntimeError, match="differ"):
        env.check_modules_at_head({"src/genriesz/a.py": H40, "src/x.py": H40}, blobs)


def test_runner_end_to_end_in_a_fresh_kernel(fake_repo) -> None:
    root, exp_dir = fake_repo
    path = runner.run(16, "stage0", "e16.ipynb")
    manifest = json.loads(path.read_text())
    outputs.validate_manifest(manifest, 16, "stage0")
    assert manifest["threads"]["variables"] == THREADS["variables"]
    assert manifest["threads"]["torch"] == {"num_threads": 1, "deterministic": True}
    assert "src/genriesz/__init__.py" in manifest["genriesz"]["loaded_modules"]
    assert set(manifest["genriesz"]["computation"]) >= set(env.COMPUTATION_TREES)
    d = path.parent
    assert (d / "executed.ipynb").is_file() and not (d / "RUNNING.json").exists()
    assert not _git(root, "status", "--porcelain", "--", "notebooks/experiments/e16.ipynb")
    with pytest.raises(FileExistsError):
        runner.run(16, "stage0", "e16.ipynb")
    with pytest.raises(runner.RunError, match="needs a finished"):
        runner.run(12, "stage1", "e16.ipynb")


def test_environment_matches_lock_and_lock_covers_registration() -> None:
    record = env.check_environment()
    pins = env.read_lock()
    assert record["lock_pins_verified"] == len(pins)
    assert {"clarabel", "torch", "cvxpy", "numpy"} <= set(pins)


def test_submodule_genriesz_is_the_imported_source() -> None:
    mod = env.use_submodule_genriesz()
    assert Path(mod.__file__).resolve().parents[1] == env.GENRIESZ_SRC


@pytest.fixture
def pilot_recorder(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "RESULTS_DIR", tmp_path)
    for k in parallel.THREAD_VARIABLES:
        monkeypatch.setenv(k, "1")
    d = outputs.stage_dir(13, "stage0.5")
    d.mkdir(parents=True)
    (d / "RUNNING.json").write_text(json.dumps({"token": "t"}))
    run = {"exp": 13, "stage": "stage0.5", "token": "t", "executable": sys.executable}
    monkeypatch.setenv(outputs.RUN_ENV, json.dumps(run))
    return outputs.RunRecorder(13, "stage0.5", cells=[[0], [1]], arms=["SQ", "UKL"])


def _timing(workers, seconds, cell=(0,)):
    return pd.DataFrame({"cell": [list(cell)], "workers": [workers], "seconds": [seconds]})


def _counts(arm, status, count):
    return pd.DataFrame({"cell": [[0]], "arm": [arm], "status": [status], "count": [count]})


@pytest.mark.parametrize(
    "name,frame",
    [
        ("timing.csv", _timing(3, 1.0)),
        ("timing.csv", _timing(1, np.inf)),
        ("timing.csv", _timing(1, 1.0, cell=(7,))),
        ("status_counts.csv", _counts("SQ", "ok", 1.5)),
        ("status_counts.csv", _counts("SQ", "0.93", 1)),
        ("status_counts.csv", _counts("1.02 est", "ok", 1)),
        ("status_counts.csv", _counts("not_a_registered_arm", "ok", 1)),
    ],
)
def test_pilot_files_carry_no_estimates(pilot_recorder, name, frame) -> None:
    with pytest.raises(ValueError):
        pilot_recorder.write_pilot(name, frame)


def test_valid_pilot_files_are_written(pilot_recorder) -> None:
    timing = pd.DataFrame({"cell": [[0], [1]], "workers": [1, 12], "seconds": [1.0, 0.2]})
    pilot_recorder.write_pilot("timing.csv", timing)
    pilot_recorder.write_pilot("status_counts.csv", _counts("SQ", "ok", 10))


def _commit_stage(stage_path, manifest):
    stage_path.mkdir(parents=True)
    for name in list(manifest["outputs_sha256"]):
        (stage_path / name).parent.mkdir(parents=True, exist_ok=True)
        (stage_path / name).write_text("x")
        manifest["outputs_sha256"][name] = env.sha256_file(stage_path / name)
    (stage_path / "manifest.json").write_text(json.dumps(manifest))


def test_stage1_refuses_a_pilot_of_other_code(fake_repo) -> None:
    root, _ = fake_repo
    pilot = make_manifest(13, "stage0.5")
    stage0 = make_manifest(13, "stage0")
    stage0["outputs_sha256"] = {"stage0_pred.json": H64}
    _commit_stage(outputs.stage_dir(13, "stage0.5"), pilot)
    _commit_stage(outputs.stage_dir(13, "stage0"), stage0)
    _git(root, "add", "-f", "notebooks/experiments/results")
    _git(root, "commit", "-qm", "earlier stages")
    sha = _git(root, "rev-parse", "HEAD").strip()
    with pytest.raises(runner.RunError, match="code changed after the pilot"):
        runner.check_prerequisites(13, "stage1", sha, {"src/genriesz": "c" * 40})
    digests, cells = runner.check_prerequisites(13, "stage1", sha, pilot["genriesz"]["computation"])
    assert cells == [[0], [1]] and set(digests) == {"stage0", "stage0.5"}
