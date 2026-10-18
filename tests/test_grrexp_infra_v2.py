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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from grrexp import env, parallel  # noqa: E402
from grrexp import outputs_v2 as outputs  # noqa: E402
from grrexp import runner_v2 as runner  # noqa: E402


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

    module = importlib.import_module(outputs.MODULES[exp])
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


def test_one_stage_manifest_and_source_provenance():
    assert outputs.STAGE_DIRS == {"run": "run"}
    assert set(outputs.APP_FILES) == {12, 24, 25}
    assert outputs.MODULES == {12: "grrexp.e12", 24: "grrexp.e24_v2", 25: "grrexp.e25"}
    manifest = _manifest(25)
    outputs.validate_manifest(manifest, 25)
    with pytest.raises(ValueError):
        outputs.validate_manifest(manifest, 24)
    sources = outputs.source_versions(24)
    for name in (
        "experiments/grrexp/e24_v2.py", "experiments/grrexp/outputs_v2.py",
        "experiments/grrexp/runner_v2.py", "experiments/run_registered_v2.py",
        "experiments/apps/e24_v2.py", "experiments/grrexp/riesznet.py",
    ):
        assert sources[name] == env.sha256_file(env.GENRIESZ_ROOT / name)
    outputs.validate_manifest(_manifest(24), 24)
    assert outputs.stage_dir(24).parts[-3:] == ("programme_20261018_uv_marimo_v2", "E24", "run")


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
    from grrexp import e24_v2 as e24
    from grrexp import study

    def fake(task):
        return {
            "folds": [0, 1],
            "arms": {
                "SQ|ARW_cf": study.failed(
                    "optimizer_failure", fits=[{"message": "library termination"}]
                )
            },
        }

    monkeypatch.setattr(e24, "replicate", fake)
    def run_fake_tasks(fn, tasks, n_jobs, *, worker_records):
        assert len(tasks) == e24.R * len(e24.CELLS)
        worker_records.append(_worker_record())
        return [fn(task) for task in tasks]

    monkeypatch.setattr(parallel, "run_tasks", run_fake_tasks)
    destination = tmp_path / "run"
    path = outputs.execute(24, destination, n_jobs=1)
    manifest = json.loads(path.read_text())
    assert manifest["R"] == e24.R and manifest["cells"] == e24.CELLS
    assert manifest["failures"] == {
        "('SQ|ARW_cf', 'optimizer_failure')": e24.R * len(e24.CELLS)
    }
    assert manifest["uv_lock_sha256"] == env.sha256_file(env.LOCK_FILE)
    assert manifest["genriesz"]["kind"] == "git_checkout"
    assert manifest["genriesz"]["dirty"] == bool(manifest["genriesz"]["status_porcelain"])
    assert manifest["parent_repository"]["gitlink"]
    assert manifest["pip_freeze"] and manifest["cpu"] and manifest["blas"]
    assert manifest["wall_seconds"] > 0 and manifest["threads"]["workers"]
    assert manifest["warning_counts"] == {}
    assert (destination / "e24_v2.py").read_bytes() == (
        env.EXPERIMENTS_DIR / outputs.APP_FILES[24]
    ).read_bytes()
    assert "library termination" in (destination / "replications.json").read_text()
    assert (destination / "tab_E24.tex").is_file() and (destination / "fig_E24.pdf").is_file()
    for name, digest in manifest["outputs_sha256"].items():
        assert env.sha256_file(destination / name) == digest
    with pytest.raises(FileExistsError):
        outputs.execute(24, destination)


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
    assert seen["cmd"] == [sys.executable, str(env.EXPERIMENTS_DIR / "apps/e25_v2.py")]
    assert all(seen["env"][v] == "1" for v in parallel.THREAD_VARIABLES)
    request = json.loads(seen["env"][outputs.RUN_ENV])
    assert request == {"experiment": 25, "directory": str(tmp_path / "run"), "n_jobs": 2}
    manifest = json.loads(manifest_path.read_text())
    for name in ("stdout.txt", "stderr.txt"):
        assert env.sha256_file(tmp_path / "run" / name) == manifest["outputs_sha256"][name]
    with pytest.raises(ValueError):
        runner.run(25, "apps/e24.py")
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


def test_runner_records_manifest_validation_failure(tmp_path, monkeypatch):
    directory = tmp_path / "run"

    def process(command, **kwargs):
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps({"experiment": "E-24"}))
        return subprocess.CompletedProcess(command, 0, "", "")

    def reject(*args):
        raise ValueError("required provenance missing")

    monkeypatch.setattr(subprocess, "run", process)
    monkeypatch.setattr(outputs, "validate_manifest", reject)
    with pytest.raises(ValueError, match="required provenance missing"):
        runner.run(24, directory=directory)
    assert "ValueError: required provenance missing" in (directory / "FAILED.txt").read_text()


def test_runner_records_process_interruption(tmp_path, monkeypatch):
    directory = tmp_path / "run"

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.run(24, directory=directory)
    assert "KeyboardInterrupt" in (directory / "FAILED.txt").read_text()


@pytest.mark.parametrize("exp", [19, 22])
def test_inactive_experiments_are_not_execution_choices(exp, tmp_path):
    with pytest.raises(ValueError, match="choose E-12, E-24 or E-25"):
        runner.run(exp, directory=tmp_path / "run")
    with pytest.raises(ValueError, match="choose E-12, E-24 or E-25"):
        outputs.execute(exp, tmp_path / "run")
    with pytest.raises(SystemExit):
        runner.main([f"E-{exp}"])
    assert not (tmp_path / "run").exists()


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


@pytest.mark.parametrize("exp", [12, 24, 25])
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
from grrexp import env, parallel
from grrexp import outputs_v2 as outputs

def inspect_full_design(fn, tasks, n_jobs, *, worker_records):
    import genriesz
    from grrexp import e24_v2 as e24
    assert Path(genriesz.__file__).resolve().parents[1] == env.GENRIESZ_SRC
    assert len(tasks) == len(e24.CELLS) * e24.R
    assert n_jobs == 2
    raise RuntimeError("full design initialized before computation")

parallel.run_tasks = inspect_full_design
directory = Path(sys.argv[2])
try:
    outputs.execute(24, directory, n_jobs=2)
except RuntimeError as error:
    assert str(error) == "full design initialized before computation", str(error)
else:
    raise AssertionError("the computation boundary was not reached")

record = json.loads((directory / "RUNNING.json").read_text())
assert record["R"] == 500 and record["cells"] == [[1000]]
assert record["environment"]["lock_sha256"] == env.sha256_file(env.LOCK_FILE)
assert record["environment"]["lock_pins_verified"] > 0
assert record["genriesz"]["sha"] and record["parent_repository"]["sha"]
assert record["genriesz"]["app_sha256"] == env.sha256_file(env.EXPERIMENTS_DIR / "apps/e24_v2.py")
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
    files = ("experiments/apps/e25_v2.py", "experiments/run_registered.py",
             "experiments/run_registered_v2.py",
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
    monkeypatch.setattr(env, "LOCK_FILE", experiments / "uv.lock")
    monkeypatch.setattr(env, "_git", lambda *a, **kw: pytest.fail("ancestor git must not be read"))
    state = env.checkout_state(root / files[0])
    assert state["kind"] == "source_snapshot" and state["sha"] is None
    assert state["source_git_state"] == metadata["source_genriesz"]
    assert env.parent_state()["source_git_state"] == metadata["source_parent_repository"]
    assert "experiments/run_registered_v2.py" in outputs.source_versions(25)
    omitted = metadata["files"].pop("experiments/run_registered_v2.py")
    (tmp_path / "SOURCE_SHA256.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="lacks source files"):
        outputs.source_versions(25)
    metadata["files"]["experiments/run_registered_v2.py"] = omitted
    (tmp_path / "SOURCE_SHA256.json").write_text(json.dumps(metadata))
    (root / files[0]).write_text("changed source")
    with pytest.raises(ValueError, match="differs from its manifest"):
        env.checkout_state(root / files[0])


def test_source_directory_without_snapshot_cannot_borrow_parent_commit(tmp_path, monkeypatch):
    root = tmp_path / "genriesz"
    app = root / "experiments/apps/e25_v2.py"
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
