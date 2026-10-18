"""One-stage paper experiment outputs, with failures and source provenance."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import env, parallel, study
from .seeds import CONFIRMATORY_ENTROPY

RUN_ENV = "GRREXP_RUN"
STAGE_DIRS = {"run": "run"}
STEM = re.compile(r"^[A-Za-z0-9_-]+$")
MODULES = {12: "grrexp.e12", 24: "grrexp.e24_v2", 25: "grrexp.e25"}
APP_FILES = {exp: f"apps/e{exp}_v2.py" for exp in MODULES}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(
        json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def source_versions(exp):
    paths = list(env.GENRIESZ_SRC.rglob("*.py")) + list(
        (env.EXPERIMENTS_DIR / "grrexp").glob("*.py")
    )
    paths += [
        env.EXPERIMENTS_DIR / "run_registered_v2.py",
        env.EXPERIMENTS_DIR / APP_FILES[exp],
        env.EXPERIMENTS_DIR / "pyproject.toml",
        env.GENRIESZ_ROOT / "pyproject.toml",
        env.LOCK_FILE,
    ]
    sources = {
        p.relative_to(env.GENRIESZ_ROOT).as_posix(): env.sha256_file(p) for p in sorted(paths)
    }
    if not (env.GENRIESZ_ROOT / ".git").exists():
        path, snapshot = env._source_snapshot()
        missing = set(sources) - set(snapshot["files"])
        if missing:
            raise ValueError(f"{path} lacks source files: {sorted(missing)}")
    return sources


def package_versions():
    return {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
        if d.metadata["Name"]
    }


def stage_dir(exp, stage="run"):
    if exp not in APP_FILES or stage != "run":
        raise ValueError("only the three retained experiments and stage 'run' are supported")
    return env.RESULTS_DIR / "programme_20261018_uv_marimo_v2" / f"E{exp}" / "run"


def validate_manifest(manifest, exp, stage="run"):
    if manifest["experiment"] != f"E-{exp}" or stage != "run" or manifest["stage"] != "run":
        raise ValueError("experiment or stage mismatch")
    for name in (
        "source_sha256",
        "packages",
        "entropy",
        "R",
        "cells",
        "outputs_sha256",
        "failures",
        "started_utc",
        "finished_utc",
        "environment",
        "genriesz",
        "genriesz_at_finish",
        "parent_repository",
        "parent_repository_at_finish",
        "pip_freeze",
        "pip_freeze_sha256",
        "pip_freeze_command",
        "blas",
        "cpu",
        "wall_seconds",
        "threads",
        "warning_counts",
        "warning_count_scope",
    ):
        if name not in manifest:
            raise ValueError(f"missing manifest field: {name}")
    if not manifest["outputs_sha256"] or manifest["R"] < 1:
        raise ValueError("empty outputs or replication count")
    module = importlib.import_module(MODULES[exp])
    if manifest["R"] != module.R or manifest["cells"] != module.CELLS:
        raise ValueError("manifest does not contain the full experiment design")
    if not manifest["pip_freeze"] or hashlib.sha256(
        manifest["pip_freeze"].encode()
    ).hexdigest() != manifest["pip_freeze_sha256"]:
        raise ValueError("missing or inconsistent installed-package freeze")
    if not manifest["cpu"] or not manifest["blas"] or not (
        np.isfinite(manifest["wall_seconds"]) and manifest["wall_seconds"] >= 0
    ):
        raise ValueError("missing numerical environment or elapsed time")
    for name in ("genriesz", "genriesz_at_finish", "parent_repository",
                 "parent_repository_at_finish"):
        state = manifest[name]
        if state["kind"] == "source_snapshot":
            if state["sha"] is not None or not state["source_git_state"]:
                raise ValueError("a source snapshot must preserve its origin separately")
            if not re.fullmatch(r"[0-9a-f]{64}", state["source_manifest_sha256"]):
                raise ValueError("missing source-snapshot hash")
        elif state["kind"] == "git_checkout":
            if not re.fullmatch(r"[0-9a-f]{40}", state["sha"]):
                raise ValueError("missing git commit")
            if state["dirty"] != bool(state["status_porcelain"]):
                raise ValueError("git dirty state contradicts its recorded status")
            if name.startswith("parent_") and not re.fullmatch(
                r"[0-9a-f]{40}", state["gitlink"]
            ):
                raise ValueError("missing parent gitlink")
        else:
            raise ValueError("unknown source provenance")
    for name in ("genriesz", "parent_repository"):
        before, after = manifest[name], manifest[name + "_at_finish"]
        if before["kind"] != after["kind"]:
            raise ValueError(f"{name} provenance changed during the run")
        keys = (
            ("source_manifest_sha256",)
            if before["kind"] == "source_snapshot"
            else (("sha", "gitlink") if name == "parent_repository" else ("sha",))
        )
        if any(before[key] != after[key] for key in keys):
            raise ValueError(f"{name} commit or source snapshot changed during the run")
    records = manifest["threads"]["workers"]
    if not records:
        raise ValueError("no observed replication-process thread records")
    for record in records:
        if record["scope"] != "replication process initialization" or record["pid"] < 1:
            raise ValueError("invalid replication-process observation")
        threads = record["threads"]
        if not all(threads["variables"][name] == "1" for name in parallel.THREAD_VARIABLES):
            raise ValueError("worker thread environment is not single-threaded")
        if threads["torch"] != {"num_threads": 1, "deterministic": True}:
            raise ValueError("worker torch configuration is not deterministic and single-threaded")
        if any(pool["num_threads"] != 1 for pool in threads["observed_pools"]):
            raise ValueError("observed worker thread pool is not single-threaded")
    if not isinstance(manifest["warning_counts"], dict):
        raise ValueError("warning counts are missing")


def warning_counts(value):
    """Count warning records retained in the nested replication output."""
    counts = Counter()
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "warnings":
                counts.update(nested)
            else:
                counts.update(warning_counts(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            counts.update(warning_counts(nested))
    return dict(sorted(counts.items()))


def execute(exp, directory, *, n_jobs=2):
    """Execute the full experiment design and retain its computational provenance."""
    if exp not in APP_FILES:
        raise ValueError("choose E-12, E-24 or E-25")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(f"output directory is not empty: {directory}")
    clock = time.perf_counter()
    workers = []
    manifest = {
        "experiment": f"E-{exp}",
        "stage": "run",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "threads": {"scope": "replication process initialization", "workers": workers},
    }
    try:
        env.use_submodule_genriesz()
        environment = env.check_environment()
        module = importlib.import_module(MODULES[exp])
        sources = source_versions(exp)
        freeze, freeze_digest = env.pip_freeze()
        manifest.update({
            "source_sha256": sources,
            "genriesz": env.checkout_state(env.EXPERIMENTS_DIR / APP_FILES[exp]),
            "parent_repository": env.parent_state(),
            "packages": package_versions(),
            "pip_freeze": freeze,
            "pip_freeze_sha256": freeze_digest,
            "pip_freeze_command": [
                "uv", "--no-cache", "pip", "freeze", "--python", sys.executable,
            ],
            "python": sys.version,
            "platform": platform.platform(),
            "environment": environment,
            "blas": env.blas_config(),
            "cpu": env.cpu_description(),
            "data_sha256": env.verify_data(env.REGISTERED_DATA.get(exp, ())),
            "entropy": CONFIRMATORY_ENTROPY,
            "seed_key": ["experiment", "cell", "rep", "stream"],
            "R": module.R,
            "cells": module.CELLS,
            "n_jobs": n_jobs,
            "worker_count_file": os.environ.get(parallel.MAX_WORKERS_FILE_ENV),
            "thread_environment": {
                name: os.environ.get(name) for name in parallel.THREAD_VARIABLES
            },
            "uv_lock_sha256": env.sha256_file(env.LOCK_FILE),
            "pyproject_sha256": env.sha256_file(env.EXPERIMENTS_DIR / "pyproject.toml"),
            "app": APP_FILES[exp],
        })
        write_json(directory / "RUNNING.json", manifest)
        tasks = module.tasks_for()
        results = parallel.run_tasks(module.replicate, tasks, n_jobs=n_jobs, worker_records=workers)
        # Raw JSON keeps failed fits, library termination messages, tuning and folds.
        write_json(
            directory / "replications.json",
            [{"task": task, "result": result} for task, result in zip(tasks, results, strict=True)],
        )
        manifest["warning_counts"] = warning_counts(results)
        manifest["warning_count_scope"] = "warning records retained in replication results"
        raw = study.raw_frame(module, tasks, results)
        raw.to_csv(directory / "raw.csv", index=False)
        summary = module.summarise(raw)
        summary.to_csv(directory / "summary.csv", index=False)
        tables = module.tables(summary) if hasattr(module, "tables") else study.tables(summary, exp)
        for name, content in tables.items():
            (directory / (name + ".tex")).write_text(content, encoding="utf-8")
        fig = module.figure(summary) if hasattr(module, "figure") else study.figure(summary)
        fig.savefig(directory / f"fig_E{exp}.pdf", bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(fig)
        if exp == 25:
            write_json(
                directory / "moment_facts.json", [module.moment_facts(b) for b in module.B_VALUES]
            )
        elif exp in (12, 24):
            write_json(
                directory / "target.json",
                {"theta0": module.THETA0},
            )
        app_source = env.EXPERIMENTS_DIR / APP_FILES[exp]
        (directory / app_source.name).write_bytes(app_source.read_bytes())
        if source_versions(exp) != sources:
            raise RuntimeError("source or uv environment files changed during the run")
        manifest.update(
            genriesz_at_finish=env.checkout_state(app_source),
            parent_repository_at_finish=env.parent_state(),
            finished_utc=datetime.now(timezone.utc).isoformat(),
            wall_seconds=time.perf_counter() - clock,
            failures=raw[raw.status != "ok"].groupby(["arm", "status"]).size().to_dict(),
        )
        manifest["failures"] = {str(key): int(value) for key, value in manifest["failures"].items()}
        manifest["outputs_sha256"] = {
            path.name: env.sha256_file(path)
            for path in sorted(directory.iterdir())
            if path.is_file() and path.name != "RUNNING.json"
        }
        validate_manifest(manifest, exp)
        write_json(directory / "manifest.json", manifest)
        (directory / "RUNNING.json").unlink()
    except BaseException as error:
        manifest.update(
            finished_utc=datetime.now(timezone.utc).isoformat(),
            wall_seconds=time.perf_counter() - clock,
            error=f"{type(error).__name__}: {error}",
        )
        write_json(directory / "RUNNING.json", manifest)
        (directory / "FAILED.txt").write_text(
            f"{type(error).__name__}: {error}\n", encoding="utf-8"
        )
        raise
    return directory / "manifest.json"


def execute_app(exp):
    """Execute the full paper design only when the runner supplies a request."""
    raw_request = os.environ.get(RUN_ENV)
    if raw_request is None:
        return None
    request = json.loads(raw_request)
    if request["experiment"] != exp:
        raise ValueError("marimo app does not match runner request")
    if set(request) != {"experiment", "directory", "n_jobs"}:
        raise ValueError("runner request must specify the full paper design")
    return execute(
        exp,
        request["directory"],
        n_jobs=request["n_jobs"],
    )
