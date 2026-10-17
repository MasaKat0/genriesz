"""One-stage paper experiment outputs, with failures and source provenance."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import env, parallel, study
from .seeds import CONFIRMATORY_ENTROPY

RUN_ENV = "GRREXP_RUN"
STAGE_DIRS = {"run": "run"}
STEM = re.compile(r"^[A-Za-z0-9_-]+$")
NOTEBOOKS = {
    12: "10_E12_omitted_direction_balance.ipynb",
    19: "16_E19_ame_inference.ipynb",
    22: "19_E22_lalonde.ipynb",
    24: "22_E24_controlled_nonlinear_benchmark.ipynb",
    25: "23_E25_weak_overlap_moments.ipynb",
}


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
    paths += [env.EXPERIMENTS_DIR / "run_registered.py", env.EXPERIMENTS_DIR / NOTEBOOKS[exp]]
    return {p.relative_to(env.GENRIESZ_ROOT).as_posix(): env.sha256_file(p) for p in sorted(paths)}


def package_versions():
    return {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
        if d.metadata["Name"]
    }


def stage_dir(exp, stage="run"):
    if exp not in NOTEBOOKS or stage != "run":
        raise ValueError("only the five retained experiments and stage 'run' are supported")
    return env.RESULTS_DIR / "programme_20261018" / f"E{exp}" / "run"


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
    ):
        if name not in manifest:
            raise ValueError(f"missing manifest field: {name}")
    if not manifest["outputs_sha256"] or manifest["R"] < 1:
        raise ValueError("empty outputs or replication count")


def execute(exp, directory, *, reps=None, n_jobs=2, cells=None):
    """Execute the complete specified cells; rep overrides support ordinary development tests."""
    module = importlib.import_module(f"grrexp.e{exp}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(f"output directory is not empty: {directory}")
    started = datetime.now(timezone.utc).isoformat()
    sources = source_versions(exp)
    manifest = {
        "experiment": f"E-{exp}",
        "stage": "run",
        "started_utc": started,
        "source_sha256": sources,
        "packages": package_versions(),
        "python": sys.version,
        "platform": platform.platform(),
        "data_sha256": env.verify_data(env.REGISTERED_DATA.get(exp, ())),
        "entropy": CONFIRMATORY_ENTROPY,
        "seed_key": ["experiment", "cell", "rep", "stream"],
        "R": module.R if reps is None else reps,
        "cells": module.CELLS if cells is None else [module.CELLS[c] for c in cells],
        "n_jobs": n_jobs,
        "worker_count_file": os.environ.get(parallel.MAX_WORKERS_FILE_ENV),
        "thread_environment": {v: os.environ.get(v) for v in parallel.THREAD_VARIABLES},
        "requirements_lock_sha256": env.sha256_file(env.LOCK_FILE),
    }
    write_json(directory / "RUNNING.json", manifest)
    try:
        tasks = module.tasks_for(reps=reps, cells=cells)
        results = parallel.run_tasks(module.replicate, tasks, n_jobs=n_jobs)
        # Raw JSON keeps failed fits, library termination messages, tuning and folds.
        write_json(
            directory / "replications.json",
            [{"task": t, "result": r} for t, r in zip(tasks, results, strict=True)],
        )
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
        elif exp in (12, 19, 24):
            write_json(
                directory / "target.json",
                {"theta0": module.true_target() if exp == 19 else module.THETA0},
            )
        manifest.update(
            finished_utc=datetime.now(timezone.utc).isoformat(),
            failures=raw[raw.status != "ok"].groupby(["arm", "status"]).size().to_dict(),
        )
        manifest["failures"] = {str(k): int(v) for k, v in manifest["failures"].items()}
        manifest["outputs_sha256"] = {
            p.name: env.sha256_file(p)
            for p in sorted(directory.iterdir())
            if p.is_file() and p.name != "RUNNING.json"
        }
        validate_manifest(manifest, exp)
        write_json(directory / "manifest.json", manifest)
        (directory / "RUNNING.json").unlink()
    except BaseException as error:
        (directory / "FAILED.txt").write_text(
            f"{type(error).__name__}: {error}\n", encoding="utf-8"
        )
        raise
    return directory / "manifest.json"


def execute_notebook(exp):
    request = json.loads(os.environ[RUN_ENV])
    if request["experiment"] != exp:
        raise ValueError("notebook does not match runner request")
    return execute(
        exp,
        request["directory"],
        reps=request.get("reps"),
        n_jobs=request["n_jobs"],
        cells=request.get("cells"),
    )
