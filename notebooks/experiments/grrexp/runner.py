"""Execute one paper experiment notebook and preserve its executed copy."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import env, outputs
from .parallel import THREAD_VARIABLES


class RunError(RuntimeError):
    pass


def run(exp, notebook=None, *, directory=None, n_jobs=2, reps=None, cells=None):
    if exp not in outputs.NOTEBOOKS:
        raise ValueError("choose E-12, E-19, E-22, E-24 or E-25")
    if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or not 1 <= n_jobs <= 4:
        raise ValueError("n_jobs must be an integer in 1..4")
    notebook = outputs.NOTEBOOKS[exp] if notebook is None else notebook
    if notebook != outputs.NOTEBOOKS[exp]:
        raise ValueError("notebook does not match the experiment")
    destination = Path(directory).resolve() if directory else outputs.stage_dir(exp)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "experiment": exp,
        "directory": str(destination),
        "n_jobs": n_jobs,
        "reps": reps,
        "cells": cells,
    }
    kernel_env = {
        **os.environ,
        **{v: "1" for v in THREAD_VARIABLES},
        outputs.RUN_ENV: json.dumps(request),
    }
    command = [
        sys.executable,
        "-m",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=-1",
        "--ExecutePreprocessor.kernel_name=python3",
        "--output",
        "executed.ipynb",
        "--output-dir",
        str(destination),
        str(env.EXPERIMENTS_DIR / notebook),
    ]
    result = subprocess.run(
        command, cwd=env.EXPERIMENTS_DIR, env=kernel_env, capture_output=True, text=True
    )
    if result.returncode:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "FAILED.txt").write_text(
            result.stdout + "\n" + result.stderr, encoding="utf-8"
        )
        raise RunError(f"execution failed; see {destination / 'FAILED.txt'}")
    path = destination / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["outputs_sha256"]["executed.ipynb"] = env.sha256_file(destination / "executed.ipynb")
    outputs.write_json(path, manifest)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one of the five paper experiments.")
    parser.add_argument("experiment", choices=[f"E-{e}" for e in outputs.NOTEBOOKS])
    parser.add_argument("--n-jobs", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    path = run(int(args.experiment[2:]), directory=args.output, n_jobs=args.n_jobs)
    print(f"finished: {path}")
    return 0
