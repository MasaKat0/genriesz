"""Run a complete paper experiment through its marimo application."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from . import env
from . import outputs_v2 as outputs
from .parallel import THREAD_VARIABLES


class RunError(RuntimeError):
    pass


def run(exp, app_file=None, *, directory=None, n_jobs=2):
    if exp not in outputs.APP_FILES:
        raise ValueError("choose E-12, E-24 or E-25")
    if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or not 1 <= n_jobs <= 4:
        raise ValueError("n_jobs must be an integer in 1..4")
    app_file = outputs.APP_FILES[exp] if app_file is None else app_file
    if app_file != outputs.APP_FILES[exp]:
        raise ValueError("marimo app does not match the experiment")
    destination = Path(directory).resolve() if directory else outputs.stage_dir(exp)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = {"experiment": exp, "directory": str(destination), "n_jobs": n_jobs}
    process_env = {
        **os.environ,
        **{variable: "1" for variable in THREAD_VARIABLES},
        outputs.RUN_ENV: json.dumps(request),
    }
    command = [sys.executable, str(env.EXPERIMENTS_DIR / app_file)]
    try:
        result = subprocess.run(
            command, cwd=env.GENRIESZ_ROOT, env=process_env, capture_output=True, text=True
        )
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "stdout.txt").write_text(result.stdout, encoding="utf-8")
        (destination / "stderr.txt").write_text(result.stderr, encoding="utf-8")
        path = destination / "manifest.json"
        if result.returncode or not path.is_file():
            with (destination / "FAILED.txt").open("a", encoding="utf-8") as stream:
                stream.write(
                    f"marimo process exit code: {result.returncode}\n"
                    + result.stdout
                    + "\n"
                    + result.stderr
                )
                if not path.is_file():
                    stream.write("\nThe experiment did not produce manifest.json.\n")
            raise RunError(f"execution failed; see {destination / 'FAILED.txt'}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        outputs.validate_manifest(manifest, exp)
        for name in ("stdout.txt", "stderr.txt"):
            manifest["outputs_sha256"][name] = env.sha256_file(destination / name)
        warning_counts = {}
        for category, message in re.findall(r"\b([A-Za-z]+Warning): ([^\n]*)", result.stderr):
            key = f"{category}: {message}"
            warning_counts[key] = warning_counts.get(key, 0) + 1
        manifest["stderr_warning_counts"] = warning_counts
        outputs.validate_manifest(manifest, exp)
        outputs.write_json(path, manifest)
        return path
    except BaseException as error:
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "FAILED.txt").open("a", encoding="utf-8") as stream:
            stream.write(f"{type(error).__name__}: {error}\n")
        raise



def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one of the three complete paper experiments.")
    parser.add_argument("experiment", choices=[f"E-{exp}" for exp in outputs.APP_FILES])
    parser.add_argument("--n-jobs", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    path = run(int(args.experiment[2:]), directory=args.output, n_jobs=args.n_jobs)
    print(f"finished: {path}")
    return 0
