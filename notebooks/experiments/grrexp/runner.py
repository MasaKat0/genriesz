"""The only way to start a registered run (registration §0.2, §1.2, §1.7).

    python3 notebooks/experiments/run_registered.py E-16 stage0 14_E16_exact_examples.ipynb

The runner

1. checks the platform and lock, the data snapshots, and that the genriesz
   checkout is clean at a commit that contains the notebook;
2. for Stage 1, checks that the earlier stages (Stage 0.5, and Stage 0 where
   §0.2 freezes predictions) are committed, unchanged and valid, and records
   their manifest digests;
3. reserves the stage directory and writes ``RUNNING.json`` with a token;
4. executes the committed notebook in a fresh kernel whose environment has the
   single-thread variables and the token before Python starts, saving the
   executed copy as ``executed.ipynb`` in the stage directory (the tracked
   notebook is never written);
5. checks that the checkout did not change, that every module the kernel loaded
   from this checkout equals its blob at the commit, and that the stage
   directory holds exactly the outputs the kernel registered; then writes and
   validates ``manifest.json`` and removes ``RUNNING.json``.

A failed execution leaves the directory in place with ``FAILED.txt``; it is
archived (:func:`grrexp.outputs.archive_run`), never deleted, before a re-run.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import env, outputs
from .parallel import THREAD_VARIABLES


class RunError(RuntimeError):
    pass


def _committed_and_clean(path: Path) -> None:
    rel = path.relative_to(env.GENRIESZ_ROOT).as_posix()
    status = env._git(
        env.GENRIESZ_ROOT, "status", "--porcelain", "--untracked-files=all", "--", rel
    )
    if status.strip():
        raise RunError(f"{rel} has uncommitted changes")
    if not env._git(env.GENRIESZ_ROOT, "ls-files", "--", f"{rel}/manifest.json").strip():
        raise RunError(f"{rel}/manifest.json is not committed")


def check_prerequisites(exp: int, stage: str, commit: str) -> tuple[dict, list | None]:
    """Validate the committed earlier stages; return their digests and the pilot cells."""
    out, pilot_cells = {}, None
    for prior in outputs.prerequisite_stages(exp, stage):
        d = outputs.stage_dir(exp, prior)
        if not (d / "manifest.json").is_file():
            raise RunError(f"Stage 1 of E-{exp} needs a finished {prior}")
        _committed_and_clean(d)
        raw = (d / "manifest.json").read_bytes()
        manifest = json.loads(raw)
        try:
            outputs.validate_manifest(manifest, exp, prior)
        except ValueError as e:
            raise RunError(f"{prior} manifest is invalid: {e}") from e
        for name, digest in manifest["outputs_sha256"].items():
            if env.sha256_file(d / name) != digest:
                raise RunError(f"{prior}/{name} does not match its manifest")
        out[prior] = {"manifest_sha256": hashlib.sha256(raw).hexdigest(), "commit": commit}
        if prior == "stage0.5":
            pilot_cells = manifest["cells"]
    return out, pilot_cells


def run(exp: int, stage: str, notebook: str) -> Path:
    nb_path = env.EXPERIMENTS_DIR / notebook
    if nb_path.parent != env.EXPERIMENTS_DIR or nb_path.suffix != ".ipynb":
        raise RunError("give the file name of a notebook in notebooks/experiments/")
    environment = env.check_environment()
    data = env.verify_data(env.REGISTERED_DATA.get(exp, ()))
    state = env.checkout_state(nb_path)
    prerequisites, pilot_cells = check_prerequisites(exp, stage, state["sha"])

    d = outputs.stage_dir(exp, stage)
    d.parent.mkdir(parents=True, exist_ok=True)
    d.mkdir()  # reservation: fails if this stage was run before
    token = secrets.token_hex(16)
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    running = {"token": token, "started_utc": started.isoformat(), "genriesz": state["sha"]}
    with open(d / "RUNNING.json", "x", encoding="utf-8") as f:
        json.dump(running, f)

    kernel_env = dict(os.environ)
    kernel_env.update({k: "1" for k in THREAD_VARIABLES})
    kernel_env[outputs.RUN_ENV] = json.dumps({"exp": exp, "stage": stage, "token": token})
    cmd = [
        sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
        "--ExecutePreprocessor.timeout=-1", "--output", "executed.ipynb",
        "--output-dir", str(d), str(nb_path),
    ]  # fmt: skip
    proc = subprocess.run(
        cmd, cwd=env.EXPERIMENTS_DIR, env=kernel_env, capture_output=True, text=True
    )
    if proc.returncode != 0:
        (d / "FAILED.txt").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
        raise RunError(f"execution failed; see {d / 'FAILED.txt'}")

    kernel = json.loads((d / "kernel.json").read_text(encoding="utf-8"))
    if env.checkout_state(nb_path) != state:
        raise RunError("the checkout changed during the run")
    try:
        env.check_modules_at_head(kernel["loaded_modules"], env.head_blobs())
    except RuntimeError as e:
        raise RunError(str(e)) from e
    present = sorted(
        p.relative_to(d).as_posix()
        for p in d.rglob("*")
        if p.is_file() and p.relative_to(d).as_posix() not in outputs.RUNNER_FILES
    )
    if present != kernel["outputs"]:
        raise RunError(
            f"stage directory holds {present}, the kernel registered {kernel['outputs']}"
        )
    if pilot_cells is not None and kernel["cells"] != pilot_cells:
        raise RunError("Stage 1 cells differ from the cells the pilot covered")

    freeze, freeze_hash = env.pip_freeze()
    manifest = {
        "experiment": f"E-{exp}",
        "stage": stage,
        "parent_repository": env.parent_state(),
        "genriesz": {**state, "loaded_modules": kernel["loaded_modules"]},
        "environment": environment,
        "threads": kernel["threads"],
        "pip_freeze": freeze,
        "pip_freeze_sha256": freeze_hash,
        "blas": env.blas_config(),
        "cpu": env.cpu_description(),
        "data_sha256": data,
        "entropy": outputs.STAGE_ENTROPY[stage],
        "R": kernel["R"],
        "n": kernel["n"],
        "cells": kernel["cells"],
        "prerequisites": prerequisites,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - t0,
        "warnings": kernel["warnings"],
        "failures": kernel["failures"],
        "outputs_sha256": {rel: env.sha256_file(d / rel) for rel in present},
        "executed_notebook_sha256": env.sha256_file(d / "executed.ipynb"),
    }
    if "parallel_identity" in kernel:
        manifest["parallel_identity"] = kernel["parallel_identity"]
    clash = set(kernel["extra"]) & set(manifest)
    if clash:
        raise RunError(f"extra keys clash with registered fields: {sorted(clash)}")
    manifest.update(kernel["extra"])
    try:
        outputs.validate_manifest(manifest, exp, stage)
    except ValueError as e:
        raise RunError(f"invalid manifest: {e}") from e
    with open(d / "manifest.json", "x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
    (d / "RUNNING.json").unlink()
    return d / "manifest.json"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Start a registered run of E-12..E-24.")
    parser.add_argument("experiment", help="E-12 ... E-24")
    parser.add_argument("stage", choices=tuple(outputs.STAGE_DIRS))
    parser.add_argument("notebook", help="file name in notebooks/experiments/")
    args = parser.parse_args(argv)
    if not (args.experiment.startswith("E-") and args.experiment[2:].isdigit()):
        parser.error("experiment must look like E-16")
    try:
        path = run(int(args.experiment[2:]), args.stage, args.notebook)
    except (RunError, RuntimeError, ValueError, FileExistsError) as e:
        print(f"run stopped: {e}", file=sys.stderr)
        return 1
    print(f"finished: {path}")
    return 0
