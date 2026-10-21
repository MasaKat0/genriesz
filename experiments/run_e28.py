"""Execute all nine E-28 conditions and save complete scientific results."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import re
import subprocess
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from run_e27 import THREAD_VARIABLES, _worker, _write_shard


def _verify_source(root, expected_commit, extra_paths=()):
    if (
        not isinstance(expected_commit, str)
        or re.fullmatch(r"[0-9a-fA-F]{40}", expected_commit) is None
    ):
        raise ValueError("expected_source_commit must be an exact 40-digit Git commit")
    expected_commit = expected_commit.lower()
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    if current != expected_commit:
        raise RuntimeError("HEAD differs from the expected reviewed source commit")
    prefixes = ("src/genriesz", "experiments/grrexp")
    pins = (
        "pyproject.toml",
        "experiments/run_e28.py",
        "experiments/run_e27.py",
        "experiments/pyproject.toml",
        "experiments/uv.lock",
        "experiments/.python-version",
    )
    tracked = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", expected_commit, "--", *prefixes],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    committed_paths = {p for p in tracked if p.endswith(".py")}
    current_paths = {
        str(p.relative_to(root)) for prefix in prefixes for p in (root / prefix).rglob("*.py")
    }
    if current_paths != committed_paths:
        raise RuntimeError("Python source files differ from the expected reviewed source commit")
    hashes = {}
    for name in sorted(committed_paths | set(pins) | set(extra_paths)):
        committed = subprocess.run(
            ["git", "show", f"{expected_commit}:{name}"],
            cwd=root,
            capture_output=True,
            check=True,
        ).stdout
        actual = (root / name).read_bytes()
        if committed != actual:
            raise RuntimeError(
                f"scientific input differs from the expected reviewed commit: {name}"
            )
        hashes[name] = hashlib.sha256(actual).hexdigest()
    return hashes


def _publish_derived(directory, summary, tables, figures, verify_source):
    """Close all companions before publishing them, with the summary published last."""
    import matplotlib.pyplot as plt

    staged = []
    try:
        for name, table in tables.items():
            target = directory / (name + ".tex")
            partial = target.with_name("." + target.name)
            staged.append((partial, target))
            partial.write_text(table, encoding="utf-8")
        for name, figure in figures.items():
            target = directory / (name + ".pdf")
            partial = target.with_name("." + target.name)
            staged.append((partial, target))
            figure.savefig(partial, format="pdf")
        target = directory / "summary.csv"
        partial = target.with_name("." + target.name)
        staged.append((partial, target))
        summary.to_csv(partial, index=False)
        verify_source()
        for partial, target in staged:
            os.replace(partial, target)
    finally:
        for figure in figures.values():
            plt.close(figure)
        for partial, _ in staged:
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                pass


def _compute(task):
    from grrexp import e28

    return e28.replicate(task)


def run(*, directory, expected_source_commit, n_jobs=4):
    if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    root = Path(__file__).resolve().parent.parent
    source_hashes = _verify_source(root, expected_source_commit)
    os.environ.update({key: "1" for key in THREAD_VARIABLES})
    from grrexp import env

    env.use_submodule_genriesz()
    environment = env.check_environment()
    import pandas as pd
    from grrexp import e28

    destination = Path(directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    raw_directory = destination / "raw"
    raw_directory.mkdir()
    manifest = dict(
        experiment="E-28",
        replications=e28.R,
        conditions=len(e28.CELLS),
        source_commit=expected_source_commit.lower(),
        environment=environment,
        source_sha256=source_hashes,
        cells=[list(c) for c in e28.CELLS],
        upper=e28.UPPER,
        penalties=e28.PENALTIES.tolist(),
    )
    manifest_partial = destination / ".manifest.json"
    manifest_partial.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(manifest_partial, destination / "manifest.json")
    shards, numerical_rows = [], []
    failure, storage_failed = None, False
    try:
        with ProcessPoolExecutor(
            max_workers=n_jobs, mp_context=multiprocessing.get_context("spawn"), initializer=_worker
        ) as pool:
            pending = iter(e28.tasks_for())
            active = {}
            while True:
                while failure is None and len(active) < n_jobs:
                    task = next(pending, None)
                    if task is None:
                        break
                    try:
                        active[pool.submit(_compute, task)] = task
                    except BaseException as error:
                        failure = error
                if not active:
                    break
                completed, _ = wait(active, return_when=FIRST_COMPLETED)
                completed_rows = []
                for future in tuple(active):
                    if future not in completed:
                        continue
                    active.pop(future)
                    try:
                        completed_rows.extend(future.result())
                    except BaseException as error:
                        if failure is None:
                            failure = error
                if not storage_failed:
                    numerical_rows.extend(completed_rows)
                    if len(numerical_rows) >= 256:
                        rows, numerical_rows = numerical_rows, []
                        try:
                            shards.append(_write_shard(raw_directory, len(shards) + 1, rows))
                        except BaseException as error:
                            storage_failed = True
                            if failure is None:
                                failure = error
    except BaseException as error:
        if failure is None:
            failure = error
    finally:
        if numerical_rows and not storage_failed:
            try:
                shards.append(_write_shard(raw_directory, len(shards) + 1, numerical_rows))
            except BaseException as error:
                if failure is None:
                    failure = error
    if failure is not None:
        raise failure
    _verify_source(root, expected_source_commit)
    raw = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True).sort_values(
        ["cell", "rep", "exponent_mode", "regression", "loss"]
    )
    summary = e28.summarise(raw)
    _publish_derived(
        destination,
        summary,
        e28.tables(summary),
        e28.figures(summary),
        lambda: _verify_source(root, expected_source_commit),
    )
    return destination / "summary.csv"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run nine E-28 conditions, 2000 replications each."
    )
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    args = parser.parse_args(argv)
    print(
        run(
            directory=args.output,
            n_jobs=args.n_jobs,
            expected_source_commit=args.expected_source_commit,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
