"""Execute all six E-30 conditions and save complete scientific results."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from run_e27 import THREAD_VARIABLES, _worker, _write_shard
from run_e28 import _publish_derived, _verify_source

EXTRA_PATHS = ("experiments/run_e30.py", "experiments/apps/e30.py")


def _compute(task):
    from grrexp import e30

    return e30.replicate(task)


def run(*, directory, expected_source_commit, n_jobs=4):
    if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    root = Path(__file__).resolve().parent.parent
    source_hashes = _verify_source(root, expected_source_commit, EXTRA_PATHS)
    os.environ.update({key: "1" for key in THREAD_VARIABLES})
    from grrexp import env

    env.use_submodule_genriesz()
    environment = env.check_environment()
    import pandas as pd
    from grrexp import e30

    destination = Path(directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    raw_directory = destination / "raw"
    raw_directory.mkdir()
    manifest = dict(
        experiment="E-30",
        replications=e30.R,
        conditions=len(e30.CELLS),
        source_commit=expected_source_commit.lower(),
        environment=environment,
        source_sha256=source_hashes,
        cells=[list(c) for c in e30.CELLS],
        methods=[list(method) for method in e30.METHODS],
        entropy=20261007,
        outer_folds=e30.K,
        inner_folds=2,
        ratio_cells="ceil(log(M)/2)",
        fitting_fraction=0.5,
        coefficient_bounds=[list(row) for row in e30.DEFAULT_BOUNDS],
        observational_error="Rademacher",
        ratio_penalties="0 and M**(-1/2), shared inner CV",
        regression_families=list(e30.REGRESSIONS),
        regression_grids="isotropic M**(1/5); anisotropic M**(2/5), M**(1/10), M**(1/10)",
        selection_threshold="1/log(log(N+exp(e)))",
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
            pending = iter(e30.tasks_for())
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
    _verify_source(root, expected_source_commit, EXTRA_PATHS)
    raw = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True).sort_values(
        ["cell", "rep", "regression", "loss"]
    )
    summary = e30.summarise(raw)
    _publish_derived(
        destination,
        summary,
        e30.tables(summary),
        e30.figures(summary),
        lambda: _verify_source(root, expected_source_commit, EXTRA_PATHS),
    )
    return destination / "summary.csv"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run six E-30 conditions, 2000 replications each.")
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
