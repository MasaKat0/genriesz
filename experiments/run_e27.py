"""Execute all twelve E-27 conditions and save scientific results."""

from __future__ import annotations

import argparse
import multiprocessing
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _worker():
    from grrexp import env
    from grrexp.parallel import configure_worker

    env.use_submodule_genriesz()
    configure_worker()


def _compute(task):
    from grrexp import e27

    return e27.raw_rows(task, e27.replicate(task))


def _write_shard(directory, number, rows):
    """Publish a Parquet shard only after its writer has closed successfully."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = directory / f"part-{number:06d}.parquet"
    partial = directory / ("." + path.name)
    table = pa.Table.from_pylist(rows)
    try:
        writer = pq.ParquetWriter(partial, table.schema, compression="zstd")
        write_failed = False
        try:
            writer.write_table(table)
        except BaseException:
            write_failed = True
            raise
        finally:
            try:
                writer.close()
            except BaseException:
                if not write_failed:
                    raise
        partial.rename(path)
    except BaseException:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return path


def run(*, directory, n_jobs=2):
    """Run the fixed design and preserve submitted successes after a compute failure.

    Storage failures propagate without retry; only previously closed shards are
    guaranteed to remain readable after a storage failure.
    """
    if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    os.environ.update({key: "1" for key in THREAD_VARIABLES})
    from grrexp import env

    env.use_submodule_genriesz()
    env.check_environment()
    import pandas as pd
    from grrexp import e27

    destination = Path(directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    raw_directory = destination / "raw"
    raw_directory.mkdir()
    shards = []
    numerical_rows = []
    failure = None
    storage_failed = False
    try:
        with ProcessPoolExecutor(
            max_workers=n_jobs, mp_context=multiprocessing.get_context("spawn"), initializer=_worker
        ) as pool:
            pending = iter(e27.tasks_for())
            active = {}
            while True:
                while failure is None and len(active) < n_jobs:
                    following = next(pending, None)
                    if following is None:
                        break
                    try:
                        active[pool.submit(_compute, following)] = following
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
    if not shards:
        raise RuntimeError("E-27 returned no numerical results")
    raw = (
        pd.concat([pd.read_parquet(path) for path in shards], ignore_index=True)
        .sort_values(["cell", "rep", "arm"])
        .reset_index(drop=True)
    )
    if raw.duplicated(["cell", "rep", "arm"]).any():
        raise ValueError("E-27 raw results contain duplicate replications and losses")
    summary = e27.summarise(raw)
    summary.to_csv(destination / "summary.csv", index=False)
    for name, text in e27.tables(summary).items():
        (destination / (name + ".tex")).write_text(text, encoding="utf-8")
    import matplotlib.pyplot as plt

    for name, figure in e27.figures(summary).items():
        figure.savefig(destination / (name + ".pdf"))
        plt.close(figure)
    return destination / "summary.csv"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run all twelve E-27 conditions, 2000 replications each."
    )
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(run(directory=args.output, n_jobs=args.n_jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
