"""Process-level parallelism of registration §1.2.

Workers are spawned (not forked) with ``OMP_NUM_THREADS``,
``OPENBLAS_NUM_THREADS`` and ``VECLIB_MAXIMUM_THREADS`` set to 1 in their
environment before Python starts, so the BLAS of every worker is single-threaded
from its first import. Every worker, and a serial run, then sets
``torch.set_num_threads(1)`` and ``torch.use_deterministic_algorithms(True)`` and
verifies with threadpoolctl that every loaded BLAS/OpenMP runtime uses one
thread before each task.

``run_tasks`` returns results in the order of ``tasks`` whatever the completion
order, so a run with 1 worker and a run with several workers can be compared bit for
bit (Stage 0.5, :func:`verify_parallel_identity`).

The number of workers can be lowered or raised while a run is going on: when the
environment variable ``GRREXP_MAX_WORKERS_FILE`` names a file holding an integer,
the parent reads it each time a task finishes and keeps at most that many tasks
(and worker processes) running, never more than ``n_jobs``. A worker that is no
longer needed finishes its current task and exits; nothing is cut short, so the
results do not depend on it (author, 2026-10-17: reduce load one or two processes
at a time instead of stopping a run).
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import pickle
import queue
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from typing import Any

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
MAX_WORKERS_FILE_ENV = "GRREXP_MAX_WORKERS_FILE"


def desired_workers(n_jobs: int) -> int:
    """The number of workers to keep running: ``n_jobs``, or the integer in the control file
    named by ``GRREXP_MAX_WORKERS_FILE`` (clamped to ``1..n_jobs``) when it can be read."""
    path = os.environ.get(MAX_WORKERS_FILE_ENV)
    if not path:
        return n_jobs
    try:
        with open(path, encoding="utf-8") as f:
            value = int(f.read().strip())
    except (OSError, ValueError):
        return n_jobs
    return max(1, min(n_jobs, value))


@contextmanager
def _single_thread_environment():
    saved = {k: os.environ.get(k) for k in THREAD_VARIABLES}
    os.environ.update({k: "1" for k in THREAD_VARIABLES})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def configure_worker() -> None:
    """Pin the thread counts inside a worker or a serial run, then verify them.

    torch is registered (§1.7), so an import failure stops the run.
    """
    os.environ.update({k: "1" for k in THREAD_VARIABLES})
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    verify_single_thread()


def verify_single_thread() -> None:
    """Raise unless the variables, every loaded BLAS/OpenMP runtime and torch use 1 thread."""
    if not all(os.environ.get(k) == "1" for k in THREAD_VARIABLES):
        raise RuntimeError("single-thread environment variables are not set")
    from threadpoolctl import threadpool_info

    busy = [
        (i["internal_api"], i["num_threads"]) for i in threadpool_info() if i["num_threads"] != 1
    ]
    if busy:
        raise RuntimeError(f"thread pools not pinned to 1: {busy}")
    import torch

    if torch.get_num_threads() != 1 or not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("torch is not single-threaded and deterministic")


def _call(fn: Callable[[Any], Any], task: Any) -> Any:
    verify_single_thread()
    return fn(task)


def run_tasks(fn: Callable[[Any], Any], tasks: Sequence[Any], n_jobs: int) -> list[Any]:
    """Apply ``fn`` to each task; ``fn`` must be importable (defined in a module).

    ``n_jobs=1`` runs in this process after :func:`configure_worker`; the kernel
    must have been started with the single-thread variables, or the check of the
    loaded BLAS runtime fails. Exceptions propagate and stop the run.
    """
    tasks = list(tasks)
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1")
    if not tasks:
        raise ValueError("no tasks")
    if n_jobs == 1:
        configure_worker()
        return [_call(fn, t) for t in tasks]
    with _single_thread_environment():
        return _run_elastic(fn, tasks, n_jobs, mp.get_context("spawn"))


def _worker_loop(fn, task_q, result_q, wid: int) -> None:
    configure_worker()
    while True:
        item = task_q.get()
        if item is None:
            return
        i, task = item
        try:
            result_q.put((i, True, _call(fn, task), wid))
        except BaseException as e:  # sent to the parent, which re-raises it
            try:
                pickle.dumps(e)
            except Exception:
                e = RuntimeError(f"{type(e).__name__}: {e}")
            result_q.put((i, False, e, wid))
            return


def _run_elastic(fn, tasks, n_jobs, ctx):
    """Dispatch one task at a time to spawned workers; keep ``desired_workers`` running."""
    result_q = ctx.Queue()
    workers: dict[int, tuple[Any, Any]] = {}
    results: list[Any] = [None] * len(tasks)
    state = {"next": 0, "wid": 0}

    def start() -> int:
        wid = state["wid"]
        state["wid"] += 1
        q = ctx.Queue()
        proc = ctx.Process(target=_worker_loop, args=(fn, q, result_q, wid), daemon=True)
        proc.start()
        workers[wid] = (proc, q)
        return wid

    def assign(wid: int) -> None:
        i = state["next"]
        state["next"] += 1
        workers[wid][1].put((i, tasks[i]))

    def retire(wid: int) -> None:
        proc, q = workers.pop(wid)
        q.put(None)
        proc.join()

    busy: set[int] = set()
    done = 0
    try:
        for _ in range(min(desired_workers(n_jobs), len(tasks))):
            w = start()
            assign(w)
            busy.add(w)
        while done < len(tasks):
            try:
                i, ok, payload, w = result_q.get(timeout=5)
            except queue.Empty:
                dead = [k for k in busy if not workers[k][0].is_alive()]
                if dead:
                    raise RuntimeError(f"worker process(es) {dead} exited without returning a result")
                continue
            if not ok:
                raise payload
            results[i] = payload
            done += 1
            busy.discard(w)
            target = desired_workers(n_jobs)
            if state["next"] < len(tasks) and len(busy) < target:
                assign(w)
                busy.add(w)
            else:
                retire(w)
            while state["next"] < len(tasks) and len(busy) < target:
                w = start()
                assign(w)
                busy.add(w)
    finally:
        for proc, q in workers.values():
            if proc.is_alive():
                q.put(None)
        for proc, _ in list(workers.values()):
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
    return results


def result_digest(results: Iterable[Any]) -> str:
    """SHA-256 of the pickled results (protocol 5); equal digests mean equal bytes."""
    h = hashlib.sha256()
    for r in results:
        h.update(pickle.dumps(r, protocol=5))
    return h.hexdigest()


def assert_bit_identical(serial: Sequence[Any], parallel: Sequence[Any], *, expected: int) -> str:
    """Raise unless both runs returned ``expected`` (> 0) results with equal bytes."""
    if expected < 1:
        raise ValueError("expected must be positive")
    if len(serial) != expected or len(parallel) != expected:
        raise AssertionError(
            f"expected {expected} results, got {len(serial)} serial and {len(parallel)} parallel"
        )
    a, b = result_digest(serial), result_digest(parallel)
    if a != b:
        raise AssertionError(f"serial and parallel runs differ: {a} != {b}")
    return a


def verify_parallel_identity(
    fn: Callable[[Any], Any], tasks: Sequence[Any], n_jobs: int = 4
) -> str:
    """Stage 0.5: run every task with 1 and with ``n_jobs`` workers and compare bytes."""
    tasks = list(tasks)
    serial = run_tasks(fn, tasks, 1)
    parallel = run_tasks(fn, tasks, n_jobs)
    return assert_bit_identical(serial, parallel, expected=len(tasks))


def thread_record(variables_at_start: dict[str, str | None]) -> dict:
    """What the manifest records about threads (§1.2).

    threadpoolctl observes OpenBLAS, MKL, BLIS and OpenMP pools but not Apple
    Accelerate; for Accelerate the evidence is ``VECLIB_MAXIMUM_THREADS=1`` in the
    environment the runner gave the kernel before any import, which is recorded
    separately from the observed pools.
    """
    import numpy as np
    import torch
    from threadpoolctl import threadpool_info

    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    return {
        "variables": dict(sorted(variables_at_start.items())),
        "blas_backend": str(blas.get("name", "")),
        "observed_pools": [
            {"internal_api": i["internal_api"], "num_threads": i["num_threads"]}
            for i in threadpool_info()
        ],
        "torch": {
            "num_threads": torch.get_num_threads(),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
        },
    }
